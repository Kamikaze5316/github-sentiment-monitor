from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from analytics import (
    SENTIMENT_LABELS,
    activity_by_time,
    cluster_messages,
    detect_negative_spikes,
    prepare_frame,
)
from database import Database
from demo_data import build_demo_messages
from github_client import GitHubAPIError, GitHubClient, sync_repository
from sentiment import DEFAULT_MODEL, SentimentAnalyzer


load_dotenv()

st.set_page_config(
    page_title="GitHub Sentiment Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.getenv("DATABASE_PATH", "data/messages.db")
MODEL_NAME = os.getenv("SENTIMENT_MODEL", DEFAULT_MODEL)
db = Database(DB_PATH)


@st.cache_resource(show_spinner=False)
def get_analyzer(model_name: str) -> SentimentAnalyzer:
    return SentimentAnalyzer(model_name)


def parse_repositories(raw: str) -> list[str]:
    repositories: list[str] = []
    for line in raw.replace(",", "\n").splitlines():
        repo = line.strip().strip("/")
        if not repo:
            continue
        parts = repo.split("/")
        if len(parts) != 2 or any(part in {".", ".."} for part in parts):
            raise ValueError(f"Некорректное имя репозитория: {repo}. Формат: owner/repository")
        if repo not in repositories:
            repositories.append(repo)
    return repositories


def run_sync(repositories: list[str], token: str, lookback_days: int, max_pages: int) -> tuple[int, list[str]]:
    if not repositories:
        return 0, ["Укажите хотя бы один репозиторий в формате owner/repository."]

    client = GitHubClient(token=token or None)
    changed_messages: list[dict] = []
    logs: list[str] = []
    for repo in repositories:
        try:
            result = sync_repository(
                client=client,
                database=db,
                repo=repo,
                lookback_days=lookback_days,
                max_issue_pages=max_pages,
                max_comment_pages=3,
            )
            logs.append(
                f"{repo}: Issues просмотрено — {result['issues_seen']}, "
                f"комментариев получено — {result['comments_seen']}, "
                f"новых/изменённых сообщений — {result['messages_changed']}, "
                f"страниц получено — {result['pages_fetched']}"
                + ("; продолжение загрузки запланировано" if result["more_pages_pending"] else "")
            )
        except GitHubAPIError as exc:
            logs.append(f"{repo}: ошибка GitHub API — {exc}")
        except Exception as exc:  # Keep one bad repository from blocking others.
            logs.append(f"{repo}: непредвиденная ошибка — {type(exc).__name__}: {exc}")

    # Classify only text that is new or has changed since the previous sync.
    raw_df = db.load_messages()
    pending = raw_df[raw_df["sentiment"].isna()] if not raw_df.empty else pd.DataFrame()
    if not pending.empty:
        try:
            analyzer = get_analyzer(MODEL_NAME)
            with st.spinner(f"Определение тональности: {len(pending)} сообщений..."):
                predictions = analyzer.predict(pending["text"].tolist())
            updates = []
            for (_, message), prediction in zip(pending.iterrows(), predictions):
                updates.append(
                    {
                        "message_id": message["message_id"],
                        "sentiment": prediction["sentiment"],
                        "score": prediction["score"],
                    }
                )
            db.update_sentiments(updates)
            logs.append(f"Модель обработала сообщений: {len(updates)}.")
        except Exception as exc:
            logs.append(
                "Тональность пока не рассчитана: "
                f"{type(exc).__name__}: {exc}. Повторите синхронизацию после устранения ошибки."
            )
    return len(pending), logs


def show_overview(frame: pd.DataFrame) -> None:
    total = len(frame)
    negative = int((frame["sentiment"] == "negative").sum())
    positive = int((frame["sentiment"] == "positive").sum())
    neutral = int((frame["sentiment"] == "neutral").sum())
    negative_share = (negative / total * 100) if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего сообщений", f"{total:,}".replace(",", " "))
    c2.metric("Отрицательные", f"{negative:,}".replace(",", " "), f"{negative_share:.1f}% от всех")
    c3.metric("Положительные", f"{positive:,}".replace(",", " "))
    c4.metric("Нейтральные", f"{neutral:,}".replace(",", " "))


def show_dashboard(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("Нет сообщений, соответствующих выбранным фильтрам.")
        return

    show_overview(frame)
    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Активность и тональность во времени")
        activity = activity_by_time(frame, "1h")
        if not activity.empty:
            long_activity = activity.melt(
                id_vars="time",
                value_vars=["total", "negative", "positive", "neutral"],
                var_name="series",
                value_name="messages",
            )
            labels = {"total": "Все сообщения", "negative": "Негативные", "positive": "Позитивные", "neutral": "Нейтральные"}
            long_activity["series"] = long_activity["series"].map(labels)
            fig = px.line(long_activity, x="time", y="messages", color="series", markers=True)
            fig.update_layout(legend_title_text="Тип сообщений", xaxis_title="Время (UTC)", yaxis_title="Количество")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Распределение тональности")
        sentiment_counts = (
            frame["sentiment"].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
            .rename_axis("sentiment").reset_index(name="messages")
        )
        sentiment_counts["label"] = sentiment_counts["sentiment"].map(SENTIMENT_LABELS)
        pie = px.pie(sentiment_counts, names="label", values="messages", hole=0.48)
        pie.update_layout(showlegend=True, legend_title_text="Тональность")
        st.plotly_chart(pie, use_container_width=True)

    st.subheader("Всплески отрицательной активности")
    spikes = detect_negative_spikes(frame)
    detected = spikes[spikes["is_spike"]].copy() if not spikes.empty else spikes
    if detected.empty:
        st.success("На выбранном интервале всплески негатива по текущим критериям не обнаружены.")
    else:
        st.warning(f"Обнаружено временных интервалов со всплеском негатива: {len(detected)}")
        display_spikes = detected.head(10).copy()
        display_spikes["time"] = display_spikes["time"].dt.strftime("%Y-%m-%d %H:%M UTC")
        display_spikes["negative_share"] = (display_spikes["negative_share"] * 100).round(1).astype(str) + "%"
        display_spikes = display_spikes.rename(columns={
            "time": "Время начала", "total": "Всего сообщений", "negative": "Негативных",
            "negative_share": "Доля негатива", "baseline_mean": "Среднее по базе", "threshold": "Порог", "is_spike": "Всплеск",
        })
        st.dataframe(display_spikes, use_container_width=True, hide_index=True)
    with st.expander("Как определяется всплеск?"):
        st.write(
            "Данные разбиваются на 15-минутные интервалы. Для каждого интервала сравнивается "
            "число отрицательных сообщений с историческим средним и стандартным отклонением "
            "предыдущих 24 интервалов. Всплеск фиксируется, если найдено не менее 3 отрицательных "
            "сообщений, их доля составляет не менее 60%, а количество превышает рассчитанный порог. "
            "Это эвристика для мониторинга, а не статистически подтверждённый инцидент."
        )

    st.subheader("Группировка похожих обращений")
    clusters = cluster_messages(frame)
    if clusters.empty:
        st.caption("Для кластеризации нужно не менее четырёх сообщений и повторяющиеся текстовые признаки.")
    else:
        shown = clusters.copy()
        shown["negative_share"] = (shown["negative_share"] * 100).round(1).astype(str) + "%"
        shown = shown.rename(columns={
            "cluster": "Группа", "keywords": "Ключевые слова", "messages": "Сообщений",
            "negative_share": "Доля негатива", "examples": "Примеры сообщений",
        })
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.caption("Группы формируются автоматически методом TF-IDF + KMeans. Названия групп — ключевые слова, а не заранее заданные категории.")

    st.subheader("Сообщения")
    visible = frame.copy()
    visible["created_at"] = visible["created_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
    visible = visible.rename(columns={
        "created_at": "Дата", "repo": "Репозиторий", "message_type_ru": "Тип",
        "issue_number": "№ Issue", "title": "Заголовок", "text_preview": "Текст",
        "username": "Автор", "sentiment_ru": "Тональность", "sentiment_score": "Уверенность", "html_url": "Ссылка",
    })
    columns = ["Дата", "Репозиторий", "Тип", "№ Issue", "Заголовок", "Текст", "Автор", "Тональность", "Уверенность", "Ссылка"]
    st.dataframe(visible[columns], use_container_width=True, hide_index=True)
    csv_data = frame.drop(columns=["labels_json", "labels"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
    st.download_button("Скачать CSV", data=csv_data, file_name="github_messages.csv", mime="text/csv")


st.title("📡 GitHub Sentiment Monitor")
st.caption("Потоковый мониторинг Issues и комментариев публичных репозиториев: тональность, активность и всплески негатива.")

with st.sidebar:
    st.header("Источники данных")
    repositories_raw = st.text_area(
        "Репозитории GitHub (через запятую или с новой строки)",
        value=os.getenv("GITHUB_REPOSITORIES", "pallets/flask"),
        height=100,
        help="Указывайте каждый репозиторий в формате owner/repository. Несколько репозиториев можно разделять запятыми или переносами строк. Пример: pallets/flask,streamlit/streamlit,psf/requests",
    )
    github_token = st.text_input(
        "GitHub token (необязательно)",
        value=os.getenv("GITHUB_TOKEN", ""),
        type="password",
        help="Для публичных репозиториев можно работать без токена, но с ним выше лимит запросов.",
    )
    lookback_days = st.slider("Глубина первой загрузки, дней", min_value=1, max_value=90, value=14)
    max_pages = st.slider("Страниц Issues на репозиторий за цикл", min_value=1, max_value=5, value=2)
    st.divider()
    st.header("Потоковая обработка")
    auto_refresh = st.toggle("Автоматическое обновление", value=False)
    refresh_seconds = st.select_slider("Интервал обновления", options=[30, 60, 120, 300], value=60, format_func=lambda x: f"{x} сек.")
    st.caption("Автообновление работает, пока вкладка приложения открыта.")
    st.divider()
    st.caption(f"Модель тональности: `{MODEL_NAME}`")
    st.caption(f"SQLite: `{DB_PATH}`")

try:
    repositories = parse_repositories(repositories_raw)
except ValueError as exc:
    repositories = []
    st.sidebar.error(str(exc))

run_every = f"{refresh_seconds}s" if auto_refresh and repositories else None


@st.fragment(run_every=run_every)
def live_dashboard() -> None:
    manual_sync_performed = False
    if st.button("🔄 Синхронизировать сейчас", type="primary", use_container_width=True):
        if not repositories:
            st.error("Укажите хотя бы один репозиторий в формате owner/repository.")
        else:
            _, logs = run_sync(repositories, github_token, lookback_days, max_pages)
            st.session_state["last_sync_logs"] = logs
            st.session_state["last_sync_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            st.session_state["last_auto_sync"] = datetime.now(timezone.utc)
            manual_sync_performed = True

    if auto_refresh and repositories and not manual_sync_performed:
        # The fragment reruns at the configured interval while this browser session stays active.
        last_auto = st.session_state.get("last_auto_sync")
        now = datetime.now(timezone.utc)
        if last_auto is None or (now - last_auto).total_seconds() >= refresh_seconds - 2:
            _, logs = run_sync(repositories, github_token, lookback_days, max_pages)
            st.session_state["last_sync_logs"] = logs
            st.session_state["last_sync_at"] = now.strftime("%Y-%m-%d %H:%M:%S UTC")
            st.session_state["last_auto_sync"] = now

    if st.session_state.get("last_sync_logs"):
        with st.expander("Результат последней синхронизации", expanded=False):
            st.caption(st.session_state.get("last_sync_at", ""))
            for log in st.session_state["last_sync_logs"]:
                st.write(f"• {log}")

    raw_frame = db.load_messages()
    frame = prepare_frame(raw_frame)
    if not frame.empty:
        st.caption(f"Последнее обновление данных: {st.session_state.get('last_sync_at', 'ещё не выполнялась')} · Все даты отображаются в UTC.")
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.4])
        with filter_col1:
            repo_options = ["Все репозитории"] + sorted(frame["repo"].dropna().unique().tolist())
            selected_repo = st.selectbox("Репозиторий", repo_options)
        with filter_col2:
            period = st.selectbox("Период", ["Всё время", "Последние 24 часа", "Последние 7 дней", "Последние 30 дней"], index=2)
        with filter_col3:
            selected_sentiments = st.multiselect(
                "Тональность", ["negative", "neutral", "positive"],
                default=["negative", "neutral", "positive"],
                format_func=lambda value: SENTIMENT_LABELS[value],
            )
        filtered = frame.copy()
        if selected_repo != "Все репозитории":
            filtered = filtered[filtered["repo"] == selected_repo]
        if period != "Всё время":
            days = {"Последние 24 часа": 1, "Последние 7 дней": 7, "Последние 30 дней": 30}[period]
            filtered = filtered[filtered["created_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
        filtered = filtered[filtered["sentiment"].isin(selected_sentiments)]
        show_dashboard(filtered)
    else:
        st.info("База данных пока пуста. Выполните синхронизацию или загрузите демонстрационные данные ниже.")


live_dashboard()

if st.button("Загрузить демонстрационные данные", help="Добавит синтетические примеры в SQLite, чтобы проверить графики и аналитику без GitHub API."):
    inserted = db.insert_demo_messages(build_demo_messages())
    st.success(f"Добавлено демонстрационных сообщений: {inserted}. Перезагрузите отображение кнопкой синхронизации или обновите страницу.")
    st.rerun()

with st.expander("Ограничения и методика"):
    st.markdown(
        """
        - GitHub REST API опрашивается периодически. Это имитация потоковой обработки, а не WebSocket-подписка.
        - На первой синхронизации загружаются обращения, обновлённые за выбранную глубину истории; далее используются временные курсоры.
        - Pull Requests исключаются, так как endpoint Issues GitHub возвращает и Issues, и Pull Requests.
        - Модель обучалась преимущественно на сообщениях из социальных сетей. Её оценки для технических Issue следует проверять на размеченной выборке.
        - Всплеск негатива — эвристический сигнал для дальнейшей проверки, а не доказательство ухудшения качества проекта.
        """
    )
