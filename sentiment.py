"""Hugging Face sentiment inference."""

from __future__ import annotations

from typing import Any


DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def select_inference_device() -> tuple[int, str]:
    """Select CUDA GPU when PyTorch can access one; otherwise use CPU.

    Hugging Face Transformers uses device=0 for the first CUDA GPU and
    device=-1 for CPU. The returned label is suitable for the UI.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - requirements install PyTorch
        return -1, "CPU (PyTorch не найден)"

    try:
        cuda_available = torch.cuda.is_available()
    except Exception:  # pragma: no cover - defensive driver/runtime fallback
        return -1, "CPU (CUDA недоступна)"

    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - defensive driver/runtime fallback
            gpu_name = "CUDA GPU"
        return 0, f"GPU / CUDA — {gpu_name}"

    return -1, "CPU (CUDA GPU не обнаружен)"


class SentimentAnalyzer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_index: int | None = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - depends on optional runtime packages
            raise RuntimeError(
                "Не установлен пакет transformers. Выполните: pip install -r requirements.txt"
            ) from exc

        self.model_name = model_name
        detected_device, self.device_label = select_inference_device()
        self.device_index = detected_device if device_index is None else device_index
        if self.device_index == -1:
            self.device_label = "CPU"

        try:
            self.classifier = pipeline(
                task="text-classification",
                model=model_name,
                tokenizer=model_name,
                device=self.device_index,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось загрузить модель {model_name}. Проверьте подключение к интернету "
                "и доступность модели Hugging Face."
            ) from exc

        self.id2label = getattr(self.classifier.model.config, "id2label", {}) or {}

    @staticmethod
    def _normalize_label(label: str) -> str:
        normalized = label.lower()
        if "neg" in normalized:
            return "negative"
        if "neu" in normalized:
            return "neutral"
        if "pos" in normalized:
            return "positive"
        return normalized

    def predict(self, texts: list[str], batch_size: int = 8) -> list[dict[str, Any]]:
        cleaned = [str(text).replace("\x00", " ").strip()[:5000] for text in texts]
        if not cleaned:
            return []

        predictions = self.classifier(
            cleaned,
            batch_size=batch_size,
            truncation=True,
            max_length=512,
        )
        # A list is returned for a list of texts. Keep a defensive single-result branch.
        if predictions and isinstance(predictions[0], dict):
            items = predictions
        else:  # pragma: no cover - defensive compatibility branch
            items = [predictions]

        results: list[dict[str, Any]] = []
        for prediction in items:
            raw_label = str(prediction.get("label", "neutral"))
            if raw_label.upper().startswith("LABEL_"):
                try:
                    label_id = int(raw_label.split("_")[-1])
                    raw_label = str(self.id2label.get(label_id, self.id2label.get(str(label_id), raw_label)))
                except ValueError:
                    pass
            sentiment = self._normalize_label(raw_label)
            if sentiment not in {"negative", "neutral", "positive"}:
                # Cardiff NLP's model card reports named labels; unknown labels are not guessed.
                sentiment = "neutral"
            results.append({"sentiment": sentiment, "score": float(prediction.get("score", 0.0))})
        return results
