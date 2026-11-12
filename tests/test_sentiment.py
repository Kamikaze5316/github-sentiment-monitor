import sys
from types import SimpleNamespace

from sentiment import select_inference_device


def test_select_inference_device_uses_cpu_when_cuda_unavailable(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device_index, device_label = select_inference_device()

    assert device_index == -1
    assert device_label.startswith("CPU")


def test_select_inference_device_uses_cuda_when_available(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "NVIDIA T4",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device_index, device_label = select_inference_device()

    assert device_index == 0
    assert "NVIDIA T4" in device_label


def test_sentiment_analyzer_passes_cuda_device_to_transformers(monkeypatch):
    calls = {}

    class FakeClassifier:
        class Model:
            class Config:
                id2label = {0: "negative", 1: "neutral", 2: "positive"}

            config = Config()

        model = Model()

    def fake_pipeline(**kwargs):
        calls.update(kwargs)
        return FakeClassifier()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "NVIDIA T4",
        )
    )
    fake_transformers = SimpleNamespace(pipeline=fake_pipeline)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer(device_index=0)

    assert analyzer.device_index == 0
    assert calls["device"] == 0
    assert "NVIDIA T4" in analyzer.device_label



def test_select_inference_device_falls_back_when_cuda_check_fails(monkeypatch):
    def unavailable_cuda_check():
        raise RuntimeError("CUDA driver is unavailable")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=unavailable_cuda_check)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device_index, device_label = select_inference_device()

    assert device_index == -1
    assert "CPU" in device_label
