import configparser

import pytest

from dataset.nlp.HuggingFace import HuggingFaceDataset


def _base_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.add_section("data")
    cfg.set("data", "hf_dataset_source", "hub")
    cfg.set("data", "hf_dataset_id", "dummy/dataset")
    return cfg


class TestHuggingFaceDataset:
    def test_valid_mode_uses_validation_split(self, monkeypatch):
        captured = {}

        def fake_load_dataset(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return [{"sentence1": "a", "sentence2": "b", "label": 1}]

        monkeypatch.setattr("dataset.nlp.HuggingFace.load_dataset", fake_load_dataset)

        cfg = _base_config()
        ds = HuggingFaceDataset(cfg, "valid")

        assert len(ds) == 1
        assert captured["kwargs"]["split"] == "validation"

    def test_maps_common_pair_columns(self, monkeypatch):
        def fake_load_dataset(*args, **kwargs):
            return [{"sentence1": "qa", "sentence2": "cb", "label": 1}]

        monkeypatch.setattr("dataset.nlp.HuggingFace.load_dataset", fake_load_dataset)

        cfg = _base_config()
        ds = HuggingFaceDataset(cfg, "train")
        item = ds[0]

        assert item["text_a"] == "qa"
        assert item["text_b"] == "cb"
        assert item["label"] == 1
        assert item["guid"].startswith("train_")

    def test_normalizes_numeric_idx_guid(self, monkeypatch):
        def fake_load_dataset(*args, **kwargs):
            return [{"idx": 123, "sentence1": "qa", "sentence2": "cb", "label": 1}]

        monkeypatch.setattr("dataset.nlp.HuggingFace.load_dataset", fake_load_dataset)

        cfg = _base_config()
        ds = HuggingFaceDataset(cfg, "train")
        item = ds[0]

        assert item["guid"] == "123_0"

    def test_requires_label_for_train(self, monkeypatch):
        def fake_load_dataset(*args, **kwargs):
            return [{"sentence1": "qa", "sentence2": "cb"}]

        monkeypatch.setattr("dataset.nlp.HuggingFace.load_dataset", fake_load_dataset)

        cfg = _base_config()
        ds = HuggingFaceDataset(cfg, "train")

        with pytest.raises(ValueError, match="label"):
            _ = ds[0]

    def test_uses_configured_columns(self, monkeypatch):
        def fake_load_dataset(*args, **kwargs):
            return [{"left": "qa", "right": "cb", "y": 0, "uid": "abc"}]

        monkeypatch.setattr("dataset.nlp.HuggingFace.load_dataset", fake_load_dataset)

        cfg = _base_config()
        cfg.set("data", "hf_text_a_col", "left")
        cfg.set("data", "hf_text_b_col", "right")
        cfg.set("data", "hf_label_col", "y")
        cfg.set("data", "hf_guid_col", "uid")

        ds = HuggingFaceDataset(cfg, "train")
        item = ds[0]

        assert item == {"guid": "abc", "text_a": "qa", "text_b": "cb", "label": 0}

    def test_requires_pair_columns(self, monkeypatch):
        def fake_load_dataset(*args, **kwargs):
            return [{"text": "single", "label": 1}]

        monkeypatch.setattr("dataset.nlp.HuggingFace.load_dataset", fake_load_dataset)

        cfg = _base_config()
        ds = HuggingFaceDataset(cfg, "train")

        with pytest.raises(ValueError, match="pares de texto"):
            _ = ds[0]
