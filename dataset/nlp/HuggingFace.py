"""Dataset baseado em Hugging Face Datasets para uso no pipeline BERT-PLI.

Permite duas fontes de dados configuráveis na secao ``[data]``:

- ``hf_dataset_source = hub`` (padrao): baixa/carrega dataset do Hub.
- ``hf_dataset_source = local_json``: carrega JSON/JSONL local via ``load_dataset('json', ...)``.

A saida e normalizada para o schema esperado pelos formatters atuais:
``guid``, ``text_a``, ``text_b`` e ``label`` (exceto em ``mode=test``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

from torch.utils.data import Dataset
from datasets import load_dataset

from tools.dataset_tool import dfs_search
from utils.paths import PathManager


class HuggingFaceDataset(Dataset):
    """Dataset PyTorch que usa ``datasets.load_dataset`` como backend."""

    _DEFAULT_TEXT_A_COLS = ("text_a", "sentence1", "premise", "question", "text", "query")
    _DEFAULT_TEXT_B_COLS = ("text_b", "sentence2", "hypothesis", "context", "candidate", "document")
    _DEFAULT_LABEL_COLS = ("label", "labels", "target", "gold")

    def __init__(self, config, mode: str, *args, **params):
        self.config = config
        self.mode = mode

        self.source = config.get("data", "hf_dataset_source", fallback="hub").strip().lower()
        self.dataset_id = config.get("data", "hf_dataset_id", fallback="").strip()
        self.dataset_config = config.get("data", "hf_dataset_config", fallback="").strip() or None

        self.text_a_col = config.get("data", "hf_text_a_col", fallback="").strip() or None
        self.text_b_col = config.get("data", "hf_text_b_col", fallback="").strip() or None
        self.label_col = config.get("data", "hf_label_col", fallback="").strip() or None
        self.guid_col = config.get("data", "hf_guid_col", fallback="").strip() or None

        split_name = self._resolve_split_name(config, mode)
        cache_dir = str(PathManager.HF_HUB_CACHE_DIR)

        if self.source in ("hub", "hf", "huggingface"):
            if not self.dataset_id:
                raise ValueError("hf_dataset_id deve ser definido quando hf_dataset_source=hub")

            kwargs: Dict[str, Any] = {
                "split": split_name,
                "cache_dir": cache_dir,
            }
            if self.dataset_config:
                kwargs["name"] = self.dataset_config

            self.dataset = load_dataset(self.dataset_id, **kwargs)

        elif self.source in ("local_json", "json", "local"):
            data_files = self._resolve_local_data_files(config, mode)
            data_key = f"{mode}_local"
            self.dataset = load_dataset(
                "json",
                data_files={data_key: data_files},
                split=data_key,
                cache_dir=cache_dir,
            )
        else:
            raise ValueError(
                f"hf_dataset_source invalido: '{self.source}'. "
                "Use 'hub' ou 'local_json'."
            )

    def _resolve_split_name(self, config, mode: str) -> str:
        if mode == "train":
            return config.get("data", "hf_train_split", fallback="train")
        if mode == "valid":
            return config.get("data", "hf_valid_split", fallback="validation")
        if mode == "test":
            return config.get("data", "hf_test_split", fallback="test")
        return mode

    def _resolve_local_data_files(self, config, mode: str) -> List[str]:
        data_path = Path(config.get("data", f"{mode}_data_path", fallback=str(PathManager.DATA_DIR)))
        filename_list = config.get("data", f"{mode}_file_list", fallback="").replace(" ", "").split(",")
        recursive = config.getboolean("data", "recursive", fallback=False)

        files = []
        for name in filename_list:
            if not name:
                continue
            files.extend(dfs_search(data_path / name, recursive))

        if not files:
            raise ValueError(
                f"Nenhum arquivo encontrado para mode={mode} em hf_dataset_source=local_json"
            )

        return [str(Path(f)) for f in sorted(files)]

    def _pick_col(self, row: Dict[str, Any], configured: Optional[str], defaults: Iterable[str]) -> Optional[str]:
        if configured:
            return configured if configured in row else None
        for candidate in defaults:
            if candidate in row:
                return candidate
        return None

    def _normalize_guid(self, value: Any, item: int) -> str:
        """Normaliza guid para o formato qid_cid esperado pela avaliacao atual."""
        if value is None:
            return f"{self.mode}_{item}"

        guid = str(value).strip()
        if not guid:
            return f"{self.mode}_{item}"

        sep_count = guid.count("_")
        if sep_count == 1:
            return guid
        if sep_count == 0:
            return f"{guid}_0"

        # Alguns datasets podem trazer ids com varios underscores; reduzimos
        # para dois segmentos para manter compatibilidade com eval_micro_query.
        first, second, *_ = guid.split("_")
        return f"{first}_{second}"

    def __getitem__(self, item: int) -> Dict[str, Any]:
        row = self.dataset[int(item)]

        col_text_a = self._pick_col(row, self.text_a_col, self._DEFAULT_TEXT_A_COLS)
        col_text_b = self._pick_col(row, self.text_b_col, self._DEFAULT_TEXT_B_COLS)
        col_label = self._pick_col(row, self.label_col, self._DEFAULT_LABEL_COLS)
        col_guid = self._pick_col(row, self.guid_col, ("guid", "id", "idx"))

        if col_text_a is None or col_text_b is None:
            raise ValueError(
                "Dataset HF precisa fornecer pares de texto (text_a/text_b). "
                "Configure hf_text_a_col e hf_text_b_col conforme o schema."
            )

        raw_guid = row[col_guid] if col_guid is not None else None
        if self.guid_col:
            guid_value = str(raw_guid) if raw_guid is not None else f"{self.mode}_{item}"
        else:
            guid_value = self._normalize_guid(raw_guid, item)

        sample: Dict[str, Any] = {
            "guid": guid_value,
            "text_a": row[col_text_a],
            "text_b": row[col_text_b],
        }

        if self.mode != "test":
            if col_label is None:
                raise ValueError(
                    "Coluna de label nao encontrada para treino/validacao. "
                    "Configure hf_label_col na secao [data]."
                )
            sample["label"] = row[col_label]

        return sample

    def __len__(self) -> int:
        return len(self.dataset)
