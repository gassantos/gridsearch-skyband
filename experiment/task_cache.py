"""Cache persistente de resultados bem-sucedidos de tarefas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .helpers import METRICS_DIR
from .workflow import TaskDefinition


class TaskCache:
    """Armazena métricas e artefatos por assinatura determinística de tarefa."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or METRICS_DIR / "workflow_cache"

    @staticmethod
    def signature(task: TaskDefinition, code_version: str) -> str:
        """Gera chave SHA-256 para configuração, entradas e versão do código."""
        payload = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "config": task.config,
            "input_signatures": task.input_signatures,
            "code_version": code_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, signature: str) -> dict[str, Any] | None:
        """Retorna a entrada persistida, ou ``None`` quando não há acerto."""
        path = self._cache_dir / f"{signature}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def put(
        self,
        signature: str,
        *,
        metrics: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> Path:
        """Persiste um resultado reutilizável de tarefa bem-sucedida."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{signature}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "artifacts": artifacts}, f, indent=2)
        return path