"""Atomic local files and confined artifact paths."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

JOB_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,63}", re.I)


def artifact_path(directory: Path, job_id: str, suffix: str) -> Path:
    # Accept legacy short IDs, but never separators, dots or absolute paths.
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise ValueError("Identificador de resultado inválido.")
    if suffix not in {".png", ".json", ".workflow.json", ".thumb.jpg", ".zip"}:
        raise ValueError("Tipo de arquivo inválido.")
    root = directory.resolve()
    path = root / f"{job_id}{suffix}"
    if path.resolve().parent != root:
        raise ValueError("Arquivo fora do diretório de resultados.")
    return path


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
