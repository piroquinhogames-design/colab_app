"""Resolve model versions and files from authoritative Civitai metadata."""
from __future__ import annotations

import os
import threading
import time
from typing import Any

import requests

_cache: dict[int, tuple[float, dict[str, Any]]] = {}
_lock = threading.Lock()


def version_metadata(version_id: int) -> dict[str, Any]:
    if not isinstance(version_id, int) or not 0 < version_id <= 2**63 - 1:
        raise ValueError("ID de versão Civitai inválido.")
    with _lock:
        cached = _cache.get(version_id)
        if cached and time.monotonic() - cached[0] < 900:
            return cached[1]
    headers = {"User-Agent": "ModelLab-Studio/3.0"}
    if token := os.environ.get("CIVITAI_TOKEN", "").strip():
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(f"https://civitai.com/api/v1/model-versions/{version_id}", headers=headers, timeout=25)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or data.get("id") != version_id:
        raise ValueError("O Civitai retornou uma versão inesperada.")
    with _lock:
        if len(_cache) >= 256:
            _cache.pop(next(iter(_cache)))
        _cache[version_id] = (time.monotonic(), data)
    return data


def model_file(data: dict[str, Any], *, kind: str, model_id: int | None = None, file_id: int | None = None) -> dict[str, Any]:
    if model_id is not None and int(data.get("modelId") or 0) != model_id:
        raise ValueError("A versão não pertence ao modelo selecionado.")
    actual_kind = str((data.get("model") or {}).get("type") or data.get("modelType") or "").lower()
    if actual_kind != kind.lower():
        raise ValueError(f"O recurso selecionado não é {kind}.")
    if "anima" not in str(data.get("baseModel", "")).lower():
        raise ValueError(f"{kind} {data.get('id')} ({(data.get('model') or {}).get('name') or data.get('name') or 'sem nome'}): base {data.get('baseModel') or 'não informada'} incompatível com Anima. Selecione uma versão Anima desse recurso ou remova-o da seleção.")
    files = [item for item in data.get("files", []) if isinstance(item, dict)
             and str(item.get("type", "")).lower() == "model"
             and str(item.get("name", "")).lower().endswith(".safetensors")]
    if file_id is not None:
        files = [item for item in files if item.get("id") == file_id]
    files.sort(key=lambda item: not bool(item.get("primary")))
    if not files:
        raise ValueError("Nenhum SafeTensor compatível foi encontrado nessa versão.")
    selected = files[0]
    sha = str((selected.get("hashes") or {}).get("SHA256", "")).lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("O Civitai não informou um SHA-256 válido para este arquivo.")
    return {
        "file_id": int(selected["id"]), "file_name": selected["name"], "sha256": sha,
        # Civitai reports KiB with limited precision; hash is the final authority.
        "size_bytes": int(float(selected.get("sizeKB") or 0) * 1024),
        "url": f"https://civitai.com/api/download/models/{int(data['id'])}?fileId={int(selected['id'])}",
    }
