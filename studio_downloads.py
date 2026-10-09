"""Resumable verified downloads; never promote an unverified partial."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from studio_storage import atomic_json, read_json

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def digest_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def safetensor_valid(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            if not 2 <= length <= min(100 * 1024 * 1024, path.stat().st_size - 8):
                return False
            metadata = json.loads(handle.read(length))
            tensors = [item for key, item in metadata.items() if key != "__metadata__"]
            payload_size = path.stat().st_size - 8 - length
            offsets = []
            for item in tensors:
                a, b = item["data_offsets"]
                if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a <= b <= payload_size:
                    return False
                offsets.append((a, b))
            offsets.sort()
            return bool(offsets) and offsets[0][0] == 0 and offsets[-1][1] == payload_size and all(
                left[1] == right[0] for left, right in zip(offsets, offsets[1:]))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def ensure_download(url: str, destination: Path, minimum_bytes: int = 1, report=None,
                    *, sha256: str | None = None, expected_bytes: int = 0,
                    cancelled=None, retries: int = 3) -> Path:
    key = str(destination.resolve())
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        destination.parent.mkdir(parents=True, exist_ok=True)
        receipt = destination.with_suffix(destination.suffix + ".verified.json")
        def valid(path):
            if not path.is_file() or path.stat().st_size < minimum_bytes:
                return False
            if path.suffix == ".safetensors" or destination.suffix == ".safetensors":
                if not safetensor_valid(path):
                    return False
            return not sha256 or digest_file(path) == sha256.lower()
        if destination.exists():
            stat = destination.stat()
            saved = read_json(receipt, {})
            fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256, "url": url}
            if saved == fingerprint or valid(destination):
                atomic_json(receipt, fingerprint)
                if report: report(100)
                return destination
            destination.unlink()
        partial = destination.with_suffix(destination.suffix + ".part")
        partial_meta = partial.with_suffix(partial.suffix + ".json")
        if read_json(partial_meta, {}).get("url") != url:
            partial.unlink(missing_ok=True)
        atomic_json(partial_meta, {"url": url})
        last_error = None
        for attempt in range(retries):
            if cancelled and cancelled(): raise InterruptedError("Download cancelado.")
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "ModelLab-Studio/3.0"}
            if urlparse(url).hostname == "civitai.com" and (token := os.environ.get("CIVITAI_TOKEN", "").strip()):
                headers["Authorization"] = f"Bearer {token}"
            if offset: headers["Range"] = f"bytes={offset}-"
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(20, 30)) as response:
                    if response.status_code == 416:
                        if valid(partial):
                            break
                        partial.unlink(missing_ok=True)
                        raise requests.RequestException("Retomada rejeitada; reiniciando download.")
                    response.raise_for_status()
                    append = bool(offset and response.status_code == 206)
                    if response.status_code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset:
                            partial.unlink(missing_ok=True)
                            raise requests.RequestException("Content-Range incompatível com o parcial.")
                        total = int(match[3])
                    else:
                        total = int(response.headers.get("Content-Length") or 0)
                        offset = 0
                    required = max(total, expected_bytes) - offset
                    if shutil.disk_usage(destination.parent).free < max(required, 0) + 256 * 1024 * 1024:
                        raise RuntimeError("Espaço livre insuficiente para baixar o modelo.")
                    downloaded = offset
                    with partial.open("ab" if append else "wb") as handle:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if cancelled and cancelled(): raise InterruptedError("Download cancelado.")
                            if chunk:
                                handle.write(chunk); downloaded += len(chunk)
                                if report and total: report(min(99, int(downloaded * 100 / total)))
                    if total and downloaded != total:
                        raise requests.RequestException("Download terminou antes do tamanho informado.")
                if not valid(partial):
                    partial.unlink(missing_ok=True)
                    raise requests.RequestException("Arquivo inválido ou SHA-256 divergente; tentando novamente.")
                break
            except InterruptedError:
                raise
            except (requests.RequestException, OSError, ValueError) as exc:
                last_error = exc
                if attempt == retries - 1: raise RuntimeError(f"Falha ao baixar {destination.name}: {exc}") from exc
                time.sleep(min(2 ** attempt, 4))
        else:
            raise RuntimeError(str(last_error))
        partial.replace(destination)
        partial_meta.unlink(missing_ok=True)
        stat = destination.stat()
        atomic_json(receipt, {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256, "url": url})
        if report: report(100)
        return destination
