"""Small security controls for a single-user public tunnel."""
from __future__ import annotations

import io
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlparse, urljoin

import requests
from PIL import Image


class LoginLimiter:
    def __init__(self):
        self.lock = threading.Lock()
        self.attempts = defaultdict(deque)

    def allow(self, key: str, *, maximum=10, window=300) -> bool:
        with self.lock:
            now = time.monotonic()
            values = self.attempts[key]
            while values and values[0] <= now - window: values.popleft()
            if len(values) >= maximum: return False
            if len(self.attempts) > 2048:
                for name in list(self.attempts):
                    if not self.attempts[name] or self.attempts[name][-1] <= now - window:
                        del self.attempts[name]
            values.append(now)
            return True


def bounded_civitai_image(url: str, limit: int) -> bytes:
    allowed = {"image.civitai.com", "images.civitai.com"}
    for _ in range(4):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in allowed or parsed.username or parsed.password or parsed.port not in {None, 443}:
            raise ValueError("Origem de imagem não autorizada.")
        with requests.get(url, headers={"User-Agent": "ModelLab-Studio/3.0"}, stream=True,
                          allow_redirects=False, timeout=(10, 15)) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                url = urljoin(url, response.headers.get("Location", "")); continue
            response.raise_for_status()
            if int(response.headers.get("Content-Length") or 0) > limit:
                raise ValueError("A imagem excede o limite de tamanho.")
            chunks = bytearray()
            for chunk in response.iter_content(64 * 1024):
                chunks.extend(chunk)
                if len(chunks) > limit: raise ValueError("A imagem excede o limite de tamanho.")
            with Image.open(io.BytesIO(chunks)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"} or image.width * image.height > 40_000_000:
                    raise ValueError("Imagem inválida ou grande demais.")
                image.load()
                output = io.BytesIO(); image.convert("RGB").save(output, "JPEG", quality=92)
                if output.tell() > limit: raise ValueError("A imagem processada excede o limite.")
                return output.getvalue()
    raise ValueError("Muitos redirecionamentos na imagem.")
