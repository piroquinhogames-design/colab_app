"""Compatibility helpers for MEGA's HTTP 402 Hashcash challenge.

MEGA can answer API requests with HTTP 402 and an X-Hashcash challenge.
Older mega.py releases do not understand this challenge. This module keeps
that compatibility logic isolated so the rest of the application can use the
existing mega.py client.
"""
from __future__ import annotations

import base64
import hashlib
import re
import struct
from typing import Mapping


# MEGA sends: 1:<easiness>:<resource>:<base64url token>
_CHALLENGE_RE = re.compile(r"^1:([0-9]+):([^:]*):([A-Za-z0-9_-]+)$")


def _threshold(easiness: int) -> int:
    """Match MEGA's gencash() threshold calculation."""
    return (((easiness & 63) << 1) + 1) << ((easiness >> 6) * 7 + 3)


def solve_hashcash(challenge: str) -> str:
    """Solve MEGA's X-Hashcash challenge and return its response header.

    MEGA's proof is not the usual ``sha256(prefix:nonce)`` construction.
    The challenge contains a 48-byte token. MEGA builds a 4-byte little-endian
    counter followed by 262144 copies of that token, hashes the whole buffer
    with SHA-256, and accepts a counter when the first 32 hash bits are below
    the challenge-derived threshold.
    """
    match = _CHALLENGE_RE.fullmatch(challenge.strip())
    if not match:
        raise ValueError("Formato X-Hashcash do MEGA inválido.")

    easiness = int(match.group(1))
    token_text = match.group(3)
    if not 0 <= easiness <= 255:
        raise ValueError("Dificuldade X-Hashcash do MEGA fora do intervalo permitido.")

    padding = "=" * (-len(token_text) % 4)
    try:
        token = base64.urlsafe_b64decode(token_text + padding)
    except ValueError as exc:
        raise ValueError("Token X-Hashcash do MEGA inválido.") from exc
    if len(token) != 48:
        raise ValueError("Token X-Hashcash do MEGA deve conter 48 bytes.")

    threshold = _threshold(easiness)
    body = bytearray(4 + 262144 * 48)
    body[4:] = token * 262144
    view = memoryview(body)

    for counter in range(1, 0xFFFFFFFF):
        struct.pack_into("<I", body, 0, counter)
        digest = hashlib.sha256(view).digest()
        if int.from_bytes(digest[:4], "big") <= threshold:
            encoded_counter = base64.urlsafe_b64encode(struct.pack("<I", counter)).decode("ascii").rstrip("=")
            return f"1:{token_text}:{encoded_counter}"

    raise RuntimeError("Não foi possível resolver o desafio X-Hashcash do MEGA.")


def challenge_from_headers(headers: Mapping[str, str]) -> str | None:
    """Return X-Hashcash from a case-insensitive header mapping."""
    for key, value in headers.items():
        if key.lower() == "x-hashcash":
            return value.strip() or None
    return None
