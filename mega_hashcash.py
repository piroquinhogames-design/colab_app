"""Compatibility helpers for MEGA's HTTP 402 Hashcash challenge.

MEGA can answer API requests with HTTP 402 and an X-Hashcash challenge.
Older mega.py releases do not understand this challenge. This module keeps
that compatibility logic isolated so the rest of the application can use the
existing mega.py client.
"""
from __future__ import annotations

import hashlib
import re
from typing import Mapping


_CHALLENGE_RE = re.compile(r"^([0-9a-fA-F]+):([0-9]+)$")


def solve_hashcash(challenge: str) -> str:
    """Solve an MEGA X-Hashcash challenge and return the header value.

    The challenge is expected to contain a hexadecimal prefix followed by a
    decimal difficulty, separated by ':'. The solution is the first decimal
    nonce whose SHA-256 digest has the requested number of leading zero bits.
    """
    match = _CHALLENGE_RE.fullmatch(challenge.strip())
    if not match:
        raise ValueError("Formato X-Hashcash inválido.")

    prefix, difficulty_text = match.groups()
    difficulty = int(difficulty_text)
    if difficulty < 0 or difficulty > 256:
        raise ValueError("Dificuldade X-Hashcash fora do intervalo permitido.")

    nonce = 0
    while True:
        candidate = f"{prefix}:{nonce}"
        digest = hashlib.sha256(candidate.encode("ascii")).digest()
        value = int.from_bytes(digest, "big")
        if value < (1 << (256 - difficulty)):
            return candidate
        nonce += 1


def challenge_from_headers(headers: Mapping[str, str]) -> str | None:
    """Return X-Hashcash from a case-insensitive header mapping."""
    for key, value in headers.items():
        if key.lower() == "x-hashcash":
            return value.strip() or None
    return None
