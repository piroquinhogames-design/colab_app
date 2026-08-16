"""Teste público do desafio Hashcash do MEGA; não envia credenciais de usuário."""
import base64
import hashlib
import json

import requests

URL = "https://g.api.mega.co.nz/cs"
PAYLOAD = [{"a": "us", "user": "hashcash-smoke@example.invalid", "uh": "0" * 43}]
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


def solve_hashcash(challenge: str) -> str:
    parts = challenge.split(":")
    if len(parts) != 4 or parts[0] != "1":
        raise ValueError("desafio X-Hashcash não suportado")
    easiness = int(parts[1])
    token = base64.urlsafe_b64decode(parts[3] + "=" * (-len(parts[3]) % 4))
    base = ((easiness & 63) << 1) + 1
    shifts = (easiness >> 6) * 7 + 3
    threshold = base << shifts
    work = bytearray(4 + 262144 * 48)
    for offset in range(4, len(work), 48):
        work[offset : offset + len(token)] = token
    while True:
        digest = hashlib.sha256(work).digest()
        if int.from_bytes(digest[:4], "big") <= threshold:
            nonce = base64.urlsafe_b64encode(bytes(work[:4])).decode("ascii").rstrip("=")
            return f"1:{parts[3]}:{nonce}"
        cursor = 0
        while True:
            work[cursor] = (work[cursor] + 1) & 0xFF
            if work[cursor]:
                break
            cursor += 1
            if cursor >= 4:
                raise RuntimeError("espaço de nonce X-Hashcash esgotado")


def main() -> None:
    first = requests.post(URL, params={"id": 1}, data=json.dumps(PAYLOAD), headers=HEADERS, timeout=30)
    challenge = first.headers.get("X-Hashcash", "")
    print(f"PRIMEIRA_RESPOSTA HTTP={first.status_code} HASHCASH={bool(challenge)}")
    if first.status_code != 402 or not challenge:
        raise SystemExit("O endpoint não apresentou desafio Hashcash; teste público inconclusivo.")
    fields = challenge.split(":")
    if len(fields) != 4:
        raise SystemExit("O desafio X-Hashcash público tem formato inesperado.")
    token_bytes = len(base64.urlsafe_b64decode(fields[3] + "=" * (-len(fields[3]) % 4)))
    print(f"DESAFIO VERSAO={fields[0]} DIFICULDADE={fields[1]} TOKEN_BYTES={token_bytes}")
    proof = solve_hashcash(challenge)
    second = requests.post(
        URL,
        params={"id": 1},
        data=json.dumps(PAYLOAD),
        headers={**HEADERS, "X-Hashcash": proof},
        timeout=30,
    )
    print(f"RESPOSTA_COM_PROVA HTTP={second.status_code} BYTES={len(second.text or '')}")
    retry_challenge = second.headers.get("X-Hashcash", "")
    print(f"DESAFIO_REPETIDO={bool(retry_challenge)} DESAFIO_IGUAL={retry_challenge == challenge if retry_challenge else False}")
    if second.status_code == 402:
        raise SystemExit("A API ainda recusou a prova Hashcash.")
    print("HASHCASH_PUBLICO_OK")


if __name__ == "__main__":
    main()
