"""DNA storage encoding/decoding utilities (toy implementation).

Implements:
- encoding bytes into A/C/G/T strings,
- simple parity-based error detection,
- decoding back into bytes.
"""

from __future__ import annotations

from typing import Iterable, List


_MAP = ["A", "C", "G", "T"]
_REV = {c: i for i, c in enumerate(_MAP)}


def dna_encode_bytes(data: bytes) -> str:
    """Encode bytes into DNA bases (2 bits per base) with per-byte parity base."""
    out: List[str] = []
    for b in data:
        # 4 bases for the 8 bits.
        for shift in (6, 4, 2, 0):
            out.append(_MAP[(b >> shift) & 0b11])
        # Parity base (xor of 2-bit chunks).
        p = ((b >> 6) & 3) ^ ((b >> 4) & 3) ^ ((b >> 2) & 3) ^ ((b >> 0) & 3)
        out.append(_MAP[p & 3])
    return "".join(out)


def dna_decode_bytes(seq: str) -> bytes:
    s = str(seq).strip().upper()
    if len(s) % 5 != 0:
        raise ValueError("invalid dna sequence length (expected 5 bases per byte)")
    out = bytearray()
    for i in range(0, len(s), 5):
        chunk = s[i : i + 5]
        vals = [_REV[c] for c in chunk]
        b = (vals[0] << 6) | (vals[1] << 4) | (vals[2] << 2) | vals[3]
        parity = vals[0] ^ vals[1] ^ vals[2] ^ vals[3]
        if (parity & 3) != (vals[4] & 3):
            raise ValueError("parity check failed")
        out.append(b & 0xFF)
    return bytes(out)


__all__ = [
    "dna_decode_bytes",
    "dna_encode_bytes",
]

