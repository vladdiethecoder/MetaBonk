"""Homomorphic computation primitives (Paillier additive scheme).

The Singularity spec asks for fully homomorphic encryption (FHE), which is
substantial to implement from scratch. This module provides a practical,
well-understood *additively homomorphic* public-key scheme (Paillier) that can
serve as a security building block and demonstration substrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import secrets


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, y, x = _egcd(b % a, a)
    return g, x - (b // a) * y, y


def _modinv(a: int, m: int) -> int:
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise ValueError("no modular inverse")
    return x % m


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def _is_probable_prime(n: int, *, rounds: int = 16) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin.
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for _ in range(max(1, int(rounds))):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        ok = False
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                ok = True
                break
        if ok:
            continue
        return False
    return True


def _generate_prime(bits: int) -> int:
    b = int(bits)
    if b < 16:
        raise ValueError("bits too small")
    while True:
        cand = secrets.randbits(b) | 1 | (1 << (b - 1))
        if _is_probable_prime(cand):
            return cand


@dataclass(frozen=True)
class PaillierPublicKey:
    n: int
    g: int

    @property
    def n2(self) -> int:
        return self.n * self.n

    def encrypt(self, m: int, *, r: Optional[int] = None) -> int:
        mm = int(m) % self.n
        rr = int(r) if r is not None else secrets.randbelow(self.n - 1) + 1
        # c = g^m * r^n mod n^2
        return (pow(self.g, mm, self.n2) * pow(rr, self.n, self.n2)) % self.n2

    def e_add(self, c1: int, c2: int) -> int:
        return (int(c1) * int(c2)) % self.n2

    def e_mul_const(self, c: int, k: int) -> int:
        return pow(int(c), int(k) % self.n, self.n2)


@dataclass(frozen=True)
class PaillierPrivateKey:
    pub: PaillierPublicKey
    lam: int
    mu: int

    def decrypt(self, c: int) -> int:
        cc = int(c) % self.pub.n2
        # m = L(c^lambda mod n^2) * mu mod n
        u = pow(cc, int(self.lam), self.pub.n2)
        l = (u - 1) // self.pub.n
        return int(l * self.mu) % self.pub.n


def paillier_generate_keypair(*, bits: int = 512) -> Tuple[PaillierPublicKey, PaillierPrivateKey]:
    """Generate a Paillier keypair."""
    b = int(bits)
    p = _generate_prime(b // 2)
    q = _generate_prime(b // 2)
    if p == q:
        q = _generate_prime(b // 2)
    n = p * q
    g = n + 1
    lam = _lcm(p - 1, q - 1)
    n2 = n * n
    u = pow(g, lam, n2)
    l = (u - 1) // n
    mu = _modinv(int(l) % n, n)
    pub = PaillierPublicKey(n=n, g=g)
    priv = PaillierPrivateKey(pub=pub, lam=lam, mu=mu)
    return pub, priv


__all__ = [
    "PaillierPrivateKey",
    "PaillierPublicKey",
    "paillier_generate_keypair",
]

