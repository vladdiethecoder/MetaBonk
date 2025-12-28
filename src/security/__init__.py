"""Security primitives for Singularity-style hardening."""

from .homomorphic import PaillierPrivateKey, PaillierPublicKey, paillier_generate_keypair
from .adversarial import fgsm_attack, sanitize_numeric_input

__all__ = [
    "PaillierPrivateKey",
    "PaillierPublicKey",
    "fgsm_attack",
    "paillier_generate_keypair",
    "sanitize_numeric_input",
]

