"""Stub package required by `outlines` (used by SGLang constrained decoding).

Upstream `outlines` imports `pyairports.airports.AIRPORT_LIST`, but the upstream
`pyairports` dataset package is not reliably available on PyPI. For MetaBonk's
cognitive server we do not rely on airport-specific schemas, so an empty list is
enough to satisfy the import and keep SGLang functional.
"""

