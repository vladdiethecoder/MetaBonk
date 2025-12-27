#!/usr/bin/env python3
"""Pre-flight checks before deployment."""

from __future__ import annotations

import importlib
import os
import sys


def _truthy(val: str | None) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def check_python_version() -> bool:
    version = sys.version_info
    ok = version >= (3, 10)
    if ok:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python 3.10+ required (found {version.major}.{version.minor}.{version.micro})")
    return ok


def check_gpu() -> bool:
    require = _truthy(os.environ.get("METABONK_REQUIRE_CUDA", "1"))
    try:
        import torch  # type: ignore

        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            return True
        if require:
            print("❌ No GPU detected (torch.cuda.is_available() is False)")
            return False
        print("⚠️  No GPU detected (METABONK_REQUIRE_CUDA=0)")
        return True
    except Exception as e:
        if require:
            print(f"❌ GPU check failed: {e}")
            return False
        print(f"⚠️  GPU check skipped/failed: {e}")
        return True


def _check_import(name: str, *, required: bool) -> bool:
    try:
        importlib.import_module(name)
        print(f"✅ {name}")
        return True
    except Exception as e:
        if required:
            print(f"❌ {name} missing: {e}")
            return False
        print(f"⚠️  {name} not installed: {e}")
        return True


def check_dependencies() -> bool:
    # Required for basic MetaBonk runtime/discovery.
    required = ["torch", "numpy", "evdev", "cv2"]
    ok = True
    for dep in required:
        ok = _check_import(dep, required=True) and ok

    # Optional production tooling.
    optional = ["prometheus_client", "psutil", "GPUtil"]
    for dep in optional:
        _check_import(dep, required=False)
    return ok


def main() -> int:
    print("\n🔍 MetaBonk Pre-flight Checks\n")
    ok = True
    ok = check_python_version() and ok
    ok = check_gpu() and ok
    ok = check_dependencies() and ok
    if ok:
        print("\n✅ All checks passed!")
        return 0
    print("\n❌ Some checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

