from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _iter_repo_text_files() -> list[Path]:
    roots = [
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT / "configs",
        REPO_ROOT / "launch.py",
        REPO_ROOT / "README.txt",
        REPO_ROOT / "LAUNCHER_README.md",
    ]
    exts = {".py", ".md", ".txt", ".json", ".sh"}

    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in exts:
                files.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts:
                continue
            files.append(path)
    return files


def test_no_menu_bootstrap_strings_in_runtime_code():
    banned = (
        "menu_bootstrap",
        "METABONK_INPUT_MENU_BOOTSTRAP",
        "METABONK_PURE_VISION_ALLOW_MENU_BOOTSTRAP",
        "_run_menu_bootstrap",
        "_input_should_bootstrap",
    )

    violations: list[str] = []
    for path in _iter_repo_text_files():
        # Avoid self-referential matches in this enforcement test.
        if path.name == "test_pure_vision_enforcement.py":
            continue
        if path.name == "validate_pure_vision.sh":
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for needle in banned:
            if needle in content:
                violations.append(f"{path}: contains {needle!r}")

    assert not violations, "Found pure-vision violations:\n" + "\n".join(sorted(violations))


def test_verify_script_does_not_depend_on_scene_labels():
    verify = REPO_ROOT / "scripts" / "verify_running_stack.py"
    assert verify.exists()
    content = verify.read_text(encoding="utf-8", errors="replace")
    assert "scene_type" not in content
