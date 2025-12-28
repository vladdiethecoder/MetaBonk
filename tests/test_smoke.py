from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest


def test_code2logic_parses_minimal_dump():
    from src.cognitive.code2logic import Code2LogicExtractor

    dump = """
// Minimal Il2CppDumper-like snippet
public class DamageUpgrade {
    public float baseDamage = 10f;
    public float scaling = 1.2f;
    public int maxLevel = 5;
}

public class TankEnemy {
    public float hp = 50f;
    public float damage = 10f;
    public float speed = 1f;
}

public class Formulas {
    public float DPS(float baseDamage, float fireRate) {
        return baseDamage / fireRate;
    }
}
"""

    ext = Code2LogicExtractor()
    ext._parse_upgrades(dump)
    ext._parse_enemies(dump)
    ext._parse_formulas(dump)

    assert "damage" in ext.logic.upgrades
    assert ext.logic.upgrades["damage"]["max_level"] == 5
    assert "tank" in ext.logic.enemies
    assert "dps" in ext.logic.formulas


def test_megabonk_gym_env_steps():
    # This is an integration test: MegabonkEnv requires a running UnityBridge backend
    # and a trained reward model checkpoint. We intentionally do not provide mock
    # fallbacks because they destroy troubleshooting signal.
    if os.environ.get("METABONK_ENABLE_INTEGRATION_TESTS", "0") not in ("1", "true", "True"):
        pytest.skip("set METABONK_ENABLE_INTEGRATION_TESTS=1 to run environment integration tests")
    if not os.path.exists(os.environ.get("METABONK_VIDEO_REWARD_CKPT", "checkpoints/video_reward_model.pt")):
        pytest.skip("reward model checkpoint missing; run scripts/video_pretrain.py --phase reward_train")

    from src.env.megabonk_gym import MegabonkEnv, MegabonkEnvConfig

    # Button keys are intentionally user-configured (no hard-coded mapping).
    if os.environ.get("METABONK_BUTTON_KEYS", "").strip() == "":
        pytest.skip("set METABONK_BUTTON_KEYS (comma-separated) for action primitives")

    env = MegabonkEnv(MegabonkEnvConfig(frame_size=(32, 32), frame_stack=2, capture_fps=5))
    obs, info = env.reset(seed=0)
    assert obs.shape == (2, 3, 32, 32)
    assert "hp" in info

    a = env.action_space.sample()
    obs2, r, term, trunc, info2 = env.step(a)
    assert obs2.shape == (2, 3, 32, 32)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "position" in info2
    env.close()


def test_megabonk_yolo_env_steps_without_capture(tmp_path, monkeypatch: pytest.MonkeyPatch):
    # Integration test: MegaBonkEnv requires a reward-from-video checkpoint and
    # an actual capture backend. Skip unless explicitly enabled.
    if os.environ.get("METABONK_ENABLE_INTEGRATION_TESTS", "0") not in ("1", "true", "True"):
        pytest.skip("set METABONK_ENABLE_INTEGRATION_TESTS=1 to run environment integration tests")
    if not os.path.exists(os.environ.get("METABONK_VIDEO_REWARD_CKPT", "checkpoints/video_reward_model.pt")):
        pytest.skip("reward model checkpoint missing; run scripts/video_pretrain.py --phase reward_train")

    # MegaBonkEnv requires a capture source. Prefer an explicit replay source to make this
    # test runnable on headless/CI hosts (no desktop capture).
    if not os.environ.get("METABONK_CAPTURE_VIDEO") and not os.environ.get("METABONK_CAPTURE_IMAGES_DIR"):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            pytest.skip("no capture source and ffmpeg not available; set METABONK_CAPTURE_VIDEO or install ffmpeg")

        video_path = tmp_path / "capture.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=15",
                "-t",
                "2",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        monkeypatch.setenv("METABONK_CAPTURE_VIDEO", str(video_path))

    from src.env.megabonk_env import MegaBonkEnv

    env = MegaBonkEnv()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert info == {}
    out, r, term, trunc, info2 = env.step(env.action_space.sample())
    assert "obs" in out and "action_mask" in out
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "action_mask" in info2
