from __future__ import annotations

import pytest


def test_testing_spec_neuro_genie_imports_smoke() -> None:
    import torch  # noqa: F401

    from src.neuro_genie.mamba_policy import MambaConfig, MambaPolicy  # noqa: F401
    from src.neuro_genie.mixture_of_reasonings import MixtureOfReasonings, MoRConfig  # noqa: F401
    from src.neuro_genie.omega_protocol import OmegaConfig, OmegaOrchestrator  # noqa: F401
    from src.neuro_genie.test_time_compute import TestTimeCompute, TTCConfig  # noqa: F401
    from src.neuro_genie.generative_world_model import GenerativeWorldModel, GWMConfig  # noqa: F401


def test_testing_spec_mamba_policy_forward_smoke_cpu() -> None:
    torch = pytest.importorskip("torch")

    from src.neuro_genie.mamba_policy import MambaConfig, MambaPolicy

    cfg = MambaConfig(
        d_model=64,
        d_state=8,
        d_conv=2,
        expand=2,
        n_layers=2,
        dropout=0.0,
        num_actions=6,
    )
    policy = MambaPolicy(cfg)
    obs = torch.randn(2, 3, 64, 64)
    out = policy(obs)

    assert out["action_logits"].shape == (2, 6)
    assert out["action_probs"].shape == (2, 6)
    assert out["value"].shape == (2,)

    action = policy.get_action(obs, deterministic=True)
    assert action.shape == (2,)


def test_testing_spec_mor_reasoning_smoke_without_embed_backend() -> None:
    torch = pytest.importorskip("torch")

    from src.neuro_genie.mixture_of_reasonings import MixtureOfReasonings

    # MoR requires a real embedding backend only when `problem_embedding` is not provided.
    mor = MixtureOfReasonings(embed_dim=64)
    trace = mor.reason("unit test", problem_embedding=torch.zeros(64))

    assert trace.final_conclusion
    assert len(trace.steps) > 0


def test_testing_spec_generative_world_model_generate_frame_smoke_cpu() -> None:
    torch = pytest.importorskip("torch")

    from src.neuro_genie.generative_world_model import GWMConfig, GenerativeWorldModel, VideoTokenizerConfig

    cfg = GWMConfig(
        tokenizer_config=VideoTokenizerConfig(
            frame_height=32,
            frame_width=32,
            hidden_channels=[8, 16],
            num_codes=256,
            code_dim=32,
        ),
        num_latent_actions=32,
        latent_action_dim=8,
        embed_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        max_context_frames=4,
        use_text_conditioning=False,
    )
    model = GenerativeWorldModel(cfg)

    context = torch.rand(1, 2, 3, 32, 32)
    action = torch.randn(1, 8)
    next_frame = model.generate_frame(context, action, temperature=1.0, top_k=16)

    assert next_frame.shape == (1, 3, 32, 32)
