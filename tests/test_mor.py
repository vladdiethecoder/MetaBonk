from __future__ import annotations


def test_mor_deliberation_runs_with_precomputed_embedding():
    import torch

    from src.neuro_genie.mixture_of_reasonings import MixtureOfReasonings, MoRConfig

    mor = MixtureOfReasonings(cfg=MoRConfig(num_candidate_strategies=2))
    emb = torch.zeros(512)

    trace = mor.reason("avoid dying to corner traps", problem_embedding=emb, context={"hp": 10})

    assert trace.steps
    assert isinstance(trace.final_conclusion, str)
    # Tool grounding should emit at least one action even without an llm_fn.
    assert trace.grounded_plan

