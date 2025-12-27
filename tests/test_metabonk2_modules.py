import pytest


torch = pytest.importorskip("torch")


def test_reflexive_policy_forward_returns_skill():
    import torch

    from src.agent.action_space.hierarchical import Skill
    from src.agent.reasoning.system1 import ReflexivePolicy, ReflexivePolicyConfig
    from src.agent.skills.library import SkillLibrary

    lib = SkillLibrary()
    lib.add_skill(Skill(name="noop", duration=1, success_probability=1.0))
    lib.add_skill(Skill(name="move_random", duration=2, success_probability=0.2))

    pol = ReflexivePolicy(skill_library=lib, cfg=ReflexivePolicyConfig(feature_dim=32, max_skills=16))
    obs = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    skill, conf, features = pol(obs)

    assert skill.name in ("noop", "move_random")
    assert 0.0 <= float(conf) <= 1.0
    assert features.shape == (1, 32)


def test_deliberative_planner_plan_returns_intents():
    import torch

    from src.agent.reasoning.system2 import DeliberativePlanner, IntentSpace

    intent_space = IntentSpace(["idle", "explore_area", "combat_combo"])
    planner = DeliberativePlanner(state_dim=32, intent_space=intent_space, hidden_dim=64)
    state = torch.zeros((32,), dtype=torch.float32)
    plan = planner.plan(state, num_simulations=8, depth=3)

    assert isinstance(plan, list)
    assert len(plan) >= 1
    assert all(hasattr(p, "name") for p in plan)


def test_metacognitive_controller_fast_and_slow_paths():
    import torch

    from src.agent.action_space.hierarchical import Skill
    from src.agent.reasoning.metacognition import MetacognitiveController, MetacognitionConfig
    from src.agent.reasoning.system1 import ReflexivePolicy, ReflexivePolicyConfig
    from src.agent.reasoning.system2 import DeliberativePlanner, IntentSpace
    from src.agent.skills.library import SkillLibrary

    lib = SkillLibrary()
    lib.add_skill(Skill(name="noop", duration=1, success_probability=1.0))
    sys1 = ReflexivePolicy(skill_library=lib, cfg=ReflexivePolicyConfig(feature_dim=16, max_skills=8))
    sys2 = DeliberativePlanner(state_dim=16, intent_space=IntentSpace(["idle", "explore_area"]), hidden_dim=32)

    meta_fast = MetacognitiveController(system1=sys1, system2=sys2, cfg=MetacognitionConfig(fast_time_ms=200.0))
    obs = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    intent, mode, dbg = meta_fast(obs, time_budget_ms=5.0)
    assert mode == "fast"
    assert intent.name
    assert isinstance(dbg, dict)

    meta_slow = MetacognitiveController(
        system1=sys1,
        system2=sys2,
        cfg=MetacognitionConfig(confidence_threshold=1.1, novelty_threshold=-1.0, fast_time_ms=0.0),
    )
    intent2, mode2, dbg2 = meta_slow(obs, time_budget_ms=500.0)
    assert mode2 == "slow"
    assert intent2.name
    assert isinstance(dbg2.get("plan", []), list)


def test_learned_reward_model_smoke():
    import torch

    from src.agent.rewards.learned_rewards import LearnedRewardModel

    model = LearnedRewardModel(state_dim=8, action_dim=3, memory_size=32)
    s = torch.zeros((8,), dtype=torch.float32)
    a = torch.zeros((3,), dtype=torch.float32)
    ns = torch.zeros((8,), dtype=torch.float32)
    out = model(s, a, ns)

    assert set(out.keys()) == {"preference", "novelty", "empowerment", "total"}
    assert all(isinstance(v, float) for v in out.values())


def test_universal_game_encoder_forward_smoke():
    import torch

    from src.agent.generalization.universal_encoder import UniversalGameEncoder

    enc = UniversalGameEncoder(output_dim=64)
    x = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
    y = enc(x, game_id="megabonk")
    assert y.shape == (2, 64)

