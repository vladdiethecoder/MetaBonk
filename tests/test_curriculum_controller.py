from src.training.curriculum_controller import AgenticCurriculum


def test_curriculum_advances():
    curriculum = AgenticCurriculum(num_strips=4, auto_advance=True, auto_regress=False)
    # Force enough successful episodes to advance.
    for _ in range(150):
        curriculum.record_episode(success=True, episode_reward=10.0, episode_length=200)
    assert curriculum.current_strip_id > 0


def test_curriculum_regresses():
    curriculum = AgenticCurriculum(num_strips=4, auto_advance=False, auto_regress=True)
    curriculum.current_strip_id = 2
    # Force failures to trigger regression.
    for _ in range(50):
        curriculum.record_episode(success=False, episode_reward=-5.0, episode_length=200)
    assert curriculum.current_strip_id < 2


def test_reward_shaping_scales():
    curriculum = AgenticCurriculum(num_strips=4, reward_shaping=True)
    low = curriculum.shape_reward(raw_reward=1.0, success=True, episode_length=100)
    curriculum.current_strip_id = 3
    high = curriculum.shape_reward(raw_reward=1.0, success=True, episode_length=100)
    assert high > low
