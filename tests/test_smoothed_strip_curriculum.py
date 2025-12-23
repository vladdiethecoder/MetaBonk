from src.training.smoothed_strip_curriculum import SmoothedStripCurriculum


def test_strip_curriculum_advances():
    cur = SmoothedStripCurriculum(min_strip_length=1, max_strip_length=3, min_episodes_per_strip=5)
    for _ in range(6):
        cur.record_episode(success=True, prediction_accuracy=0.9)
    assert cur.get_strip_length() >= 2


def test_strip_curriculum_regresses():
    cur = SmoothedStripCurriculum(min_strip_length=1, max_strip_length=3, min_episodes_per_strip=5)
    cur.current_strip_length = 3
    cur.episodes_at_current = 51
    for _ in range(10):
        cur.record_episode(success=False, prediction_accuracy=0.1)
    assert cur.get_strip_length() <= 2
