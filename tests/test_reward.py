from src.reward import RewardEngine

def test_compute_judge_reward():
    engine = RewardEngine()
    
    r = engine.compute_judge_reward(1.0, 1.0)
    assert r.total == 0.60
    assert r.judgment_component == 0.50
    assert r.critique_component == 0.10
    
    r = engine.compute_judge_reward(0.0, 0.0)
    assert r.total == 0.0

def test_compute_refine_reward():
    engine = RewardEngine()
    
    r = engine.compute_refine_reward(0.0)
    assert r.total == -0.03   # no improvement, paid cost
    
    r = engine.compute_refine_reward(0.5)
    assert abs(r.total - 0.095) < 0.001   # 0.25*0.5 - 0.03

def test_compute_submit_reward():
    engine = RewardEngine()
    
    r = engine.compute_submit_reward(1, 1, 0.91, 0.91)
    assert abs(r.budget_component - 0.10) < 0.001  # full early bonus
    assert r.penalty_component == 0.0              # no gap
    
    r = engine.compute_submit_reward(0, 2, 0.70, 0.91)
    assert r.budget_component == 0.0               # no bonus, used all budget
    assert abs(r.penalty_component - (-0.042)) < 0.001  # -0.2 * 0.21

def test_compute_wrong_phase_reward():
    engine = RewardEngine()
    r = engine.compute_wrong_phase_reward()
    assert r.total == -0.1
    assert r.penalty_component == -0.1