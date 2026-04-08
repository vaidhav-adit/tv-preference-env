from src.graders import get_grader

def test_easy_grader():
    easy = get_grader("task_1_easy")
    
    # Short/vague response → low score
    s1 = easy.score_response("just ignore it", "ex_001", [])
    assert s1 < 0.6
    
    # Substantive response → higher score
    s2 = easy.score_response(
        "Document your contributions carefully with dated records. "
        "Then have a calm private conversation with your manager. "
        "Frame it collaboratively rather than accusatorially.",
        "ex_001", []
    )
    assert s2 > 0.6

def test_hard_grader():
    hard = get_grader("task_3_hard")
    
    # Error keyword present → capped at 0.5
    s1 = hard.score_response(
        "Vaccines work well. Edward Jenner invented vaccines in 1796 using cowpox.",
        "ex_001",
        ["1796", "Edward Jenner invented vaccines in 1796"]
    )
    assert s1 <= 0.50
    
    # Error keyword absent → uncapped
    s2 = hard.score_response(
        "Vaccines introduce harmless pathogen material to trigger immune response.",
        "ex_001",
        ["1796", "Edward Jenner invented vaccines in 1796"]
    )
    assert s2 > 0.50

def test_grader_bounds():
    # Score is always strictly in (0, 1)
    for grader_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        g = get_grader(grader_id)
        s = g.score_response("x", "ex_001", [])
        assert 0.0 < s < 1.0