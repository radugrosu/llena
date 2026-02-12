import math

from scripts.eval import anls_score, normalize_answer, vqa_accuracy


def test_normalize_answer_basic() -> None:
    assert normalize_answer("  Hello, World! ") == "hello world"


def test_vqa_accuracy_thresholds() -> None:
    answers = ["yes", "yes", "no", "yes"]
    assert vqa_accuracy("yes", answers) == 1.0
    assert math.isclose(vqa_accuracy("no", answers), 1.0 / 3.0)
    assert vqa_accuracy("maybe", answers) == 0.0


def test_anls_score() -> None:
    answers = ["contract", "invoice"]
    assert anls_score("contract", answers) == 1.0
    assert anls_score("contracts", answers) > 0.5
    assert anls_score("xxxx", answers) == 0.0


def test_anls_uses_max_pred_or_answer_length() -> None:
    # dist=4 between "abcdxxxx" and "abcd": similarity should be 1 - 4/8 = 0.5
    assert anls_score("abcdxxxx", ["abcd"]) == 0.5
