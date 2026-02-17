from data.synthetic import SyntheticVQADataset


def test_synthetic_dataset_uses_fixed_question_when_provided() -> None:
    ds = SyntheticVQADataset(num_samples=2, image_size=64, seed=1, fixed_question="fixed question")
    assert ds[0]["question"] == "fixed question"
    assert ds[1]["question"] == "fixed question"


def test_synthetic_dataset_cycles_templates_when_not_fixed() -> None:
    ds = SyntheticVQADataset(
        num_samples=3,
        image_size=64,
        seed=1,
        question_templates=("q1", "q2"),
    )
    assert ds[0]["question"] == "q1"
    assert ds[1]["question"] == "q2"
    assert ds[2]["question"] == "q1"


def test_synthetic_dataset_rejects_blank_fixed_question() -> None:
    try:
        SyntheticVQADataset(num_samples=1, fixed_question="   ")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "fixed_question" in str(exc)
