"""Tests for prompt construction, sampling and scoring.

Two kinds of failure are covered: a prompt that quietly makes the task
impossible, and a comparison that quietly stops being fair.
"""

from __future__ import annotations

import pytest

from src.data import (
    Example,
    cap,
    label_distribution,
    majority_accuracy,
    majority_label,
    stratified_sample,
)
from src.evaluate import (
    Result,
    accuracy,
    breakeven_queries,
    format_parser_effect,
    format_results,
    per_class_accuracy,
    unparsed_rate,
    worst_classes,
)
from src.labels import UNPARSED
from src.prompting import (
    INSTRUCTION,
    build_prompt,
    build_training_text,
    format_label_list,
    prompt_token_estimate,
    select_shots,
)

LABELS = [f"intent_{i:02d}" for i in range(10)]


def pool(per_class: int = 3) -> list[Example]:
    return [
        Example(text=f"message {label} number {i}", label=label)
        for label in LABELS
        for i in range(per_class)
    ]


# --- sampling --------------------------------------------------------------


def test_stratified_sampling_takes_the_same_number_from_each_class():
    """A plain random sample over 77 classes leaves some with nothing, and a
    model cannot be blamed for a class it never saw."""
    counts = label_distribution(stratified_sample(pool(5), per_class=2))
    assert set(counts.values()) == {2}


def test_stratified_sampling_takes_what_exists_when_a_class_is_short():
    thin = pool(3) + [Example(text="only one", label="rare_intent")]
    counts = label_distribution(stratified_sample(thin, per_class=3))
    assert counts["rare_intent"] == 1


def test_sampling_is_seeded_so_two_runs_agree():
    assert stratified_sample(pool(5), 2, seed=1) == stratified_sample(pool(5), 2, seed=1)


def test_a_different_seed_gives_a_different_sample():
    assert stratified_sample(pool(5), 2, seed=1) != stratified_sample(pool(5), 2, seed=2)


def test_asking_for_zero_per_class_is_rejected():
    with pytest.raises(ValueError, match="per_class must be positive"):
        stratified_sample(pool(), per_class=0)


def test_cap_limits_the_number_of_examples():
    assert len(cap(pool(5), limit=7)) == 7


def test_cap_returns_everything_when_the_limit_is_larger():
    examples = pool(2)
    assert len(cap(examples, limit=1000)) == len(examples)


def test_cap_with_no_limit_changes_nothing():
    examples = pool(2)
    assert len(cap(examples, limit=None)) == len(examples)


# --- the floor -------------------------------------------------------------


def test_majority_accuracy_is_the_share_of_the_commonest_class():
    examples = [Example("a", "x")] * 3 + [Example("b", "y")]
    assert majority_accuracy(examples) == pytest.approx(0.75)


def test_majority_accuracy_over_balanced_classes_is_tiny():
    """On 77 balanced classes this is about 1.3%, which is the number any
    reported accuracy should be read against."""
    assert majority_accuracy(pool(3)) == pytest.approx(0.1)


def test_majority_label_is_the_commonest_one():
    assert majority_label([Example("a", "x")] * 3 + [Example("b", "y")]) == "x"


def test_majority_of_nothing_is_rejected():
    with pytest.raises(ValueError, match="majority of nothing"):
        majority_label([])


# --- prompts ---------------------------------------------------------------


def test_the_prompt_lists_every_available_label():
    """A model cannot pick from a set it was never shown. Omitting the list
    would test label-guessing rather than classification."""
    prompt = build_prompt("my card has not arrived", LABELS)
    for label in LABELS:
        assert label in prompt


def test_the_prompt_contains_the_message():
    assert "my card has not arrived" in build_prompt("my card has not arrived", LABELS)


def test_the_prompt_asks_for_the_label_only():
    prompt = build_prompt("x", LABELS)
    assert INSTRUCTION in prompt
    assert "No explanation" in prompt


def test_a_zero_shot_prompt_carries_no_worked_examples():
    assert "Examples:" not in build_prompt("x", LABELS)


def test_a_few_shot_prompt_carries_its_worked_examples():
    shots = select_shots(pool(3), LABELS, shots_per_class=1)
    prompt = build_prompt("x", LABELS, shots=shots)

    assert "Examples:" in prompt
    for shot in shots:
        assert shot.text in prompt


def test_shots_cover_as_many_classes_as_possible():
    """Coverage matters more than count when choosing between many classes."""
    shots = select_shots(pool(5), LABELS, shots_per_class=1)
    assert len({shot.label for shot in shots}) == len(LABELS)


def test_shots_respect_the_maximum():
    shots = select_shots(pool(5), LABELS, shots_per_class=1, max_examples=3)
    assert len(shots) == 3


def test_a_few_shot_prompt_is_much_longer_than_a_zero_shot_one():
    """Which is the cost being measured: in-context learning pays for its
    examples on every query, forever."""
    shots = select_shots(pool(3), LABELS, shots_per_class=1)

    zero = prompt_token_estimate(build_prompt("x", LABELS))
    few = prompt_token_estimate(build_prompt("x", LABELS, shots=shots))
    assert few > zero * 1.3


def test_training_text_uses_the_zero_shot_prompt_plus_the_answer():
    """Training on the few-shot prompt would teach the model to expect 77
    examples at inference, destroying the saving that justifies fine-tuning."""
    example = Example(text="where is my card", label="intent_03")
    text = build_training_text(example, LABELS)

    assert text.endswith("intent_03")
    assert "Examples:" not in text


def test_training_text_and_inference_prompt_share_a_format():
    """If one method is asked for a label and the other for a sentence, the
    difference measured is formatting, not learning."""
    example = Example(text="where is my card", label="intent_03")
    training = build_training_text(example, LABELS)
    inference = build_prompt("where is my card", LABELS)

    assert training.startswith(inference)


def test_the_label_list_wraps_rather_than_running_to_one_line():
    assert "\n" in format_label_list(LABELS, per_line=3)


# --- scoring ---------------------------------------------------------------


def test_accuracy_counts_exact_agreement():
    assert accuracy(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)


def test_an_unparsed_prediction_counts_as_wrong_not_as_skipped():
    """Dropping unanswered rows would let a method that answers a fifth of
    queries perfectly outscore one that answers all of them well."""
    assert accuracy([UNPARSED, "b"], ["a", "b"]) == pytest.approx(0.5)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="must line up"):
        accuracy(["a"], ["a", "b"])


def test_accuracy_of_nothing_is_zero_not_a_crash():
    assert accuracy([], []) == 0.0


def test_unparsed_rate_counts_unreadable_predictions():
    assert unparsed_rate([UNPARSED, "a", UNPARSED, "b"]) == pytest.approx(0.5)


def test_per_class_accuracy_separates_the_classes():
    predicted = ["a", "a", "x", "b"]
    actual = ["a", "a", "a", "b"]
    scores = per_class_accuracy(predicted, actual)

    assert scores["a"] == pytest.approx(2 / 3)
    assert scores["b"] == pytest.approx(1.0)


def test_worst_classes_surfaces_the_failures_an_average_hides():
    predicted = ["a"] * 3 + ["x"]
    actual = ["a"] * 3 + ["b"]
    assert worst_classes(predicted, actual, n=1) == [("b", 0.0)]


# --- cost ------------------------------------------------------------------


def test_breakeven_is_where_training_cost_is_repaid():
    """600s of training, saving 0.3s a query, pays for itself at 2,000."""
    assert breakeven_queries(600.0, 0.2, 0.5) == pytest.approx(2000.0)


def test_breakeven_is_infinite_when_fine_tuning_is_not_faster():
    """A real outcome, and it should not be reported as a very large number."""
    assert breakeven_queries(600.0, 0.5, 0.3) == float("inf")


def test_breakeven_is_infinite_when_the_two_cost_the_same():
    assert breakeven_queries(600.0, 0.4, 0.4) == float("inf")


def test_breakeven_is_zero_when_training_was_free():
    assert breakeven_queries(0.0, 0.2, 0.5) == pytest.approx(0.0)


# --- results ---------------------------------------------------------------


def result(method: str, accuracy_: float, strict: float) -> Result:
    return Result(
        method=method, model="m", accuracy=accuracy_, strict_accuracy=strict,
        unparsed_rate=0.1, strict_unparsed_rate=0.4, n_examples=100,
        train_seconds=300.0, inference_seconds=50.0,
    )


def test_parser_gain_is_what_exact_matching_would_have_discarded():
    assert result("in-context", 0.70, 0.40).parser_gain == pytest.approx(0.30)


def test_seconds_per_query_divides_by_the_number_of_examples():
    assert result("in-context", 0.7, 0.4).seconds_per_query == pytest.approx(0.5)


def test_trainable_fraction_does_not_divide_by_zero():
    assert Result(method="m", model="m", accuracy=0, strict_accuracy=0,
                  unparsed_rate=0, strict_unparsed_rate=0,
                  n_examples=0).trainable_fraction == 0.0


def test_results_render_as_a_table():
    text = format_results([result("in-context", 0.7, 0.4), result("lora", 0.8, 0.79)])
    assert "accuracy" in text and "in-context" in text and "lora" in text


def test_an_empty_result_set_renders_without_crashing():
    assert format_results([]) == "No results."


def test_the_parser_effect_table_shows_both_scorings():
    text = format_parser_effect([result("in-context", 0.7, 0.4)])
    assert "strict" in text and "lenient" in text
