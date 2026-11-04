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
    fit_shots,
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


# --- the truncation bug ----------------------------------------------------
#
# A smoke run scored few-shot at 2% against zero-shot's 44%. The cause was not
# the model: the prompt overran the token limit and the tokenizer cut from the
# right, which removes the end of the prompt -- and the end of the prompt is
# the question. The model was shown 76 worked examples and asked nothing.


def test_the_question_is_the_last_thing_in_the_prompt():
    """Which is why right-side truncation destroys it."""
    prompt = build_prompt("where is my card", LABELS)
    assert prompt.rstrip().endswith("Intent:")
    assert prompt.index("where is my card") > prompt.index("Available intents:")


def test_the_question_survives_when_the_prompt_is_cut_from_the_left():
    """The fix, stated as a test: cut the oldest example, keep the question."""
    shots = select_shots(pool(3), LABELS, shots_per_class=1)
    prompt = build_prompt("where is my card", LABELS, shots=shots)

    kept = prompt[len(prompt) // 2:]
    assert "where is my card" in kept
    assert kept.rstrip().endswith("Intent:")


def realistic_banking_prompt() -> str:
    """A full 77-class few-shot prompt with messages of a realistic length."""
    many_labels = [
        "card_arrival", "balance_not_updated_after_bank_transfer",
        "apple_pay_or_google_pay", "card_payment_fee_charged",
    ] + [f"intent_{i:02d}" for i in range(73)]
    big_pool = [
        Example(
            text=f"hello i have a question about my account and {label} "
                 f"could you please help me understand what is going on here",
            label=label,
        )
        for label in many_labels
    ]
    shots = select_shots(big_pool, many_labels, shots_per_class=1)
    return build_prompt("where is my card", many_labels, shots=shots)


def test_a_full_few_shot_prompt_is_well_over_a_thousand_tokens():
    """The condition that triggered the bug, so it is not forgotten."""
    assert prompt_token_estimate(realistic_banking_prompt()) > 1500


def test_the_word_based_estimate_undercounts_snake_case_labels():
    """The reason the overrun was missed.

    `prompt_token_estimate` counts words times 1.3. A label like
    `balance_not_updated_after_bank_transfer` is one word and about eight real
    tokens, so the estimate reported 1,587 for a prompt that truly ran past
    2,048 and was silently cut. The estimate is fine for comparing two prompts
    and must never be used to set a budget -- hence `MAX_PROMPT_TOKENS` has
    room to spare rather than being fitted to this number.
    """
    label = "balance_not_updated_after_bank_transfer"
    assert prompt_token_estimate(label) <= 2
    assert len(label.split("_")) >= 6  # really six-plus tokens, not one


# --- fitting the prompt to the budget --------------------------------------


def words(text: str) -> int:
    """A stand-in token counter: one token per whitespace-separated word."""
    return len(text.split())


def test_shots_are_dropped_until_the_prompt_fits():
    shots = select_shots(pool(3), LABELS, shots_per_class=1)
    full = words(build_prompt("q", LABELS, shots=shots))

    kept = fit_shots("q", LABELS, shots, count_tokens=words, budget=full // 2)
    assert 0 < len(kept) < len(shots)
    assert words(build_prompt("q", LABELS, shots=kept)) <= full // 2


def test_nothing_is_dropped_when_the_prompt_already_fits():
    shots = select_shots(pool(3), LABELS, shots_per_class=1)
    assert fit_shots("q", LABELS, shots, count_tokens=words, budget=100_000) == shots


def test_fitting_stops_at_zero_shots_rather_than_looping():
    """A budget too small even for the label list must terminate."""
    shots = select_shots(pool(3), LABELS, shots_per_class=1)
    assert fit_shots("q", LABELS, shots, count_tokens=words, budget=1) == []


def test_the_question_survives_however_many_shots_are_dropped():
    """The whole point. Examples give way; the question never does."""
    shots = select_shots(pool(3), LABELS, shots_per_class=1)
    kept = fit_shots("where is my card", LABELS, shots, count_tokens=words, budget=5)

    prompt = build_prompt("where is my card", LABELS, shots=kept)
    assert "where is my card" in prompt
    assert prompt.rstrip().endswith("Intent:")


# --- surviving a library upgrade -------------------------------------------


def test_unsupported_arguments_are_dropped_rather_than_crashing():
    """A run died on `unexpected keyword argument 'warmup_ratio'` after an
    hour of setup, because Colab had a newer transformers than the code
    expected."""
    from src.finetune import supported_arguments

    class Older:
        def __init__(self, output_dir, learning_rate=1e-4):
            pass

    kept = supported_arguments(Older, {
        "output_dir": "x", "learning_rate": 2e-4, "warmup_ratio": 0.03,
    })
    assert kept == {"output_dir": "x", "learning_rate": 2e-4}


def test_everything_supported_is_kept():
    from src.finetune import supported_arguments

    class Newer:
        def __init__(self, output_dir, learning_rate=1e-4, warmup_ratio=0.0):
            pass

    wanted = {"output_dir": "x", "learning_rate": 2e-4, "warmup_ratio": 0.03}
    assert supported_arguments(Newer, wanted) == wanted
