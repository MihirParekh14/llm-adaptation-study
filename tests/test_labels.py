"""Tests for turning what a model says back into a label.

This is where the study's result is decided. A parser that is strict with one
method and lenient with the other produces a clean-looking comparison of the
wrong thing, and nothing about that failure raises an error.
"""

from __future__ import annotations

import pytest

from src.labels import UNPARSED, LabelParser, first_line, normalise, similarity

LABELS = [
    "card_arrival",
    "card_payment",
    "card_payment_fee_charged",
    "balance_not_updated_after_bank_transfer",
    "apple_pay_or_google_pay",
    "age_limit",
    "atm_support",
]


@pytest.fixture
def parser() -> LabelParser:
    return LabelParser(LABELS)


# --- normalising -----------------------------------------------------------


def test_capitalisation_and_spacing_become_the_label_shape():
    assert normalise("Card Arrival") == "card_arrival"
    assert normalise("card arrival") == "card_arrival"
    assert normalise("  CARD   ARRIVAL  ") == "card_arrival"


def test_punctuation_is_stripped():
    assert normalise('"card_arrival".') == "card_arrival"
    assert normalise("card_arrival!") == "card_arrival"


def test_hyphens_and_slashes_become_underscores():
    assert normalise("card-arrival") == "card_arrival"
    assert normalise("apple pay/google pay") == "apple_pay_google_pay"


def test_repeated_separators_collapse():
    assert normalise("card___arrival") == "card_arrival"
    assert normalise("card -- arrival") == "card_arrival"


def test_empty_input_normalises_to_empty():
    assert normalise("") == ""
    assert normalise("   ") == ""


def test_first_line_ignores_an_explanation_underneath():
    """Models often answer correctly and then explain. The essay is not
    evidence that the answer was wrong."""
    text = "card_arrival\n\nExplanation: the customer is asking when their card..."
    assert first_line(text) == "card_arrival"


def test_first_line_skips_leading_blank_lines():
    assert first_line("\n\n  card_arrival  ") == "card_arrival"


def test_similarity_is_one_for_identical_strings():
    assert similarity("card_arrival", "card_arrival") == pytest.approx(1.0)


# --- the strict parser -----------------------------------------------------


def test_strict_accepts_an_exact_label(parser):
    assert parser.parse_strict("card_arrival") == "card_arrival"


def test_strict_accepts_a_capitalised_label(parser):
    assert parser.parse_strict("Card Arrival") == "card_arrival"


def test_strict_rejects_a_label_inside_a_sentence(parser):
    """This is the behaviour the study is about. A prompted model answers in
    sentences, and exact matching scores that as a failure."""
    assert parser.parse_strict("The intent is card_arrival.") == UNPARSED


def test_strict_rejects_a_near_miss(parser):
    assert parser.parse_strict("card arrivals") == UNPARSED


# --- the fair parser -------------------------------------------------------


def test_an_exact_label_is_accepted(parser):
    assert parser.parse("card_arrival") == "card_arrival"


def test_a_label_wrapped_in_a_sentence_is_found(parser):
    assert parser.parse("The intent is card_arrival.") == "card_arrival"


def test_a_quoted_label_is_found(parser):
    assert parser.parse('"card_arrival"') == "card_arrival"


def test_a_label_followed_by_an_explanation_is_found(parser):
    text = "card_arrival\n\nThis is because the customer asks about delivery."
    assert parser.parse(text) == "card_arrival"


def test_a_near_miss_is_accepted_when_it_is_close_enough(parser):
    assert parser.parse("card arrivals") == "card_arrival"


def test_something_unrelated_is_left_unparsed_not_snapped_to_a_label(parser):
    """Guessing here would hide the unparsed rate, which is itself a result."""
    assert parser.parse("I am not sure what you mean") == UNPARSED


def test_empty_output_is_unparsed(parser):
    assert parser.parse("") == UNPARSED
    assert parser.parse("   ") == UNPARSED


# --- the prefix trap -------------------------------------------------------


def test_a_longer_label_is_not_shadowed_by_its_own_prefix(parser):
    """"card_payment" is a prefix of "card_payment_fee_charged". Searching
    shortest-first returns the wrong label inside a correct answer, and it
    looks exactly like the model being wrong."""
    assert parser.parse("card_payment_fee_charged") == "card_payment_fee_charged"


def test_the_prefix_label_itself_still_resolves(parser):
    assert parser.parse("card_payment") == "card_payment"


def test_the_longer_label_is_found_inside_a_sentence_too(parser):
    text = "The intent here is card_payment_fee_charged, I think."
    assert parser.parse(text) == "card_payment_fee_charged"


def test_the_longest_matching_label_wins_in_a_sentence(parser):
    text = "This is balance_not_updated_after_bank_transfer."
    assert parser.parse(text) == "balance_not_updated_after_bank_transfer"


# --- batch behaviour -------------------------------------------------------


def test_parse_all_returns_one_label_per_input(parser):
    outputs = ["card_arrival", "nonsense here", "The intent is age_limit."]
    parsed = parser.parse_all(outputs)

    assert parsed == ["card_arrival", UNPARSED, "age_limit"]


def test_strict_and_lenient_disagree_on_conversational_output(parser):
    """The measurement the whole study rests on."""
    outputs = ["The intent is card_arrival.", "I'd say age_limit", "atm_support"]

    strict = parser.parse_all(outputs, strict=True)
    lenient = parser.parse_all(outputs)

    assert strict.count(UNPARSED) == 2
    assert lenient.count(UNPARSED) == 0


def test_unparsed_rate_counts_what_could_not_be_read(parser):
    outputs = ["card_arrival", "nonsense", "also nonsense", "age_limit"]
    assert parser.unparsed_rate(outputs) == pytest.approx(0.5)


def test_unparsed_rate_of_an_empty_batch_does_not_divide_by_zero(parser):
    assert parser.unparsed_rate([]) == 0.0


# --- construction ----------------------------------------------------------


def test_a_parser_with_no_labels_is_rejected():
    with pytest.raises(ValueError, match="at least one label"):
        LabelParser([])


def test_a_stricter_threshold_rejects_a_looser_match():
    loose = LabelParser(LABELS, fuzzy_threshold=0.5)
    tight = LabelParser(LABELS, fuzzy_threshold=0.99)

    assert loose.parse("card arival") == "card_arrival"
    assert tight.parse("card arival") == UNPARSED


def test_every_real_label_parses_to_itself(parser):
    """A parser that cannot recognise its own labels would fail silently."""
    for label in LABELS:
        assert parser.parse(label) == label
        assert parser.parse_strict(label) == label
