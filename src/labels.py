"""Map what a model actually says back onto one of the 77 labels.

This module decides the result of the study, which is not obvious until you
look at what the models emit. Asked to classify a message into one of 77
banking intents, a model answers with any of::

    card_arrival
    Card arrival
    "card_arrival"
    The intent is card_arrival.
    card_arrival

    Explanation: the customer is asking when...

A fine-tuned model learns the exact label format from its training examples and
emits it cleanly. A prompted model does not -- it was shown the format once and
writes like a chat assistant. So scoring with exact string matching punishes
in-context learning for its manners rather than its understanding, and the
study concludes that fine-tuning won when what actually happened is that the
parser was unfair to one side.

Both methods are therefore scored through the same parser, and the gap between
strict and lenient scoring is reported rather than hidden.

Nothing here snaps an arbitrary string onto the nearest label. Text that is not
close enough to anything is returned as unparsed, and the unparsed rate is a
reported number -- a method that produces unusable output 30% of the time has
told you something, and quietly rounding it to the nearest class would erase
exactly that.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from src import config

_PUNCTUATION = re.compile(r"[^a-z0-9\s_]+")
_WHITESPACE = re.compile(r"[\s\-/]+")

UNPARSED = "<unparsed>"


def normalise(text: str) -> str:
    """Reduce a string to the shape a label is written in.

    "Card Arrival." and "card arrival" and "card_arrival" all become
    ``card_arrival``, so a model is not marked wrong for capitalising.
    """
    if not text:
        return ""
    lowered = str(text).strip().lower()
    lowered = _PUNCTUATION.sub(" ", lowered)
    lowered = _WHITESPACE.sub("_", lowered.strip())
    return re.sub(r"_+", "_", lowered).strip("_")


def first_line(text: str) -> str:
    """The first non-empty line.

    Models often answer correctly and then explain themselves. The answer is
    the first line; the essay after it is not evidence of being wrong.
    """
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return ""


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class LabelParser:
    """Resolves generated text to one of a fixed set of labels."""

    def __init__(self, labels: list[str], fuzzy_threshold: float = config.FUZZY_THRESHOLD):
        if not labels:
            raise ValueError("a parser needs at least one label")

        self.labels = list(labels)
        self.fuzzy_threshold = fuzzy_threshold
        self._normalised = {normalise(label): label for label in labels}

        # Longest first. "card_payment" is a prefix of
        # "card_payment_fee_charged", so searching shortest-first would match
        # the wrong label inside a correct answer -- a bug that looks like the
        # model being wrong.
        self._by_length = sorted(self._normalised, key=len, reverse=True)

    # --- the two scoring modes --------------------------------------------

    def parse_strict(self, text: str) -> str:
        """Exact match only, after lowercasing.

        This is the naive scorer. It is kept so the study can report what it
        would have concluded, which is the point of the comparison.
        """
        candidate = normalise(first_line(text))
        return self._normalised.get(candidate, UNPARSED)

    def parse(self, text: str) -> str:
        """The fair parser: exact, then contained, then close enough."""
        if not text or not str(text).strip():
            return UNPARSED

        head = normalise(first_line(text))
        if head in self._normalised:
            return self._normalised[head]

        whole = normalise(text)
        if whole in self._normalised:
            return self._normalised[whole]

        contained = self._find_contained(head) or self._find_contained(whole)
        if contained:
            return contained

        return self._find_closest(head) or UNPARSED

    # --- the steps ---------------------------------------------------------

    def _find_contained(self, candidate: str) -> str | None:
        """A label sitting inside a longer sentence."""
        if not candidate:
            return None
        for label in self._by_length:
            if label and label in candidate:
                return self._normalised[label]
        return None

    def _find_closest(self, candidate: str) -> str | None:
        """The nearest label, if it is near enough to be believable."""
        if not candidate:
            return None

        best, best_score = None, 0.0
        for label in self._normalised:
            score = similarity(candidate, label)
            if score > best_score:
                best, best_score = label, score

        if best is not None and best_score >= self.fuzzy_threshold:
            return self._normalised[best]
        return None

    # --- batch -------------------------------------------------------------

    def parse_all(self, texts: list[str], strict: bool = False) -> list[str]:
        parse = self.parse_strict if strict else self.parse
        return [parse(text) for text in texts]

    def unparsed_rate(self, texts: list[str], strict: bool = False) -> float:
        parsed = self.parse_all(texts, strict=strict)
        return sum(label == UNPARSED for label in parsed) / max(len(parsed), 1)
