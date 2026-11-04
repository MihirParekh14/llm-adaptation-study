"""Score a method and account for what it cost.

Accuracy alone does not answer the question this study asks. Fine-tuning pays
a large cost once and a small cost per query afterwards. In-context learning
pays nothing up front and carries 77 worked examples into every single query
for the rest of time. Which is cheaper depends entirely on how many queries
there will be, and that number is computable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.labels import UNPARSED


@dataclass(frozen=True)
class Result:
    """What one method achieved on one model, and what it cost."""

    method: str
    model: str
    accuracy: float
    strict_accuracy: float
    unparsed_rate: float
    strict_unparsed_rate: float
    n_examples: int
    train_seconds: float = 0.0
    inference_seconds: float = 0.0
    prompt_tokens_per_query: int = 0
    peak_memory_gb: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def seconds_per_query(self) -> float:
        return self.inference_seconds / max(self.n_examples, 1)

    @property
    def parser_gain(self) -> float:
        """How much accuracy exact-match scoring would have thrown away."""
        return self.accuracy - self.strict_accuracy

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_params / max(self.total_params, 1)


def accuracy(predicted: list[str], actual: list[str]) -> float:
    """Share of predictions that are right.

    An unparsed prediction counts as wrong, never as skipped. Dropping the
    rows a method failed to answer would let a method that answers 20% of
    queries perfectly outscore one that answers all of them well.
    """
    if len(predicted) != len(actual):
        raise ValueError(
            f"{len(predicted)} predictions for {len(actual)} examples; they must line up"
        )
    if not actual:
        return 0.0
    return sum(p == a for p, a in zip(predicted, actual, strict=True)) / len(actual)


def unparsed_rate(predicted: list[str]) -> float:
    if not predicted:
        return 0.0
    return sum(label == UNPARSED for label in predicted) / len(predicted)


def per_class_accuracy(predicted: list[str], actual: list[str]) -> dict[str, float]:
    """Accuracy within each true class.

    With 77 classes an overall number hides a lot. A method can look fine
    while failing every one of a dozen rare intents.
    """
    right: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    for p, a in zip(predicted, actual, strict=True):
        total[a] += 1
        right[a] += int(p == a)

    return {label: right[label] / total[label] for label in sorted(total)}


def worst_classes(predicted: list[str], actual: list[str], n: int = 5) -> list[tuple[str, float]]:
    scores = per_class_accuracy(predicted, actual)
    return sorted(scores.items(), key=lambda item: item[1])[:n]


def breakeven_queries(
    train_seconds: float,
    finetuned_seconds_per_query: float,
    incontext_seconds_per_query: float,
) -> float:
    """How many queries before fine-tuning has paid for its training.

    Fine-tuning costs ``train_seconds`` up front and then saves
    ``incontext - finetuned`` on every query, because its prompt no longer
    carries 77 examples. Below the break-even point, prompting is the cheaper
    choice however the accuracy comparison came out.

    Returns infinity when fine-tuning is not actually faster per query, which
    is a real outcome and should not be reported as a very large number.
    """
    saving = incontext_seconds_per_query - finetuned_seconds_per_query
    if saving <= 0:
        return float("inf")
    return train_seconds / saving


def format_results(results: list[Result]) -> str:
    """A table comparing every method and model that was run."""
    if not results:
        return "No results."

    header = (
        f"  {'model':<14} {'method':<12} {'accuracy':>9} {'strict':>8} "
        f"{'unparsed':>9} {'train':>8} {'s/query':>8}"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]

    for result in sorted(results, key=lambda r: (r.model, r.method)):
        lines.append(
            f"  {result.model:<14} {result.method:<12} "
            f"{result.accuracy:>8.1%} {result.strict_accuracy:>7.1%} "
            f"{result.unparsed_rate:>8.1%} {result.train_seconds:>7.0f}s "
            f"{result.seconds_per_query:>7.2f}s"
        )
    return "\n".join(lines)


def format_parser_effect(results: list[Result]) -> str:
    """How much each method gained from being scored fairly.

    The headline of the study. If one method gains far more than the other,
    exact-match scoring was not measuring understanding -- it was measuring
    which model happened to answer in the expected format.
    """
    lines = ["  method        model           strict   lenient    gained"]
    lines.append("  " + "-" * 54)

    for result in sorted(results, key=lambda r: (r.method, r.model)):
        lines.append(
            f"  {result.method:<13} {result.model:<14} "
            f"{result.strict_accuracy:>6.1%} {result.accuracy:>9.1%} "
            f"{result.parser_gain:>+9.1%}"
        )
    return "\n".join(lines)
