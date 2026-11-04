"""Load banking77 and build the splits both methods are judged on.

The two methods see exactly the same data. Fine-tuning trains on the training
subset; in-context learning draws its examples from that same subset. Both are
evaluated on the same held-out rows. Letting either one see anything the other
did not would make the comparison meaningless while still producing numbers.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class Example:
    """One customer message and its intent."""

    text: str
    label: str


def load_banking77(split: str) -> tuple[list[Example], list[str]]:
    """Return the examples in ``split`` and the full list of label names.

    ``datasets`` is imported inside the function so this module, and its
    tests, stay importable without it.
    """
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset(config.HF_DATASET, split=split)
    label_names = list(dataset.features["label"].names)

    examples = [
        Example(text=row["text"], label=label_names[row["label"]])
        for row in dataset
    ]
    return examples, label_names


def stratified_sample(
    examples: list[Example],
    per_class: int,
    seed: int = config.RANDOM_SEED,
) -> list[Example]:
    """Take up to ``per_class`` examples of each label.

    Stratified rather than random. With 77 classes a plain random sample of a
    few hundred rows leaves some classes with one example and some with none,
    and a model cannot be blamed for failing a class it never saw.
    """
    if per_class <= 0:
        raise ValueError("per_class must be positive")

    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    rng = random.Random(seed)
    picked: list[Example] = []
    for label in sorted(by_label):
        pool = by_label[label][:]
        rng.shuffle(pool)
        picked.extend(pool[:per_class])

    rng.shuffle(picked)
    return picked


def cap(
    examples: list[Example],
    limit: int | None,
    seed: int = config.RANDOM_SEED,
) -> list[Example]:
    """Shuffle and take at most ``limit`` examples."""
    if limit is None or limit >= len(examples):
        return list(examples)

    pool = list(examples)
    random.Random(seed).shuffle(pool)
    return pool[:limit]


def label_distribution(examples: list[Example]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[example.label] += 1
    return dict(counts)


def majority_label(examples: list[Example]) -> str:
    """The most common label -- the floor any method has to beat."""
    counts = label_distribution(examples)
    if not counts:
        raise ValueError("cannot take a majority of nothing")
    return max(sorted(counts), key=lambda label: counts[label])


def majority_accuracy(examples: list[Example]) -> float:
    """What you get by always guessing the most common class.

    On 77 roughly balanced classes this is about 1.3%, which is worth printing:
    it is the number any reported accuracy should be read against.
    """
    if not examples:
        return 0.0
    counts = label_distribution(examples)
    return max(counts.values()) / len(examples)
