"""Build the prompts the in-context method is judged on.

Two things decide whether this comparison is fair.

The prompt must list all 77 labels. A model cannot pick from a set it has not
been shown, and omitting the list would be testing whether it can guess the
label vocabulary rather than whether it can classify.

And the fine-tuned model is trained on the same instruction and the same
answer format. If one method is asked for ``card_arrival`` and the other for
"the intent", the difference measured is formatting, not learning.
"""

from __future__ import annotations

from src import config
from src.data import Example

INSTRUCTION = (
    "Classify the customer's message into exactly one banking intent.\n"
    "Reply with the intent label only. No explanation, no punctuation."
)


def format_label_list(labels: list[str], per_line: int = 4) -> str:
    """The available labels, wrapped so the prompt stays readable."""
    lines = []
    for start in range(0, len(labels), per_line):
        lines.append(", ".join(labels[start : start + per_line]))
    return "\n".join(lines)


def format_example(example: Example) -> str:
    return f"Message: {example.text}\nIntent: {example.label}"


def select_shots(
    pool: list[Example],
    labels: list[str],
    shots_per_class: int = config.SHOTS_PER_CLASS,
    max_examples: int = config.MAX_PROMPT_EXAMPLES,
    seed: int = config.RANDOM_SEED,
) -> list[Example]:
    """Pick the worked examples to show, covering as many classes as fit.

    Coverage is the thing to optimise, not count. Ten examples of one class
    teach less than one example each of ten classes when the model has to
    choose between 77.
    """
    from src.data import stratified_sample  # noqa: PLC0415  (avoids a cycle)

    chosen = stratified_sample(pool, per_class=shots_per_class, seed=seed)
    by_label = {}
    for example in chosen:
        by_label.setdefault(example.label, example)

    ordered = [by_label[label] for label in labels if label in by_label]
    return ordered[:max_examples]


def build_prompt(
    message: str,
    labels: list[str],
    shots: list[Example] | None = None,
    instruction: str = INSTRUCTION,
) -> str:
    """Assemble the instruction, the label list, any examples, and the query."""
    parts = [instruction, "", "Available intents:", format_label_list(labels)]

    if shots:
        parts += ["", "Examples:"]
        parts += [format_example(shot) for shot in shots]

    parts += ["", f"Message: {message}", "Intent:"]
    return "\n".join(parts)


def build_training_text(example: Example, labels: list[str]) -> str:
    """The text a fine-tuned model is trained on.

    Deliberately the zero-shot prompt plus the answer. Training on the
    few-shot prompt would teach the model to expect 77 examples at inference
    time, and the saving that makes fine-tuning worth its cost -- a short
    prompt on every query afterwards -- would disappear.
    """
    return build_prompt(example.text, labels) + f" {example.label}"


def prompt_token_estimate(prompt: str) -> int:
    """A rough token count, for comparing prompt cost without a tokenizer.

    English runs about 1.3 tokens per word. This is an estimate and is only
    used for reporting relative prompt sizes; the real token counts come from
    the tokenizer during a run.
    """
    return int(len(prompt.split()) * 1.3)
