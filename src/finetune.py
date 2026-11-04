"""Fine-tune a quantised model with LoRA.

LoRA freezes the original weights and trains two small matrices whose product
is added to them. The number worth reporting is the fraction of parameters
that are actually trained: a few tenths of a percent, which is what makes this
possible on a free 16 GB card at all.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass

from src import config
from src.data import Example
from src.prompting import build_prompt, build_training_text
from src.runtime import count_parameters, peak_memory_gb, supports_bfloat16


@dataclass
class TrainingOutcome:
    seconds: float
    trainable_params: int
    total_params: int
    peak_memory_gb: float
    final_loss: float


def supported_arguments(cls, wanted: dict) -> dict:
    """Keep only the arguments ``cls`` actually accepts, and say what was dropped.

    `TrainingArguments` changes between transformers releases, and Colab
    upgrades underneath you. A run here died with
    ``unexpected keyword argument 'warmup_ratio'`` after an hour of setup --
    not because the setting was wrong, but because the installed version had
    moved on.

    Filtering on the real signature means a version change costs a warning
    instead of the session. The dropped names are printed rather than
    swallowed, because a training run that quietly ignored the learning rate
    would be worse than one that crashed.
    """
    accepted = set(inspect.signature(cls.__init__).parameters)
    kept = {name: value for name, value in wanted.items() if name in accepted}

    dropped = sorted(set(wanted) - set(kept))
    if dropped:
        print(f"  note: this transformers version does not accept {', '.join(dropped)} "
              f"-- continuing without them")
    return kept


def lora_config():
    from peft import LoraConfig  # noqa: PLC0415

    return LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=list(config.LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )


IGNORE_INDEX = -100


def mask_prompt(input_ids: list[int], attention: list[int], prompt_length: int) -> list[int]:
    """Loss targets: the answer only, with the prompt and padding ignored.

    Training on the whole sequence is what the obvious implementation does, and
    on this task it is close to useless. The prompt is a 700-token list of 77
    label names, identical in every example; the answer is about five tokens.
    Scoring the whole thing means roughly 99% of the gradient is the model
    learning to recite a constant, which it does almost immediately -- training
    loss fell from 1.185 to 0.0046 inside two epochs while accuracy stayed at
    42%. The loss curve looked excellent and the model had learned nothing.

    Masking the prompt leaves only the part that is actually different between
    examples.
    """
    masked = list(input_ids)
    for position in range(len(masked)):
        if position < prompt_length or not attention[position]:
            masked[position] = IGNORE_INDEX
    return masked


def build_dataset(examples: list[Example], labels: list[str], tokenizer):
    """Tokenise the training texts and mask everything but the answer.

    Also checks that no example was cut short. A truncated training example
    silently drops the answer off the end, so the model trains on a prompt that
    leads nowhere -- which is exactly what happened at the old sequence length.
    """
    from datasets import Dataset  # noqa: PLC0415

    prompts = [build_prompt(example.text, labels) for example in examples]
    texts = [build_training_text(example, labels) for example in examples]

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        padding="max_length",
    )
    prompt_lengths = [
        len(ids) for ids in tokenizer(prompts, truncation=True,
                                      max_length=config.MAX_SEQ_LENGTH)["input_ids"]
    ]

    untruncated = tokenizer(texts, truncation=False)["input_ids"]
    cut = sum(len(ids) > config.MAX_SEQ_LENGTH for ids in untruncated)
    longest = max(len(ids) for ids in untruncated)
    if cut:
        print(f"  WARNING: {cut} of {len(texts)} training examples exceed "
              f"MAX_SEQ_LENGTH={config.MAX_SEQ_LENGTH} (longest is {longest}). "
              f"Their answers were cut off -- raise it, these results mean nothing.")

    encoded["labels"] = [
        mask_prompt(ids, attention, prompt_length)
        for ids, attention, prompt_length in zip(
            encoded["input_ids"], encoded["attention_mask"], prompt_lengths, strict=True
        )
    ]
    return Dataset.from_dict(dict(encoded))


def finetune(
    model,
    tokenizer,
    examples: list[Example],
    labels: list[str],
    output_dir: str,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    grad_accumulation: int = config.GRAD_ACCUMULATION,
) -> tuple[object, TrainingOutcome]:
    """Train LoRA adapters and return the adapted model plus what it cost."""
    from peft import get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import Trainer, TrainingArguments  # noqa: PLC0415

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config())
    model.config.use_cache = False

    trainable, total = count_parameters(model)
    dataset = build_dataset(examples, labels, tokenizer)

    # A T4 has no usable bfloat16. Asking for it there is the most common way
    # a notebook that worked elsewhere fails here, so the choice is made from
    # what the card reports rather than from a constant.
    use_bf16 = supports_bfloat16() and config.USE_BF16

    arguments = TrainingArguments(**supported_arguments(TrainingArguments, {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accumulation,
        "learning_rate": config.LEARNING_RATE,
        "warmup_ratio": config.WARMUP_RATIO,
        "logging_steps": 25,
        "save_strategy": "no",
        "report_to": [],
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "optim": "paged_adamw_8bit",
        "seed": config.RANDOM_SEED,
    }))

    trainer = Trainer(model=model, args=arguments, train_dataset=dataset)

    started = time.time()
    history = trainer.train()
    seconds = time.time() - started

    model.config.use_cache = True
    model.eval()

    return model, TrainingOutcome(
        seconds=seconds,
        trainable_params=trainable,
        total_params=total,
        peak_memory_gb=peak_memory_gb(),
        final_loss=float(getattr(history, "training_loss", 0.0) or 0.0),
    )
