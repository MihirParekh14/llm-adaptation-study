"""Loading models and generating from them.

Everything in this module needs a GPU, so nothing in it is unit tested. The
logic that decides the study's outcome lives in :mod:`src.labels`,
:mod:`src.prompting` and :mod:`src.evaluate`, which are tested without one.

The settings here are written for a free Colab T4, which is where this study
is meant to run. That constraint shapes two choices that would otherwise look
arbitrary, and both are called out below.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

from src import config


def torch_dtype():
    import torch  # noqa: PLC0415

    return torch.float16 if config.COMPUTE_DTYPE == "float16" else torch.bfloat16


def supports_bfloat16() -> bool:
    """A T4 is a Turing card and has no usable bfloat16.

    Training in bf16 there either fails outright or silently falls back and
    runs at a crawl. This is the most common reason a notebook that worked on
    an A100 does not work on free Colab, so it is checked rather than assumed.
    """
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def quantisation_config():
    from transformers import BitsAndBytesConfig  # noqa: PLC0415

    return BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=torch_dtype(),
    )


def load_model_and_tokenizer(model_id: str, for_training: bool = False):
    """Load a quantised model and its tokenizer.

    Gemma 2 uses attention soft-capping, which the fused attention kernels do
    not implement. Training it without ``attn_implementation="eager"`` produces
    a model that trains without error and predicts badly, which is the worst
    kind of failure. The flag costs some speed and buys a correct model.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left padding for generation: with right padding, a batched generate()
    # continues from pad tokens and the outputs are rubbish for every sequence
    # that was shorter than the longest one.
    tokenizer.padding_side = "right" if for_training else "left"

    kwargs = {
        "quantization_config": quantisation_config(),
        "device_map": "auto",
        "dtype": torch_dtype(),
    }
    if "gemma" in model_id.lower():
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    if not for_training:
        model.eval()
        torch.cuda.empty_cache()
    return model, tokenizer


def free_memory(*objects) -> None:
    """Release a model so the next one fits. Colab gives you one GPU."""
    import torch  # noqa: PLC0415

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024**3
    except Exception:
        return 0.0


def count_parameters(model) -> tuple[int, int]:
    """(trainable, total). The ratio is what LoRA is for."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@dataclass
class GenerationOutput:
    texts: list[str]
    seconds: float
    prompt_tokens: int


def generate(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> GenerationOutput:
    """Generate a completion for each prompt, greedily.

    Greedy rather than sampled: this is classification. Sampling would add
    variance to a measurement without adding anything to the answer.

    Only the newly generated tokens are decoded. Returning the prompt as part
    of the answer would let the label list in the prompt satisfy the parser,
    and every method would score suspiciously well.
    """
    import torch  # noqa: PLC0415

    outputs: list[str] = []
    prompt_tokens = 0
    started = time.time()

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048).to(model.device)
        prompt_tokens += int(encoded["input_ids"].numel())

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = generated[:, encoded["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    return GenerationOutput(
        texts=outputs,
        seconds=time.time() - started,
        prompt_tokens=prompt_tokens // max(len(prompts), 1),
    )
