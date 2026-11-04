"""Shared settings for the adaptation study.

Every tunable value lives here so an experiment changes one file, not five.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ADAPTER_DIR = PROJECT_ROOT / "adapters"

# --- Task ------------------------------------------------------------------
#
# banking77: customer service messages labelled with one of 77 banking intents.
#
# The class count is the reason for choosing it. Seventy-seven labels with
# worked examples do not fit in a few-shot prompt -- there is not room -- so
# this measures the real trade-off between showing a model examples and
# training it on them. A three-class task would have produced a tie and taught
# nothing.
HF_DATASET = "PolyAI/banking77"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

# --- Models ----------------------------------------------------------------
#
# Add or remove entries here; nothing else needs to change. The first two are
# gated on Hugging Face and need a licence accepted and an HF_TOKEN set. The
# others are not, which is what makes this list worth having: a run can
# continue when an approval has not come through.
MODELS: dict[str, str] = {
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-2-2b": "google/gemma-2-2b-it",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "qwen-2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
}

UNGATED = ("mistral-7b", "qwen-2.5-3b")

# --- Quantisation and precision --------------------------------------------
#
# A free Colab T4 is a Turing card. It has no usable bfloat16, so training in
# bf16 either fails or silently falls back and runs at a crawl. fp16 is the
# correct choice there, and this is the single most common reason a notebook
# that worked on an A100 does not work on free Colab.
LOAD_IN_4BIT = True
BNB_QUANT_TYPE = "nf4"
BNB_DOUBLE_QUANT = True
COMPUTE_DTYPE = "float16"
USE_FP16 = True
USE_BF16 = False

# --- LoRA ------------------------------------------------------------------

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

# --- Training --------------------------------------------------------------

TRAIN_EXAMPLES = 2000
EVAL_EXAMPLES = 400
EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
MAX_SEQ_LENGTH = 256

# --- In-context learning ---------------------------------------------------
#
# Examples per class, not examples in total. With 77 classes even one example
# each is a long prompt, which is itself part of what is being measured: an
# in-context approach pays for its examples on every single query, while a
# fine-tuned one pays once.
SHOTS_PER_CLASS = 1
MAX_PROMPT_EXAMPLES = 77

# --- Generation ------------------------------------------------------------

MAX_NEW_TOKENS = 16
TEMPERATURE = 0.0  # greedy: this is classification, not writing

# --- Label parsing ---------------------------------------------------------
#
# How close a generated string has to be to a real label before it is accepted
# as that label. Below this it is counted as unparsed rather than guessed --
# an unparsed rate is a result, and silently snapping distant text onto the
# nearest label would hide it.
FUZZY_THRESHOLD = 0.82

RANDOM_SEED = 42


def ensure_dirs() -> None:
    for directory in (RESULTS_DIR, ADAPTER_DIR):
        directory.mkdir(parents=True, exist_ok=True)
