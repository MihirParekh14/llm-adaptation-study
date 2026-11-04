# LLM Adaptation Study

[![CI](https://github.com/MihirParekh14/llm-adaptation-study/actions/workflows/ci.yml/badge.svg)](https://github.com/MihirParekh14/llm-adaptation-study/actions/workflows/ci.yml)

**LoRA fine-tuning against in-context learning**, on a task where the two are
genuinely different, measured for accuracy *and* for cost.

Four models — Llama 3.2 3B, Gemma 2 2B, Mistral 7B, Qwen 2.5 3B — all in 4-bit,
all on a free Colab T4.

The interesting part is not which method wins. It is that **the answer depends
on how you score it**, and a scorer most people would write by default gets it
wrong.

---

## The task, and why this one

[banking77](https://huggingface.co/datasets/PolyAI/banking77): customer service
messages labelled with one of **77** banking intents.

The class count is the whole reason for the choice. Seventy-seven labels with
worked examples do not fit in a few-shot prompt — there is not room. So this
measures the real trade-off between showing a model examples and training it on
them. A three-class sentiment task would have produced a tie and taught nothing.

Always guessing the commonest class scores about **1.3%**. That is the floor
every number below should be read against.

---

## The defect: scoring is part of the experiment

Ask a model to pick one of 77 labels and it answers in whatever way it feels
like:

```
card_arrival
Card arrival
"card_arrival"
The intent is card_arrival.
card_arrival

Explanation: the customer is asking when their card will...
```

A **fine-tuned** model learns the exact output format from its training
examples and emits it cleanly. A **prompted** model does not — it was shown the
format once and otherwise writes like a chat assistant.

So scoring with exact string matching punishes in-context learning for its
manners rather than its understanding. You get a clean-looking table showing
fine-tuning winning by a wide margin, and the margin is mostly formatting.

Both methods are therefore scored through the same parser, and **both scorings
are reported**:

| | what it accepts |
|---|---|
| **strict** | exact match after lowercasing — what a naive scorer does |
| **lenient** | exact, then a label found inside a sentence, then a near miss above a similarity threshold |

The gap between them is a reported number, per method. If one method gains far
more from fair parsing than the other, the strict comparison was measuring
output formatting, not capability.

**Nothing is snapped onto the nearest label.** Text too far from anything is
recorded as unparsed, and the unparsed rate is reported — a method that
produces unusable output 30% of the time has told you something real, and
quietly rounding it to the closest class would erase exactly that. An unparsed
prediction counts as **wrong**, never as skipped; dropping unanswered rows
would let a method that answers a fifth of queries perfectly outscore one that
answers all of them well.

---

## A second defect, found by running it

The first smoke run scored few-shot at **2%** against zero-shot's **44%**. The
obvious reading is that in-context learning fails when there are 77 classes.

That reading is wrong. The prompt was over the token limit, and a tokenizer
truncates from the **right** by default — so the end of the prompt was deleted,
and the end of the prompt is the question. The model was shown 76 worked
examples and then asked nothing.

Two things made it hard to see:

- It raises nothing. Truncation is a normal operation on a normal input.
- The project's own `prompt_token_estimate` said 1,587, comfortably under
  2,048. It counts words times 1.3, and a label like
  `balance_not_updated_after_bank_transfer` is one word and about eight real
  tokens. The estimate undercounts exactly the thing that made the prompt long.

**Truncating from the left is not the fix either.** That removes the
instruction and the list of 77 labels, so the model no longer knows what it is
choosing between. Neither end of this prompt is disposable.

The fix is to not truncate. `fit_shots` drops worked examples until the prompt
fits, measured with the **real tokenizer**, against the **longest** message in
the evaluation set so every prompt fits rather than the average one. Examples
are the only part that degrades gracefully — with fewer of them the task is
harder but it is still the same task. How many were dropped is printed.

Anything still over the limit is counted and reported as a warning, because a
truncated prompt invalidates a measurement rather than weakening it.

---

## A third defect, same shape, a run later

With the prompt fixed, LoRA came back at 42% — and its training loss had
fallen from 1.185 to **0.0046** in under two epochs. A loss that good with
accuracy that ordinary means the model learned something other than the task.

Two causes, both the same mistake as before in a different place.

**The loss was computed over the whole sequence.** A training example is the
prompt plus the answer. The prompt is a ~700-token list of 77 label names,
identical in every example; the answer is about five tokens. Scoring all of it
means roughly 99% of the gradient teaches the model to recite a constant,
which it does almost immediately. The loss curve looks excellent and the model
has not learned to classify anything. Prompt tokens are now masked out with
`-100` so only the answer counts.

**`MAX_SEQ_LENGTH` was 256.** Against a ~700-token prompt, every training
example was cut before its answer — the model was fine-tuned on truncated
label lists ending nowhere. Raised to 1,024, and `build_dataset` now counts
how many examples exceed it and says so rather than silently cutting them.

The word-based estimate hid this one too: 77 snake_case labels read as 77
words and tokenise as roughly 550.

Three defects, all the same underlying shape — **a limit smaller than the
thing it was limiting, failing without an error.**

---

## The other thing accuracy does not tell you

Fine-tuning costs a lot once and little afterwards. In-context learning costs
nothing up front and carries its worked examples into **every query, forever**.

Which is cheaper is not a matter of opinion:

```
break-even queries = training seconds / (seconds saved per query)
```

The run prints that number per model. Below it, prompting is the cheaper choice
however the accuracy comparison came out. The function returns infinity when
fine-tuning is not actually faster per query, because that is a real outcome
and should not be reported as a very large number.

This is also why the fine-tuned model is trained on the **zero-shot** prompt
plus its answer. Training on the few-shot prompt would teach it to expect 77
examples at inference time, and the saving that justifies the training cost
would vanish.

---

## Results

300 held-out messages, 500 training examples, 4-bit, one T4.

| model | method | accuracy | strict | unparsed | train | s/query |
|---|---|---|---|---|---|---|
| llama-3.2-3b | **lora** | **77.0%** | 76.0% | 0.3% | 1,767s | 0.45s |
| llama-3.2-3b | few-shot | 59.3% | 55.3% | 2.0% | — | 1.26s |
| llama-3.2-3b | zero-shot | 41.0% | 41.0% | 2.7% | — | 0.31s |
| gemma-2-2b | few-shot | 57.0% | 54.7% | 5.0% | — | 1.40s |
| gemma-2-2b | zero-shot | 24.3% | 9.0% | **51.3%** | — | 0.32s |
| qwen-2.5-3b | lora | 72.0% | 72.0% | 0.0% | 1,716s | 0.45s |
| qwen-2.5-3b | few-shot | 53.0% | 52.0% | 4.0% | — | 1.17s |
| qwen-2.5-3b | zero-shot | 44.0% | 32.0% | 3.0% | — | 0.31s |

Always guessing the commonest class scores about 1.3%.

Fine-tuning wins wherever it ran, and the break-even on Llama is **2,167
queries**: 1,767 seconds of training, repaid at 0.82 seconds saved per query
because the prompt no longer carries 77 worked examples.

### What the scoring was actually measuring

| model | zero-shot strict | zero-shot lenient | gained | unparsed |
|---|---|---|---|---|
| gemma-2-2b | 9.0% | 24.3% | **+15.3** | 51.3% |
| qwen-2.5-3b | 32.0% | 44.0% | **+12.0** | 3.0% |
| llama-3.2-3b | 41.0% | 41.0% | **+0.0** | 2.7% |

The original hypothesis was that exact-match scoring penalises prompted
methods. The first model tested agreed. Llama disagreed completely.

It is not the method. It is **instruction-following**. All three were told to
reply with the bare label. Llama did, so strict and lenient scoring agree
exactly. Gemma ignored it half the time — 51.3% of its answers could not be
read as a label at all — so strict scoring reported 9% for a model that got
24% right.

So a scorer that only accepts exact matches is grading obedience to a
formatting instruction and reporting the number as accuracy. On one model that
costs 15 points. On another it costs nothing. Which means you cannot know
whether your comparison is fair without measuring both.

Few-shot and LoRA narrow the gap to 1–4 points for exactly the reason you would
expect: both show the model the output format, one by example and one by
training.

---

## Running it

The code lives in `src/`, not in a notebook. `notebooks/colab.ipynb` clones
this repository, installs the dependencies and calls it — so the logic is
version controlled and tested, and the notebook is something you can re-run
from scratch.

**Open `notebooks/colab.ipynb` in Colab, set Runtime → T4 GPU, and run the
cells.**

Locally, with a GPU:

```bash
pip install -r requirements.txt

python -m src.run --models qwen-2.5-3b --eval-examples 100   # quick check
python -m src.run --models llama-3.2-3b gemma-2-2b mistral-7b qwen-2.5-3b
python -m src.run --ungated-only                             # skip gated models
```

Llama 3.2 and Gemma 2 need a licence accepted on Hugging Face and an
`HF_TOKEN`. Mistral and Qwen need neither — which is why the model list is a
config value rather than hard-coded. A run can continue when an approval has
not come through.

---

## Two things that will cost you a session if you do not know them

**A T4 has no usable bfloat16.** It is a Turing card. Training in bf16 there
either fails outright or silently falls back and crawls. This is the single
most common reason a notebook that worked on an A100 does not work on free
Colab. The code checks `torch.cuda.is_bf16_supported()` rather than assuming.

**Gemma 2 needs eager attention.** It uses attention soft-capping, which the
fused kernels do not implement. Train it without
`attn_implementation="eager"` and you get a model that trains with no error and
predicts badly — the worst kind of failure, because nothing tells you.

**Gemma 2 runs out of memory where a larger model does not.** It is the
smallest model here at 2B, and it was the only one to OOM on a 16 GB card. The
cause is its vocabulary: 256,000 tokens against roughly 32,000–150,000 for the
others. Output logits are (batch × sequence × vocabulary), so at batch 4 and
1,024 tokens that single tensor is about 2.1 GB in fp16 before any gradients —
exactly the allocation that failed. Training memory does not track parameter
count. `TRAINING_OVERRIDES` gives it batch 1 with 16 accumulation steps, which
holds the effective batch size constant and only changes the footprint.

A fourth, smaller one: generation uses **left padding**. With right padding, a
batched `generate()` continues from pad tokens and every sequence shorter than
the longest in its batch comes back as rubbish.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest --cov=src --cov-report=term-missing
ruff check src tests
```

87 tests. CI runs them on Python 3.10, 3.11 and 3.12 on every push.

**No test loads a model, touches a GPU, or reaches the network**, and CI does
not install torch — a 2 GB wheel to check string parsing would be absurd. The
parts that decide the study's outcome are plain Python, and that is where the
tests are:

- **Label parsing.** That a label inside a sentence is found, that an
  explanation underneath does not count as being wrong, that `card_payment`
  does not shadow `card_payment_fee_charged` (searching shortest-first returns
  the wrong label inside a correct answer and looks exactly like model error),
  that unrelated text stays unparsed rather than being snapped to the nearest
  class, and that every real label parses to itself.
- **Fairness of the comparison.** That the training text and the inference
  prompt share a format, that the prompt lists all 77 labels, that the
  fine-tuning text is built from the zero-shot prompt, and that the few-shot
  prompt really is substantially longer — which is the cost being measured.
- **Scoring.** That unparsed counts as wrong, that mismatched lengths are
  rejected rather than silently zipped, that per-class accuracy surfaces the
  classes an average hides.
- **Break-even.** That it is infinite when fine-tuning is not faster, rather
  than a large number that reads like a result.
- **The truncation bug.** That the question is the last thing in the prompt
  and so is what right-truncation destroys, that `fit_shots` drops examples
  until the prompt fits, that it stops at zero rather than looping, and that
  the question survives however many examples are dropped.
- **What the fine-tuned model is scored on.** That prompt tokens and padding
  are masked out of the loss, that a training example cut at the prompt leaves
  no targets at all, and that the sequence length exceeds a full training
  example with room to spare.
- **Per-model memory.** That a large-vocabulary model gets a smaller batch,
  and that the override keeps the effective batch size identical so only the
  footprint changes, not the optimisation.
- **Surviving a library upgrade.** `TrainingArguments` changes between
  transformers releases and Colab upgrades underneath you; a run here died on
  `unexpected keyword argument 'warmup_ratio'` after an hour of setup.
  Arguments are now filtered against the real signature, and the dropped names
  are printed rather than swallowed.

---

## Project layout

```
src/
  config.py      models, LoRA settings, thresholds, task
  data.py        banking77 loading, stratified splits, the majority baseline
  labels.py      parsing generated text back to one of 77 labels
  prompting.py   prompt construction, and fitting it to a token budget
  runtime.py     4-bit loading and batched generation (needs a GPU)
  finetune.py    LoRA training (needs a GPU)
  evaluate.py    metrics, cost accounting, break-even
  run.py         command line: run the comparison
tests/           pytest suite, no GPU and no network
notebooks/
  colab.ipynb    clones this repo and runs it on a free T4
```

---

## Licence

MIT
