# LLM Adaptation Study

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

*Fill in from your run — `python -m src.run` prints all three tables.*

| model | method | accuracy | strict | unparsed | train | s/query |
|---|---|---|---|---|---|---|
| | | | | | | |

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

A third, smaller one: generation uses **left padding**. With right padding, a
batched `generate()` continues from pad tokens and every sequence shorter than
the longest in its batch comes back as rubbish.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest --cov=src --cov-report=term-missing
ruff check src tests
```

70 tests. CI runs them on Python 3.10, 3.11 and 3.12 on every push.

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

---

## Project layout

```
src/
  config.py      models, LoRA settings, thresholds, task
  data.py        banking77 loading, stratified splits, the majority baseline
  labels.py      parsing generated text back to one of 77 labels
  prompting.py   prompt construction for both methods
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
