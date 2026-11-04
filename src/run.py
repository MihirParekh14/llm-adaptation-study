"""Run the comparison and print the results.

Usage:
    python -m src.run --models qwen-2.5-3b
    python -m src.run --models llama-3.2-3b gemma-2-2b mistral-7b qwen-2.5-3b
    python -m src.run --models qwen-2.5-3b --methods zero-shot few-shot
    python -m src.run --ungated-only          # skip models needing a licence
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from src import config
from src.data import cap, load_banking77, majority_accuracy, stratified_sample
from src.evaluate import (
    Result,
    accuracy,
    breakeven_queries,
    format_parser_effect,
    format_results,
    unparsed_rate,
    worst_classes,
)
from src.labels import LabelParser
from src.prompting import build_prompt, fit_shots, prompt_token_estimate, select_shots

METHODS = ("zero-shot", "few-shot", "lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning versus in-context learning.")
    parser.add_argument("--models", nargs="+", default=["qwen-2.5-3b"],
                        choices=list(config.MODELS), help="Which models to run.")
    parser.add_argument("--ungated-only", action="store_true",
                        help="Only models that need no Hugging Face licence.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--train-examples", type=int, default=config.TRAIN_EXAMPLES)
    parser.add_argument("--eval-examples", type=int, default=config.EVAL_EXAMPLES)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Generation batch size. Lower it if you run out of memory.")
    parser.add_argument("--out", default=None, help="Where to write results JSON.")
    return parser.parse_args()


def evaluate_generations(outputs, parser, truth, method, model, **cost) -> Result:
    """Score one set of generations both strictly and fairly."""
    lenient = parser.parse_all(outputs)
    strict = parser.parse_all(outputs, strict=True)

    return Result(
        method=method,
        model=model,
        accuracy=accuracy(lenient, truth),
        strict_accuracy=accuracy(strict, truth),
        unparsed_rate=unparsed_rate(lenient),
        strict_unparsed_rate=unparsed_rate(strict),
        n_examples=len(truth),
        **cost,
    )


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    models = list(config.UNGATED) if args.ungated_only else args.models

    print("Loading banking77 ...")
    train_all, labels = load_banking77(config.TRAIN_SPLIT)
    test_all, _ = load_banking77(config.TEST_SPLIT)

    train = cap(train_all, args.train_examples)
    test = cap(test_all, args.eval_examples)

    # Worked examples are chosen from the full training split, not the capped
    # one. Picking them from a 500-row sample left one class with no example at
    # all, so the prompt silently offered 76 of 77 and the model could not have
    # got that class right.
    shots = select_shots(train_all, labels)
    parser = LabelParser(labels)
    truth = [example.label for example in test]

    zero_prompt = build_prompt(test[0].text, labels)
    few_prompt = build_prompt(test[0].text, labels, shots=shots)

    print(f"  classes                  {len(labels)}")
    print(f"  training examples        {len(train):,}")
    print(f"  held-out examples        {len(test):,}")
    print(f"  worked examples in prompt {len(shots)} of {len(labels)}")
    print(f"  always-guess-commonest   {majority_accuracy(test):.1%}")
    print(f"  prompt size, zero-shot   ~{prompt_token_estimate(zero_prompt):,} tokens")
    print(f"  prompt size, few-shot    ~{prompt_token_estimate(few_prompt):,} tokens\n")

    from src.finetune import finetune  # noqa: PLC0415
    from src.runtime import free_memory, generate, load_model_and_tokenizer, peak_memory_gb

    results: list[Result] = []

    for name in models:
        model_id = config.MODELS[name]
        print(f"\n{'=' * 70}\n{name}  ({model_id})\n{'=' * 70}")

        for method in args.methods:
            started = time.time()
            try:
                if method == "lora":
                    base, tokenizer = load_model_and_tokenizer(model_id, for_training=True)
                    settings = config.training_settings(name)
                    if settings["batch_size"] != config.BATCH_SIZE:
                        print(f"  {'':<11} note: training {name} at batch "
                              f"{settings['batch_size']} x {settings['grad_accumulation']} "
                              f"accumulation (large vocabulary)")

                    adapted, outcome = finetune(
                        base, tokenizer, stratified_sample(train, per_class=99),
                        labels, output_dir=str(config.ADAPTER_DIR / name),
                        epochs=args.epochs, **settings,
                    )
                    tokenizer.padding_side = "left"
                    prompts = [build_prompt(e.text, labels) for e in test]
                    generated = generate(adapted, tokenizer, prompts,
                                         batch_size=args.batch_size)
                    cost = dict(
                        train_seconds=outcome.seconds,
                        inference_seconds=generated.seconds,
                        prompt_tokens_per_query=generated.prompt_tokens,
                        peak_memory_gb=outcome.peak_memory_gb,
                        trainable_params=outcome.trainable_params,
                        total_params=outcome.total_params,
                    )
                    free_memory(adapted, base)
                else:
                    model, tokenizer = load_model_and_tokenizer(model_id)

                    use_shots = None
                    if method == "few-shot":
                        # Fit to the real tokenizer, not a word estimate. The
                        # longest message is used so every prompt fits, not
                        # just the average one.
                        longest = max(test, key=lambda e: len(e.text))
                        def count_tokens(text: str, tok=tokenizer) -> int:
                            return len(tok(text)["input_ids"])

                        use_shots = fit_shots(
                            longest.text, labels, shots,
                            count_tokens=count_tokens,
                            budget=config.MAX_PROMPT_TOKENS - config.MAX_NEW_TOKENS,
                        )
                        if len(use_shots) < len(shots):
                            print(f"  {'':<11} note: {len(shots) - len(use_shots)} worked "
                                  f"examples dropped to fit {config.MAX_PROMPT_TOKENS} tokens "
                                  f"({len(use_shots)} kept)")

                    prompts = [build_prompt(e.text, labels, shots=use_shots) for e in test]
                    generated = generate(model, tokenizer, prompts,
                                         batch_size=args.batch_size)
                    cost = dict(
                        inference_seconds=generated.seconds,
                        prompt_tokens_per_query=generated.prompt_tokens,
                        peak_memory_gb=peak_memory_gb(),
                    )
                    free_memory(model)

                if generated.truncated:
                    print(f"  {'':<11} WARNING: {generated.truncated} of {len(test)} prompts "
                          f"were too long and were cut. Raise MAX_PROMPT_TOKENS -- "
                          f"these results do not mean anything.")

                result = evaluate_generations(generated.texts, parser, truth,
                                              method, name, **cost)
                results.append(result)
                print(f"  {method:<11} accuracy {result.accuracy:>6.1%}  "
                      f"strict {result.strict_accuracy:>6.1%}  "
                      f"unparsed {result.unparsed_rate:>5.1%}  "
                      f"({time.time() - started:.0f}s)")

            except Exception as error:  # noqa: BLE001
                print(f"  {method:<11} FAILED: {type(error).__name__}: {error}")

    if not results:
        raise SystemExit("\nNothing completed. Check GPU memory and model access.")

    print(f"\n\n{'=' * 70}\nRESULTS\n{'=' * 70}\n")
    print(format_results(results))

    print("\n\nWhat the parser was worth")
    print("(strict = exact string match only, lenient = the full parser)\n")
    print(format_parser_effect(results))

    print("\n\nWhen fine-tuning pays for itself\n")
    by_model: dict[str, dict[str, Result]] = {}
    for result in results:
        by_model.setdefault(result.model, {})[result.method] = result

    for model_name, methods in sorted(by_model.items()):
        if "lora" in methods and "few-shot" in methods:
            lora, few = methods["lora"], methods["few-shot"]
            point = breakeven_queries(lora.train_seconds, lora.seconds_per_query,
                                      few.seconds_per_query)
            if point == float("inf"):
                print(f"  {model_name:<14} never -- fine-tuning is not faster per query")
            else:
                print(f"  {model_name:<14} after {point:,.0f} queries "
                      f"({lora.train_seconds:.0f}s training, saving "
                      f"{few.seconds_per_query - lora.seconds_per_query:.2f}s each)")

    best = max(results, key=lambda r: r.accuracy)
    print(f"\n\nWorst classes for the best run ({best.model}, {best.method}):")
    print("  " + ", ".join(f"{label} ({rate:.0%})"
                           for label, rate in worst_classes(
                               parser.parse_all(generated.texts), truth, n=5)))

    out = args.out or str(config.RESULTS_DIR / "results.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump([vars(r) for r in results], handle, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
        sys.exit(130)
