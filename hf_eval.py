import json
import argparse
from tqdm import tqdm
import random

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)


def shuffle_rules_in_prompt(nl_prompt: str) -> str:
    lines = nl_prompt.splitlines()
    split_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "" and i + 1 < len(lines) and lines[i + 1].startswith("And the following facts:"):
            split_idx = i
            break
    if split_idx is None:
        return nl_prompt

    header = lines[:1]
    rule_lines = lines[1:split_idx]
    footer = lines[split_idx:]
    random.shuffle(rule_lines)
    return "\n".join(header + rule_lines + footer)


def run_hf_inference(model_id: str, tasks: list, out_path: str, batch_size: int = 8, shuffle_rules: bool = False):
    # Determine if gpu available
    device = 0 if torch.cuda.is_available() else -1

    # Load tokeniser + model
    print(f"Loading tokeniser and model for {model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)

    # Create a text2text-generation pipeline
    print("Creating text2text-generation pipeline…")
    generator = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # Build list of (possibly‐shuffled) prompts
    prompts = []
    for t in tasks:
        base_prompt = t["natural language"].strip()
        if shuffle_rules:
            prompts.append(shuffle_rules_in_prompt(base_prompt))
        else:
            prompts.append(base_prompt)

    total_prompts = len(prompts)
    print(f"Will generate {total_prompts} outputs in batches of {batch_size}…")

    # Generate in mini‐batches
    all_generated_texts = []
    n_chunks = (total_prompts + batch_size - 1) // batch_size

    for chunk_idx in tqdm(range(n_chunks), desc="Chunks"):
        start = chunk_idx * batch_size
        end = min(start + batch_size, total_prompts)
        batch_prompts = prompts[start:end]

        outputs = generator(
            batch_prompts,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False,
        )
        for out in outputs:
            all_generated_texts.append(out["generated_text"].strip())

    assert len(all_generated_texts) == total_prompts

    # Parse each generated_text as JSON, compare to ground truth
    results = []
    for idx, task in enumerate(tqdm(tasks, desc="Parsing outputs")):
        generated = all_generated_texts[idx]
        try:
            parsed = json.loads(generated)
            pred_answer = parsed.get("answer", "").strip()
            pred_rationale = parsed.get("rationale", "").strip()
        except json.JSONDecodeError:
            pred_answer = "ERROR"
            pred_rationale = generated

        rec = {
            **task["metadata"],
            "gold": task["t"],
            "pred": pred_answer,
            "rationale": pred_rationale
        }
        results.append(rec)

    print(f"Writing {len(results)} records to {out_path} …")
    with open(out_path, "w") as fout:
        json.dump(results, fout, indent=2)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        required=True,
        help="Hugging Face model ID (e.g. google/flan-t5-xl)."
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Path to your JSON file containing the list of tasks (e.g. benchmark.json)."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output file path (e.g. flan_t5_xl_unshuffled.json)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="How many prompts to generate at once (default: 8)."
    )
    parser.add_argument(
        "--shuffle_rules",
        action="store_true",
        help="If set, shuffle the rule‐lines in each prompt before generation."
    )

    args = parser.parse_args()

    # Load tasks from JSON
    tasks_list = json.load(open(args.tasks, "r"))

    # Run inference + save outputs
    run_hf_inference(
        model_id=args.model_id,
        tasks=tasks_list,
        out_path=args.out,
        batch_size=args.batch_size,
        shuffle_rules=args.shuffle_rules
    )


# Usage example:
# python hf_eval.py \
#   --model_id google/flan-t5-xl \
#   --tasks benchmark.json \
#   --out flan_t5_shuffled.json \
#   --batch_size 8 \
#   --shuffle_rules
