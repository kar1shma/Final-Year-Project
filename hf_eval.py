import json
import re
import random
import argparse
import pathlib
from tqdm import tqdm
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ---------- helpers ----------------------------------------------------------
# Accept YES/NO/TRUE/FALSE in any case, then normalize
YESNO_RE = re.compile(r"\b(YES|NO|TRUE|FALSE)\b", re.I)

def shuffle_rules(nl_prompt: str) -> str:
    """
    Shuffle only the rule lines (between the opening line and the blank line
    before 'And the following facts:'). Leave everything else untouched.
    """
    lines = nl_prompt.splitlines()
    split = None
    for i, ln in enumerate(lines):
        if (
            not ln.strip()
            and i + 1 < len(lines)
            and lines[i + 1].startswith("And the following facts:")
        ):
            split = i
            break
    if split is None:
        return nl_prompt

    header = lines[:1]
    rule_lines = lines[1:split]
    footer = lines[split:]
    random.shuffle(rule_lines)
    return "\n".join(header + rule_lines + footer)


def parse_answer(text: str) -> Tuple[str, str]:
    """
    Extract the first YES/NO/TRUE/FALSE token and return (answer, remainder).
    Normalize to lowercase "true"/"false".
    """
    m = YESNO_RE.search(text)
    if not m:
        return "ERROR", text.strip()

    token = m.group(1).upper()
    if token in {"YES", "TRUE"}:
        ans = "true"
    else:
        ans = "false"
    rationale = text[m.end():].lstrip(" .,\n")
    return ans, rationale


def run(
    model_id: str,
    tasks: list[dict],
    out_path: pathlib.Path,
    batch_size: int = 8,
    shuffle: bool = False,
) -> None:
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {model_id} on device {device} ...")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)

    gen = pipeline("text2text-generation", model=model, tokenizer=tok, device=device)

    # ===== updated instruction: ask for true/false only =====
    instr = "Answer exactly true or false.\n\n### Question:\n"

    prompts = []
    for t in tasks:
        p = t["natural language"].strip()
        if shuffle:
            p = shuffle_rules(p)
        prompts.append(instr + p + "\n\n### Answer:")

    outputs: list[str] = []
    chunks = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(chunks), desc="Generation"):
        start, end = i * batch_size, min((i + 1) * batch_size, len(prompts))
        outs = gen(
            prompts[start:end],
            max_new_tokens=32,
            num_beams=4,
            do_sample=False,
        )
        outputs.extend(o["generated_text"].strip() for o in outs)

    assert len(outputs) == len(tasks)

    results = []
    for task, text in tqdm(list(zip(tasks, outputs)), desc="Parsing outputs"):
        ans, rat = parse_answer(text)
        rec = {
            "q": task["q"],
            "c": task["c"],
            "natural language": task["natural language"],
            "t": task["t"],
            "metadata": task["metadata"],
            "pred": ans,
            "rationale": rat,
        }
        results.append(rec)

    json.dump(results, out_path.open("w"), indent=2)
    print(f"Done – {len(results)} records written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_id",
        required=True,
        help="HF model ID, e.g. google/flan-t5-xl",
    )
    ap.add_argument(
        "--tasks",
        required=True,
        help="benchmark.json produced by your generator",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="where to write the results JSON",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--shuffle_rules",
        action="store_true",
        help="shuffle the rule lines inside each prompt",
    )
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))
    run(
        args.model_id,
        tasks,
        pathlib.Path(args.out),
        batch_size=args.batch_size,
        shuffle=args.shuffle_rules,
    )


# Example usage:
# python hf_eval.py --model_id google/flan-t5-xl \
#                   --tasks benchmark.json \
#                   --out results.json \
#                   --batch_size 8 \
#                   --shuffle_rules
