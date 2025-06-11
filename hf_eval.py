#!/usr/bin/env python3
import json
import argparse
import random
import pathlib
import re
from tqdm import tqdm
from typing import List, Dict, Any

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
)


TF_ENUM = ["true", "false"]
# regex to extract the enum
TF_RE = re.compile(r'"answer"\s*:\s*"(\w+)"')

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": TF_ENUM
        }
    },
    "required": ["answer"]
}
SCHEMA_PROMPT = json.dumps(SCHEMA, indent=2)

def shuffle_rules(nl: str) -> str:
    lines = nl.splitlines()
    split = None
    for i, ln in enumerate(lines):
        if not ln.strip() and i+1 < len(lines) and lines[i+1].startswith("And the following facts"):
            split = i
            break
    if split is None:
        return nl
    header, rules, footer = lines[:1], lines[1:split], lines[split:]
    random.shuffle(rules)
    return "\n".join(header + rules + footer)

def build_prompt(nl: str, shuffle: bool) -> str:
    p = nl.strip()
    if shuffle:
        p = shuffle_rules(p)
    return (
        "Please reply with exactly one JSON object matching this schema:\n"
        f"{SCHEMA_PROMPT}\n\n"
        "### Question:\n"
        f"{p}\n\n"
        "### Answer:"
    )

def extract_answer(text: str) -> str:
    if isinstance(text, bool):
        return "true" if text else "false"

    txt = text if isinstance(text, str) else str(text)

    try:
        obj = json.loads(txt)
        ans = obj.get("answer", "").lower()
        if ans in TF_ENUM:
            return ans
    except Exception:
        pass

    m = TF_RE.search(txt)
    if m and m.group(1).lower() in TF_ENUM:
        return m.group(1).lower()

    return "ERROR"



def run(
    model_id: str,
    tasks: List[Dict[str, Any]],
    out_path: pathlib.Path,
    batch_size: int,
    shuffle: bool,
):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {model_id} on device {device} ...")
    cfg = AutoConfig.from_pretrained(model_id)
    is_causal = cfg.architectures and any(a.lower().startswith(("llama","gpt","gemma")) for a in cfg.architectures)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if is_causal:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        generator = pipeline("text-generation", model=model, tokenizer=tok, device=device, pad_token_id=tok.eos_token_id, trust_remote_code=False, return_full_text=False)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        generator = pipeline("text2text-generation", model=model, tokenizer=tok, device=device, pad_token_id=tok.eos_token_id, trust_remote_code=False)
    

    prompts = [build_prompt(t["natural language"], shuffle) for t in tasks]

    outputs: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        outs = generator(batch, max_new_tokens=32, do_sample=False, num_beams=4, temperature=0.0)
        for idx_in_batch, o in enumerate(outs):
            text = o["generated_text"] if isinstance(o, dict) else str(o)
            text = text.strip()

            global_idx = i + idx_in_batch
            tqdm.write(f"[{global_idx:04d}] → {text!r}")

            outputs.append(text)

    assert len(outputs) == len(tasks)

    results = []
    for task, out in zip(tasks, outputs):
        ans = extract_answer(out)
        rec = {
            **task,
            "pred": ans,
            "raw_output": out,
        }
        results.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done: wrote {len(results)} records to {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a set of logic-reasoning tasks on any HF model."
    )
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--tasks", required=True, help="JSON file of tasks")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--shuffle_rules",
        action="store_true",
        help="Randomly permute the rule lines in each prompt",
    )
    args = parser.parse_args()

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
#                   --tasks first_order_benchmark.json \
#                   --out flan_fol_shuffled_results.json \
#                   --batch_size 8 \
#                   --shuffle_rules