import json
import argparse
from tqdm import tqdm
from typing import Literal
import random

import llama_cpp
import instructor
from instructor import Mode
from pydantic import BaseModel


def shuffle_rules_in_prompt(nl_prompt: str) -> str:
    """
    Shuffle only the lines between
    "You are given the following information:" and the blank line
    right before "And the following facts:".
    """
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


def setup_gguf(path: str, ctx: int = 4096):
    return llama_cpp.Llama(
        model_path=path,
        chat_format="chatml",
        n_ctx=ctx,
        verbose=False
    )


class YesNo(BaseModel):
    answer: Literal["YES", "NO"]


def eval_gguf(llm, prompt: str) -> YesNo:
    instr = (
        "Reply with exactly one JSON object, no extra text or newlines:\n"
        '{"answer":"YES" or "NO"}\n\n'
        "### Question:\n" + prompt + "\n\n### Answer:"
    )
    resp: YesNo = llm.create(
        messages=[{"role": "user", "content": instr}],
        max_tokens=64,
        temperature=0.0,
        response_model=YesNo,
    )
    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tasks with a local GGUF model.")
    parser.add_argument(
        "--model",
        choices=["llama2"],
        required=True,
        help="Must be 'llama2' (llama_cpp only)."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to your .gguf file (e.g. /path/to/llama2-13b-chat.gguf)."
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Path to JSON file containing benchmark tasks (e.g. benchmark.json)."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON file where predictions + metadata will be saved."
    )
    parser.add_argument(
        "--shuffle_rules",
        action="store_true",
        help="If set, shuffle the rule-lines inside each prompt before sending to the model."
    )
    args = parser.parse_args()

    if args.model == "llama2":
        # Instantiate llama_cpp client
        llm = setup_gguf(args.model_path, ctx=4096)
        llm.create = instructor.patch(
            create=llm.create_chat_completion_openai_v1,
            mode=Mode.JSON_SCHEMA,
        )
        evaluator = lambda prompt: eval_gguf(llm, prompt)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Load all tasks
    tasks = json.load(open(args.tasks, "r"))
    results = []

    # Iterate and call the model
    for idx, t in enumerate(tqdm(tasks, desc="llama2")):
        base_prompt = t["natural language"]
        if args.shuffle_rules:
            prompt_to_model = shuffle_rules_in_prompt(base_prompt)
        else:
            prompt_to_model = base_prompt

        resp = evaluator(prompt_to_model)
        tqdm.write(f"#{idx:03d} → answer={resp.answer!r}")
        results.append({
            **t["metadata"],
            "pred": resp.answer,
            "gold": t["t"],
        })

    # Write a single JSON array containing all result‐objects
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Predictions written to {args.out}")


# Usage example:
# python llama_cpp_eval.py \
#   --model llama2 \
#   --model_path /path/to/llama2-13b-chat.gguf \
#   --tasks benchmark.json \
#   --out llama2_shuffled.json
#   --shuffle_rules