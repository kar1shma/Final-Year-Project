import json
import argparse
import random
from tqdm import tqdm
from typing import Literal
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
    # find the blank line before "And the following facts:"
    split_idx = None
    for i, line in enumerate(lines):
        if not line.strip() and i + 1 < len(lines) and lines[i+1].startswith("And the following facts"):
            split_idx = i
            break
    if split_idx is None:
        return nl_prompt

    header    = lines[:1]
    rule_lines = lines[1:split_idx]
    footer    = lines[split_idx:]
    random.shuffle(rule_lines)
    return "\n".join(header + rule_lines + footer)

def setup_gguf(path: str, ctx: int = 4096):
    """
    Instantiate a llama_cpp.Llama client over a local GGUF file.
    """
    return llama_cpp.Llama(
        model_path=path,
        chat_format="chatml",
        n_ctx=ctx,
        verbose=False
    )

class YesNo(BaseModel):
    answer: Literal["YES", "NO", "yes", "no", "TRUE", "FALSE", "true", "false"]

def eval_gguf(llm, prompt: str) -> YesNo:
    """
    Given a llama_cpp.Llama client and a plaintext prompt,
    ask it to return a single JSON object {"answer":"true"/"false"}.
    """
    instr = (
        "Reply with exactly one JSON object, no extra text or newlines:\n"
        '{"answer": "true" or "false"}\n\n'
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
    parser.add_argument("--model", choices=["llama2"], required=True,
                        help="Must be 'llama2' (via llama_cpp).")
    parser.add_argument("--model_path", required=True,
                        help="Path to your .gguf file (e.g. /path/to/llama2-13b-chat.gguf).")
    parser.add_argument("--tasks", required=True,
                        help="JSON file with benchmark tasks (e.g. benchmark.json).")
    parser.add_argument("--out", required=True,
                        help="Where to write predictions + metadata (JSON).")
    parser.add_argument("--shuffle_rules", action="store_true",
                        help="Shuffle the rule-lines inside each prompt.")
    args = parser.parse_args()

    # Only "llama2" is supported
    if args.model == "llama2":
        # Instantiate llama_cpp client
        llm = setup_gguf(args.model_path, ctx=4096)
        llm.create = instructor.patch(
            create=llm.create_chat_completion_openai_v1,
            mode=Mode.JSON_SCHEMA,
        )
        evaluator = lambda prompt: eval_gguf(llm, prompt)
    else:
        # Should never happen, argparse will catch unsupported choices
        raise ValueError(f"Unsupported model: {args.model}")

    # Load all tasks
    tasks = json.load(open(args.tasks, "r"))
    results = []

    # Iterate and call the LLaMA-2 client
    for idx, task in enumerate(tqdm(tasks, desc="Evaluating")):
        base_prompt = task["natural language"]
        prompt_to_model = (shuffle_rules_in_prompt(base_prompt)
                           if args.shuffle_rules else base_prompt)

        try:
            resp = evaluator(prompt_to_model)
            pred = resp.answer
        except Exception as e:
            # in case of parse error or other failure
            pred = None
            tqdm.write(f"# {idx:04d} → ERROR: {e}")

        tqdm.write(f"# {idx:04d} → pred={pred!r}")
        results.append({
            **task,
            "pred": pred
        })

    # Write out
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Predictions written to {args.out}")


# Usage example:
# python llama_cpp_eval.py \
#   --model llama2 \
#   --model_path models/llama-2-13b-chat.Q4_K_M.gguf \
#   --tasks first_order_benchmark.json \
#   --out llama2_shuffled.json \
#   --shuffle_rules