import json
import argparse
from tqdm import tqdm
from typing import Literal

import llama_cpp
import instructor
from instructor import Mode
from pydantic import BaseModel



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

class YesNoWithRationale(BaseModel):
    answer: Literal["YES", "NO"]
    rationale: str

def eval_gguf(llm, prompt: str) -> YesNoWithRationale:
    """
    Given a llama_cpp.Llama client and a plaintext prompt,
    ask it to return a single JSON object {"answer":"YES"/"NO","rationale":"..."}.
    """
    instr = (
        "Reply with exactly one JSON object, no extra text or newlines:\n"
        '{"answer": "YES" or "NO", "rationale": "<single-line string with no control chars>"}\n\n'
        "### Question:\n" + prompt + "\n\n### Answer:"
    )
    resp: YesNoWithRationale = llm.create(
        messages=[{"role": "user", "content": instr}],
        max_tokens=256,
        temperature=0.0,
        response_model=YesNoWithRationale,
    )
    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tasks with a local GGUF model (llama2).")
    parser.add_argument(
        "--model",
        choices=["llama2"],
        required=True,
        help="Must be 'llama2' (we only support llama-cpp here)"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to your .gguf file (e.g. /path/to/llama2-7b.gguf)."
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Path to JSON file containing list of benchmark tasks (e.g. benchmark.json)."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON file where predictions+metadata will be saved."
    )
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

    # Iterate and call the LLaMA‐2 client
    for idx, t in enumerate(tqdm(tasks, desc="llama2")):
        resp = evaluator(t["natural language"])
        tqdm.write(f"#{idx:03d} → answer={resp.answer!r}, rationale={resp.rationale!r}")
        results.append({
            **t["metadata"],
            "pred": resp.answer,
            "rationale": resp.rationale,
            "gold": t["t"],
        })

    # Write a single JSON array containing 1,088 result‐objects
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Predictions written to {args.out}")
