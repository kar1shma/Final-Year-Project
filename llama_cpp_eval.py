import json
import argparse
from tqdm import tqdm
from typing import Literal

import llama_cpp
import instructor
from instructor import Mode
from pydantic import BaseModel


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
        help="Must be 'llama2' (we only support llama-cpp here)"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to your .gguf file (e.g. /path/to/llama2-13b-chat.gguf)."
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

    if args.model == "llama2":
        llm = setup_gguf(args.model_path, ctx=4096)
        llm.create = instructor.patch(
            create=llm.create_chat_completion_openai_v1,
            mode=Mode.JSON_SCHEMA,
        )
        evaluator = lambda prompt: eval_gguf(llm, prompt)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    tasks = json.load(open(args.tasks, "r"))
    results = []

    for idx, t in enumerate(tqdm(tasks, desc="llama2")):
        resp = evaluator(t["natural language"])
        tqdm.write(f"#{idx:03d} → answer={resp.answer!r}")
        results.append({
            **t["metadata"],
            "pred": resp.answer,
            "gold": t["t"],
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Predictions written to {args.out}")
