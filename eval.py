import json, argparse
from tqdm import tqdm
from typing import Literal

import llama_cpp
import instructor
from instructor import Mode
from pydantic import BaseModel


def setup_gguf(path, ctx=4096):
    return llama_cpp.Llama(
        model_path=path,
        chat_format="chatml",
        n_ctx=ctx,
        verbose=False
    )

class YesNoWithRationale(BaseModel):
    answer: Literal["YES","NO"]
    rationale: str


def eval_gguf(llm, prompt):
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
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   choices=["llama2", "mistral"],
                   required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--tasks", required=True,
                   help="deduction.json or abduction.json")
    p.add_argument("--out", required=True, help="results.json")
    args = p.parse_args()

    # pick loader ------------------------------------------------------------
    if args.model in ["llama2", "mistral"]:
        llm = setup_gguf(args.model_path)

        # patch its OpenAI-compatible client to JSON_SCHEMA mode:
        llm.create = instructor.patch(
            create=llm.create_chat_completion_openai_v1,
            mode=Mode.JSON_SCHEMA,
        )

        evaluator = lambda pr: eval_gguf(llm, pr)
    else:
        raise ValueError(f"Unknown model {args.model}")

    # run benchmark ----------------------------------------------------------
    tasks   = json.load(open(args.tasks))

    results = []
    for idx, t in enumerate(tqdm(tasks, desc=args.model)):
        resp = evaluator(t["natural language"])
        tqdm.write(f"#{idx:03d} → answer={resp.answer!r}, rationale={resp.rationale!r}")
        results.append({
            **t["metadata"],
            "pred": resp.answer,
            "rationale": resp.rationale,
            "gold": t["t"],
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


# Example usage:
# python eval.py \
#   --model mistral \
#   --model_path models/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
#   --tasks deduction.json \
#   --out mistral_deduction_results.json


