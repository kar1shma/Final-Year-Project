import json
import argparse
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)


def run_hf_inference(model_id: str, tasks: list, out_path: str, batch_size: int = 8):
    
	# check device
    device = 0 if torch.cuda.is_available() else -1

    # Load tokeniser and model
    print(f"Loading tokenizer and model for {model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)

    # build a text2text‐generation pipeline.
    print("Creating text2text-generation pipeline…")
    generator = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # extract all prompts into a list
    prompts = [t["natural language"].strip() for t in tasks]
    total_prompts = len(prompts)
    print(f"Will generate {total_prompts} outputs in batches of {batch_size}…")

    # for each chunk of size batch_size, call the pipeline
    all_generated_texts = []
    n_chunks = (total_prompts + batch_size - 1) // batch_size

    for chunk_idx in tqdm(range(n_chunks), desc="Chunks"):
        start = chunk_idx * batch_size
        end = min(start + batch_size, total_prompts)
        batch_prompts = prompts[start:end]

        # call the generator on this mini‐batch
        outputs = generator(
            batch_prompts,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False,
        )
        # collect the generated_text for each element
        for out in outputs:
            all_generated_texts.append(out["generated_text"].strip())

    assert len(all_generated_texts) == total_prompts

    # parse each generated_text as json and compare to ground truth
    results = []
    for idx, task in enumerate(tqdm(tasks, desc="Parsing outputs")):
        generated = all_generated_texts[idx]

        # attempt to parse JSON
        try:
            parsed = json.loads(generated)
            pred_answer = parsed.get("answer", "").strip()
            pred_rationale = parsed.get("rationale", "").strip()
        except json.JSONDecodeError:
            pred_answer = "ERROR"
            pred_rationale = generated

        # Combine metadata + gold label + prediction
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
        help="Output file path (e.g. flan_t5_xl_results.json)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="How many prompts to generate at once (default: 8)."
    )
    args = parser.parse_args()

    # load all tasks from the provided JSON file
    tasks_list = json.load(open(args.tasks, "r"))

    # run inference and save outputs
    run_hf_inference(args.model_id, tasks_list, args.out, batch_size=args.batch_size)
