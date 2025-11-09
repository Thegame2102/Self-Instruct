import os
import json
import re
import argparse
import tqdm
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.instance_gen_template import output_first_template_for_clf, input_first_template_for_gen

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--classification_tasks_only", action="store_true")
    parser.add_argument("--generation_tasks_only", action="store_true")
    parser.add_argument("--engine", type=str, default="accounts/fireworks/models/llama-v3p1-8b-instruct")
    parser.add_argument("--request_batch_size", type=int, default=2)
    parser.add_argument("--max_instances_to_generate", type=int, default=3)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--organization", type=str)
    return parser.parse_args()

def parse_instances(text):
    """Extract Input–Output pairs using regex."""
    pairs = re.findall(r"Input\s*:\s*(.*?)\s*Output\s*:\s*(.*?)(?=(?:Input\s*:|$))", text, re.DOTALL | re.IGNORECASE)
    results = []
    for inp, out in pairs:
        inp, out = inp.strip(), out.strip()
        if len(out) < 2 or len(out.split()) > 200:
            continue
        if inp.lower() == out.lower():
            continue
        results.append({"input": inp, "output": out})
    return results

if __name__ == "__main__":
    args = parse_args()

    # Load input tasks
    with open(os.path.join(args.batch_dir, args.input_file)) as fin:
        tasks = [json.loads(l) for l in fin]

    # Mode
    if args.classification_tasks_only:
        mode = "classification"
        template = output_first_template_for_clf
    elif args.generation_tasks_only:
        mode = "generation"
        template = input_first_template_for_gen
    else:
        raise ValueError("Specify --classification_tasks_only or --generation_tasks_only")

    if not args.output_file:
        args.output_file = f"{mode}_finetune_ready_clean.jsonl"

    out_path = os.path.join(args.batch_dir, args.output_file)
    fout = open(out_path, "w", encoding="utf-8")

    progress = tqdm.tqdm(total=len(tasks), desc=f"Generating {mode} instances")
    for i in range(0, len(tasks), args.request_batch_size):
        batch = tasks[i:i+args.request_batch_size]
        prompts = []
        for t in batch:
            instruction = t["instruction"].strip()
            prompt = (
                f"{template}\n"
                "Generate 3 high-quality examples for the following task.\n"
                "Each example must strictly follow this format:\n"
                "Input: ...\nOutput: ...\n\n"
                f"Task: {instruction}"
            )
            prompts.append(prompt)

        results = make_gpt3_requests(
            engine=args.engine,
            prompts=prompts,
            max_tokens=400,
            temperature=0.3,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0.8,
            stop_sequences=["Example", "Task:"],
            logprobs=1,
            n=1,
            best_of=1,
            api_key=args.api_key,
            organization=args.organization
        )

        for j, r in enumerate(results):
            resp_text = r["response"]["choices"][0]["text"] if r["response"] else ""
            pairs = parse_instances(resp_text)
            for p in pairs:
                fout.write(json.dumps({
                    "instruction": batch[j]["instruction"].strip(),
                    "input": p["input"],
                    "output": p["output"]
                }, ensure_ascii=False) + "\n")

        progress.update(len(batch))

    fout.close()
    print(f"\nClean aligned dataset saved to {out_path}")
