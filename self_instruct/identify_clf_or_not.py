import os
import json
import random
import tqdm
import time
import argparse
from collections import OrderedDict

# Fireworks-compatible API helper
from gpt3_api import make_requests as make_gpt3_requests
from templates.clf_task_template import template_1

random.seed(42)

templates = {"template_1": template_1}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=str, required=True)
    parser.add_argument("--num_instructions", type=int)
    parser.add_argument("--template", type=str, default="template_1")
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--request_batch_size", type=int, default=1)  # safest default
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--organization", type=str)
    return parser.parse_args()


def clean_response(text):
    """Normalize model responses into 'Yes' or 'No'."""
    if not text:
        return "UNKNOWN"
    text = text.strip().lower()
    if "yes" in text:
        return "Yes"
    elif "no" in text:
        return "No"
    elif "classification" in text and "not" not in text:
        return "Yes"
    elif "not" in text and "classification" in text:
        return "No"
    elif "api_generation_failed" in text:
        return "FAILED"
    else:
        return "UNKNOWN"


def very_safe_make_requests(prompts, args, delay_between_calls=8, max_retries=5):
    """
    Sends API requests safely, pausing between each batch to avoid 429s.
    Automatically retries with backoff if Fireworks rate-limits.
    """
    for attempt in range(max_retries):
        try:
            results = make_gpt3_requests(
                engine=args.engine,
                prompts=prompts,
                max_tokens=5,
                temperature=0,
                top_p=0,
                frequency_penalty=0,
                presence_penalty=0,
                stop_sequences=["\n", "Task"],
                logprobs=1,
                n=1,
                best_of=1,
                api_key=args.api_key,
                organization=args.organization,
            )

            # Respect delay after *each* successful request
            time.sleep(delay_between_calls)
            return results

        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = delay_between_calls * (attempt + 1)
                print(f"Fireworks rate-limited (429). Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Unexpected error: {e}")
                time.sleep(5)

    print("Max retries reached. Skipping this batch.")
    return [{"response": None} for _ in prompts]


if __name__ == "__main__":
    args = parse_args()

    batch_file = os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")
    if not os.path.exists(batch_file):
        raise FileNotFoundError(f" Could not find input file: {batch_file}")

    with open(batch_file) as fin:
        lines = fin.readlines()
        if args.num_instructions:
            lines = lines[:args.num_instructions]

    output_path = os.path.join(
        args.batch_dir,
        f"is_clf_or_not_{os.path.basename(args.engine).replace('/', '_')}.jsonl",
    )

    # Resume progress
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in fin:
                try:
                    data = json.loads(line)
                    existing[data["instruction"]] = data["is_classification"]
                except:
                    pass
        print(f"Resuming: found {len(existing)} existing classifications.")

    progress = tqdm.tqdm(total=len(lines), desc="Classifying tasks", dynamic_ncols=True)

    with open(output_path, "a", encoding="utf-8") as fout:
        for batch_idx in range(0, len(lines), args.request_batch_size):
            batch = [json.loads(line) for line in lines[batch_idx: batch_idx + args.request_batch_size]]

            # Skip already completed
            batch = [b for b in batch if b["instruction"] not in existing]
            if not batch:
                progress.update(args.request_batch_size)
                continue

            prefix = templates[args.template]
            prompts = [f"{prefix}\nTask: {d['instruction'].strip()}\nIs it classification?" for d in batch]

            # Wait before sending (throttling)
            time.sleep(5)
            results = very_safe_make_requests(prompts, args, delay_between_calls=8)

            for i, result in enumerate(results):
                instruction = batch[i]["instruction"]
                raw_text = (
                    result.get("response", {})
                    .get("choices", [{}])[0]
                    .get("text", "")
                )
                cleaned = clean_response(raw_text)

                fout.write(
                    json.dumps(
                        OrderedDict(instruction=instruction, is_classification=cleaned),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()
                existing[instruction] = cleaned

            # Enforce cooldown between batches
            time.sleep(10)
            progress.update(len(batch))

    progress.close()
    print(f"\n Classification complete → {output_path}")
