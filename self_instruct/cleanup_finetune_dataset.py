import json
import argparse
import os
import random
from tqdm import tqdm
from statistics import mean

random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Clean, merge, and analyze fine-tuning JSONL data")
    parser.add_argument("--input_files", nargs="+", required=True,
                        help="List of fine-tune-ready JSONL files to merge and clean")
    parser.add_argument("--output_file", type=str, default="final_finetune_merged_clean.jsonl",
                        help="Cleaned output file path")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output examples")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of final samples")
    return parser.parse_args()


# -------------------- Helper Functions --------------------

def is_valid(entry):
    """Check if JSON entry has meaningful content."""
    if not entry:
        return False
    if not entry.get("instruction") or not entry.get("output"):
        return False
    if len(entry["output"].strip()) < 2:
        return False
    if entry["instruction"].strip().lower() == entry["output"].strip().lower():
        return False
    return True


def normalize(entry):
    """Clean whitespace and ensure all fields exist."""
    entry["instruction"] = entry["instruction"].strip()
    entry["input"] = entry.get("input", "").strip()
    entry["output"] = entry["output"].strip()
    return entry


def detect_task_type(entry):
    """Simple heuristic: classify as classification/generation."""
    instr = entry["instruction"].lower()
    clf_keywords = ["classify", "label", "decide", "detect", "choose", "predict"]
    return any(k in instr for k in clf_keywords)


# -------------------- Main Cleanup Function --------------------

def cleanup_dataset(input_files, output_file, shuffle=True, max_samples=None):
    data = []
    seen = set()
    total_raw = 0
    stats = {"classification": 0, "generation": 0}
    input_lens, output_lens = [], []

    for path in input_files:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        print(f"Reading {path}")
        with open(path, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=f"Processing {os.path.basename(path)}"):
                total_raw += 1
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry = normalize(entry)
                key = (entry["instruction"], entry["input"], entry["output"])

                if not is_valid(entry) or key in seen:
                    continue

                seen.add(key)
                data.append(entry)

                # Track stats
                if detect_task_type(entry):
                    stats["classification"] += 1
                else:
                    stats["generation"] += 1

                input_lens.append(len(entry["input"].split()))
                output_lens.append(len(entry["output"].split()))

    if not data:
        print("No valid entries found.")
        return

    print(f"\nLoaded {len(data)} valid examples (from {total_raw} raw).")

    if shuffle:
        random.shuffle(data)
        print("Shuffled data.")

    if max_samples:
        data = data[:max_samples]
        print(f"Limited to {max_samples} samples.")

    with open(output_file, "w", encoding="utf-8") as fout:
        for d in data:
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nCleaned dataset saved → {output_file}")
    print(f"Total samples: {len(data)}")
    print(f"Classification tasks: {stats['classification']}")
    print(f"Generation tasks: {stats['generation']}")
    print(f"Avg input length: {mean(input_lens):.1f} words")
    print(f"Avg output length: {mean(output_lens):.1f} words")


# -------------------- Entry Point --------------------

if __name__ == "__main__":
    args = parse_args()
    cleanup_dataset(args.input_files, args.output_file, args.shuffle, args.max_samples)
