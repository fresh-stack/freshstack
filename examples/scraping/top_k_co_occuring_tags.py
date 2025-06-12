"""
This code saves the top-k tags as a JSONL file computed by counting the Stack Overflow posts.
export DIR=<your_output_dir>
for keyword in laravel; do
    python top_k_co_occuring_tags.py \
        --input_filepath ${DIR}/${keyword}/Posts.${keyword}.jsonl \
        --output_filepath ${DIR}/${keyword}/top25.tags.${keyword}.txt \
        --top_k 25
done
"""

import argparse
import json
import os
from collections import Counter


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_filepath", type=str, help="Path to the JSONL file")
    argparser.add_argument("--output_filepath", type=str, help="Path to the output file")
    argparser.add_argument("--top_k", type=int, default=25, help="Number of top tags to extract")
    args = argparser.parse_args()

    with open(args.input_filepath) as fin:
        data = [json.loads(line) for line in fin]

    tags = Counter()
    for item in data:
        if "tags" in item:
            tags.update(item["tags"])

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    with open(args.output_filepath, "w") as fout:
        for tag, count in tags.most_common(args.top_k):
            fout.write(f"{tag}\t{count}\n")


if __name__ == "__main__":
    main()
