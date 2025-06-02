"""
for keyword in angular17; do
    python merge_corpus.py \
        --input_dir /Users/nandan.thakur/Desktop/stackoverflow/corpus/processed/beir/${keyword}
done
"""

import json, os
import argparse
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path to the jsonl file")
    args = parser.parse_args()

    input_filenames = [file for file in os.listdir(args.input_dir) if file.endswith(".jsonl") and file != "corpus.jsonl"]
    output_filename = os.path.join(args.input_dir, "corpus.jsonl")

    with open(output_filename, 'w') as fout:
        for input_filename in input_filenames:

            keyword = input_filename.split('.')[1]
            pattern = re.compile(r'^corpus\.(.*?)\.jsonl$')
            keyword = pattern.match(input_filename).group(1)
            print(f"Keyword: {keyword}")
            input_filepath = os.path.join(args.input_dir, input_filename)
            with open(input_filepath, 'r') as fin:
                for line in fin:
                    row = json.loads(line)
                    row["_id"] = f"{keyword}/{row['_id']}"
                    fout.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    # convert_jsonl_to_readme(args.jsonl_file, args.readme_file)

if __name__ == "__main__":
    main()