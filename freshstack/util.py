import json
import logging
import os
import re

logger = logging.getLogger(__name__)


def merge_corpus(
    input_dir: str,
    output_filename: str | None = None,
    exclude_filename: str = "corpus.jsonl",
    file_pattern: str = r"^corpus\.(.*?)\.jsonl$",
) -> str:
    """
    Merge all .jsonl files in `input_dir` (except `exclude_filename`) into one JSONL.

    Each line in each source file is parsed as JSON, its "_id" is
    prefixed with "keyword/" (keyword extracted via file_pattern)
    and then dumped to the output file.

    Args:
      input_dir: directory containing your .jsonl files.
      output_filename: path to write the merged corpus. Defaults to
                       input_dir/corpus.jsonl
      exclude_filename: filename to skip (default "corpus.jsonl").
      file_pattern: regex with one capture group to extract the keyword
                    from filenames like "corpus.angular17.jsonl".

    Returns:
      The full path to the merged JSONL file.
    """
    if output_filename is None:
        output_filename = os.path.join(input_dir, exclude_filename)

    # list all .jsonl files except the final corpus
    input_files: list[str] = [
        f
        for f in os.listdir(input_dir)
        if f.endswith(".jsonl") and f != exclude_filename
    ]

    pattern = re.compile(file_pattern)

    with open(output_filename, "w", encoding="utf-8") as fout:
        for fname in input_files:
            m = pattern.match(fname)
            if not m:
                # skip files that don’t match the naming scheme
                continue

            keyword = m.group(1)
            logger.info(f"Merging '{fname}' under keyword '{keyword}'...")

            with open(os.path.join(input_dir, fname), encoding="utf-8") as fin:
                for line in fin:
                    row = json.loads(line)
                    # prefix the _id
                    row["_id"] = f"{keyword}/{row['_id']}"
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Done: merged corpus written to {output_filename}")
    return output_filename
