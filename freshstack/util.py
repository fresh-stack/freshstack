from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import numpy as np
from collections import Counter, defaultdict

if importlib.util.find_spec("py7zr") is not None:
    import py7zr  # type: ignore[import]

logger = logging.getLogger(__name__)


def extract_7z(archive_path: str, extract_to: str = "."):
    """Extract a 7z archive to the specified directory."""
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=extract_to)


def _get_top_k_tags(input_filepath: str, top_k: int = 25) -> list[tuple[str, int]]:
    """
    Read a JSONL file (one JSON object per line, each with a 'tags' list),
    count all tags, and return the top_k most common as (tag, count) pairs.
    """
    tags = Counter()
    with open(input_filepath, encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            # Ensure 'tags' is a list before updating the Counter
            if "tags" in row and isinstance(row["tags"], list):
                tags.update(row["tags"])
    return tags.most_common(top_k)


def _write_top_k_tags(tag_counts: list[tuple[str, int]], output_filepath: str) -> str:
    """
    Write a list of (tag, count) pairs to a file, one per line,
    as tab-separated values. Returns the output_filepath.
    """
    with open(output_filepath, "w", encoding="utf-8") as fout:
        for tag, count in tag_counts:
            fout.write(f"{tag}\t{count}\n")
    return output_filepath


def extract_top_k_tags(input_filepath: str, output_filepath: str, top_k: int = 25) -> str:
    """
    Full pipeline: read input_filepath, get its top_k tags,
    write them to output_filepath. Returns the path written.
    """
    tag_counts = _get_top_k_tags(input_filepath, top_k)
    return _write_top_k_tags(tag_counts, output_filepath)


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
    input_files: list[str] = [f for f in os.listdir(input_dir) if f.endswith(".jsonl") and f != exclude_filename]

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

def fusion(
    results_list: list[dict[str, dict[str, float]]],
    method: str = "normalized_sum",
    depth: int = 50) -> dict[str, dict[str, float]]:
    """
    Fuse multiple retrieval results using the specified fusion method.

    Args:
        results_list: List of retrieval results dictionaries to fuse.
        method: Fusion method to use. Currently supports 'reciprocal_rank_fusion'.
        depth: Depth of documents to consider for fusion from each set.

    Returns:
        A fused retrieval results dictionary.
    """
    fused_results = {}

    if method == "normalized_sum":
        for results in results_list:
            for query_id, doc_scores in results.items():
                doc_ids = list(doc_scores.keys())[:depth]

                # Normalize the scores and sum them up
                scores = np.array(list(doc_scores.values())[:depth])
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

                for score, doc_id in sorted(zip(list(scores), doc_ids), reverse=True):
                    if query_id not in fused_results:
                        fused_results[query_id] = {doc_id: float(score)}
                    elif doc_id not in fused_results[query_id]:
                        fused_results[query_id][doc_id] = float(score)
                    else:
                        fused_results[query_id][doc_id] += float(score)
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    for query_id in fused_results:
        fused_results[query_id] = {doc_id: round(score, 3) for doc_id, score in sorted(fused_results[query_id].items(), key=lambda x: x[1], reverse=True)}

    return fused_results

def save_results(
    output_file: str,
    alpha_ndcg: dict[str, float],
    coverage: dict[str, float],
    recall: dict[str, float],
    ndcg: dict[int, float] | None = None,
    hole: dict[int, float] | None = None,
):
    optional_names = ["ndcg", "hole"]

    with open(output_file, "w") as f:
        results = {"alpha_ndcg": alpha_ndcg, "coverage": coverage, "recall": recall}

        # Add optional metrics
        for idx, metric in enumerate([ndcg, hole]):
            if metric:
                results.update({optional_names[idx]: metric})

        json.dump(results, f, indent=4)

    logger.info(f"Saved evaluation results to {output_file}")


def load_runfile(
    runfile: str,
) -> dict[str, dict[str, float]]:
    """
    Load a TREC runfile and return a dictionary with query IDs as keys
    and dictionaries of document IDs and their scores as values.
    """
    results = defaultdict(dict)
    with open(runfile) as f:
        for line in f:
            parts = line.strip().split()
            # there are document ids with spaces in them
            if len(parts) >= 7:
                query_id = parts[0]
                score = parts[-2]
                doc_id = " ".join(parts[2:-3])
            else:
                query_id, _, doc_id, _, score, _ = parts[:6]

            results[query_id][doc_id] = float(score)
    return results

def save_runfile(
    output_file: str,
    results: dict[str, dict[str, float]],
    run_name: str = "beir",
    top_k: int = 1000,
):
    with open(output_file, "w") as fOut:
        for qid, doc_dict in results.items():
            sorted_docs = sorted(doc_dict.items(), key=lambda item: item[1], reverse=True)[:top_k]
            for doc_id, score in sorted_docs:
                fOut.write(f"{qid} Q0 {doc_id} 0 {score} {run_name}\n")

