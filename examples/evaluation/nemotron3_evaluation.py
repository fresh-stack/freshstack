"""
Evaluate NVIDIA Nemotron-3-Embed-8B-BF16 on FreshStack using BEIR.
Requires CUDA-enabled PyTorch and:
`pip install beir "transformers>=5.2.0" "sentence-transformers>=5.4.1"`.
FlashAttention-2 must be installed, or use `--attn_implementation sdpa`.
Model weights are cached in ~/huggingface-cache (override with --cache_dir).

export CUDA_VISIBLE_DEVICES=0
for topic in langchain yolo godot angular laravel; do
    python examples/evaluation/nemotron3_evaluation.py \
        --model_name_or_path "nvidia/Nemotron-3-Embed-8B-BF16" \
        --max_length 2048 \
        --batch_size 1 \
        --queries "freshstack/queries-oct-2024" \
        --corpus "freshstack/corpus-oct-2024" \
        --topic $topic \
        --output_dir "./results/nemotron3_8B_results/"
done
"""

import argparse
import json
import logging
import os
import pathlib

import torch
from batch_tuning import tune_batch_size
from beir import LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval as BEIREval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nemotron-3 embeddings on FreshStack")
    parser.add_argument("--model_name_or_path", type=str, default="nvidia/Nemotron-3-Embed-8B-BF16")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(pathlib.Path.home() / "huggingface-cache"),
        help="Hugging Face cache directory for model weights and configuration",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum tokens for both queries and documents, including prompts"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for encoding the queries and corpus documents"
    )
    parser.add_argument(
        "--query_prompt",
        type=str,
        default=None,
        help="Override the model's saved query prompt text (default: use the model's 'query: ' prompt)",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["flash_attention_2", "sdpa"],
        default="flash_attention_2",
        help="Use sdpa if FlashAttention-2 is not installed",
    )
    parser.add_argument("--score_function", type=str, default="cos_sim")
    parser.add_argument("--queries", type=str, required=True, help="The dataset to evaluate")
    parser.add_argument(
        "--corpus", type=str, default=None, help="The corpus dataset to evaluate, if different from the main dataset"
    )
    parser.add_argument("--topic", type=str, default="langchain")
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[5, 10, 20, 50], help="List of k values for evaluation metrics"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--auto_batch_size", action=argparse.BooleanOptionalAction, default=True,
        help="Benchmark larger batches before evaluation; disable with --no-auto_batch_size",
    )

    args = parser.parse_args()
    cache_dir = pathlib.Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    ### Load the nugget qrels
    dataloader = DataLoader(queries_repo=args.queries, corpus_repo=args.corpus, topic=args.topic)
    corpus, queries, nuggets = dataloader.load(split="test")
    qrels_nuggets, qrels_query, query_to_nuggets = dataloader.load_qrels(split="test")

    # You can also load the freshstack answers as follows:
    # answers = dataloader.load_answers(split="test")

    ## Parameters
    model_name_or_path = args.model_name_or_path
    max_length = args.max_length

    # BEIR calls encode(), so select the saved prompts used by encode_query/encode_document explicitly.
    dense_model = models.SentenceBERT(
        model_name_or_path,
        max_length=max_length,
        cache_folder=str(cache_dir),
        prompt_names={"query": "query", "passage": "document"},
        device="cuda",
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": args.attn_implementation},
        processor_kwargs={"padding_side": "left"},
    )
    if args.query_prompt is not None:
        dense_model.q_model.prompts["query"] = args.query_prompt
    if args.auto_batch_size:
        args.batch_size, measurements = tune_batch_size(dense_model, args.batch_size, args.max_length)
        tuning_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(__file__).parent / "results"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        (tuning_dir / f"{args.topic}.batch_tuning.json").write_text(
            json.dumps({"selected_batch_size": args.batch_size, "measurements": measurements}, indent=2) + "\n"
        )

    ### Load the Sentence Transformers model
    model = DRES(
        dense_model,
        batch_size=args.batch_size,
    )

    retriever = BEIREval(model, score_function=args.score_function)
    results = retriever.retrieve(corpus=corpus, queries=queries)

    ### Evaluate the retrieval results
    evaluator = EvaluateRetrieval(k_values=args.k_values)
    alpha_ndcg, coverage, recall = evaluator.evaluate(
        qrels_nuggets=qrels_nuggets, query_to_nuggets=query_to_nuggets, qrels_query=qrels_query, results=results
    )

    ### Store the evaluation results and metrics
    results = {query_id: results[query_id] for query_id in results if query_id in qrels_query}
    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)

    ### Save the evaluation results & metrics
    util.save_runfile(os.path.join(results_dir, f"{args.topic}.run.trec"), results)
    util.save_results(os.path.join(results_dir, f"{args.topic}.json"), alpha_ndcg, coverage, recall)


if __name__ == "__main__":
    main()
