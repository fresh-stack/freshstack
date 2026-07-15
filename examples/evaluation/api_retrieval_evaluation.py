"""
This script shows an example on how to evaluate an OpenAI-compatible embedding API (e.g., voyage-4-large) on FreshStack.
Make sure you have the BEIR and OpenAI repositories installed: `pip install beir openai`.

export VOYAGE_API_KEY=<your_key>
for topic in langchain yolo godot angular laravel; do
    python -m api_retrieval_evaluation \
        --model_name_or_path "voyage-4-large" \
        --base_url "https://api.voyageai.com/v1" \
        --api_key_env "VOYAGE_API_KEY" \
        --query_input_type "query" \
        --document_input_type "document" \
        --batch_size 128 \
        --queries "freshstack/queries-oct-2024" \
        --corpus "freshstack/corpus-oct-2024" \
        --topic $topic \
        --output_dir "./results/voyage_4_large_results/"
done
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import time

import numpy as np
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval as BEIREval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from openai import OpenAI

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


class OpenAIEmbeddingModel:
    """OpenAI-compatible embedding API encoder for BEIR dense retrieval."""

    def __init__(
        self,
        model_name_or_path: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        query_input_type: str | None = None,
        document_input_type: str | None = None,
        output_dimension: int | None = None,
        request_batch_size: int = 128,
        max_retries: int = 5,
    ):
        self.model_name_or_path = model_name_or_path
        self.query_input_type = query_input_type
        self.document_input_type = document_input_type
        self.output_dimension = output_dimension
        self.request_batch_size = request_batch_size
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env])

    def _embed(self, texts: list[str], input_type: str | None) -> np.ndarray:
        extra_body = {}
        if input_type is not None:
            extra_body["input_type"] = input_type
        if self.output_dimension is not None:
            extra_body["output_dimension"] = self.output_dimension

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.request_batch_size):
            batch = texts[start : start + self.request_batch_size]
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name_or_path, input=batch, extra_body=extra_body
                    )
                    ordered = sorted(response.data, key=lambda item: item.index)
                    embeddings.extend(row.embedding for row in ordered)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait = 2**attempt
                    logging.warning(f"Embedding request failed ({e}); retrying in {wait}s...")
                    time.sleep(wait)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_queries(self, queries: list[str], batch_size: int = 128, **kwargs) -> np.ndarray:
        return self._embed(queries, self.query_input_type)

    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int = 128, **kwargs) -> np.ndarray:
        texts = [(doc.get("title", "") + " " + doc.get("text", "")).strip() for doc in corpus]
        return self._embed(texts, self.document_input_type)


def main():
    parser = argparse.ArgumentParser(description="Compute the retrieval metrics")
    parser.add_argument("--model_name_or_path", type=str, required=True, default="voyage-4-large")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL of the OpenAI-compatible embedding API")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY", help="Env variable storing the API key")
    parser.add_argument(
        "--query_input_type", type=str, default=None, help="Input type for queries (e.g., 'query' for Voyage)"
    )
    parser.add_argument(
        "--document_input_type", type=str, default=None, help="Input type for documents (e.g., 'document' for Voyage)"
    )
    parser.add_argument("--output_dimension", type=int, default=None, help="Optional output embedding dimension")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for encoding the queries and corpus documents"
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

    args = parser.parse_args()

    ### Load the nugget qrels
    dataloader = DataLoader(queries_repo=args.queries, corpus_repo=args.corpus, topic=args.topic)
    corpus, queries, nuggets = dataloader.load(split="test")
    qrels_nuggets, qrels_query, query_to_nuggets = dataloader.load_qrels(split="test")

    # You can also load the freshstack answers as follows:
    # answers = dataloader.load_answers(split="test")

    ### Load the OpenAI-compatible embedding model
    model = DRES(
        OpenAIEmbeddingModel(
            model_name_or_path=args.model_name_or_path,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            query_input_type=args.query_input_type,
            document_input_type=args.document_input_type,
            output_dimension=args.output_dimension,
            request_batch_size=args.batch_size,
        ),
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
        results_dir = os.path.dirname(args.output_dir)
    else:
        results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)

    ### Save the evaluation results & metrics
    util.save_runfile(os.path.join(results_dir, f"{args.topic}.run.trec"), results)
    util.save_results(os.path.join(results_dir, f"{args.topic}.json"), alpha_ndcg, coverage, recall)


if __name__ == "__main__":
    main()
