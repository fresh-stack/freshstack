"""
This script shows an example on how to use evaluate a ColBERT model (e.g., GTE-ModernColBERT-v1) on FreshStack using PyLate documentation.
Make sure you have the PyLate repository installed: `pip install pylate`.
"""

import argparse
import logging
import os
import pathlib

from pylate import indexes, models, retrieve

from freshstack import LoggingHandler, util
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
    parser = argparse.ArgumentParser(description="Compute the retrieval metrics")
    parser.add_argument("--model_name_or_path", type=str, required=True, default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for encoding the queries and corpus documents"
    )
    parser.add_argument(
        "--query_prompt_name",
        type=str,
        default=None,
        help="The query prompt name to use for the model, default is 'default'",
    )
    parser.add_argument("--score_function", type=str, default="cos_sim")
    parser.add_argument("--queries", type=str, required=True, help="The dataset to evaluate")
    parser.add_argument(
        "--corpus", type=str, default=None, help="The corpus dataset to evaluate, if different from the main dataset"
    )
    parser.add_argument(
        "--version", type=str, default="oct-2024", help="The version of the dataset to evaluate, default is 'oct-2024'"
    )
    parser.add_argument("--topic", type=str, default="langchain")
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[5, 10, 20, 50], help="List of k values for evaluation metrics"
    )
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    ### Load the nugget qrels
    dataloader = DataLoader(queries_repo=args.queries, corpus_repo=args.corpus, version=args.version, topic=args.topic)
    corpus, queries, nuggets = dataloader.load(split="test")
    qrels_nuggets, qrels_query, query_to_nuggets = dataloader.load_qrels(split="test")

    # You can also load the freshstack answers as follows:
    # answers = dataloader.load_answers(split="test")

    if args.output_dir:
        results_dir = os.path.dirname(args.output_dir)
    else:
        results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Load the ColBERT model
    model = models.ColBERT(
        model_name_or_path=args.model_name_or_path,
        query_length=args.max_length,
        document_length=args.max_length,
    )

    # Step 2: Initialize the Voyager index
    index_folder = os.path.join(results_dir, f"{args.topic}_index")
    os.makedirs(index_folder, exist_ok=True)

    logging.info(f"Using index folder: {index_folder}")
    index = indexes.Voyager(
        index_folder=index_folder,
        index_name="index",
        override=False,  # This overwrites the existing index if any
    )

    # Step 3: Encode the documents
    documents_ids = list(corpus.keys())
    documents = [doc["text"] for doc in corpus.values()]

    if not os.path.exists(os.path.join(index_folder, "index.voyager")):
        logging.info("Index does not exist, creating a new index...")

        documents_embeddings = model.encode(
            documents,
            batch_size=args.batch_size,
            is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
            show_progress_bar=True,
        )

        # Step 4: Add document embeddings to the index by providing embeddings and corresponding ids
        index.add_documents(
            documents_ids=documents_ids,
            documents_embeddings=documents_embeddings,
        )

    logging.info("Starting retrieval evaluation...")

    # Step 5: Initialize the retriever
    retriever = retrieve.ColBERT(index=index)

    query_ids = list(queries.keys())
    queries_texts = list(queries.values())

    queries_embeddings = model.encode(
        queries_texts,
        batch_size=args.batch_size,
        is_query=True,  #  # Ensure that it is set to False to indicate that these are queries
        show_progress_bar=True,
    )

    scores = retriever.retrieve(
        queries_embeddings=queries_embeddings,
        k=max(args.k_values),  # Retrieve top-k results based on the maximum k value specified
        batch_size=1, # We have kept a batch size of 1 to avoid memory issues as 2048 tokens for query and documents can be large.
        device="cpu", # Use CPU for inference, change to "cuda" if you have a GPU available
    )

    # Step 6: Prepare the results in the required BEIR format
    results = {}
    for query_id, doc_scores in zip(query_ids, scores):
        results[query_id] = {}
        for doc_id, score in doc_scores:
            results[query_id][doc_id] = score

    ### Evaluate the retrieval results
    evaluator = EvaluateRetrieval(k_values=args.k_values)
    alpha_ndcg, coverage, recall = evaluator.evaluate(
        qrels_nuggets=qrels_nuggets, query_to_nuggets=query_to_nuggets, qrels_query=qrels_query, results=results
    )

    ### Save the evaluation results & metrics
    util.save_runfile(os.path.join(results_dir, f"{args.topic}.run.trec"), results)
    util.save_results(os.path.join(results_dir, f"{args.topic}.json"), alpha_ndcg, coverage, recall)


if __name__ == "__main__":
    main()
