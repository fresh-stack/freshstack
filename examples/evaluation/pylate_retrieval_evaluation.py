"""Evaluate ColBERT models on FreshStack using PyLate.

This script demonstrates how to evaluate a ColBERT model (e.g., GTE-ModernColBERT-v1)
on FreshStack datasets using the PyLate library. The evaluation computes retrieval
metrics including alpha-nDCG, coverage, and recall.

Requirements:
    PyLate library must be installed: `pip install pylate`

Example:
    python pylate_retrieval_evaluation.py \\
        --model_name_or_path lightonai/GTE-ModernColBERT-v1 \\
        --queries path/to/queries \\
        --topic langchain \\
        --device cuda
"""

import argparse
import logging
import os
import pathlib

from pylate import indexes, models, retrieve
from freshstack import LoggingHandler, util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval

# Configure logging to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def main():
    """Execute the ColBERT retrieval evaluation pipeline.
    
    This function performs the following steps:
    1. Parse command-line arguments
    2. Load queries, corpus, and nuggets from FreshStack datasets
    3. Initialize and load the ColBERT model
    4. Create PLAID index and encode documents
    5. Encode queries and retrieve relevant documents
    6. Evaluate retrieval results using FreshStack metrics
    7. Save results and metrics to output directory
    """
    parser = argparse.ArgumentParser(
        description="Compute retrieval metrics for ColBERT models on FreshStack datasets"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model name or local path to the ColBERT model"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for queries and documents"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding queries and corpus documents"
    )
    parser.add_argument(
        "--search_batch_size",
        type=int,
        default=1,
        help="Batch size for encoding queries and corpus documents"
    )
    parser.add_argument(
        "--query_prompt_name",
        type=str,
        default=None,
        help="Query prompt name to use for the model (default: None)"
    )
    parser.add_argument(
        "--score_function",
        type=str,
        default="cos_sim",
        help="Scoring function to use (default: cos_sim)"
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path or identifier for the queries dataset"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path or identifier for the corpus dataset (if different from queries dataset)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="langchain",
        help="Topic name for the evaluation dataset"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="List of k values for computing retrieval metrics"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./results)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (cuda or cpu)"
    )

    args = parser.parse_args()

    # Load queries, corpus, nuggets, and qrels from FreshStack datasets
    dataloader = DataLoader(
        queries_repo=args.queries,
        corpus_repo=args.corpus,
        topic=args.topic
    )
    corpus, queries, nuggets = dataloader.load(split="test")
    qrels_nuggets, qrels_query, query_to_nuggets = dataloader.load_qrels(split="test")

    # Note: FreshStack answers can be loaded with:
    # answers = dataloader.load_answers(split="test")

    # Set up output directory for results
    if args.output_dir:
        results_dir = os.path.dirname(args.output_dir)
    else:
        results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Load the ColBERT model
    logging.info(f"Loading ColBERT model: {args.model_name_or_path}")
    model = models.ColBERT(
        model_name_or_path=args.model_name_or_path,
        query_length=args.max_length,
        document_length=args.max_length,
    )

    # Step 2: Initialize the PLAID index
    index_folder = os.path.join(results_dir, f"{args.topic}")
    os.makedirs(index_folder, exist_ok=True)
    logging.info(f"Using index folder: {index_folder}")

    index = indexes.PLAID(
        index_folder=index_folder,
        index_name="fast_plaid_index",
        override=True,  # Set to True to overwrite existing index
        use_triton=False,
        device=args.device,
    )

    # Step 3: Encode corpus documents
    logging.info("Encoding corpus documents...")
    documents_ids = list(corpus.keys())
    documents = [doc["text"] for doc in corpus.values()]

    documents_embeddings = model.encode(
        documents,
        batch_size=args.batch_size,
        is_query=False,  # Set to False for document encoding
        show_progress_bar=True,
    )

    # Step 4: Add document embeddings to the PLAID index
    logging.info("Adding document embeddings to PLAID index...")
    index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=documents_embeddings,
    )

    # Step 5: Initialize the retriever and encode queries
    logging.info("Starting retrieval evaluation...")
    retriever = retrieve.ColBERT(index=index)

    query_ids = list(queries.keys())
    queries_texts = list(queries.values())

    logging.info("Encoding queries...")
    queries_embeddings = model.encode(
        queries_texts,
        batch_size=args.batch_size,
        is_query=True,  # Set to True for query encoding
        show_progress_bar=True,
    )

    # Step 6: Retrieve relevant documents for queries
    logging.info(f"Retrieving top-{max(args.k_values)} documents for each query...")
    scores = retriever.retrieve(
        queries_embeddings=queries_embeddings,
        k=max(args.k_values),  # Retrieve top-k results based on maximum k value
        batch_size=args.search_batch_size,  # Batch size for search
        device=args.device,
    )

    # Step 7: Format retrieval results in BEIR-compatible format
    logging.info("Formatting retrieval results...")
    results = {}
    for query_id, doc_scores in zip(query_ids, scores):
        results[query_id] = {}
        for doc_score in list(doc_scores): 
            results[query_id][doc_score["id"]] = float(doc_score["score"])

    # Step 8: Evaluate retrieval performance using FreshStack metrics
    logging.info("Evaluating retrieval results...")
    evaluator = EvaluateRetrieval(k_values=args.k_values)
    alpha_ndcg, coverage, recall = evaluator.evaluate(
        qrels_nuggets=qrels_nuggets,
        query_to_nuggets=query_to_nuggets,
        qrels_query=qrels_query,
        results=results
    )

    # Step 9: Save evaluation results and metrics to disk
    logging.info(f"Saving results to {results_dir}")
    util.save_runfile(os.path.join(results_dir, f"{args.topic}.run.trec"), results)
    util.save_results(os.path.join(results_dir, f"{args.topic}.json"), alpha_ndcg, coverage, recall)
    
    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
