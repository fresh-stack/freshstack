"""
This script shows an example on how to use evaluate a Sentence Transformer model (e.g., Stella) on FreshStack using BEIR dataset.
Make sure you have the BEIR repository installed: `pip install beir`.
"""
import argparse
import logging
import os
import pathlib

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
    parser = argparse.ArgumentParser(description='Compute the retrieval metrics')
    parser.add_argument('--model_name_or_path', type=str, required=True, default="NovaSearch/stella_en_1.5B_v5")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoding the queries and corpus documents")
    parser.add_argument("--query_prompt_name", type=str, default=None,
                        help="The query prompt name to use for the model, default is 'default'")
    parser.add_argument("--score_function", type=str, default="cos_sim")
    parser.add_argument('--queries', type=str, required=True, help='The dataset to evaluate')
    parser.add_argument("--corpus", type=str, default=None,
                        help="The corpus dataset to evaluate, if different from the main dataset")
    parser.add_argument("--version", type=str, default="oct-2024",
                        help="The version of the dataset to evaluate, default is 'oct-2024'")
    parser.add_argument("--topic", type=str, default="langchain")
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20, 50],
                        help='List of k values for evaluation metrics')
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    ### Load the nugget qrels
    dataloader = DataLoader(
        queries_repo=args.queries,
        corpus_repo=args.corpus,
        version=args.version,
        topic=args.topic
    )
    corpus, queries, nuggets = dataloader.load(split="test")
    qrels_nuggets, qrels_query, query_to_nuggets = dataloader.load_qrels(split="test")

    # You can also load the freshstack answers as follows:
    # answers = dataloader.load_answers(split="test")

    ### Load the Sentence Transformers model
    model = DRES(models.SentenceBERT(
        args.model_name_or_path,
        max_length=args.max_length,
        prompt_names={"query": args.query_prompt_name, "passage": None},
        trust_remote_code=True,
    ), batch_size=args.batch_size)

    retriever = BEIREval(model, score_function=args.score_function)
    results = retriever.retrieve(corpus=corpus, queries=queries)

    ### Evaluate the retrieval results
    evaluator = EvaluateRetrieval(k_values=args.k_values)
    alpha_ndcg, coverage, recall = evaluator.evaluate(
        qrels_nuggets=qrels_nuggets,
        query_to_nuggets=query_to_nuggets,
        qrels_query=qrels_query,
        results=results
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