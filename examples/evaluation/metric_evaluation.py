import argparse
import logging
import os
import pathlib

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
    parser = argparse.ArgumentParser(description='Compute the retrieval metrics')
    parser.add_argument('--queries', type=str, required=True, help='The dataset to evaluate')
    parser.add_argument("--corpus", type=str, default=None,
                        help="The corpus dataset to evaluate, if different from the main dataset")
    parser.add_argument("--version", type=str, default="oct-2024",
                        help="The version of the dataset to evaluate, default is 'oct-2024'")
    parser.add_argument("--topic", type=str, default="langchain")
    parser.add_argument("--results_filepath", type=str, default=None,
                        help="The file path to the results to evaluate")
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20, 50],
                        help='List of k values for evaluation metrics')
    parser.add_argument('--output_results', type=str, default=None)
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

    ### Load the results
    if args.results_filepath is None:
        raise ValueError("Please provide the results file path to evaluate.")

    # retrieval results format: query_id: {doc_id: score}
    retrieval_results = util.load_results_from_json(args.results_filepath)
    logging.info(f"Loaded {len(retrieval_results)} results from {args.results_filepath}")

    ### Evaluate the retrieval results
    evaluator = EvaluateRetrieval(k_values=args.k_values)
    alpha_ndcg, coverage, recall = evaluator.evaluate(
        qrels_nuggets=qrels_nuggets,
        query_to_nuggets=query_to_nuggets,
        qrels_query=qrels_query,
        results=retrieval_results
    )

    ### Store the evaluation results and metrics
    retrieval_results = {query_id: retrieval_results[query_id] for query_id in retrieval_results if query_id in qrels_query}
    if args.output_results:
        results_dir = os.path.dirname(args.output_results)
    else:
        results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)

    ### Save the evaluation results & metrics
    util.save_runfile(os.path.join(results_dir, f"{args.topic}.run.trec"), retrieval_results)
    util.save_results(os.path.join(results_dir, f"{args.topic}.json"), alpha_ndcg, coverage, recall)

if __name__ == "__main__":
    main()