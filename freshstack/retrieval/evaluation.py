from __future__ import annotations

import logging

from .metrics import alpha_ndcg, coverage, hole, ndcg, recall
from .util import round_and_log

logger = logging.getLogger(__name__)


class EvaluateRetrieval:
    def __init__(self, k_values: list[int] = [1, 3, 5, 10, 100, 1000]):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.results = {}

    def evaluate(
        self,
        qrels_nuggets: dict[str, dict[str, int]],
        query_to_nuggets: dict[str, list[str]],
        qrels_query: dict[str, list[str]],
        results: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Evaluate the retrieval results using various metrics."""
        if not qrels_nuggets or not qrels_query or not results:
            raise ValueError("qrels and results must be provided for evaluation.")
        logger.info("Evaluating retrieval results...")

        ### Filter results to only include queries present in qrels_query
        results = {query_id: results[query_id] for query_id in results if query_id in qrels_query}

        ### qrels in the form of query_ids by summing up nugget qrels
        _alpha_ndcg = alpha_ndcg(
            qrels_nuggets, query_to_nuggets, results, self.k_values
        )  # pyndeval only allows k upto 20
        _coverage = coverage(qrels_nuggets, query_to_nuggets, results, self.k_values)
        _recall = recall(qrels_query, results, self.k_values)
        all_metrics = [_alpha_ndcg, _coverage, _recall]

        for metric in all_metrics:
            metric = round_and_log(metric)

        return _alpha_ndcg, _coverage, _recall

    def evaluate_custom(
        self,
        qrels_query: dict[str, list[str]],
        results: dict[str, dict[str, float]],
        metric: str = "ndcg" in ["ndcg", "hole"],
    ) -> dict[str, float]:
        """Evaluate the retrieval results using a custom metric."""
        if not qrels_query or not results:
            raise ValueError("qrels and results must be provided for evaluation.")

        ### Filter results to only include queries present in qrels_query
        results = {query_id: results[query_id] for query_id in results if query_id in qrels_query}

        if metric.lower() == "ndcg":
            return round_and_log(ndcg(qrels_query, results, self.k_values), precision=4)
        elif metric.lower() == "hole":
            return round_and_log(hole(qrels_query, results, self.k_values), precision=4)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
