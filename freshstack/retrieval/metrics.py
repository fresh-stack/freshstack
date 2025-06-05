from __future__ import annotations

import pyndeval
import pytrec_eval
from pyndeval import ScoredDoc, SubtopicQrel


# Evaluation metric for alpha-nDCG@k metric
def alpha_ndcg(
    qrels_nuggets: dict[str, dict[str, int]],
    query_to_nuggets: dict[str, list[str]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    _alpha_ndcg = {}
    qrels = []

    # pndeval only allows k values up to 20
    k_values = [k for k in k_values if k <= 20]
    k_max = max(k_values)
    for k in k_values:
        _alpha_ndcg[f"alpha-nDCG@{k}"] = 0.0

    # Build the qrels first for the subtopics
    for query_id in query_to_nuggets:
        for nugget_id in query_to_nuggets[query_id]:
            if nugget_id in qrels_nuggets:
                # populate qrels with the nugget_id
                for doc_id, score in qrels_nuggets[nugget_id].items():
                    qrels.append(SubtopicQrel(query_id, nugget_id, doc_id, score))

    # relevance evaluator for qrels (using default alpha value of 0.5)
    ev = pyndeval.RelevanceEvaluator(qrels, measures=[f"alpha-nDCG@{k}" for k in k_values])

    for query_id, doc_scores in results.items():
        if query_id not in query_to_nuggets:
            continue
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        runs = [ScoredDoc(query_id, doc_id, score) for doc_id, score in top_hits]
        scores = ev.evaluate(runs)
        for k in k_values:
            if f"alpha-nDCG@{k}" in scores[query_id]:
                _alpha_ndcg[f"alpha-nDCG@{k}"] += scores[query_id][f"alpha-nDCG@{k}"]

    # Average the alpha-nDCG for all queries
    for k in k_values:
        if len(results) > 0:
            _alpha_ndcg[f"alpha-nDCG@{k}"] = round(_alpha_ndcg[f"alpha-nDCG@{k}"] / len(query_to_nuggets), 4)
        else:
            _alpha_ndcg[f"alpha-nDCG@{k}"] = 0.0

    return _alpha_ndcg


# Evaluation metric for nugget coverage
def coverage(
    qrels_nuggets: dict[str, dict[str, int]],
    query_to_nuggets: dict[str, list[str]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    # compute the coverage for the given k values
    Coverage, scores = {}, {}
    k_max = max(k_values)
    for k in k_values:
        Coverage[f"Coverage@{k}"] = 0.0

    for query_id in query_to_nuggets.keys():
        scores[query_id] = {}
        doc_scores = results.get(query_id, {})
        coverage_per_query = {k: 0 for k in k_values}

        # Get the top k documents & nugget ids
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        nugget_ids = query_to_nuggets[query_id]

        for nugget_id in nugget_ids:
            for k in k_values:
                retrieved_docs = [row[0] for row in top_hits[0:k] if qrels_nuggets[nugget_id].get(row[0], 0) > 0]
                coverage_per_query[k] += 1 if len(retrieved_docs) > 0 else 0

        for k in k_values:
            scores[query_id].update({f"Coverage@{k}": round(coverage_per_query[k] / len(nugget_ids), 5)})

    # Average the coverage for all queries
    for k in k_values:
        Coverage[f"Coverage@{k}"] = round(
            sum([scores[query_id][f"Coverage@{k}"] for query_id in scores]) / len(query_to_nuggets), 4
        )

    return Coverage


def recall(
    qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k_values: list[int]
) -> dict[str, float]:
    Recall = {}
    for k in k_values:
        Recall[f"Recall@{k}"] = 0.0

    # compute the Recall for the given k values
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            Recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    # average the Recall for all queries
    for k in k_values:
        Recall[f"Recall@{k}"] = round(Recall[f"Recall@{k}"] / len(qrels), 4)

    return Recall


def ndcg(
    qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k_values: list[int]
) -> dict[str, float]:
    Ndcg = {}
    for k in k_values:
        Ndcg[f"NDCG@{k}"] = 0.0

    # compute the ndcg for the given k values
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            Ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]

    # average the ndcg for all queries
    for k in k_values:
        Ndcg[f"NDCG@{k}"] = round(Ndcg[f"NDCG@{k}"] / len(qrels), 5)

    return Ndcg


def hole(
    qrels_query: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k_values: list[int]
) -> dict[str, float]:
    Hole = {}

    annotated_corpus = set()
    for _, docs in qrels_query.items():
        for doc_id, _ in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)

    for query_id, scores in results.items():
        Hole[query_id] = {}
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        for k in k_values:
            hole_docs = [row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus]
            Hole[query_id].update({f"Hole@{k}": len(hole_docs) / k})

    return Hole
