from __future__ import annotations


def nuggets_to_query_qrels(
    qrels_nuggets: dict[str, dict[str, int]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """
    Convert nuggets to qrels format.
    Args:
        qrels_nuggets (dict): Dictionary containing nugget IDs as keys and dictionaries of document IDs and relevance scores as values.
    Returns:
        query_nuggets: Dictionary with query IDs as keys and lists of nugget IDs as values.
        qrels: Dictionary in qrels format with query IDs as keys and dictionaries of document IDs and relevance scores as values.
    """
    # All nuggets are in the format of query_id_0, query_id_1, etc.
    query_ids = list(set([nugget_id.split("_")[0] for nugget_id in qrels_nuggets.keys()]))
    query_nuggets = {query_id: [] for query_id in query_ids}

    for query_id in query_ids:
        nugget_ids = [nugget_id for nugget_id in qrels_nuggets.keys() if nugget_id.startswith(query_id)]
        query_nuggets[query_id] = nugget_ids

    qrels = {query_id: {} for query_id in query_ids}
    for query_id in query_ids:
        for nugget_id in query_nuggets[query_id]:
            # For each nugget, we need to get the document scores
            docs_scores = qrels_nuggets[nugget_id]
            # For each document, we need to add the score to the qrels
            for doc_id, score in docs_scores.items():
                if doc_id not in qrels[query_id]:
                    qrels[query_id][doc_id] = 0
                qrels[query_id][doc_id] += score

    return query_nuggets, qrels
