"""
Evaluate Cohere Embed v5 (Pro and Fast) on all five FreshStack topics through the Cohere API.

Settings match the other FreshStack embedding runs: queries and documents truncated at 2048
tokens (truncate=END), documents embedded as "title text", cosine similarity, exact search
over the full corpus, top 1000 per query. Embeddings are requested at 2048 dimensions as
float32 with input_type search_query / search_document.

All API embeddings are saved under <output_dir>/embeddings/<topic>/{queries,corpus}/ as
float32 .npy chunks (row order given by query_ids.json / corpus_ids.json). The chunks double
as a resume cache, so an interrupted run does not re-embed (and re-bill) finished chunks.

export COHERE_API_KEY=...
python examples/evaluation/cohere_embed_evaluation.py --model embed-v5.0-pro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cohere
import numpy as np
import torch
from beir import LoggingHandler
from beir.retrieval.models.util import extract_corpus_sentences
from cohere.core.api_error import ApiError

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval

MAX_TOKENS = 2048
DIMENSION = 2048
TOP_K = 1000
API_BATCH = 96  # API maximum texts per call
CHUNK = API_BATCH * 50
TOPICS = ("langchain", "yolo", "godot", "angular", "laravel")
K_VALUES = [5, 10, 20, 50]
MODELS = {"embed-v5.0-pro": "cohere_embed_v5_pro_results", "embed-v5.0-fast": "cohere_embed_v5_fast_results"}
RETRYABLE = (429, 500, 502, 503, 504)
LOGGER = logging.getLogger(__name__)


class CohereEmbedder:
    def __init__(self, model: str, workers: int):
        self.client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"], timeout=300)
        self.model = model
        self.workers = workers
        self.input_tokens = 0

    def _call(self, texts: list[str], input_type: str) -> np.ndarray:
        for attempt in range(10):
            try:
                response = self.client.embed(
                    model=self.model,
                    input_type=input_type,
                    texts=texts,
                    embedding_types=["float"],
                    output_dimension=DIMENSION,
                    max_tokens=MAX_TOKENS,
                    truncate="END",
                )
                break
            except ApiError as error:
                if error.status_code not in RETRYABLE or attempt == 9:
                    raise
                wait = min(2 ** attempt, 60)
                LOGGER.warning("Cohere API %s, retrying in %ss", error.status_code, wait)
                time.sleep(wait)
            except Exception as error:  # network errors and timeouts
                if attempt == 9:
                    raise
                wait = min(2 ** attempt, 60)
                LOGGER.warning("Cohere request failed (%s), retrying in %ss", error, wait)
                time.sleep(wait)
        embeddings = np.asarray(response.embeddings.float_, dtype=np.float32)
        if embeddings.shape != (len(texts), DIMENSION) or not np.isfinite(embeddings).all():
            raise ValueError(f"Unexpected embeddings shape {embeddings.shape} for {len(texts)} texts")
        if response.meta and response.meta.billed_units and response.meta.billed_units.input_tokens:
            self.input_tokens += int(response.meta.billed_units.input_tokens)
        return embeddings

    def embed(self, texts: list[str], input_type: str) -> np.ndarray:
        # The API rejects empty strings; a single space keeps positions aligned.
        texts = [text if text.strip() else " " for text in texts]
        batches = [texts[i:i + API_BATCH] for i in range(0, len(texts), API_BATCH)]
        with ThreadPoolExecutor(self.workers) as pool:
            parts = list(pool.map(lambda batch: self._call(batch, input_type), batches))
        return np.concatenate(parts) if parts else np.zeros((0, DIMENSION), dtype=np.float32)

    def embed_cached(self, texts: list[str], input_type: str, cache_dir: Path) -> np.ndarray:
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256("\x00".join(texts).encode()).hexdigest()
        marker = cache_dir / "inputs.sha256"
        if marker.exists() and marker.read_text().strip() != digest:
            raise ValueError(f"Cached embeddings in {cache_dir} were made from different inputs")
        marker.write_text(digest + "\n")
        parts = []
        starts = range(0, len(texts), CHUNK)
        for number, start in enumerate(starts):
            path = cache_dir / f"chunk_{start:07d}.npy"
            if path.exists():
                parts.append(np.load(path))
                continue
            chunk = self.embed(texts[start:start + CHUNK], input_type)
            np.save(path.with_suffix(".tmp.npy"), chunk)
            path.with_suffix(".tmp.npy").replace(path)
            parts.append(chunk)
            LOGGER.info("%s: chunk %d/%d embedded (%d billed tokens so far)",
                        cache_dir.name, number + 1, len(starts), self.input_tokens)
        embeddings = np.concatenate(parts)
        if embeddings.shape[0] != len(texts):
            raise ValueError(f"Cache in {cache_dir} has {embeddings.shape[0]} rows for {len(texts)} texts")
        return embeddings


def search(query_ids, query_emb, corpus_ids, corpus_emb) -> dict[str, dict[str, float]]:
    """Exact cosine search, top 1000 per query, skipping a doc whose id equals the query id (as BEIR does)."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    queries = torch.nn.functional.normalize(torch.from_numpy(query_emb).to(device), dim=-1)
    corpus = torch.nn.functional.normalize(torch.from_numpy(corpus_emb).to(device), dim=-1)
    scores = queries @ corpus.T
    values, indices = torch.topk(scores, min(TOP_K + 1, len(corpus_ids)), dim=1)
    results = {}
    for row, qid in enumerate(query_ids):
        hits = {}
        for score, index in zip(values[row].tolist(), indices[row].tolist()):
            if corpus_ids[index] != qid and len(hits) < TOP_K:
                hits[corpus_ids[index]] = score
        results[qid] = hits
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API requests")
    parser.add_argument("--results_root", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--topics", choices=TOPICS, nargs="+", default=list(TOPICS))
    parser.add_argument("--smoke_test", action="store_true", help="Embed two texts, check the output, and exit")
    args = parser.parse_args()
    if not os.environ.get("COHERE_API_KEY"):
        parser.error("COHERE_API_KEY is not set")
    embedder = CohereEmbedder(args.model, args.workers)
    if args.smoke_test:
        for input_type in ("search_query", "search_document"):
            emb = embedder._call(["A technical question", "token " * 5000], input_type)
            LOGGER.info("%s %s: shape=%s norms=%s", args.model, input_type, emb.shape, np.linalg.norm(emb, axis=1))
        LOGGER.info("Smoke test billed %d input tokens", embedder.input_tokens)
        return

    output_dir = args.results_root / MODELS[args.model]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "provider": "Cohere API v2 /embed",
        "query_max_tokens": MAX_TOKENS,
        "document_max_tokens": MAX_TOKENS,
        "truncate": "END",
        "dimension": DIMENSION,
        "embedding_type": "float",
        "input_types": {"query": "search_query", "document": "search_document"},
        "document_text": "title + ' ' + text",
        "normalize": True,
        "score_function": "cos_sim",
        "top_k": TOP_K,
        "topics": args.topics,
        "queries": "freshstack/queries-oct-2024",
        "corpus": "freshstack/corpus-oct-2024",
        "split": "test",
        "versions": {name: version(name) for name in ("cohere", "torch", "beir")},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "configuration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for topic in args.topics:
        LOGGER.info("Starting %s on %s", args.model, topic)
        loader = DataLoader(queries_repo=manifest["queries"], corpus_repo=manifest["corpus"], topic=topic)
        corpus, queries, _ = loader.load(split="test")
        qrels_nuggets, qrels_query, query_to_nuggets = loader.load_qrels(split="test")
        query_ids, corpus_ids = list(queries), list(corpus)
        cache = output_dir / "embeddings" / topic
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "query_ids.json").write_text(json.dumps(query_ids))
        (cache / "corpus_ids.json").write_text(json.dumps(corpus_ids))
        query_emb = embedder.embed_cached([queries[q] for q in query_ids], "search_query", cache / "queries")
        corpus_emb = embedder.embed_cached(
            extract_corpus_sentences(corpus=[corpus[c] for c in corpus_ids], sep=" "),
            "search_document", cache / "corpus")
        results = search(query_ids, query_emb, corpus_ids, corpus_emb)
        results = {qid: docs for qid, docs in results.items() if qid in qrels_query}
        alpha_ndcg, coverage, recall = EvaluateRetrieval(k_values=K_VALUES).evaluate(
            qrels_nuggets=qrels_nuggets, query_to_nuggets=query_to_nuggets, qrels_query=qrels_query, results=results
        )
        util.save_runfile(str(output_dir / f"{topic}.run.trec"), results)
        util.save_results(str(output_dir / f"{topic}.json"), alpha_ndcg, coverage, recall)
        LOGGER.info("Completed %s on %s (%d billed tokens this run)", args.model, topic, embedder.input_tokens)
    LOGGER.info("All requested topics complete for %s", args.model)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
