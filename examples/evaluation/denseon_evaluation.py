"""
Evaluate lightonai/DenseOn on all five FreshStack topics.

DenseOn is a standard SentenceTransformer (ModernBERT, CLS pooling, cosine similarity) with
saved "query: " / "document: " prompts. Retrieval is exact cosine search over the whole corpus.

Both sides are truncated at 2048 tokens to match the other FreshStack embedding runs
(the checkpoint saves 512). The model runs in float32, its stored precision and the model
card's default.

python examples/evaluation/denseon_evaluation.py --batch_size 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval as BEIREval
from beir.retrieval.models.util import extract_corpus_sentences
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import SentenceTransformer

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval

MODEL_ID = "lightonai/DenseOn"
REVISION = "cb9947ebccb33862d24e3c7ca2edb25e51acd887"
OUTPUT_NAME = "denseon_results"
MAX_LENGTH = 2048
DIMENSION = 768
PROMPTS = {"query": "query: ", "document": "document: "}
TOPICS = ("langchain", "yolo", "godot", "angular", "laravel")
K_VALUES = [5, 10, 20, 50]
LOGGER = logging.getLogger(__name__)


class DenseEncoder:
    """BEIR adapter applying the saved query/document prompts through encode_query/encode_document."""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs):
        return self.model.encode_query(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus, batch_size: int = 16, **kwargs):
        texts = extract_corpus_sentences(corpus=corpus, sep=" ")
        return self.model.encode_document(texts, batch_size=batch_size, **kwargs)


def load_model(cache_dir: Path) -> tuple[SentenceTransformer, int]:
    """Load in float32, returning the model with its saved length before the 2048 override."""
    model = SentenceTransformer(MODEL_ID, revision=REVISION, cache_folder=str(cache_dir),
                                model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
                                device="cuda:0")
    saved = model.max_seq_length
    model.max_seq_length = MAX_LENGTH
    model.eval()
    return model, saved


def check_model(model: SentenceTransformer) -> None:
    """Verify prompts, 2048-token truncation, output shape and batch-padding invariance on GPU."""
    if model.prompts != PROMPTS:
        raise ValueError(f"Unexpected prompts {model.prompts}")
    long_text = "token " * (MAX_LENGTH * 3)
    for task, prompt in PROMPTS.items():
        features = model.preprocess(["short text", long_text], prompt=prompt, task=task)
        expected = model.tokenizer([prompt + t for t in ("short text", long_text)],
                                   padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        if not torch.equal(features["input_ids"], expected["input_ids"]):
            raise ValueError(f"{task} preprocessing differs from '{prompt}' + text truncated at {MAX_LENGTH}")
        if features["input_ids"].shape[1] != MAX_LENGTH:
            raise ValueError(f"{task} inputs are truncated at {features['input_ids'].shape[1]}, not {MAX_LENGTH}")

    q = model.encode_query(["How do I stream tokens from a chain?"], convert_to_tensor=True)
    d = model.encode_document(["Use the streaming callback handler to receive tokens from a chain.",
                               "Unrelated: configure the build system for the renderer."], convert_to_tensor=True)
    if q.shape[1] != DIMENSION or not torch.isfinite(d).all():
        raise ValueError("Invalid dense output")
    scores = model.similarity(q, d)[0]
    if scores[0] <= scores[1]:
        raise ValueError("Relevant document did not outscore the unrelated one")

    # A document must embed identically alone and when padded next to a much longer one.
    short = "def stream(chain):\n    for chunk in chain.stream(inputs):\n        print(chunk)"
    alone = model.encode_document([short], convert_to_tensor=True)[0].float()
    batched = model.encode_document([short, "word " * 1500], batch_size=2, convert_to_tensor=True)[0].float()
    cosine = torch.nn.functional.cosine_similarity(alone, batched, dim=0).item()
    if cosine < 0.9999:
        raise ValueError(f"Embeddings change with batch padding (cosine {cosine:.5f})")
    LOGGER.info("Prompts, %d-token truncation, outputs and padding invariance confirmed (cosine %.6f)",
                MAX_LENGTH, cosine)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache_dir", type=Path, default=Path.home() / "huggingface-cache")
    parser.add_argument("--results_root", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--topics", choices=TOPICS, nargs="+", default=list(TOPICS))
    parser.add_argument("--check_model_only", action="store_true", help="Load and verify the model, then stop")
    args = parser.parse_args()
    if version("sentence-transformers").split(".")[0] != "6":
        parser.error("This adapter requires SentenceTransformers 6")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these evaluations")
    model, checkpoint_length = load_model(args.cache_dir)
    check_model(model)
    if args.check_model_only:
        return

    output_dir = args.results_root / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": MODEL_ID,
        "revision": REVISION,
        "architecture": "SentenceTransformer (ModernBERT -> CLS pooling)",
        "parameters": sum(p.numel() for p in model.parameters()),
        "query_max_length": MAX_LENGTH,
        "document_max_length": MAX_LENGTH,
        "checkpoint_lengths": {"max_seq_length": checkpoint_length},
        "prompts": PROMPTS,
        "document_text": "title + ' ' + text, stripped",
        "dimension": DIMENSION,
        "dtype": "float32",
        "attention": "sdpa",
        "batch_size": args.batch_size,
        "score_function": "cos_sim (exact)",
        "topics": args.topics,
        "queries": "freshstack/queries-oct-2024",
        "corpus": "freshstack/corpus-oct-2024",
        "split": "test",
        "gpu": torch.cuda.get_device_name(0),
        "versions": {name: version(name) for name in ("torch", "transformers", "sentence-transformers", "beir")},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "configuration.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for topic in args.topics:
        LOGGER.info("Starting %s on %s", MODEL_ID, topic)
        loader = DataLoader(queries_repo=manifest["queries"], corpus_repo=manifest["corpus"], topic=topic)
        corpus, queries, _ = loader.load(split="test")
        qrels_nuggets, qrels_query, query_to_nuggets = loader.load_qrels(split="test")
        retriever = BEIREval(DenseRetrievalExactSearch(DenseEncoder(model), batch_size=args.batch_size),
                             score_function="cos_sim")
        results = retriever.retrieve(corpus=corpus, queries=queries)
        results = {qid: docs for qid, docs in results.items() if qid in qrels_query}
        alpha_ndcg, coverage, recall = EvaluateRetrieval(k_values=K_VALUES).evaluate(
            qrels_nuggets=qrels_nuggets, query_to_nuggets=query_to_nuggets, qrels_query=qrels_query, results=results
        )
        util.save_runfile(str(output_dir / f"{topic}.run.trec"), results)
        util.save_results(str(output_dir / f"{topic}.json"), alpha_ndcg, coverage, recall)
        LOGGER.info("Completed %s on %s: alpha-nDCG@10 %.4f", MODEL_ID, topic, alpha_ndcg["alpha-nDCG@10"])
    LOGGER.info("All requested topics complete for %s", MODEL_ID)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
    )
    main()
