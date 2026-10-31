"""
Evaluate LightOn's single-vector retrievers on all five FreshStack topics:
lightonai/DenseOn (dense) and lightonai/SlideOn (learned sparse).

DenseOn is a standard SentenceTransformer (ModernBERT, CLS pooling, cosine similarity) with
saved "query: " / "document: " prompts. Retrieval is exact cosine search over the whole corpus.

SlideOn is a SparseEncoder (custom code in the checkpoint) on top of DenseOn: each normalized
token representation is mapped to its top-32 of 32,768 learned latents, ReLU'd and max-pooled
into one nonnegative sparse vector per text, scored by dot product. It uses the same saved
prompts and drops punctuation tokens on the document side. Vectors are unpruned
(max_active_dims=None, the model card's headline setting). Retrieval is an exact sparse dot
product of every query against every document on the GPU. SlideOn rounds its output weights
to float16, so document vectors are cached losslessly as float16 CSR shards under
<output_dir>/embeddings/<topic>/, and an interrupted run resumes without re-encoding.

Both sides are truncated at 2048 tokens to match the other models in
results/embedding_comparison.csv (the checkpoints save 512 for DenseOn and 8192 for SlideOn).

Both models run in float32, their stored precision and the model cards' default. In bfloat16,
SlideOn's per-token top-32 latent selection flips with kernel-level noise (a document embedded
alone vs. padded next to a long one reached only 0.989 cosine), while float32 is exact.

python examples/evaluation/lighton_evaluation.py --model denseon --batch_size 64
python examples/evaluation/lighton_evaluation.py --model slideon --batch_size 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval as BEIREval
from beir.retrieval.models.util import extract_corpus_sentences
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import SentenceTransformer, SparseEncoder

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval


@dataclass(frozen=True)
class LightOnModel:
    model_id: str
    revision: str
    output_name: str
    sparse: bool


MODELS = {
    "denseon": LightOnModel("lightonai/DenseOn", "cb9947ebccb33862d24e3c7ca2edb25e51acd887", "denseon_results", False),
    "slideon": LightOnModel("lightonai/SlideOn", "370c600e9886afc43ff8026695261e56ecfaa011", "slideon_results", True),
}
MAX_LENGTH = 2048
PROMPTS = {"query": "query: ", "document": "document: "}
TOP_K = 1000
TOPICS = ("langchain", "yolo", "godot", "angular", "laravel")
K_VALUES = [5, 10, 20, 50]
SHARD_DOCS = 8192
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


def load_model(spec: LightOnModel, cache_dir: Path) -> tuple[SentenceTransformer | SparseEncoder, dict[str, int]]:
    """Load in float32, returning the model with its saved lengths before the 2048 override."""
    model_kwargs = {"dtype": torch.float32, "attn_implementation": "sdpa"}
    if spec.sparse:
        model = SparseEncoder(spec.model_id, revision=spec.revision, cache_folder=str(cache_dir),
                              trust_remote_code=True, model_kwargs=model_kwargs, device="cuda:0")
        settings = model[0].settings
        saved = {"query_length": settings["query_length"], "document_length": settings["document_length"]}
        settings["query_length"] = settings["document_length"] = MAX_LENGTH
    else:
        model = SentenceTransformer(spec.model_id, revision=spec.revision, cache_folder=str(cache_dir),
                                    model_kwargs=model_kwargs, device="cuda:0")
        saved = {"max_seq_length": model.max_seq_length}
    model.max_seq_length = MAX_LENGTH
    model.eval()
    return model, saved


def check_model(model: SentenceTransformer | SparseEncoder, spec: LightOnModel) -> None:
    """Verify prompts, 2048-token truncation, output shape and batch-padding invariance on GPU."""
    if model.prompts != PROMPTS:
        raise ValueError(f"Unexpected prompts {model.prompts}")
    long_text = "token " * (MAX_LENGTH * 3)
    for task, prompt in PROMPTS.items():
        features = model.preprocess(["short text", long_text], prompt=prompt, task=task)
        expected = model.tokenizer([(prompt + t).strip() if spec.sparse else prompt + t for t in ("short text", long_text)],
                                   padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        if not torch.equal(features["input_ids"], expected["input_ids"]):
            raise ValueError(f"{task} preprocessing differs from '{prompt}' + text truncated at {MAX_LENGTH}")
        if features["input_ids"].shape[1] != MAX_LENGTH:
            raise ValueError(f"{task} inputs are truncated at {features['input_ids'].shape[1]}, not {MAX_LENGTH}")

    query = ["How do I stream tokens from a chain?"]
    documents = ["Use the streaming callback handler to receive tokens from a chain.",
                 "Unrelated: configure the build system for the renderer."]
    if spec.sparse:
        q = model.encode_query(query, convert_to_sparse_tensor=True)
        d = model.encode_document(documents, convert_to_sparse_tensor=True)
        if q.shape[1] != 32768 or not (d.coalesce().values() >= 0).all():
            raise ValueError("Invalid sparse output")
        values = d.coalesce().values()
        if not torch.equal(values, values.half().float()):
            raise ValueError("Sparse weights are not float16-representable; float16 caching would be lossy")
        scores = model.similarity(q, d)[0]
        dense_of = lambda x: x.to_dense().float()
    else:
        q = model.encode_query(query, convert_to_tensor=True)
        d = model.encode_document(documents, convert_to_tensor=True)
        if q.shape[1] != 768 or not torch.isfinite(d).all():
            raise ValueError("Invalid dense output")
        scores = model.similarity(q, d)[0]
        dense_of = lambda x: torch.nn.functional.normalize(x.float(), dim=-1)
    if scores[0] <= scores[1]:
        raise ValueError("Relevant document did not outscore the unrelated one")

    # A document must embed identically alone and when padded next to a much longer one.
    short = "def stream(chain):\n    for chunk in chain.stream(inputs):\n        print(chunk)"
    kwargs = {"convert_to_sparse_tensor": True} if spec.sparse else {"convert_to_tensor": True}
    alone = dense_of(model.encode_document([short], **kwargs))[0]
    batched = dense_of(model.encode_document([short, "word " * 1500], batch_size=2, **kwargs))[0]
    cosine = torch.nn.functional.cosine_similarity(alone, batched, dim=0).item()
    if cosine < 0.9999:
        raise ValueError(f"Embeddings change with batch padding (cosine {cosine:.5f})")
    LOGGER.info("Prompts, %d-token truncation, outputs and padding invariance confirmed (cosine %.6f)",
                MAX_LENGTH, cosine)


def encode_sparse_corpus(model: SparseEncoder, texts: list[str], ids: list[str], cache: Path, batch_size: int):
    """Encode documents into float16 CSR shards, resuming finished shards. Returns (indptr, indices, values)."""
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256("\x00".join(f"{i}\x01{t}" for i, t in zip(ids, texts)).encode()).hexdigest()
    meta_path = cache / "meta.json"
    meta = {"fingerprint": fingerprint, "documents": len(ids), "shard_docs": SHARD_DOCS, "max_length": MAX_LENGTH}
    if meta_path.is_file() and json.loads(meta_path.read_text()) != meta:
        raise ValueError(f"{cache} holds embeddings for different inputs or settings; remove it to re-encode")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    (cache / "corpus_ids.json").write_text(json.dumps(ids))

    counts, indices, values = [], [], []
    for shard, start in enumerate(range(0, len(texts), SHARD_DOCS)):
        path = cache / f"shard_{shard:04d}.npz"
        if not path.is_file():
            began = time.time()
            embedded = model.encode_document(texts[start:start + SHARD_DOCS], batch_size=batch_size,
                                             convert_to_sparse_tensor=True, show_progress_bar=False)
            csr = embedded.coalesce().to_sparse_csr().cpu()
            np.savez(path.with_suffix(".tmp.npz"), counts=np.diff(csr.crow_indices().numpy()).astype(np.int32),
                     indices=csr.col_indices().numpy().astype(np.int32),
                     values=csr.values().numpy().astype(np.float16))
            path.with_suffix(".tmp.npz").replace(path)
            done = min(start + SHARD_DOCS, len(texts))
            LOGGER.info("Encoded documents %d-%d of %d (%.1f docs/s, %.0f nonzeros/doc)", start, done, len(texts),
                        (done - start) / (time.time() - began), csr.values().numel() / (done - start))
            del embedded, csr
        with np.load(path) as shard_data:
            counts.append(shard_data["counts"])
            indices.append(shard_data["indices"])
            values.append(shard_data["values"])
    counts = np.concatenate(counts)
    indptr = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    if len(counts) != len(ids) or indptr[-1] != sum(len(i) for i in indices):
        raise ValueError("Cached shards do not cover every document")
    return indptr, np.concatenate(indices), np.concatenate(values)


def sparse_search(queries: torch.Tensor, indptr: np.ndarray, indices: np.ndarray, values: np.ndarray) -> torch.Tensor:
    """Exact dot product of every query against every document; returns a (queries, documents) float32 matrix."""
    device = queries.device
    dense_queries = queries.to_dense().float().T.contiguous()  # (latents, queries)
    documents = len(indptr) - 1
    scores = torch.empty(queries.shape[0], documents, dtype=torch.float32, device=device)
    for start in range(0, documents, SHARD_DOCS * 4):
        end = min(start + SHARD_DOCS * 4, documents)
        lo, hi = indptr[start], indptr[end]
        block = torch.sparse_csr_tensor(
            torch.from_numpy(indptr[start:end + 1] - lo).to(device),
            torch.from_numpy(indices[lo:hi].astype(np.int64)).to(device),
            torch.from_numpy(values[lo:hi]).to(device, torch.float32),
            size=(end - start, dense_queries.shape[0]),
        )
        scores[:, start:end] = (block @ dense_queries).T
    return scores


def check_sparse_search(model: SparseEncoder, queries: torch.Tensor, indptr, indices, values) -> None:
    """Compare the CSR search against SentenceTransformers' own similarity on a document sample."""
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(indptr) - 1, size=min(300, len(indptr) - 1), replace=False))
    rows, cols, vals = [], [], []
    for row, doc in enumerate(sample):
        lo, hi = indptr[doc], indptr[doc + 1]
        rows.append(np.full(hi - lo, row))
        cols.append(indices[lo:hi])
        vals.append(values[lo:hi])
    reference_docs = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([np.concatenate(rows), np.concatenate(cols)])),
        torch.from_numpy(np.concatenate(vals).astype(np.float32)), size=(len(sample), queries.shape[1]),
    ).to(queries.device)
    subset = torch.stack([queries[i] for i in range(min(20, queries.shape[0]))])
    reference = model.similarity(subset, reference_docs).float()
    sub_indptr = np.concatenate([[0], np.cumsum([indptr[d + 1] - indptr[d] for d in sample])])
    ours = sparse_search(subset, sub_indptr, np.concatenate(cols), np.concatenate(vals))
    error = (reference - ours).abs().max().item() / reference.abs().max().item()
    if error > 1e-4:
        raise ValueError(f"CSR search differs from SentenceTransformers similarity (relative error {error:.2e})")
    LOGGER.info("CSR search matches SentenceTransformers similarity (max relative error %.1e)", error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache_dir", type=Path, default=Path.home() / "huggingface-cache")
    parser.add_argument("--results_root", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--topics", choices=TOPICS, nargs="+", default=list(TOPICS))
    parser.add_argument("--check_model_only", action="store_true", help="Load and verify the model, then stop")
    args = parser.parse_args()
    if version("sentence-transformers").split(".")[0] != "6":
        parser.error("This adapter requires SentenceTransformers 6")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these evaluations")
    spec = MODELS[args.model]
    model, checkpoint_lengths = load_model(spec, args.cache_dir)
    check_model(model, spec)
    if args.check_model_only:
        return

    output_dir = args.results_root / spec.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": spec.model_id,
        "revision": spec.revision,
        "architecture": ("SparseEncoder (DenseOn backbone -> top-32 latents/token -> ReLU -> max pooling)"
                         if spec.sparse else "SentenceTransformer (ModernBERT -> CLS pooling)"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "query_max_length": MAX_LENGTH,
        "document_max_length": MAX_LENGTH,
        "checkpoint_lengths": checkpoint_lengths,
        "prompts": PROMPTS,
        "document_text": "title + ' ' + text, stripped",
        "dimension": 32768 if spec.sparse else 768,
        "max_active_dims": None if spec.sparse else "n/a",
        "skiplist": "punctuation, documents only (checkpoint)" if spec.sparse else "none",
        "dtype": "float32" + (" (head under the checkpoint's bfloat16 autocast), float16 output weights"
                              if spec.sparse else ""),
        "attention": "sdpa",
        "batch_size": args.batch_size,
        "score_function": "dot (exact sparse)" if spec.sparse else "cos_sim (exact)",
        "top_k": TOP_K,
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
        LOGGER.info("Starting %s on %s", spec.model_id, topic)
        loader = DataLoader(queries_repo=manifest["queries"], corpus_repo=manifest["corpus"], topic=topic)
        corpus, queries, _ = loader.load(split="test")
        qrels_nuggets, qrels_query, query_to_nuggets = loader.load_qrels(split="test")
        if spec.sparse:
            doc_ids = list(corpus)
            texts = extract_corpus_sentences(corpus=[corpus[i] for i in doc_ids], sep=" ")
            query_ids = [q for q in queries if q in qrels_query]
            indptr, indices, values = encode_sparse_corpus(model, texts, doc_ids, output_dir / "embeddings" / topic,
                                                           args.batch_size)
            query_embeddings = model.encode_query([queries[q] for q in query_ids], batch_size=args.batch_size,
                                                  convert_to_sparse_tensor=True)
            if topic == args.topics[0]:
                check_sparse_search(model, query_embeddings, indptr, indices, values)
            scores = sparse_search(query_embeddings, indptr, indices, values)
            top_scores, top_index = scores.topk(min(TOP_K, len(doc_ids)), dim=1)
            results = {
                qid: {doc_ids[i]: float(s) for s, i in zip(row_scores.tolist(), row_index.tolist())}
                for qid, row_scores, row_index in zip(query_ids, top_scores.cpu(), top_index.cpu())
            }
            del scores
        else:
            retriever = BEIREval(DenseRetrievalExactSearch(DenseEncoder(model), batch_size=args.batch_size),
                                 score_function="cos_sim")
            results = retriever.retrieve(corpus=corpus, queries=queries)
            results = {qid: docs for qid, docs in results.items() if qid in qrels_query}
        alpha_ndcg, coverage, recall = EvaluateRetrieval(k_values=K_VALUES).evaluate(
            qrels_nuggets=qrels_nuggets, query_to_nuggets=query_to_nuggets, qrels_query=qrels_query, results=results
        )
        util.save_runfile(str(output_dir / f"{topic}.run.trec"), results)
        util.save_results(str(output_dir / f"{topic}.json"), alpha_ndcg, coverage, recall)
        LOGGER.info("Completed %s on %s: alpha-nDCG@10 %.4f", spec.model_id, topic, alpha_ndcg["alpha-nDCG@10"])
    LOGGER.info("All requested topics complete for %s", spec.model_id)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
    )
    main()
