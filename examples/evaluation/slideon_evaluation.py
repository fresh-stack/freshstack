"""
Evaluate lightonai/SlideOn (learned sparse) on all five FreshStack topics.

SlideOn is a SparseEncoder (custom code in the checkpoint) on top of DenseOn: each normalized
token representation is mapped to its top-32 of 32,768 learned latents, ReLU'd and max-pooled
into one nonnegative sparse vector per text, scored by dot product. It uses the saved
"query: " / "document: " prompts and drops punctuation tokens on the document side. Vectors
are unpruned (max_active_dims=None, the model card's headline setting). Retrieval is an exact
sparse dot product of every query against every document on the GPU. SlideOn rounds its
output weights to float16, so document vectors are cached losslessly as float16 CSR shards
under <output_dir>/embeddings/<topic>/, and an interrupted run resumes without re-encoding.

Both sides are truncated at 2048 tokens to match the other FreshStack embedding runs
(the checkpoint saves 8192).

The model runs in float32, its stored precision and the model card's default. In bfloat16,
the per-token top-32 latent selection flips with kernel-level noise (a document embedded
alone vs. padded next to a long one reached only 0.989 cosine), while float32 is exact.

python examples/evaluation/slideon_evaluation.py --batch_size 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from beir import LoggingHandler
from beir.retrieval.models.util import extract_corpus_sentences
from sentence_transformers import SparseEncoder

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval

MODEL_ID = "lightonai/SlideOn"
REVISION = "370c600e9886afc43ff8026695261e56ecfaa011"
OUTPUT_NAME = "slideon_results"
MAX_LENGTH = 2048
DIMENSION = 32768
PROMPTS = {"query": "query: ", "document": "document: "}
TOP_K = 1000
TOPICS = ("langchain", "yolo", "godot", "angular", "laravel")
K_VALUES = [5, 10, 20, 50]
SHARD_DOCS = 8192
LOGGER = logging.getLogger(__name__)


def load_model(cache_dir: Path) -> tuple[SparseEncoder, dict[str, int]]:
    """Load in float32, returning the model with its saved lengths before the 2048 override."""
    model = SparseEncoder(MODEL_ID, revision=REVISION, cache_folder=str(cache_dir), trust_remote_code=True,
                          model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"}, device="cuda:0")
    settings = model[0].settings
    saved = {"query_length": settings["query_length"], "document_length": settings["document_length"]}
    settings["query_length"] = settings["document_length"] = MAX_LENGTH
    model.max_seq_length = MAX_LENGTH
    model.eval()
    return model, saved


def check_model(model: SparseEncoder) -> None:
    """Verify prompts, 2048-token truncation, output shape and batch-padding invariance on GPU."""
    if model.prompts != PROMPTS:
        raise ValueError(f"Unexpected prompts {model.prompts}")
    long_text = "token " * (MAX_LENGTH * 3)
    for task, prompt in PROMPTS.items():
        features = model.preprocess(["short text", long_text], prompt=prompt, task=task)
        expected = model.tokenizer([(prompt + t).strip() for t in ("short text", long_text)],
                                   padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        if not torch.equal(features["input_ids"], expected["input_ids"]):
            raise ValueError(f"{task} preprocessing differs from '{prompt}' + text truncated at {MAX_LENGTH}")
        if features["input_ids"].shape[1] != MAX_LENGTH:
            raise ValueError(f"{task} inputs are truncated at {features['input_ids'].shape[1]}, not {MAX_LENGTH}")

    q = model.encode_query(["How do I stream tokens from a chain?"], convert_to_sparse_tensor=True)
    d = model.encode_document(["Use the streaming callback handler to receive tokens from a chain.",
                               "Unrelated: configure the build system for the renderer."],
                              convert_to_sparse_tensor=True)
    if q.shape[1] != DIMENSION or not (d.coalesce().values() >= 0).all():
        raise ValueError("Invalid sparse output")
    values = d.coalesce().values()
    if not torch.equal(values, values.half().float()):
        raise ValueError("Sparse weights are not float16-representable; float16 caching would be lossy")
    scores = model.similarity(q, d)[0]
    if scores[0] <= scores[1]:
        raise ValueError("Relevant document did not outscore the unrelated one")

    # A document must embed identically alone and when padded next to a much longer one.
    short = "def stream(chain):\n    for chunk in chain.stream(inputs):\n        print(chunk)"
    alone = model.encode_document([short], convert_to_sparse_tensor=True)[0].to_dense().float()
    batched = model.encode_document([short, "word " * 1500], batch_size=2,
                                    convert_to_sparse_tensor=True)[0].to_dense().float()
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
    model, checkpoint_lengths = load_model(args.cache_dir)
    check_model(model)
    if args.check_model_only:
        return

    output_dir = args.results_root / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": MODEL_ID,
        "revision": REVISION,
        "architecture": "SparseEncoder (DenseOn backbone -> top-32 latents/token -> ReLU -> max pooling)",
        "parameters": sum(p.numel() for p in model.parameters()),
        "query_max_length": MAX_LENGTH,
        "document_max_length": MAX_LENGTH,
        "checkpoint_lengths": checkpoint_lengths,
        "prompts": PROMPTS,
        "document_text": "title + ' ' + text, stripped",
        "dimension": DIMENSION,
        "max_active_dims": None,
        "skiplist": "punctuation, documents only (checkpoint)",
        "dtype": "float32 (head under the checkpoint's bfloat16 autocast), float16 output weights",
        "attention": "sdpa",
        "batch_size": args.batch_size,
        "score_function": "dot (exact sparse)",
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
        LOGGER.info("Starting %s on %s", MODEL_ID, topic)
        loader = DataLoader(queries_repo=manifest["queries"], corpus_repo=manifest["corpus"], topic=topic)
        corpus, queries, _ = loader.load(split="test")
        qrels_nuggets, qrels_query, query_to_nuggets = loader.load_qrels(split="test")
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
