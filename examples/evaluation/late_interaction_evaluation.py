"""
Evaluate late-interaction (ColBERT-style) multi-vector models on all five FreshStack topics:
perplexity-ai/pplx-embed-v2-late-{0.6b,9b}, lightonai/LateOn and lightonai/mLateOn.

Every token gets a 128-dimensional normalized vector, and a query-document score is MaxSim,
the sum over query tokens of the best-matching document token. All checkpoints load as
SentenceTransformers MultiVectorEncoder pipelines (Transformer, Dense layers -> 128,
MultiVectorMask, Normalize) with no remote code, and their saved "[Q] " / "[D] " markers are
used. pplx expects the marker as the very first token; the PyLate-trained LightOn models
expect it right after [CLS] / <bos>, which is how SentenceTransformers loads their saved
query_prefix / document_prefix. MultiVectorMask drops punctuation tokens on the document
side only (pplx and LateOn; mLateOn ships an empty skiplist).

Both sides are truncated at 2048 tokens to match the other models in
results/embedding_comparison.csv, instead of each checkpoint's own query/document lengths
(pplx 1024/4096, LateOn 32/300, mLateOn 8192/8192).

Retrieval is exact and exhaustive: every query is scored against every document with
full MaxSim on the GPU (no PLAID or other approximate index), and the top 1000 are kept.
Document token embeddings are stored as float16 shards under
<output_dir>/embeddings/<topic>/, so an interrupted run resumes without re-encoding.
bfloat16 to float16 conversion is lossless for these normalized vectors.

The pplx Qwen3.5 backbone mixes Gated DeltaNet linear-attention layers with full attention.
flash-linear-attention supplies the fast kernel for the delta rule. Without it,
transformers falls back to a much slower reference PyTorch path.

python examples/evaluation/late_interaction_evaluation.py --model pplx-0.6b --batch_size 64
python examples/evaluation/late_interaction_evaluation.py --model pplx-9b --batch_size 8
python examples/evaluation/late_interaction_evaluation.py --model lateon --batch_size 64
python examples/evaluation/late_interaction_evaluation.py --model mlateon --batch_size 64
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
from beir.retrieval.models.util import extract_corpus_sentences
from sentence_transformers import MultiVectorEncoder

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval


@dataclass(frozen=True)
class LateModel:
    model_id: str
    revision: str
    output_name: str
    marker_position: int  # index of the [Q]/[D] marker token in the tokenized input
    dense_layers: int
    punctuation_skiplist: bool
    needs_fla: bool


MODELS = {
    "pplx-0.6b": LateModel("perplexity-ai/pplx-embed-v2-late-0.6b", "8fc2de24534aa3610d85fa59c463313a5f096455",
                           "pplx_embed_v2_late_0.6b_results", 0, 1, True, True),
    "pplx-9b": LateModel("perplexity-ai/pplx-embed-v2-late-9b", "0f49a9977fe06b83377d598094c5c0204ce18ad9",
                         "pplx_embed_v2_late_9b_results", 0, 1, True, True),
    "lateon": LateModel("lightonai/LateOn", "62911e105059585d244384c7d17826e35f669c17",
                        "lateon_results", 1, 3, True, False),
    "mlateon": LateModel("lightonai/mLateOn", "bbd883ae1fadb563056ff73b1061ae8172a3fcda",
                         "mlateon_results", 1, 3, False, False),
}
MAX_LENGTH = 2048
DIMENSION = 128
TOP_K = 1000
TOPICS = ("langchain", "yolo", "godot", "angular", "laravel")
K_VALUES = [5, 10, 20, 50]
QUERY_MARKER, DOCUMENT_MARKER = "[Q] ", "[D] "
SHARD_DOCS = 8192
LOGGER = logging.getLogger(__name__)


def load_model(spec: LateModel, cache_dir: Path) -> tuple[MultiVectorEncoder, dict[str, int]]:
    """Load the checkpoint in bfloat16, returning it with its saved lengths before the 2048 override."""
    model = MultiVectorEncoder(
        spec.model_id,
        revision=spec.revision,
        cache_folder=str(cache_dir),
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        device="cuda:0",
    )
    checkpoint_lengths = {"query_length": model[0].query_length, "document_length": model[0].document_length}
    model[0].query_length = MAX_LENGTH
    model[0].document_length = MAX_LENGTH
    model.eval()
    return model, checkpoint_lengths


def check_model(model: MultiVectorEncoder, spec: LateModel) -> None:
    """Verify the pipeline, markers, 2048-token truncation, masking and batch invariance on GPU."""
    pipeline = [type(module).__name__ for module in model]
    if pipeline != ["Transformer", *["Dense"] * spec.dense_layers, "MultiVectorMask", "Normalize"]:
        raise ValueError(f"Unexpected pipeline: {pipeline}")
    if model.prompts != {"query": QUERY_MARKER, "document": DOCUMENT_MARKER}:
        raise ValueError(f"Unexpected prompts {model.prompts}")
    if spec.needs_fla:
        try:
            import fla  # noqa: F401
        except ImportError as error:
            raise RuntimeError("flash-linear-attention is required for practical encoding speed") from error

    long_text = "token " * (MAX_LENGTH * 3)
    for task, marker in (("query", QUERY_MARKER), ("document", DOCUMENT_MARKER)):
        ids = model.preprocess(["short text", long_text], prompt=marker, task=task)["input_ids"]
        if ids.shape[1] != MAX_LENGTH:
            raise ValueError(f"{task} inputs are truncated at {ids.shape[1]}, not {MAX_LENGTH}")
        found = model.tokenizer.convert_ids_to_tokens(ids[0].tolist())[spec.marker_position]
        if found.replace("\u2581", " ") != marker:
            raise ValueError(f"{task} marker is not token {spec.marker_position} (got {found!r})")

    query = model.encode_query(["How do I stream tokens from a chain?"])
    documents = model.encode_document(
        ["Use the streaming callback handler to receive tokens.",
         "Unrelated: configure the build system for the renderer."]
    )
    for embedding in [*query, *documents]:
        if embedding.shape[-1] != DIMENSION or not torch.isfinite(embedding).all():
            raise ValueError("Invalid inference output")
        norms = embedding.float().norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=0.02, rtol=0):
            raise ValueError("Token embeddings are not L2-normalized")
    punctuated = model.encode_document(["a, b. c!"])[0]
    plain = model.preprocess(["a, b. c!"], prompt=DOCUMENT_MARKER, task="document")["input_ids"][0]
    if (punctuated.shape[0] < len(plain)) != spec.punctuation_skiplist:
        raise ValueError("Document punctuation skiplist does not match the checkpoint")
    scores = model.similarity(query, documents)[0]
    if scores[0] <= scores[1]:
        raise ValueError("Relevant document did not outscore the unrelated one")

    # A document must embed identically alone and when padded next to a much longer one.
    short = "def stream(chain):\n    for chunk in chain.stream(inputs):\n        print(chunk)"
    alone = model.encode_document([short])[0].float()
    batched = model.encode_document([short, "word " * 1500], batch_size=2)[0].float()
    if alone.shape != batched.shape or (alone - batched).abs().max() > 0.05:
        raise ValueError("Embeddings change with batch padding")
    LOGGER.info(
        "Pipeline, %s/%s markers at token %d, %d-token truncation, skiplist, normalization and padding "
        "invariance confirmed", QUERY_MARKER.strip(), DOCUMENT_MARKER.strip(), spec.marker_position, MAX_LENGTH,
    )


def encode_corpus(model: MultiVectorEncoder, texts: list[str], ids: list[str], cache: Path, batch_size: int):
    """Encode documents into float16 shards (flat token matrix + per-doc lengths), resuming finished shards."""
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256("\x00".join(f"{i}\x01{t}" for i, t in zip(ids, texts)).encode()).hexdigest()
    meta_path = cache / "meta.json"
    meta = {"fingerprint": fingerprint, "documents": len(ids), "shard_docs": SHARD_DOCS, "max_length": MAX_LENGTH}
    if meta_path.is_file() and json.loads(meta_path.read_text()) != meta:
        raise ValueError(f"{cache} holds embeddings for different inputs or settings; remove it to re-encode")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    (cache / "corpus_ids.json").write_text(json.dumps(ids))

    shards, lengths = [], []
    for shard, start in enumerate(range(0, len(texts), SHARD_DOCS)):
        vec_path, len_path = cache / f"shard_{shard:04d}.f16.npy", cache / f"shard_{shard:04d}.lengths.npy"
        if not (vec_path.is_file() and len_path.is_file()):
            began = time.time()
            embedded = model.encode_document(texts[start:start + SHARD_DOCS], batch_size=batch_size, show_progress_bar=False)
            np.save(len_path.with_suffix(".tmp.npy"), np.array([e.shape[0] for e in embedded], dtype=np.int32))
            np.save(vec_path.with_suffix(".tmp.npy"), torch.cat(embedded).to(torch.float16).cpu().numpy())
            vec_path.with_suffix(".tmp.npy").replace(vec_path)
            len_path.with_suffix(".tmp.npy").replace(len_path)
            del embedded
            done = min(start + SHARD_DOCS, len(texts))
            LOGGER.info("Encoded documents %d-%d of %d (%.1f docs/s)", start, done, len(texts),
                        (done - start) / (time.time() - began))
        shards.append(np.load(vec_path, mmap_mode="r"))
        lengths.append(np.load(len_path))
    lengths = np.concatenate(lengths)
    tokens = np.concatenate(shards)
    if len(lengths) != len(ids) or lengths.min() < 1 or tokens.shape != (lengths.sum(), DIMENSION):
        raise ValueError("Cached shards do not cover every document with at least one vector")
    return tokens, lengths


@torch.no_grad()
def maxsim_search(queries: list[torch.Tensor], tokens: np.ndarray, lengths: np.ndarray,
                  token_budget: int) -> torch.Tensor:
    """Exact MaxSim of every query against every document; returns a (queries, documents) float32 matrix.

    All real query tokens are flattened into one matrix. Documents are visited in length order,
    padded per block to the block's longest document, and each block is scored with one matmul:
    max over a document's tokens, then sum over each query's tokens.
    """
    device = queries[0].device
    flat_queries = torch.cat(queries).to(torch.float16)
    owner = torch.repeat_interleave(torch.arange(len(queries), device=device),
                                    torch.tensor([q.shape[0] for q in queries], device=device))
    scores = torch.empty(len(queries), len(lengths), dtype=torch.float32, device=device)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    order = np.argsort(lengths, kind="stable")
    position = 0
    while position < len(order):
        # Lengths ascend, so sizing the block by its last (longest) member keeps it within budget.
        guess = order[position:position + max(token_budget // int(lengths[order[position]]), 1)]
        block = order[position:position + max(token_budget // int(lengths[guess[-1]]), 1)]
        longest = int(lengths[block].max())
        padded = np.zeros((len(block), longest, DIMENSION), dtype=np.float16)
        for row, doc in enumerate(block):
            padded[row, : lengths[doc]] = tokens[offsets[doc]:offsets[doc + 1]]
        documents = torch.from_numpy(padded).to(device, non_blocking=True)
        mask = torch.arange(longest, device=device)[None, :] < torch.from_numpy(lengths[block]).to(device)[:, None]
        similarity = flat_queries @ documents.reshape(-1, DIMENSION).T
        similarity = similarity.view(-1, len(block), longest).masked_fill(~mask[None], float("-inf"))
        best = similarity.amax(dim=2).float()
        block_scores = torch.zeros(len(queries), len(block), dtype=torch.float32, device=device)
        block_scores.index_add_(0, owner, best)
        scores[:, torch.from_numpy(block).to(device)] = block_scores
        position += len(block)
    return scores


def check_search(model: MultiVectorEncoder, queries: list[torch.Tensor], tokens, lengths, budget: int) -> None:
    """Compare the blocked search against SentenceTransformers' own MaxSim on a document sample."""
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(lengths), size=min(300, len(lengths)), replace=False))
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    pieces = [np.array(tokens[offsets[i]:offsets[i + 1]]) for i in sample]
    # Score the reference at the search's float16 precision (queries are encoded in bfloat16).
    reference = model.similarity([q.to(torch.float16) for q in queries[:20]],
                                 [torch.from_numpy(p).cuda() for p in pieces]).float()
    ours = maxsim_search(queries[:20], np.concatenate(pieces), lengths[sample], budget)
    error = (reference - ours).abs().max().item() / reference.abs().max().item()
    if error > 2e-3:
        raise ValueError(f"Blocked MaxSim differs from SentenceTransformers MaxSim (relative error {error:.2e})")
    LOGGER.info("Blocked MaxSim matches SentenceTransformers MaxSim (max relative error %.1e)", error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--token_budget", type=int, default=32768, help="Document tokens per scoring matmul")
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
    model_id = spec.model_id
    model, checkpoint_lengths = load_model(spec, args.cache_dir)
    check_model(model, spec)
    if args.check_model_only:
        return

    output_dir = args.results_root / spec.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model_id,
        "revision": spec.revision,
        "architecture": "MultiVectorEncoder (" + " -> ".join(type(module).__name__ for module in model) + ")",
        "parameters": sum(p.numel() for p in model.parameters()),
        "query_max_length": MAX_LENGTH,
        "document_max_length": MAX_LENGTH,
        "checkpoint_lengths": checkpoint_lengths,
        "marker_position": spec.marker_position,
        "prompts": {"query": QUERY_MARKER, "document": DOCUMENT_MARKER},
        "document_text": "title + ' ' + text, stripped",
        "skiplist": ("punctuation, documents only" if spec.punctuation_skiplist else "none")
                    + " (checkpoint MultiVectorMask)",
        "dimension": DIMENSION,
        "normalize": True,
        "dtype": "bfloat16 encoding, float16 token storage",
        "attention": "sdpa + flash-linear-attention" if spec.needs_fla else "sdpa",
        "batch_size": args.batch_size,
        "score_function": "maxsim (exact, exhaustive)",
        "top_k": TOP_K,
        "topics": args.topics,
        "queries": "freshstack/queries-oct-2024",
        "corpus": "freshstack/corpus-oct-2024",
        "split": "test",
        "gpu": torch.cuda.get_device_name(0),
        "versions": {name: version(name) for name in
                     ("torch", "transformers", "sentence-transformers", "flash-linear-attention", "beir")},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "configuration.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for topic in args.topics:
        LOGGER.info("Starting %s on %s", model_id, topic)
        loader = DataLoader(queries_repo=manifest["queries"], corpus_repo=manifest["corpus"], topic=topic)
        corpus, queries, _ = loader.load(split="test")
        qrels_nuggets, qrels_query, query_to_nuggets = loader.load_qrels(split="test")
        doc_ids = list(corpus)
        texts = extract_corpus_sentences(corpus=[corpus[i] for i in doc_ids], sep=" ")
        query_ids = [q for q in queries if q in qrels_query]
        tokens, lengths = encode_corpus(model, texts, doc_ids, output_dir / "embeddings" / topic, args.batch_size)
        query_embeddings = model.encode_query([queries[q] for q in query_ids], batch_size=args.batch_size)
        if topic == args.topics[0]:
            check_search(model, query_embeddings, tokens, lengths, args.token_budget)
        began = time.time()
        scores = maxsim_search(query_embeddings, tokens, lengths, args.token_budget)
        top_scores, top_index = scores.topk(min(TOP_K, len(doc_ids)), dim=1)
        results = {
            qid: {doc_ids[i]: float(s) for s, i in zip(row_scores.tolist(), row_index.tolist())}
            for qid, row_scores, row_index in zip(query_ids, top_scores.cpu(), top_index.cpu())
        }
        LOGGER.info("Scored %d queries x %d documents (%d vectors) in %.1fs",
                    len(query_ids), len(doc_ids), int(lengths.sum()), time.time() - began)
        alpha_ndcg, coverage, recall = EvaluateRetrieval(k_values=K_VALUES).evaluate(
            qrels_nuggets=qrels_nuggets, query_to_nuggets=query_to_nuggets, qrels_query=qrels_query, results=results
        )
        util.save_runfile(str(output_dir / f"{topic}.run.trec"), results)
        util.save_results(str(output_dir / f"{topic}.json"), alpha_ndcg, coverage, recall)
        LOGGER.info("Completed %s on %s: alpha-nDCG@10 %.4f", model_id, topic, alpha_ndcg["alpha-nDCG@10"])
        del tokens, scores
    LOGGER.info("All requested topics complete for %s", model_id)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
    )
    main()
