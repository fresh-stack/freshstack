"""
Evaluate Octen and zembed on all five FreshStack topics with 2048 total tokens.

Requires sentence-transformers>=6,<7 and transformers>=5,<6.
Uses explicit text processing (no implicit chat template), the checkpoints' saved
query/document prompts, native embedding dimensions, last-token pooling, and
normalization. Octen's saved document prompt is one space,
not the alternative "- " prefix in the 8B card's raw Transformers example.

zembed's legacy tokenize override is bypassed by SentenceTransformers 6. Rebuild
its standard Transformer/Pooling/Normalize pipeline without executing remote code
and append its closing suffix before tokenization, matching its original behavior.
As in the original implementation, right truncation can remove the closing suffix.

python examples/evaluation/additional_evaluation.py \
    --model Octen/Octen-Embedding-8B --batch_size 16
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from batch_tuning import tune_batch_size
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval as BEIREval
from beir.retrieval.models.util import extract_corpus_sentences
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import Normalize, Transformer
from sentence_transformers.sentence_transformer.modules import Pooling
from transformers import AutoTokenizer

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval

MAX_LENGTH = 2048
TOPICS = ("langchain", "yolo", "godot", "angular", "laravel")
K_VALUES = [5, 10, 20, 50]
OCTEN_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
ZEMBED_PREFIX = "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    revision: str
    dimension: int
    output_name: str
    query_prompt: str = OCTEN_QUERY
    document_prompt: str = " "
    suffix: str = ""


MODELS = {
    "Octen/Octen-Embedding-8B": ModelSpec(
        "5adcfa292e712091dfc30f0e97f0b2282e6cc66c", 4096, "octen_8B_results"
    ),
    "Octen/Octen-Embedding-4B": ModelSpec(
        "fea468fae3f0caffbae8a12ba792d1c394b6277d", 2560, "octen_4B_results"
    ),
    "Octen/Octen-Embedding-0.6B": ModelSpec(
        "d715b32ee68f057b54dff09fc93c23485bc403d3", 1024, "octen_0.6B_results"
    ),
    "zeroentropy/zembed-1-embedding": ModelSpec(
        "cf13c81f3274394053d166740294f7eea4586f7a",
        2560,
        "zembed_1_results",
        ZEMBED_PREFIX.format("query"),
        ZEMBED_PREFIX.format("document"),
        "<|im_end|>\n",
    ),
}


class RetrievalEncoder:
    def __init__(self, model: SentenceTransformer, suffix: str = ""):
        self.model = model
        self.suffix = suffix

    def format_texts(self, texts: list[str]) -> list[str]:
        return [text + self.suffix for text in texts]

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs):
        return self.model.encode_query(self.format_texts(queries), batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus, batch_size: int = 16, **kwargs):
        texts = extract_corpus_sentences(corpus=corpus, sep=" ")
        return self.model.encode_document(self.format_texts(texts), batch_size=batch_size, **kwargs)


def checkpoint_config(model_id: str, spec: ModelSpec, cache_dir: Path) -> tuple[dict, dict]:
    configs = {}
    for filename in ("config_sentence_transformers.json", "1_Pooling/config.json", "modules.json"):
        path = hf_hub_download(model_id, filename, revision=spec.revision, cache_dir=str(cache_dir))
        configs[filename] = json.loads(Path(path).read_text())
    config = configs["config_sentence_transformers.json"]
    expected_prompts = {"query": spec.query_prompt, "document": spec.document_prompt}
    if config["prompts"] != expected_prompts or config.get("suffix", "") != spec.suffix:
        raise ValueError(f"Unexpected saved formatting for {model_id}")
    pooling = configs["1_Pooling/config.json"]
    if (
        pooling["word_embedding_dimension"] != spec.dimension
        or not pooling["pooling_mode_lasttoken"]
        or not pooling["include_prompt"]
        or any(value for key, value in pooling.items() if key.startswith("pooling_mode_") and key != "pooling_mode_lasttoken")
    ):
        raise ValueError(f"Unexpected pooling configuration for {model_id}")
    transformer_type = "modeling_zembed.ZembedTransformer" if spec.suffix else "sentence_transformers.models.Transformer"
    expected_modules = [
        transformer_type,
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ]
    if [module["type"] for module in configs["modules.json"]] != expected_modules:
        raise ValueError(f"Unexpected model pipeline for {model_id}")
    return config, pooling


def check_tokenizer(model_id: str, spec: ModelSpec, cache_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=spec.revision, cache_dir=str(cache_dir), padding_side="left", truncation_side="right"
    )
    for prompt in (spec.query_prompt, spec.document_prompt):
        texts = [prompt + text + spec.suffix for text in ("short text", "token " * 4096)]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        if encoded["input_ids"].shape[1] != MAX_LENGTH:
            raise ValueError(f"Long-input token limit was not applied for {model_id}")
        lengths = encoded["attention_mask"].sum(dim=1)
        if not (lengths[0] < MAX_LENGTH and lengths[1] == MAX_LENGTH):
            raise ValueError(f"Unexpected padded/truncated lengths for {model_id}")
    LOGGER.info("Tokenizer and saved prompts confirmed for %s; both limits=%d", model_id, MAX_LENGTH)


def load_encoder(model_id: str, spec: ModelSpec, cache_dir: Path, config: dict, pooling: dict) -> RetrievalEncoder:
    snapshot = snapshot_download(model_id, revision=spec.revision, cache_dir=str(cache_dir))
    model_kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    processor_kwargs = {"padding_side": "left", "truncation_side": "right"}
    # ST6 otherwise applies the tokenizer's chat template to plain-text inputs.
    transformer = Transformer(
        snapshot,
        max_seq_length=MAX_LENGTH,
        model_kwargs=model_kwargs,
        processor_kwargs=processor_kwargs,
        modality_config={"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
        module_output_name="token_embeddings",
    )
    model = SentenceTransformer(
        modules=[transformer, Pooling(**pooling), Normalize()],
        prompts=config["prompts"],
        default_prompt_name=config.get("default_prompt_name"),
        similarity_fn_name="cosine",
        device="cuda:0",
    )
    model.max_seq_length = MAX_LENGTH
    return RetrievalEncoder(model, suffix=spec.suffix)


def check_encoder(encoder: RetrievalEncoder, spec: ModelSpec) -> None:
    model = encoder.model
    if model.max_seq_length != MAX_LENGTH or model.get_sentence_embedding_dimension() != spec.dimension:
        raise ValueError("Incorrect model sequence length or embedding dimension")
    for role, prompt in (("query", spec.query_prompt), ("document", spec.document_prompt)):
        texts = encoder.format_texts(["short text", "token " * 4096])
        actual = model.preprocess(texts, prompt=prompt, task=role)
        expected = model.tokenizer(
            [prompt + text for text in texts],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        if not torch.equal(actual["input_ids"], expected["input_ids"]):
            raise ValueError(f"{role} formatting differs from the checkpoint's reference tokenization")
        if actual["input_ids"].shape[1] != MAX_LENGTH:
            raise ValueError(f"{role} does not truncate at {MAX_LENGTH} tokens")
    for embeddings in (
        encoder.encode_queries(["A technical question", "Another question"], convert_to_tensor=True),
        encoder.encode_corpus([{"text": "A technical document"}, {"text": "Another document"}], convert_to_tensor=True),
    ):
        if embeddings.shape != (2, spec.dimension) or not torch.isfinite(embeddings).all():
            raise ValueError("Invalid inference output")
        norms = embeddings.float().norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=0.02, rtol=0):
            raise ValueError("Model embeddings are not L2-normalized")
    LOGGER.info("Actual model preprocessing and GPU inference confirmed")


def write_comparison(results_root: Path) -> None:
    entries = {
        "nvidia/Nemotron-3-Embed-1B-BF16": "nemotron3_1B_results",
        "nvidia/Nemotron-3-Embed-8B-BF16": "nemotron3_8B_results",
        **{model_id: spec.output_name for model_id, spec in MODELS.items()},
        "cohere/embed-v5.0-pro": "cohere_embed_v5_pro_results",
        "cohere/embed-v5.0-fast": "cohere_embed_v5_fast_results",
    }
    rows = []
    for model_id, directory in entries.items():
        model_rows = []
        for topic in TOPICS:
            path = results_root / directory / f"{topic}.json"
            if not path.is_file():
                continue
            metrics = json.loads(path.read_text())
            row = {
                "model": model_id,
                "topic": topic,
                "alpha-nDCG@10": metrics["alpha_ndcg"]["alpha-nDCG@10"],
                "Coverage@20": metrics["coverage"]["Coverage@20"],
                "Recall@50": metrics["recall"]["Recall@50"],
            }
            model_rows.append(row)
        rows.extend(model_rows)
        if len(model_rows) == len(TOPICS):
            rows.append({
                "model": model_id,
                "topic": "AVERAGE",
                **{key: round(sum(row[key] for row in model_rows) / len(TOPICS), 4)
                   for key in ("alpha-nDCG@10", "Coverage@20", "Recall@50")},
            })
    temporary = results_root / "embedding_comparison.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "topic", "alpha-nDCG@10", "Coverage@20", "Recall@50"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results_root / "embedding_comparison.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--auto_batch_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_dir", type=Path, default=Path.home() / "huggingface-cache")
    parser.add_argument("--results_root", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--topics", choices=TOPICS, nargs="+", default=list(TOPICS))
    parser.add_argument("--check_inputs_only", action="store_true", help="Check saved settings and tokenizers without GPU use")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if version("sentence-transformers").split(".")[0] != "6":
        parser.error("This adapter requires SentenceTransformers 6; recheck preprocessing before changing major versions")
    spec = MODELS[args.model]
    cache_dir = args.cache_dir.expanduser().resolve()
    config, pooling = checkpoint_config(args.model, spec, cache_dir)
    check_tokenizer(args.model, spec, cache_dir)
    if args.check_inputs_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these evaluations")
    output_dir = args.results_root / spec.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(args.model, spec, cache_dir, config, pooling)
    check_encoder(encoder, spec)
    measurements = []
    if args.auto_batch_size:
        args.batch_size, measurements = tune_batch_size(encoder, args.batch_size, MAX_LENGTH)
    manifest = {
        "model": args.model,
        "revision": spec.revision,
        "query_max_length": MAX_LENGTH,
        "document_max_length": MAX_LENGTH,
        "prompts": config["prompts"],
        "suffix": spec.suffix,
        "suffix_policy": "append before right truncation",
        "pooling": pooling,
        "dimension": spec.dimension,
        "normalize": True,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "padding_side": "left",
        "input_modality": "text",
        "batch_size": args.batch_size,
        "batch_tuning": measurements,
        "score_function": "cos_sim",
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
        LOGGER.info("Starting %s on %s", args.model, topic)
        loader = DataLoader(queries_repo=manifest["queries"], corpus_repo=manifest["corpus"], topic=topic)
        corpus, queries, _ = loader.load(split="test")
        qrels_nuggets, qrels_query, query_to_nuggets = loader.load_qrels(split="test")
        retriever = BEIREval(DenseRetrievalExactSearch(encoder, batch_size=args.batch_size), score_function="cos_sim")
        results = retriever.retrieve(corpus=corpus, queries=queries)
        results = {qid: docs for qid, docs in results.items() if qid in qrels_query}
        alpha_ndcg, coverage, recall = EvaluateRetrieval(k_values=K_VALUES).evaluate(
            qrels_nuggets=qrels_nuggets, query_to_nuggets=query_to_nuggets, qrels_query=qrels_query, results=results
        )
        util.save_runfile(str(output_dir / f"{topic}.run.trec"), results)
        util.save_results(str(output_dir / f"{topic}.json"), alpha_ndcg, coverage, recall)
        write_comparison(args.results_root)
        LOGGER.info("Completed %s on %s", args.model, topic)
    LOGGER.info("All requested topics complete for %s", args.model)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
    )
    main()
