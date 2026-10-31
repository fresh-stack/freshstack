"""
Evaluate google/embeddinggemma-2 on all five FreshStack topics with 2048 total tokens.

Text-only configuration. The vision (170M) and audio (300M) encoders are switched off
through config_kwargs, leaving the 270M text backbone, so nothing but text is loaded or
encoded. Embeddings use the native 768 dimensions with no Matryoshka truncation, which
the model card reports as the highest-quality setting.

The checkpoint ships a standard SentenceTransformers pipeline (Transformer, mean Pooling,
Normalize) with no remote code, so the model is loaded with SentenceTransformer directly
and its own saved prompts are used: "task: code retrieval | query: " for queries and
"title: none | text: " for documents. FreshStack asks natural-language questions over
code and documentation, which is the card's code-search case, and every FreshStack
document has an empty title, which is what the "title: none" document prompt is for.

Inference runs in bfloat16; the card warns that float16 silently returns NaN or degraded
embeddings because the activation range exceeds its dynamic range.

The 2048-token limit matches the other models in results/embedding_comparison.csv rather
than the model's full 8K window, so the comparison stays like-for-like.

python examples/evaluation/embeddinggemma2_evaluation.py --batch_size 16
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from batch_tuning import tune_batch_size
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval as BEIREval
from beir.retrieval.models.util import extract_corpus_sentences
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
OUTPUT_NAME = "embeddinggemma2_results"
MAX_LENGTH = 2048
DIMENSION = 768
TOPICS = ("langchain", "yolo", "godot", "angular", "laravel")
K_VALUES = [5, 10, 20, 50]
# Task prefixes as published on the model card. SearchQuery is the general retrieval
# prompt; CodeRetrieval is the card's prompt for code search from natural-language
# queries, which is also what FreshStack asks for.
QUERY_PROMPTS = {
    "SearchQuery": "task: search result | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "QuestionAnswering": "task: question answering | query: ",
}
DOCUMENT_PROMPT = "title: none | text: "
TEXT_ONLY_KWARGS = {"vision_config": None, "audio_config": None}
# Nine langchain documents contain these media placeholders as literal text. The
# processor reads them as markers for real images, video or audio and refuses the input,
# so the delimiters are softened; "<image>" and friends are ordinary text to the
# tokenizer. Queries contain none of them.
MEDIA_PLACEHOLDERS = re.compile(r"<\|(image|video|audio)\|>")
LOGGER = logging.getLogger(__name__)


def as_text(value: str) -> str:
    return MEDIA_PLACEHOLDERS.sub(r"<\1>", value)


class RetrievalEncoder:
    """Adapts a SentenceTransformer to the BEIR dense-search interface."""

    def __init__(self, model: SentenceTransformer, query_prompt: str):
        self.model = model
        self.query_prompt = query_prompt

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs):
        texts = [as_text(query) for query in queries]
        return self.model.encode_query(texts, prompt=self.query_prompt, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus, batch_size: int = 16, **kwargs):
        texts = [as_text(text) for text in extract_corpus_sentences(corpus=corpus, sep=" ")]
        return self.model.encode_document(texts, prompt=DOCUMENT_PROMPT, batch_size=batch_size, **kwargs)


def checkpoint_config(cache_dir: Path, query_prompt_name: str) -> dict:
    """Confirm the published prompts, pooling and pipeline before any GPU work."""
    configs = {}
    for filename in ("config_sentence_transformers.json", "1_Pooling/config.json", "modules.json"):
        path = hf_hub_download(MODEL_ID, filename, revision=REVISION, cache_dir=str(cache_dir))
        configs[filename] = json.loads(Path(path).read_text())
    prompts = configs["config_sentence_transformers.json"]["prompts"]
    if prompts.get(query_prompt_name) != QUERY_PROMPTS[query_prompt_name]:
        raise ValueError(f"Saved {query_prompt_name} prompt differs from the model card")
    if prompts.get("Document") != DOCUMENT_PROMPT:
        raise ValueError("Saved Document prompt differs from the model card")
    pooling = configs["1_Pooling/config.json"]
    if (
        pooling["embedding_dimension"] != DIMENSION
        or pooling["pooling_mode"] != "mean"
        or not pooling["include_prompt"]
    ):
        raise ValueError("Unexpected pooling configuration")
    expected_modules = [
        "sentence_transformers.base.modules.transformer.Transformer",
        "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
        "sentence_transformers.base.modules.normalize.Normalize",
    ]
    if [module["type"] for module in configs["modules.json"]] != expected_modules:
        raise ValueError("Unexpected model pipeline")
    LOGGER.info("Checkpoint prompts, mean pooling at %dd and standard pipeline confirmed", DIMENSION)
    return configs["config_sentence_transformers.json"]


def load_encoder(cache_dir: Path, query_prompt: str) -> RetrievalEncoder:
    model = SentenceTransformer(
        MODEL_ID,
        revision=REVISION,
        cache_folder=str(cache_dir),
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        config_kwargs=TEXT_ONLY_KWARGS,
        similarity_fn_name="cosine",
        device="cuda:0",
    )
    model.max_seq_length = MAX_LENGTH
    return RetrievalEncoder(model, query_prompt)


def check_encoder(encoder: RetrievalEncoder) -> None:
    """Verify text-only loading, the token limit, the prompts and real GPU inference."""
    model = encoder.model
    transformer = model[0]
    auto_model = transformer.auto_model
    for attribute in ("vision_tower", "vision_model", "audio_tower", "audio_model"):
        if getattr(auto_model, attribute, None) is not None:
            raise ValueError(f"{attribute} was loaded despite the text-only configuration")
    config = auto_model.config
    if getattr(config, "vision_config", None) is not None or getattr(config, "audio_config", None) is not None:
        raise ValueError("Vision or audio configuration survived the text-only request")
    parameters = sum(p.numel() for p in auto_model.parameters())
    if parameters > 400_000_000:
        raise ValueError(f"Loaded {parameters / 1e6:.0f}M parameters; expected the ~270M text backbone")
    if model.max_seq_length != MAX_LENGTH or model.get_embedding_dimension() != DIMENSION:
        raise ValueError("Incorrect sequence length or embedding dimension")
    if auto_model.dtype != torch.bfloat16:
        raise ValueError("Model is not running in bfloat16")

    long_text = "token " * (MAX_LENGTH * 2)
    for prompt in (encoder.query_prompt, DOCUMENT_PROMPT):
        encoded = model.preprocess(["short text", long_text], prompt=prompt, task="query")
        expected = model.tokenizer(
            [prompt + text for text in ("short text", long_text)],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        if not torch.equal(encoded["input_ids"], expected["input_ids"]):
            raise ValueError("Prompted input differs from the reference tokenization")
        if encoded["input_ids"].shape[1] != MAX_LENGTH:
            raise ValueError(f"Inputs are not truncated at {MAX_LENGTH} tokens")
        # The prompt's final token merges with the text that follows it, so compare
        # everything before it, with the end-of-sequence marker removed.
        prompt_ids = [i for i in model.tokenizer(prompt)["input_ids"] if i != model.tokenizer.eos_token_id][:-1]
        prefix = torch.tensor(prompt_ids)
        if not torch.equal(encoded["input_ids"][:, : len(prompt_ids)], prefix.expand(2, -1)):
            raise ValueError("Task prefix is missing from the encoded input")
        unprompted = model.preprocess(["short text"], task="query")["input_ids"]
        if torch.equal(unprompted[0][: len(prompt_ids)], prefix):
            raise ValueError("Prompt made no difference to the encoded input")

    query = encoder.encode_queries(["How do I stream tokens from a chain?"], convert_to_tensor=True)
    documents = encoder.encode_corpus(
        [{"title": "", "text": "Use the streaming callback handler to receive tokens."},
         {"title": "", "text": "Unrelated: configure the build system for the renderer."}],
        convert_to_tensor=True,
    )
    for embeddings in (query, documents):
        if embeddings.shape[-1] != DIMENSION or not torch.isfinite(embeddings).all():
            raise ValueError("Invalid inference output")
        norms = embeddings.float().norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=0.02, rtol=0):
            raise ValueError("Embeddings are not L2-normalized")
    scores = (query.float() @ documents.float().T)[0]
    if scores[0] <= scores[1]:
        raise ValueError("Relevant document did not outscore the unrelated one")
    literal = encoder.encode_corpus([{"title": "", "text": "Render the diagram <|image|> inline."}],
                                    convert_to_tensor=True)
    if literal.shape[-1] != DIMENSION or not torch.isfinite(literal).all():
        raise ValueError("Documents containing literal media placeholders were not encoded as text")
    LOGGER.info(
        "Text-only backbone (%.0fM parameters), bfloat16, %dd normalized output and prompts confirmed on GPU",
        parameters / 1e6, DIMENSION,
    )


def write_comparison(results_root: Path) -> None:
    """Rewrite the shared comparison table, appending this model's rows."""
    entries = {
        "nvidia/Nemotron-3-Embed-1B-BF16": "nemotron3_1B_results",
        "nvidia/Nemotron-3-Embed-8B-BF16": "nemotron3_8B_results",
        "Octen/Octen-Embedding-8B": "octen_8B_results",
        "Octen/Octen-Embedding-4B": "octen_4B_results",
        "Octen/Octen-Embedding-0.6B": "octen_0.6B_results",
        "zeroentropy/zembed-1-embedding": "zembed_1_results",
        MODEL_ID: OUTPUT_NAME,
    }
    rows = []
    for model_id, directory in entries.items():
        model_rows = []
        for topic in TOPICS:
            path = results_root / directory / f"{topic}.json"
            if not path.is_file():
                continue
            metrics = json.loads(path.read_text())
            model_rows.append({
                "model": model_id,
                "topic": topic,
                "alpha-nDCG@10": metrics["alpha_ndcg"]["alpha-nDCG@10"],
                "Coverage@20": metrics["coverage"]["Coverage@20"],
                "Recall@50": metrics["recall"]["Recall@50"],
            })
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--auto_batch_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query_prompt", choices=QUERY_PROMPTS, default="CodeRetrieval")
    parser.add_argument("--cache_dir", type=Path, default=Path.home() / "huggingface-cache")
    parser.add_argument("--results_root", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--output_name", default=None, help="Results directory name; defaults to the model's")
    parser.add_argument("--topics", choices=TOPICS, nargs="+", default=list(TOPICS))
    parser.add_argument("--check_inputs_only", action="store_true", help="Check saved settings without GPU use")
    parser.add_argument("--check_model_only", action="store_true", help="Load and verify the model, then stop")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if version("sentence-transformers").split(".")[0] != "6":
        parser.error("This adapter requires SentenceTransformers 6; recheck preprocessing before changing versions")
    cache_dir = args.cache_dir.expanduser().resolve()
    saved = checkpoint_config(cache_dir, args.query_prompt)
    if args.check_inputs_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these evaluations")
    query_prompt = QUERY_PROMPTS[args.query_prompt]
    encoder = load_encoder(cache_dir, query_prompt)
    check_encoder(encoder)
    if args.check_model_only:
        return
    output_dir = args.results_root / (args.output_name or OUTPUT_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements = []
    if args.auto_batch_size:
        args.batch_size, measurements = tune_batch_size(encoder, args.batch_size, MAX_LENGTH)
    manifest = {
        "model": MODEL_ID,
        "revision": REVISION,
        "modalities": "text only (vision and audio encoders disabled)",
        "config_kwargs": TEXT_ONLY_KWARGS,
        "parameters_loaded": sum(p.numel() for p in encoder.model[0].auto_model.parameters()),
        "query_max_length": MAX_LENGTH,
        "document_max_length": MAX_LENGTH,
        "query_prompt_name": args.query_prompt,
        "prompts": {"query": query_prompt, "document": DOCUMENT_PROMPT},
        "pooling": "mean, include_prompt=true",
        "dimension": DIMENSION,
        "matryoshka_truncation": None,
        "normalize": True,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "input_modality": "text",
        "media_placeholders": "literal <|image|>/<|video|>/<|audio|> strings softened to <image>/<video>/<audio>",
        "batch_size": args.batch_size,
        "batch_tuning": measurements,
        "score_function": "cos_sim",
        "topics": args.topics,
        "queries": "freshstack/queries-oct-2024",
        "corpus": "freshstack/corpus-oct-2024",
        "split": "test",
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_versions": saved.get("__version__"),
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
        retriever = BEIREval(DenseRetrievalExactSearch(encoder, batch_size=args.batch_size), score_function="cos_sim")
        results = retriever.retrieve(corpus=corpus, queries=queries)
        results = {qid: docs for qid, docs in results.items() if qid in qrels_query}
        alpha_ndcg, coverage, recall = EvaluateRetrieval(k_values=K_VALUES).evaluate(
            qrels_nuggets=qrels_nuggets, query_to_nuggets=query_to_nuggets, qrels_query=qrels_query, results=results
        )
        util.save_runfile(str(output_dir / f"{topic}.run.trec"), results)
        util.save_results(str(output_dir / f"{topic}.json"), alpha_ndcg, coverage, recall)
        write_comparison(args.results_root)
        LOGGER.info("Completed %s on %s", MODEL_ID, topic)
    LOGGER.info("All requested topics complete for %s", MODEL_ID)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
    )
    main()
