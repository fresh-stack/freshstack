"""
Evaluate NVIDIA Nemotron-3-Embed and Nemotron-3.5-Embed (1B / 8B, BF16) on FreshStack using BEIR.

Nemotron-3 checkpoints are SentenceTransformers models; their saved "query" / "document"
prompts are applied through encode_query / encode_document (left padding).

Nemotron-3.5 checkpoints are multimodal (Pixtral vision encoder + Nemotron-3 language model), and
their reference usage formats every input with the checkpoint's chat template:
    [{"role": "system", "content": "query:" | "passage:"}, {"role": "user", "content": [text]}]
The template joins the prefix and text with one space, escapes "[" and "<" in user text with a
zero-width non-joiner, and adds no BOS/EOS. Token states are mean-pooled and L2-normalized.

Requires CUDA-enabled PyTorch and
`pip install beir "transformers>=5.19" "sentence-transformers>=5.4.1" pillow`.
Use `--attn_implementation flash_attention_2` if FlashAttention-2 is installed.

export CUDA_VISIBLE_DEVICES=0
for topic in langchain yolo godot angular laravel; do
    python examples/evaluation/nemotron_embed_evaluation.py \
        --model_name_or_path "nvidia/Nemotron-3.5-Embed-1B-BF16" \
        --max_length 2048 \
        --batch_size 16 \
        --queries "freshstack/queries-oct-2024" \
        --corpus "freshstack/corpus-oct-2024" \
        --topic $topic \
        --output_dir "./results/nemotron3.5_1B_results/"
done
"""

import argparse
import json
import logging
import os
import pathlib
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
import transformers
from beir import LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval as BEIREval
from beir.retrieval.models.sentence_bert import extract_corpus_sentences
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from tqdm.autonotebook import trange
from transformers import AutoModel, AutoProcessor

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


class Nemotron35TextEncoder:
    """BEIR-compatible text encoder that follows the Nemotron-3.5 chat-template input format."""

    def __init__(self, model_name_or_path, cache_dir, max_length, query_prefix, document_prefix, attn_implementation):
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        ).to("cuda").eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def render(self, prefix, text):
        return self.processor.apply_chat_template(
            [
                {"role": "system", "content": prefix},
                {"role": "user", "content": [{"type": "text", "text": text}]},
            ],
            tokenize=False,
        )

    def tokenize(self, rendered):
        # The template already holds the full input, so no BOS/EOS; truncation drops the end of the text.
        return self.tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            padding_side="right",
            return_tensors="pt",
        )

    @torch.inference_mode()
    def _encode(self, texts, prefix, batch_size, show_progress_bar=True, convert_to_tensor=True, **kwargs):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        embeddings = [None] * len(texts)
        for start in trange(0, len(texts), batch_size, disable=not show_progress_bar, desc="Batches"):
            idx = order[start : start + batch_size]
            inputs = self.tokenize([self.render(prefix, texts[i]) for i in idx]).to("cuda")
            hidden = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = F.normalize(pooled.float(), dim=-1)
            for row, i in enumerate(idx):
                embeddings[i] = pooled[row]
        stacked = torch.stack(embeddings)
        return stacked if convert_to_tensor else stacked.cpu().numpy()

    def encode_queries(self, queries, batch_size=16, **kwargs):
        return self._encode(list(queries), self.query_prefix, batch_size, **kwargs)

    def encode_corpus(self, corpus, batch_size=16, **kwargs):
        return self._encode(extract_corpus_sentences(corpus=corpus, sep=" "), self.document_prefix, batch_size, **kwargs)


def check_pipeline(encoder):
    """Fail fast if the input format, truncation, or padding invariance is not as expected."""
    text = "Use List<String> or arr[0] in Godot"
    rendered = encoder.render(encoder.query_prefix, text)
    expected = encoder.query_prefix.rstrip(" ") + " " + text.replace("[", "[\u200c").replace("<", "<\u200c")
    if rendered != expected:
        raise RuntimeError(f"Unexpected chat-template rendering: {rendered!r}")
    ids = encoder.tokenize([rendered])["input_ids"][0].tolist()
    if ids[0] == encoder.tokenizer.bos_token_id or ids[-1] == encoder.tokenizer.eos_token_id:
        raise RuntimeError("BOS/EOS tokens must not be added to rendered inputs")
    long_ids = encoder.tokenize([encoder.render(encoder.document_prefix, "token " * (2 * encoder.max_length))])
    if long_ids["input_ids"].shape[1] != encoder.max_length:
        raise RuntimeError("Long documents are not truncated to --max_length")
    texts = ["short query", "a much longer document " * 40, text]
    batched = encoder._encode(texts, encoder.document_prefix, batch_size=3, show_progress_bar=False)
    single = torch.cat([encoder._encode([t], encoder.document_prefix, 1, show_progress_bar=False) for t in texts])
    similarity = (batched * single).sum(-1)
    if similarity.min() < 0.999:
        raise RuntimeError(f"Embeddings change with padding: {similarity.tolist()}")
    logger.info(
        "Pipeline checks passed: chat-template prefix + escaping, no BOS/EOS, %d-token truncation, "
        "padding invariance (min cos %.5f)", encoder.max_length, similarity.min().item(),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nemotron-3 / 3.5 embeddings on FreshStack")
    parser.add_argument("--model_name_or_path", type=str, default="nvidia/Nemotron-3.5-Embed-1B-BF16")
    parser.add_argument("--cache_dir", type=str, default=str(pathlib.Path.home() / "huggingface-cache"))
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum tokens for queries and documents, including the prompt"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--query_prefix", type=str, default="query:", help="Nemotron-3.5 only")
    parser.add_argument("--document_prefix", type=str, default="passage:", help="Nemotron-3.5 only")
    parser.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa"], default="sdpa")
    parser.add_argument("--score_function", type=str, default="cos_sim")
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--topic", type=str, default="langchain")
    parser.add_argument("--k_values", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cache_dir = pathlib.Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir or os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)

    dataloader = DataLoader(queries_repo=args.queries, corpus_repo=args.corpus, topic=args.topic)
    corpus, queries, nuggets = dataloader.load(split="test")
    qrels_nuggets, qrels_query, query_to_nuggets = dataloader.load_qrels(split="test")

    if "Nemotron-3.5" in args.model_name_or_path:
        encoder = Nemotron35TextEncoder(
            args.model_name_or_path, str(cache_dir), args.max_length,
            args.query_prefix, args.document_prefix, args.attn_implementation,
        )
        check_pipeline(encoder)
        configuration = {
            "model": args.model_name_or_path,
            "revision": encoder.revision,
            "parameters": sum(p.numel() for p in encoder.model.parameters()),
            "max_length": args.max_length,
            "prefixes": {"query": args.query_prefix, "document": args.document_prefix},
            "input_format": "checkpoint chat template (system prefix + user text), no BOS/EOS",
            "document_text": "title + ' ' + text, stripped",
            "pooling": "mean over attention mask, L2-normalized",
            "dtype": "bfloat16",
            "attention": args.attn_implementation,
            "batch_size": args.batch_size,
            "score_function": args.score_function,
            "queries": args.queries,
            "corpus": args.corpus,
            "gpu": torch.cuda.get_device_name(0),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        pathlib.Path(results_dir, "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    else:
        # BEIR calls encode(), so select the saved prompts used by encode_query/encode_document explicitly.
        encoder = models.SentenceBERT(
            args.model_name_or_path,
            max_length=args.max_length,
            cache_folder=str(cache_dir),
            prompt_names={"query": "query", "passage": "document"},
            device="cuda",
            model_kwargs={"dtype": torch.bfloat16, "attn_implementation": args.attn_implementation},
            processor_kwargs={"padding_side": "left"},
        )

    retriever = BEIREval(DRES(encoder, batch_size=args.batch_size), score_function=args.score_function)
    results = retriever.retrieve(corpus=corpus, queries=queries)

    evaluator = EvaluateRetrieval(k_values=args.k_values)
    alpha_ndcg, coverage, recall = evaluator.evaluate(
        qrels_nuggets=qrels_nuggets, query_to_nuggets=query_to_nuggets, qrels_query=qrels_query, results=results
    )

    results = {query_id: results[query_id] for query_id in results if query_id in qrels_query}
    util.save_runfile(os.path.join(results_dir, f"{args.topic}.run.trec"), results)
    util.save_results(os.path.join(results_dir, f"{args.topic}.json"), alpha_ndcg, coverage, recall)


if __name__ == "__main__":
    main()
