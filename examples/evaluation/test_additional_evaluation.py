"""CPU-only regression tests; run --check_inputs_only for each checkpoint first."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import additional_evaluation as evaluation
import batch_tuning
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import Qwen3Config, Qwen3Model


class AdditionalEvaluationTests(unittest.TestCase):
    def test_batch_tuning_stops_when_throughput_plateaus(self):
        encoder = Mock()
        with (
            patch.object(batch_tuning.torch.cuda, "empty_cache"),
            patch.object(batch_tuning.torch.cuda, "mem_get_info", return_value=(70 * 1024**3, 80 * 1024**3)),
            patch.object(batch_tuning.torch.cuda, "memory_allocated", return_value=8 * 1024**3),
            patch.object(batch_tuning.torch.cuda, "max_memory_allocated", return_value=16 * 1024**3),
            patch.object(batch_tuning.torch.cuda, "reset_peak_memory_stats"),
            patch.object(batch_tuning.torch.cuda, "synchronize"),
            patch.object(batch_tuning, "perf_counter", side_effect=[0, 2, 3, 6, 7, 13]),
        ):
            selected, measurements = batch_tuning.tune_batch_size(encoder, 16, 2048)
        self.assertEqual(selected, 32)
        self.assertEqual(len(measurements), 3)
        self.assertEqual([entry["batch_size"] for entry in measurements], [16, 32, 64])
        self.assertTrue(all(entry["memory_headroom_ok"] for entry in measurements))

    def test_batch_tuning_recovers_from_larger_batch_oom(self):
        encoder = Mock()
        encoder.encode_queries.side_effect = [None, torch.cuda.OutOfMemoryError("expected")]
        with (
            patch.object(batch_tuning.torch.cuda, "empty_cache"),
            patch.object(batch_tuning.torch.cuda, "mem_get_info", return_value=(70 * 1024**3, 80 * 1024**3)),
            patch.object(batch_tuning.torch.cuda, "memory_allocated", return_value=8 * 1024**3),
            patch.object(batch_tuning.torch.cuda, "max_memory_allocated", return_value=16 * 1024**3),
            patch.object(batch_tuning.torch.cuda, "reset_peak_memory_stats"),
            patch.object(batch_tuning.torch.cuda, "synchronize"),
            patch.object(batch_tuning, "perf_counter", side_effect=[0, 2]),
        ):
            selected, measurements = batch_tuning.tune_batch_size(encoder, 16, 2048)
        self.assertEqual(selected, 16)
        self.assertEqual(measurements[-1]["status"], "out_of_memory")

    def test_actual_preprocessing_and_encoding_for_all_checkpoints(self):
        cache = Path.home() / "huggingface-cache"
        with patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}):
            for model_id, spec in evaluation.MODELS.items():
                with self.subTest(model=model_id):
                    config_path = hf_hub_download(
                        model_id, "config.json", revision=spec.revision, cache_dir=str(cache), local_files_only=True
                    )
                    snapshot = Path(config_path).parent
                    config = json.loads((snapshot / "config_sentence_transformers.json").read_text())
                    pooling = json.loads((snapshot / "1_Pooling/config.json").read_text())
                    tiny_config = Qwen3Config(
                        vocab_size=json.loads(Path(config_path).read_text())["vocab_size"],
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        head_dim=8,
                    )
                    tiny_model = Qwen3Model(tiny_config)
                    pooling["word_embedding_dimension"] = 32

                    def cpu_sentence_transformer(**kwargs):
                        kwargs["device"] = "cpu"
                        return SentenceTransformer(**kwargs)

                    with (
                        patch.object(evaluation, "snapshot_download", return_value=str(snapshot)),
                        patch.object(evaluation.Transformer, "_load_model", return_value=tiny_model),
                        patch.object(evaluation, "SentenceTransformer", side_effect=cpu_sentence_transformer),
                    ):
                        encoder = evaluation.load_encoder(model_id, spec, cache, config, pooling)
                    self.assertEqual(set(encoder.model[0].modality_config), {"text"})
                    evaluation.check_encoder(encoder, replace(spec, dimension=32))
                    for role, text in (("query", "Question?"), ("document", "Document.")):
                        prompt = config["prompts"][role]
                        features = encoder.model.preprocess(encoder.format_texts([text]), prompt=prompt, task=role)
                        expected = encoder.model.tokenizer(
                            [prompt + text + spec.suffix], return_tensors="pt", truncation=True, max_length=2048
                        )
                        self.assertTrue(torch.equal(features["input_ids"], expected["input_ids"]))

    def test_beir_adapter_selects_roles_and_preserves_title(self):
        model = Mock()
        encoder = evaluation.RetrievalEncoder(model, suffix="<|im_end|>\n")
        encoder.encode_queries(["question"], batch_size=2, convert_to_tensor=True)
        encoder.encode_corpus([{"title": "Title", "text": "Body"}], batch_size=3, convert_to_tensor=True)
        model.encode_query.assert_called_once_with(
            ["question<|im_end|>\n"], batch_size=2, convert_to_tensor=True
        )
        model.encode_document.assert_called_once_with(
            ["Title Body<|im_end|>\n"], batch_size=3, convert_to_tensor=True
        )

    def test_comparison_only_averages_complete_models(self):
        metrics = {
            "alpha_ndcg": {"alpha-nDCG@10": 0.3},
            "coverage": {"Coverage@20": 0.4},
            "recall": {"Recall@50": 0.5},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, topics in (("nemotron3_1B_results", evaluation.TOPICS), ("octen_8B_results", ["langchain"])):
                directory = root / name
                directory.mkdir()
                for topic in topics:
                    (directory / f"{topic}.json").write_text(json.dumps(metrics))
            evaluation.write_comparison(root)
            with (root / "embedding_comparison.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 7)
            averages = [row for row in rows if row["topic"] == "AVERAGE"]
            self.assertEqual(len(averages), 1)
            self.assertEqual(averages[0]["model"], "nvidia/Nemotron-3-Embed-1B-BF16")
            self.assertEqual(float(averages[0]["alpha-nDCG@10"]), 0.3)


if __name__ == "__main__":
    unittest.main()
