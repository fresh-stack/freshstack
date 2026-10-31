"""Measure single-GPU encoding throughput without changing token limits."""

from __future__ import annotations

import gc
import logging
from time import perf_counter

import torch

LOGGER = logging.getLogger(__name__)


def tune_batch_size(encoder, initial_batch_size: int, max_length: int) -> tuple[int, list[dict]]:
    if initial_batch_size < 1:
        raise ValueError("Initial batch size must be positive")
    text = "token " * (max_length * 2)
    selected = initial_batch_size
    best_rate = 0.0
    measurements = []
    for batch_size in (initial_batch_size * factor for factor in (1, 2, 4, 8)):
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        try:
            # Include both roles because prompt processing can change allocation.
            for encode, inputs in (
                (encoder.encode_queries, [text] * batch_size),
                (encoder.encode_corpus, [{"text": text}] * batch_size),
            ):
                output = encode(inputs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
                del output
            torch.cuda.synchronize()
            start = perf_counter()
            for _ in range(2):
                output = encoder.encode_corpus(
                    [{"text": text}] * batch_size,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                del output
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
        except torch.cuda.OutOfMemoryError:
            LOGGER.warning("Batch %d exceeded GPU memory; keeping the previous safe batch", batch_size)
            measurements.append({"batch_size": batch_size, "status": "out_of_memory"})
            if not best_rate:
                raise
            break
        peak = torch.cuda.max_memory_allocated()
        rate = 2 * batch_size / elapsed
        # Leave room for other processes, retrieval tensors, and allocator overhead.
        safe = peak - allocated_before <= free - 0.15 * total
        measurements.append({
            "batch_size": batch_size,
            "documents_per_second": round(rate, 3),
            "peak_allocated_gib": round(peak / 1024**3, 3),
            "memory_headroom_ok": safe,
        })
        LOGGER.info("Batch benchmark: %s", measurements[-1])
        if not safe:
            if not best_rate:
                raise RuntimeError("Initial batch leaves less than 15% GPU memory headroom; reduce --batch_size")
            break
        if best_rate and rate <= best_rate * 1.05:
            break
        selected, best_rate = batch_size, rate
    gc.collect()
    torch.cuda.empty_cache()
    LOGGER.info("Selected batch size %d for both queries and documents (limit=%d)", selected, max_length)
    return selected, measurements
