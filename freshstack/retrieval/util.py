from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def round_and_log(metric: dict[str, float], precision: int = 4) -> float:
    logger.info("\n")
    for key, value in metric.items():
        if isinstance(value, float):
            metric[key] = round(value, precision)
            logger.info(f"{key}: {value:.{precision}f}")
    return metric
