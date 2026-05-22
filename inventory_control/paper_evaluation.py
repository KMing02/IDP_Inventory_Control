"""Shared evaluation settings for paper-style lost-sales experiments."""

from __future__ import annotations

from typing import Any


PAPER_TEST_EVALUATION_SEED_START = 30_000
PAPER_VALIDATION_EVALUATION_SEED_START = 40_000


def paper_test_product_count(config: dict[str, Any]) -> int:
    """Return the common capped test product count for paper comparisons."""
    runtime = config.get("runtime", {})
    caps = [
        int(config["test_products"]),
        int(runtime.get("baseline_eval_products", config["test_products"])),
        int(runtime.get("fcnn_test_products", config["test_products"])),
    ]
    return min(caps)
