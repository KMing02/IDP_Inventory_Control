"""Shared evaluation settings for paper-style lost-sales experiments."""

from __future__ import annotations

from typing import Any


PAPER_TEST_EVALUATION_SEED_START = 30_000
PAPER_VALIDATION_EVALUATION_SEED_START = 40_000


def paper_validation_product_count(config: dict[str, Any]) -> int:
    """Return the common capped validation product count for tuning comparisons."""
    runtime = config.get("runtime", {})
    return min(
        int(runtime.get("fcnn_validation_products", 20)),
        int(runtime.get("fcnn_test_products", config["test_products"])),
        int(config["test_products"]),
    )


def paper_test_product_count(config: dict[str, Any]) -> int:
    """Return the common capped test product count for final paper comparisons."""
    runtime = config.get("runtime", {})
    caps = [
        int(config["test_products"]),
        int(runtime.get("baseline_eval_products", config["test_products"])),
        int(runtime.get("fcnn_test_products", config["test_products"])),
    ]
    return min(caps)


def paper_evaluation_product_count(config: dict[str, Any], split: str) -> int:
    """Return the product count for a validation or final test evaluation."""
    if split == "validation":
        return paper_validation_product_count(config)
    if split == "test":
        return paper_test_product_count(config)
    raise ValueError("split must be 'validation' or 'test'")


def paper_evaluation_seed_start(split: str) -> int:
    """Return the common seed offset for validation or final test evaluation."""
    if split == "validation":
        return PAPER_VALIDATION_EVALUATION_SEED_START
    if split == "test":
        return PAPER_TEST_EVALUATION_SEED_START
    raise ValueError("split must be 'validation' or 'test'")


def require_final_test_confirmation(split: str, final_test: bool) -> None:
    """Prevent accidental repeated use of the held-out test set."""
    if split == "test" and not final_test:
        raise ValueError("Refusing test-set evaluation without --final-test. Use validation for model selection.")
