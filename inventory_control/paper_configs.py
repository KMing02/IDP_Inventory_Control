"""Configuration loading and validation for paper replication experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


REQUIRED_PAPER_LOST_SALES_FIELDS = {
    "history_length_H",
    "training_horizon_T",
    "evaluation_horizon",
    "burn_in",
    "training_products",
    "test_products",
    "batch_size",
    "epochs",
    "optimizer",
    "learning_rate",
    "fcnn",
    "demand_distribution",
    "product_features",
    "product_generation",
}


def load_paper_lost_sales_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the Maggiar-style lost-sales config."""
    config_path = Path(path or os.environ.get("PAPER_LOST_SALES_CONFIG", "configs/paper_lost_sales.yaml"))
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    validate_paper_lost_sales_config(config)
    return config


def apply_config_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a copy of ``config`` with ``dotted.key=value`` overrides applied.

    Values are parsed as YAML, so ``monokan.level_network=kan``, ``epochs=2400`` and ``monokan.roll_history=false``
    give a string, an int and a bool. Every key must already exist, so a typo cannot silently add a new field.
    """
    updated = yaml.safe_load(yaml.safe_dump(config))
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or not key:
            raise ValueError(f"override must look like 'section.key=value', got {override!r}")
        *parents, leaf = key.split(".")
        node = updated
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"unknown config section {key!r}")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[leaf] = yaml.safe_load(raw_value)
    validate_paper_lost_sales_config(updated)
    return updated


def validate_paper_lost_sales_config(config: dict[str, Any]) -> None:
    """Raise ``ValueError`` when required paper-replication fields are missing."""
    missing = sorted(REQUIRED_PAPER_LOST_SALES_FIELDS - set(config))
    if missing:
        raise ValueError(f"missing required paper_lost_sales fields: {missing}")
    fcnn = config["fcnn"]
    for field in ["hidden_layers", "hidden_units", "activation"]:
        if field not in fcnn:
            raise ValueError(f"missing fcnn.{field}")
    if config["demand_distribution"] != "Gamma":
        raise ValueError("paper lost-sales replication currently expects Gamma demand")
    if int(config["history_length_H"]) <= 0:
        raise ValueError("history_length_H must be positive")
    if int(config["training_horizon_T"]) <= 0:
        raise ValueError("training_horizon_T must be positive")
