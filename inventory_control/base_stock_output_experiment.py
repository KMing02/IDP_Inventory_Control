"""Direct base-stock-output experiments for paper-style lost sales."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -e '.[dev]'`.") from exc

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import paper_evaluation_seed_start
from inventory_control.paper_lost_sales import (
    evaluate_paper_lost_sales_policy,
    historical_demand_base_stock_policy,
    non_omniscient_gamma_base_stock_policy,
    omniscient_gamma_base_stock_policy,
    paper_evaluation_demand_path,
)
from inventory_control.evaluation.torch_policy import batched_torch_policy_metrics
from inventory_control.policies.neural.fcnn import (
    PaperFCNNFeatureStats,
    compute_paper_fcnn_feature_stats,
    empirical_newsvendor_target_ratio,
    make_paper_direct_target_stock_fcnn_model_for_products,
)
from inventory_control.policies.neural.monokan import (
    make_paper_direct_target_stock_kan_model,
    make_paper_direct_target_stock_monokan_model_for_products,
    make_paper_deep_grid7_direct_target_stock_kan_model,
    make_paper_deep_grid7_direct_target_stock_monokan_model_for_products,
)
from inventory_control.policies.neural.registry import (
    CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES,
    DIRECT_TARGET_STOCK_POLICY_NAMES,
    POLICY_SPECS_BY_NAME,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import DirectBackpropResult, train_paper_lost_sales_fcnn


@dataclass(frozen=True)
class BaseStockOutputArtifacts:
    """Artifacts from one direct base-stock-output comparison."""

    output_dir: Path
    summary: pd.DataFrame
    training_log: pd.DataFrame
    target_diagnostics: pd.DataFrame
    product_metrics: pd.DataFrame
    metadata: dict[str, Any]


def run_base_stock_output_experiment(
    config_path: str | Path = "configs/paper_lost_sales_debug.yaml",
    output_dir: str | Path = "outputs/base_stock_output_smoke",
    models: tuple[str, ...] = DIRECT_TARGET_STOCK_POLICY_NAMES,
    device: str | None = None,
    seed: int | None = None,
    activation_checkpointing: bool = True,
    shared_training_seed: bool = False,
) -> BaseStockOutputArtifacts:
    """Run a no-imitation direct base-stock-output comparison."""
    supported_models = set(DIRECT_TARGET_STOCK_POLICY_NAMES) | set(CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES)
    unknown = sorted(set(models) - supported_models)
    if unknown:
        raise ValueError(f"unsupported base-stock-output models: {unknown}")

    config = load_paper_lost_sales_config(config_path)
    settings = _compact_settings(config)
    if device is not None:
        settings["device"] = str(device)
    if seed is not None:
        settings["seed"] = int(seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(settings["training_products"], split="train")
    validation_products = generator.generate(settings["validation_products"], split="validation")
    feature_stats = (
        compute_paper_fcnn_feature_stats(train_products)
        if settings["normalize_observations"]
        else PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0)
    )
    initial_target_ratio = empirical_newsvendor_target_ratio(train_products, feature_stats)

    summary_rows: list[dict[str, Any]] = []
    log_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    product_frames: list[pd.DataFrame] = []
    rolling_history_by_model: dict[str, bool] = {}
    for model_name in models:
        spec = POLICY_SPECS_BY_NAME[model_name]
        model_seed = settings["seed"] + spec.seed_offset
        torch.manual_seed(model_seed)
        model = _make_model(model_name, config, train_products, feature_stats, initial_target_ratio)
        include_newsvendor_feature = bool(getattr(model, "paper_include_newsvendor_feature", False))
        enforce_action_cap = bool(getattr(model, "paper_enforce_action_cap", True))
        include_demand_level_feature = bool(getattr(model, "paper_include_demand_level_feature", False))
        rolling_history = bool(getattr(model, "paper_roll_history", False))
        rolling_history_by_model[model_name] = rolling_history
        training_seed = settings["seed"] if shared_training_seed else model_seed
        result = train_paper_lost_sales_fcnn(
            train_products,
            model,
            horizon=settings["training_horizon"],
            batch_size=min(settings["batch_size"], len(train_products)),
            epochs=settings["reward_epochs"],
            learning_rate=settings["learning_rate"],
            seed=training_seed,
            imitation_epochs=0,
            validation_products=validation_products,
            validation_interval=settings["validation_interval"],
            feature_stats=feature_stats,
            max_order_factor=settings["max_order_factor"],
            normalize_reward_loss=settings["normalize_reward_loss"],
            training_demand_mode=settings["training_demand_mode"],
            history_feature_mode=settings["paper_history_feature_mode"],
            include_newsvendor_feature=include_newsvendor_feature,
            include_demand_level_feature=include_demand_level_feature,
            include_terminal_salvage=settings["include_terminal_salvage"],
            initial_inventory_mode=settings["initial_inventory_mode"],
            enforce_action_cap=enforce_action_cap,
            checkpoint_rolling_target=activation_checkpointing,
            device=settings["device"],
        )
        final_state = copy.deepcopy(model.state_dict())
        final_checkpoint_filename = f"{Path(spec.checkpoint_filename).stem}_final.pt"
        torch.save(final_state, destination / final_checkpoint_filename)
        selected_checkpoint_epoch = _restore_best_validation_state(model, result)
        torch.save(model.state_dict(), destination / spec.checkpoint_filename)
        target_diagnostic = _target_diagnostics(model, model_name, validation_products, feature_stats, settings)
        product_metrics = _evaluate_model(
            model,
            model_name,
            model_seed,
            validation_products,
            feature_stats,
            settings,
        )
        summary_rows.append(
            {
                "policy_name": model_name,
                **_mean_metrics(product_metrics.to_dict("records")),
                **_target_summary(target_diagnostic),
                "training_epochs": settings["reward_epochs"],
                "imitation_epochs": 0,
                "initial_target_ratio": initial_target_ratio,
                "parameter_count": _parameter_count(model),
                "training_seconds": result.elapsed_seconds,
                "checkpoint_filename": spec.checkpoint_filename,
                "final_checkpoint_filename": final_checkpoint_filename,
                "selected_checkpoint_epoch": selected_checkpoint_epoch,
                "final_checkpoint_epoch": settings["reward_epochs"],
                "model_seed": model_seed,
                "training_seed": training_seed,
                "action_cap_enforced": enforce_action_cap,
                "persistent_newsvendor_anchor": include_newsvendor_feature,
                "rolling_history": rolling_history,
            }
        )
        log_frames.append(_training_log_frame(result, model_name))
        target_frames.append(target_diagnostic)
        product_frames.append(product_metrics)

    for policy_name, policy_factory in (
        ("HistoricalDemandBaseStock", historical_demand_base_stock_policy),
        ("NonOmniscientOptimal", non_omniscient_gamma_base_stock_policy),
        ("OmniscientOptimal", omniscient_gamma_base_stock_policy),
    ):
        rows = [
            _evaluate_baseline(product, policy_name, settings, index, policy_factory)
            for index, product in enumerate(validation_products)
        ]
        summary_rows.append({"policy_name": policy_name, **_mean_metrics(rows)})
        product_frames.append(pd.DataFrame(rows))
    summary = pd.DataFrame(summary_rows)
    training_log = pd.concat(log_frames, ignore_index=True) if log_frames else pd.DataFrame()
    target_diagnostics = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    product_metrics = pd.concat(product_frames, ignore_index=True) if product_frames else pd.DataFrame()
    metadata = {
        "config_path": str(config_path),
        "settings": settings,
        "models": list(models),
        "action_parameterization": "r_theta=softplus(model(z)); S_t=m_0*r_theta; q_t=max(S_t-I_t,0)",
        "paper_history_feature_mode": settings["paper_history_feature_mode"],
        "paper_include_newsvendor_feature": settings["paper_include_newsvendor_feature"],
        "include_terminal_salvage": settings["include_terminal_salvage"],
        "initial_inventory_mode": settings["initial_inventory_mode"],
        "normalize_observations": settings["normalize_observations"],
        "rolling_history": rolling_history_by_model,
        "initialization": "final output initialized near empirical newsvendor target ratio",
        "imitation_pretraining": False,
        "activation_checkpointing": bool(activation_checkpointing),
        "shared_training_seed": bool(shared_training_seed),
        "direct_target_stock_models": list(DIRECT_TARGET_STOCK_POLICY_NAMES),
        "capacity_comparison_models": list(CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES),
    }

    summary.to_csv(destination / "summary.csv", index=False)
    training_log.to_csv(destination / "training_log.csv", index=False)
    target_diagnostics.to_csv(destination / "target_stock_diagnostics.csv", index=False)
    product_metrics.to_csv(destination / "product_metrics.csv", index=False)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return BaseStockOutputArtifacts(destination, summary, training_log, target_diagnostics, product_metrics, metadata)


def _compact_settings(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config.get("idp_benchmark", {})
    compact = benchmark.get("compact", {})
    return {
        "seed": int(benchmark.get("seed", config.get("product_generation", {}).get("seed", 0))),
        "product_generation_seed": int(config.get("product_generation", {}).get("seed", 0)),
        "training_products": int(compact.get("training_products", config.get("training_products", 8))),
        "validation_products": int(compact.get("validation_products", config.get("test_products", 4))),
        "batch_size": int(compact.get("batch_size", config.get("batch_size", 4))),
        "training_horizon": int(compact.get("training_horizon", config.get("training_horizon_T", 8))),
        "reward_epochs": int(compact.get("reward_epochs", config.get("epochs", 2))),
        "validation_interval": int(compact.get("validation_interval", 0)),
        "evaluation_horizon": int(compact.get("evaluation_horizon", config.get("evaluation_horizon", 12))),
        "burn_in": int(compact.get("burn_in", config.get("burn_in", 0))),
        "max_order_factor": float(benchmark.get("max_order_factor", config.get("fcnn", {}).get("max_order_factor", 4.0))),
        "normalize_reward_loss": bool(benchmark.get("normalize_reward_loss", True)),
        "include_terminal_salvage": bool(
            benchmark.get("include_terminal_salvage", config.get("include_terminal_salvage", True))
        ),
        "training_demand_mode": str(benchmark.get("training_demand_mode", "fixed_future")),
        "initial_inventory_mode": str(benchmark.get("initial_inventory_mode", "historical_mean")),
        "normalize_observations": bool(benchmark.get("normalize_observations", True)),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "device": benchmark.get("device"),
        "paper_history_feature_mode": str(
            benchmark.get("paper_history_feature_mode", config.get("paper_history_feature_mode", "raw_lags"))
        ),
        "paper_include_newsvendor_feature": bool(
            benchmark.get("paper_include_newsvendor_feature", config.get("paper_include_newsvendor_feature", False))
        ),
    }


def _make_model(
    name: str,
    config: dict[str, Any],
    train_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    initial_target_ratio: float,
) -> nn.Module:
    if name == "PaperFCNNDirectTargetStock":
        return make_paper_direct_target_stock_fcnn_model_for_products(
            config,
            train_products,
            feature_stats=feature_stats,
        )
    if name in {"KANDirectTargetStock", "MonoKANDirectTargetStock"}:
        metadata = dict(config.get("monokan", {}))
        metadata["initial_normalized_target"] = initial_target_ratio
        if name == "KANDirectTargetStock":
            return make_paper_direct_target_stock_kan_model(config, metadata=metadata)
        return make_paper_direct_target_stock_monokan_model_for_products(
            config,
            train_products,
            feature_stats,
            metadata=metadata,
        )
    if name in CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES:
        metadata = dict(config.get("monokan", {}))
        metadata["initial_normalized_target"] = initial_target_ratio
        if name == "KANDirectTargetStockDeepGrid7":
            return make_paper_deep_grid7_direct_target_stock_kan_model(config, metadata=metadata)
        return make_paper_deep_grid7_direct_target_stock_monokan_model_for_products(
            config,
            train_products,
            feature_stats,
            metadata=metadata,
        )
    raise ValueError(f"unknown direct base-stock model: {name}")


def _evaluate_model(
    model: nn.Module,
    model_name: str,
    model_seed: int,
    products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
) -> pd.DataFrame:
    use_batched_evaluation = bool(
        settings["paper_history_feature_mode"] in {"raw_lags", "raw_lags_temporal"}
        and not getattr(model, "paper_include_newsvendor_feature", False)
        and not getattr(model, "paper_include_demand_level_feature", False)
    )
    if use_batched_evaluation:
        batch_settings = {
            **settings,
            "scale_action_by_history_mean": False,
            "enforce_action_cap": bool(getattr(model, "paper_enforce_action_cap", True)),
        }
        metrics = batched_torch_policy_metrics(products, model, feature_stats, batch_settings, "validation")
        rows = [
            _product_metric_row(
                metrics.iloc[index].to_dict(),
                model_name,
                product,
                settings,
                index,
                model_seed=model_seed,
            )
            for index, product in enumerate(products)
        ]
    else:
        rows = []
        for index, product in enumerate(products):
            policy = _TorchBaseStockPolicyAdapter(
                model,
                product,
                feature_stats,
                settings["max_order_factor"],
                settings["paper_history_feature_mode"],
                bool(getattr(model, "paper_include_newsvendor_feature", False)),
                bool(getattr(model, "paper_enforce_action_cap", True)),
                bool(getattr(model, "paper_include_demand_level_feature", False)),
            )
            metrics = evaluate_paper_lost_sales_policy(
                product,
                policy,
                horizon=settings["evaluation_horizon"],
                n_episodes=1,
                seed=paper_evaluation_seed_start("validation") + index,
                burn_in=settings["burn_in"],
            )
            rows.append(
                _product_metric_row(
                    metrics,
                    model_name,
                    product,
                    settings,
                    index,
                    model_seed=model_seed,
                )
            )
    return pd.DataFrame(rows)


class _TorchBaseStockPolicyAdapter:
    """Evaluate a context-target torch model through the shared order-up-to decoder."""

    def __init__(
        self,
        model: nn.Module,
        product: PaperLostSalesProduct,
        feature_stats: PaperFCNNFeatureStats,
        max_order_factor: float,
        history_feature_mode: str = "raw_lags",
        include_newsvendor_feature: bool = False,
        enforce_action_cap: bool = True,
        include_demand_level_feature: bool = False,
    ) -> None:
        self.name = getattr(model, "name", "NeuralBaseStockOrderUpTo")
        self.model = model
        self.product = product
        self.feature_stats = feature_stats
        self.max_order_factor = float(max_order_factor)
        self.history_feature_mode = history_feature_mode
        self.include_newsvendor_feature = include_newsvendor_feature
        self.enforce_action_cap = bool(enforce_action_cap)
        self.include_demand_level_feature = bool(include_demand_level_feature)
        self.initial_history = [float(value) for value in product.historical_demand]
        self.history = list(self.initial_history)
        self.roll_history = bool(getattr(model, "paper_roll_history", False))
        self.target_stock = self._predict_target_stock()

    def reset(self) -> None:
        """Reset episode-local demand history and initial target stock."""
        self.history = list(self.initial_history)
        self.target_stock = self._predict_target_stock()

    def _predict_target_stock(self) -> float:
        from inventory_control.policies.neural.fcnn import build_paper_lost_sales_features, infer_model_device

        features = torch.tensor(
            [
                build_paper_lost_sales_features(
                    inventory=0.0,
                    historical_demand=self.history,
                    price=self.product.price,
                    purchase_cost=self.product.purchase_cost,
                    holding_cost=self.product.holding_cost,
                    lost_sale_penalty=self.product.lost_sale_penalty,
                    feature_stats=self.feature_stats,
                    history_feature_mode=self.history_feature_mode,
                    include_newsvendor_feature=self.include_newsvendor_feature,
                    include_demand_level_feature=self.include_demand_level_feature,
                )
            ],
            dtype=torch.float32,
            device=infer_model_device(self.model),
        )
        self.model.eval()
        with torch.no_grad():
            target_factor = float(self.model(features[:, 1:]).item())
        historical_mean = max(sum(self.product.historical_demand) / len(self.product.historical_demand), 1.0)
        return target_factor * historical_mean

    def act(self, inventory: float, t: int, rng=None) -> float:
        if self.roll_history:
            self.target_stock = self._predict_target_stock()
        historical_mean = max(sum(self.product.historical_demand) / len(self.product.historical_demand), 1.0)
        order = max(self.target_stock - float(inventory), 0.0)
        if not self.enforce_action_cap:
            return order
        max_inventory_level = self.max_order_factor * historical_mean
        return min(order, max(max_inventory_level - float(inventory), 0.0))

    def observe_demand(self, demand: float) -> None:
        """Advance the observable history after the current demand is revealed."""
        if self.roll_history:
            self.history = [*self.history[1:], float(demand)]


def _evaluate_baseline(
    product: PaperLostSalesProduct,
    policy_name: str,
    settings: dict[str, Any],
    index: int,
    policy_factory,
) -> dict[str, float]:
    metrics = evaluate_paper_lost_sales_policy(
        product,
        policy_factory(product),
        horizon=settings["evaluation_horizon"],
        n_episodes=1,
        seed=paper_evaluation_seed_start("validation") + index,
        burn_in=settings["burn_in"],
    )
    return _product_metric_row(metrics, policy_name, product, settings, index, model_seed=None)


def _product_metric_row(
    metrics: dict[str, float],
    policy_name: str,
    product: PaperLostSalesProduct,
    settings: dict[str, Any],
    index: int,
    model_seed: int | None,
) -> dict[str, Any]:
    """Attach explicit pairing keys to one product-level evaluation result."""
    evaluation_seed = paper_evaluation_seed_start("validation") + index
    demand_path = paper_evaluation_demand_path(product, settings["evaluation_horizon"], evaluation_seed)
    demand_path_payload = json.dumps(demand_path.tolist(), separators=(",", ":")).encode("utf-8")
    demand_path_source = (
        "stored_future"
        if product.future_demand is not None and len(product.future_demand) >= settings["evaluation_horizon"]
        else "sampled"
    )
    return {
        **metrics,
        "run_seed": settings["seed"],
        "model_seed": model_seed,
        "product_generation_seed": settings["product_generation_seed"],
        "evaluation_seed": evaluation_seed,
        "demand_path_source": demand_path_source,
        "demand_path_id": hashlib.sha256(demand_path_payload).hexdigest()[:16],
        "policy_name": policy_name,
        "product_id": product.product_id,
    }


def _target_diagnostics(
    model: nn.Module,
    model_name: str,
    products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Return per-product target-stock scale diagnostics for base-stock-output models."""
    from inventory_control.policies.neural.fcnn import build_paper_lost_sales_features, infer_model_device

    rows: list[dict[str, float | str]] = []
    device = infer_model_device(model)
    include_newsvendor_feature = bool(getattr(model, "paper_include_newsvendor_feature", False))
    include_demand_level_feature = bool(getattr(model, "paper_include_demand_level_feature", False))
    enforce_action_cap = bool(getattr(model, "paper_enforce_action_cap", True))
    model.eval()
    for product in products:
        historical_mean = max(sum(product.historical_demand) / len(product.historical_demand), 1.0)
        empirical_base_stock = historical_demand_base_stock_policy(product).S
        features = torch.tensor(
            [
                build_paper_lost_sales_features(
                    inventory=0.0,
                    historical_demand=product.historical_demand,
                    price=product.price,
                    purchase_cost=product.purchase_cost,
                    holding_cost=product.holding_cost,
                    lost_sale_penalty=product.lost_sale_penalty,
                    feature_stats=feature_stats,
                    history_feature_mode=settings["paper_history_feature_mode"],
                    include_newsvendor_feature=include_newsvendor_feature,
                    include_demand_level_feature=include_demand_level_feature,
                )
            ],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            target_factor = float(model(features[:, 1:]).item())
        target_stock = target_factor * historical_mean
        rows.append(
            {
                "policy_name": model_name,
                "product_id": product.product_id,
                "target_factor": target_factor,
                "target_stock": target_stock,
                "historical_mean_demand": historical_mean,
                "historical_base_stock": float(empirical_base_stock),
                "target_minus_historical_base_stock": target_stock - float(empirical_base_stock),
                "target_to_historical_base_stock_ratio": target_stock / max(float(empirical_base_stock), 1e-6),
                "near_upper_bound": (
                    float(target_factor >= 0.95 * float(settings["max_order_factor"]))
                    if enforce_action_cap
                    else float("nan")
                ),
                "action_cap_enforced": enforce_action_cap,
            }
        )
    return pd.DataFrame(rows)


def _target_summary(target_diagnostics: pd.DataFrame) -> dict[str, float]:
    """Return compact target-scale summaries for the main results table."""
    return {
        "target_factor_mean": float(target_diagnostics["target_factor"].mean()),
        "target_factor_q90": float(target_diagnostics["target_factor"].quantile(0.9)),
        "target_stock_mean": float(target_diagnostics["target_stock"].mean()),
        "target_minus_historical_base_stock_mean": float(
            target_diagnostics["target_minus_historical_base_stock"].mean()
        ),
        "near_upper_bound_rate": float(target_diagnostics["near_upper_bound"].mean()),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    return {
        "mean_reward_per_period": float(frame["mean_reward_per_period"].mean()),
        "service_level": float(frame["service_level"].mean()),
        "lost_sales_rate": float(frame["lost_sales_rate"].mean()),
        "mean_order_quantity": float(frame["mean_order_quantity"].mean()),
        "mean_inventory": float(frame["mean_inventory"].mean()),
    }


def _training_log_frame(result: DirectBackpropResult, model_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(result.training_log or [])
    frame["policy_name"] = model_name
    return frame


def _restore_best_validation_state(model: nn.Module, result: DirectBackpropResult) -> int | None:
    """Restore the validation-selected state and return its epoch when available."""
    if result.best_state_dict is None:
        return None
    model.load_state_dict(result.best_state_dict)
    if not result.validation_rewards or not result.validation_epochs:
        return None
    best_index = max(range(len(result.validation_rewards)), key=result.validation_rewards.__getitem__)
    return int(result.validation_epochs[best_index])


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
