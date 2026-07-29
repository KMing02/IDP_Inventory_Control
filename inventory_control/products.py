"""Product-level benchmark generation for inventory generalization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from inventory_control.envs.demand import (
    DemandDistribution,
    DeterministicDemand,
    NegativeBinomialDemand,
    PoissonDemand,
)
from inventory_control.envs.lost_sales import (
    DualSourcingEnv,
    DualSourcingEnvConfig,
    LostSalesLeadTimeEnv,
    LostSalesLeadTimeEnvConfig,
    LostSalesZeroLeadTimeEnv,
    LostSalesEnvConfig,
)


@dataclass(frozen=True)
class Product:
    """A single generated product with hidden true demand parameters."""

    product_id: str
    environment_type: str
    demand_process_type: str
    demand_parameters: dict[str, float]
    historical_demand: list[int]
    holding_cost: float
    lost_sales_cost: float
    unit_order_cost: float
    fixed_order_cost: float
    initial_inventory: int
    lead_time: int | None = None
    regular_lead_time: int | None = None
    expedited_lead_time: int | None = None
    regular_unit_order_cost: float | None = None
    expedited_unit_order_cost: float | None = None
    regular_fixed_order_cost: float = 0.0
    expedited_fixed_order_cost: float = 0.0

    def observable_features(self, recent_history_length: int | None = None) -> dict[str, Any]:
        """Return product features that do not expose true demand parameters."""
        history = self.historical_demand
        if recent_history_length is not None:
            history = history[-recent_history_length:]
        features: dict[str, Any] = {
            "environment_type": self.environment_type,
            "historical_demand": list(history),
            "historical_mean_demand": float(np.mean(history)) if history else 0.0,
            "historical_std_demand": float(np.std(history, ddof=0)) if history else 0.0,
            "holding_cost": self.holding_cost,
            "lost_sales_cost": self.lost_sales_cost,
            "unit_order_cost": self.unit_order_cost,
            "fixed_order_cost": self.fixed_order_cost,
            "initial_inventory": self.initial_inventory,
            "lead_time": self.lead_time,
            "regular_lead_time": self.regular_lead_time,
            "expedited_lead_time": self.expedited_lead_time,
            "regular_unit_order_cost": self.regular_unit_order_cost,
            "expedited_unit_order_cost": self.expedited_unit_order_cost,
        }
        return features


@dataclass(frozen=True)
class PaperLostSalesProduct:
    """Product for Maggiar-style no-lead-time lost-sales replication."""

    product_id: str
    price: float
    purchase_cost: float
    holding_cost: float
    lost_sale_penalty: float
    true_mean_demand: float
    true_coefficient_of_variation: float
    gamma_shape: float
    gamma_scale: float
    demand_distribution_type: str
    historical_demand: list[int]
    initial_inventory: int
    future_demand: list[int] | None = None

    def observable_features(self) -> dict[str, Any]:
        """Return policy-observable features without true distribution parameters."""
        return {
            "historical_demand": list(self.historical_demand),
            "price": self.price,
            "purchase_cost": self.purchase_cost,
            "holding_cost": self.holding_cost,
            "lost_sale_penalty": self.lost_sale_penalty,
            "initial_inventory": self.initial_inventory,
        }

    def sample_demand(self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None):
        """Sample rounded non-negative Gamma demand from hidden true parameters."""
        samples = rng.gamma(shape=self.gamma_shape, scale=self.gamma_scale, size=size)
        return np.maximum(np.rint(samples), 0).astype(int)


class PaperLostSalesProductGenerator:
    """Generate Gamma-demand products for paper-style lost-sales replication."""

    def __init__(self, config: dict[str, Any], seed: int | None = None) -> None:
        self.config = config
        generation = config["product_generation"]
        self.seed = int(generation.get("seed", 0) if seed is None else seed)
        self.history_length = int(config["history_length_H"])

    def generate(self, n_products: int, split: str) -> list[PaperLostSalesProduct]:
        """Generate ``n_products`` independent products for one split."""
        rng = np.random.default_rng(self.seed + _split_seed_offset(split))
        return [self._generate_one(rng, split, idx) for idx in range(n_products)]

    def _generate_one(
        self,
        rng: np.random.Generator,
        split: str,
        index: int,
    ) -> PaperLostSalesProduct:
        generation = self.config["product_generation"]
        if generation.get("mode") == "paper_table_1":
            price = float(rng.exponential(float(generation.get("price_mean", 100.0))))
            purchase_cost = price * float(rng.uniform(0.0, 1.0))
            holding_cost = float(rng.exponential(float(generation.get("holding_cost_mean", 5.0))))
            lost_sale_penalty = float(generation.get("lost_sale_penalty_max", 10.0)) * float(rng.uniform(0.0, 1.0))
            mean_demand = float(rng.exponential(float(generation.get("mean_demand_mean", 100.0))))
            cv_bounds = generation.get("coefficient_of_variation_range", [0.0, 1.0])
            cv = _uniform_range(rng, cv_bounds)
            initial_inventory = 0
        else:
            price = _uniform_range(rng, generation["price_range"])
            purchase_cost = price * _uniform_range(rng, generation["purchase_cost_fraction_range"])
            holding_cost = price * _uniform_range(rng, generation["holding_cost_fraction_range"])
            lost_sale_penalty = price * _uniform_range(rng, generation["lost_sale_penalty_fraction_range"])
            mean_demand = _uniform_range(rng, generation["mean_demand_range"])
            cv = _uniform_range(rng, generation["coefficient_of_variation_range"])
            initial_inventory = int(round(mean_demand * _uniform_range(rng, generation["initial_inventory_multiplier_range"])))
        gamma_shape = 1.0 / max(cv**2, 1e-6)
        gamma_scale = mean_demand / gamma_shape
        evaluation_horizon = self.config.get("evaluation_horizon")
        trace_horizon = max(
            int(evaluation_horizon)
            if evaluation_horizon is not None
            else int(self.config.get("training_horizon_T", 0)),
            1,
        )
        demand_trace = np.maximum(
            np.rint(rng.gamma(shape=gamma_shape, scale=gamma_scale, size=self.history_length + trace_horizon)),
            0,
        ).astype(int)
        history = demand_trace[: self.history_length]
        future = demand_trace[self.history_length :]
        return PaperLostSalesProduct(
            product_id=f"{split}_{index:06d}",
            price=price,
            purchase_cost=purchase_cost,
            holding_cost=holding_cost,
            lost_sale_penalty=lost_sale_penalty,
            true_mean_demand=mean_demand,
            true_coefficient_of_variation=cv,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            demand_distribution_type="Gamma",
            historical_demand=history.tolist(),
            initial_inventory=initial_inventory,
            future_demand=future.tolist(),
        )


def _uniform_range(rng: np.random.Generator, bounds: list[float] | tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _split_seed_offset(split: str) -> int:
    return {"train": 0, "validation": 1_000_000, "test": 2_000_000}.get(split, 3_000_000)


@dataclass(frozen=True)
class ProductGeneratorConfig:
    """Configuration for heterogeneous product generation."""

    n_train: int
    n_validation: int
    n_test: int
    historical_demand_length: int
    environment_types: tuple[str, ...]
    demand_process_types: tuple[str, ...]
    mean_demand_range: tuple[float, float]
    coefficient_of_variation_range: tuple[float, float]
    holding_cost_range: tuple[float, float]
    lost_sales_cost_range: tuple[float, float]
    unit_order_cost_range: tuple[float, float]
    fixed_order_cost_range: tuple[float, float]
    initial_inventory_multiplier_range: tuple[float, float]
    lead_time_range: tuple[int, int]
    regular_lead_time_range: tuple[int, int]
    expedited_lead_time_range: tuple[int, int]
    expedited_cost_multiplier_range: tuple[float, float]


class ProductGenerator:
    """Sample heterogeneous products with reproducible train/validation/test splits."""

    def __init__(self, config: ProductGeneratorConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = seed

    def generate_splits(self) -> dict[str, list[Product]]:
        """Generate product splits."""
        seed_sequence = np.random.SeedSequence(self.seed)
        train_seed, validation_seed, test_seed = seed_sequence.spawn(3)
        return {
            "train": self._generate_split("train", self.config.n_train, np.random.default_rng(train_seed)),
            "validation": self._generate_split(
                "validation",
                self.config.n_validation,
                np.random.default_rng(validation_seed),
            ),
            "test": self._generate_split("test", self.config.n_test, np.random.default_rng(test_seed)),
        }

    def _generate_split(
        self,
        split: str,
        n_products: int,
        rng: np.random.Generator,
    ) -> list[Product]:
        return [self._generate_product(split, index, rng) for index in range(n_products)]

    def _generate_product(
        self,
        split: str,
        index: int,
        rng: np.random.Generator,
    ) -> Product:
        mean_demand = self._uniform(rng, *self.config.mean_demand_range)
        cv = self._uniform(rng, *self.config.coefficient_of_variation_range)
        demand_process_type = str(rng.choice(self.config.demand_process_types))
        environment_type = str(rng.choice(self.config.environment_types))
        demand_parameters = _demand_parameters_from_mean_cv(mean_demand, cv, demand_process_type)
        demand = make_product_demand(demand_process_type, demand_parameters)

        history_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        historical_demand = [
            demand.sample(history_rng, t)
            for t in range(self.config.historical_demand_length)
        ]

        holding_cost = self._uniform(rng, *self.config.holding_cost_range)
        lost_sales_cost = self._uniform(rng, *self.config.lost_sales_cost_range)
        unit_order_cost = self._uniform(rng, *self.config.unit_order_cost_range)
        fixed_order_cost = self._uniform(rng, *self.config.fixed_order_cost_range)
        initial_multiplier = self._uniform(rng, *self.config.initial_inventory_multiplier_range)
        initial_inventory = int(round(mean_demand * initial_multiplier))

        lead_time = None
        regular_lead_time = None
        expedited_lead_time = None
        regular_unit_order_cost = None
        expedited_unit_order_cost = None
        if environment_type == "lost_sales_lead_time":
            lead_time = int(rng.integers(self.config.lead_time_range[0], self.config.lead_time_range[1] + 1))
        elif environment_type == "dual_sourcing":
            regular_lead_time = int(
                rng.integers(
                    self.config.regular_lead_time_range[0],
                    self.config.regular_lead_time_range[1] + 1,
                )
            )
            expedited_lead_time = int(
                rng.integers(
                    self.config.expedited_lead_time_range[0],
                    self.config.expedited_lead_time_range[1] + 1,
                )
            )
            regular_unit_order_cost = unit_order_cost
            expedited_unit_order_cost = unit_order_cost * self._uniform(
                rng,
                *self.config.expedited_cost_multiplier_range
            )

        return Product(
            product_id=f"{split}_{index:04d}",
            environment_type=environment_type,
            demand_process_type=demand_process_type,
            demand_parameters=demand_parameters,
            historical_demand=historical_demand,
            holding_cost=holding_cost,
            lost_sales_cost=lost_sales_cost,
            unit_order_cost=unit_order_cost,
            fixed_order_cost=fixed_order_cost,
            initial_inventory=initial_inventory,
            lead_time=lead_time,
            regular_lead_time=regular_lead_time,
            expedited_lead_time=expedited_lead_time,
            regular_unit_order_cost=regular_unit_order_cost,
            expedited_unit_order_cost=expedited_unit_order_cost,
        )

    def _uniform(self, rng: np.random.Generator, low: float, high: float) -> float:
        return float(rng.uniform(low, high))


def product_generator_config_from_dict(config: dict[str, Any]) -> ProductGeneratorConfig:
    """Create generator config from YAML-loaded dictionary."""
    return ProductGeneratorConfig(
        n_train=int(config["splits"]["train"]),
        n_validation=int(config["splits"]["validation"]),
        n_test=int(config["splits"]["test"]),
        historical_demand_length=int(config["historical_demand_length"]),
        environment_types=tuple(config["environment_types"]),
        demand_process_types=tuple(config["demand_process_types"]),
        mean_demand_range=tuple(config["mean_demand_range"]),
        coefficient_of_variation_range=tuple(config["coefficient_of_variation_range"]),
        holding_cost_range=tuple(config["holding_cost_range"]),
        lost_sales_cost_range=tuple(config["lost_sales_cost_range"]),
        unit_order_cost_range=tuple(config["unit_order_cost_range"]),
        fixed_order_cost_range=tuple(config["fixed_order_cost_range"]),
        initial_inventory_multiplier_range=tuple(config["initial_inventory_multiplier_range"]),
        lead_time_range=tuple(config["lead_time_range"]),
        regular_lead_time_range=tuple(config["regular_lead_time_range"]),
        expedited_lead_time_range=tuple(config["expedited_lead_time_range"]),
        expedited_cost_multiplier_range=tuple(config["expedited_cost_multiplier_range"]),
    )


def make_product_demand(
    demand_process_type: str,
    demand_parameters: dict[str, float],
) -> DemandDistribution:
    """Create true simulator demand for a product."""
    if demand_process_type == "poisson":
        return PoissonDemand(lam=float(demand_parameters["lam"]))
    if demand_process_type == "negative_binomial":
        return NegativeBinomialDemand(
            mean=float(demand_parameters["mean"]),
            dispersion=float(demand_parameters["dispersion"]),
        )
    raise ValueError(f"unsupported product demand type: {demand_process_type}")


def make_product_env_factory(
    product: Product,
    horizon: int,
    demand: DemandDistribution | None = None,
):
    """Create an environment factory for a product.

    By default this uses the product's hidden true simulator demand parameters.
    Passing ``demand`` is useful for non-omniscient tuning from historical data.
    """
    if demand is None:
        demand_factory = lambda: make_product_demand(product.demand_process_type, product.demand_parameters)
    else:
        demand_factory = lambda: copy.deepcopy(demand)

    if product.environment_type == "lost_sales_zero_lead_time":
        env_config = LostSalesEnvConfig(
            horizon=horizon,
            initial_inventory=product.initial_inventory,
            holding_cost=product.holding_cost,
            lost_sales_cost=product.lost_sales_cost,
            unit_order_cost=product.unit_order_cost,
            fixed_order_cost=product.fixed_order_cost,
        )

        def factory(seed: int | None = None) -> LostSalesZeroLeadTimeEnv:
            return LostSalesZeroLeadTimeEnv(env_config, demand_factory(), seed=seed)

        return factory

    if product.environment_type == "lost_sales_lead_time":
        if product.lead_time is None:
            raise ValueError("lead_time is required for lost_sales_lead_time products")
        env_config = LostSalesLeadTimeEnvConfig(
            horizon=horizon,
            initial_inventory=product.initial_inventory,
            holding_cost=product.holding_cost,
            lost_sales_cost=product.lost_sales_cost,
            unit_order_cost=product.unit_order_cost,
            fixed_order_cost=product.fixed_order_cost,
            lead_time=product.lead_time,
        )

        def factory(seed: int | None = None) -> LostSalesLeadTimeEnv:
            return LostSalesLeadTimeEnv(env_config, demand_factory(), seed=seed)

        return factory

    if product.environment_type == "dual_sourcing":
        if (
            product.regular_lead_time is None
            or product.expedited_lead_time is None
            or product.regular_unit_order_cost is None
            or product.expedited_unit_order_cost is None
        ):
            raise ValueError("dual-sourcing products require supplier lead times and costs")
        env_config = DualSourcingEnvConfig(
            horizon=horizon,
            initial_inventory=product.initial_inventory,
            holding_cost=product.holding_cost,
            lost_sales_cost=product.lost_sales_cost,
            regular_unit_order_cost=product.regular_unit_order_cost,
            expedited_unit_order_cost=product.expedited_unit_order_cost,
            regular_fixed_order_cost=product.regular_fixed_order_cost,
            expedited_fixed_order_cost=product.expedited_fixed_order_cost,
            regular_lead_time=product.regular_lead_time,
            expedited_lead_time=product.expedited_lead_time,
        )

        def factory(seed: int | None = None) -> DualSourcingEnv:
            return DualSourcingEnv(env_config, demand_factory(), seed=seed)

        return factory

    raise ValueError(f"unsupported environment_type: {product.environment_type}")


def _demand_parameters_from_mean_cv(
    mean_demand: float,
    cv: float,
    demand_process_type: str,
) -> dict[str, float]:
    if demand_process_type == "poisson":
        return {"lam": mean_demand}
    if demand_process_type == "negative_binomial":
        variance = max((cv * mean_demand) ** 2, mean_demand + 1e-6)
        dispersion = max(mean_demand**2 / max(variance - mean_demand, 1e-6), 1e-3)
        return {"mean": mean_demand, "dispersion": dispersion}
    raise ValueError(f"unsupported demand_process_type: {demand_process_type}")
