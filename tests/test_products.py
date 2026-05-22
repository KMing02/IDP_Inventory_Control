from inventory_control.envs.demand import DeterministicDemand
from inventory_control.evaluation.runner import run_episode
from inventory_control.policies.classical import BaseStockPolicy
from inventory_control.products import (
    Product,
    ProductGenerator,
    ProductGeneratorConfig,
    make_product_env_factory,
)


def _generator_config() -> ProductGeneratorConfig:
    return ProductGeneratorConfig(
        n_train=3,
        n_validation=2,
        n_test=1,
        historical_demand_length=12,
        environment_types=("lost_sales_zero_lead_time", "lost_sales_lead_time"),
        demand_process_types=("poisson", "negative_binomial"),
        mean_demand_range=(3.0, 6.0),
        coefficient_of_variation_range=(0.5, 1.0),
        holding_cost_range=(0.5, 1.0),
        lost_sales_cost_range=(4.0, 6.0),
        unit_order_cost_range=(0.5, 1.0),
        fixed_order_cost_range=(0.0, 0.2),
        initial_inventory_multiplier_range=(0.5, 1.5),
        lead_time_range=(1, 2),
        regular_lead_time_range=(2, 3),
        expedited_lead_time_range=(0, 1),
        expedited_cost_multiplier_range=(1.5, 2.0),
    )


def test_product_generator_reproducible_splits() -> None:
    generator_a = ProductGenerator(_generator_config(), seed=7)
    generator_b = ProductGenerator(_generator_config(), seed=7)

    splits_a = generator_a.generate_splits()
    splits_b = generator_b.generate_splits()

    assert splits_a == splits_b
    assert len(splits_a["train"]) == 3
    assert len(splits_a["validation"]) == 2
    assert len(splits_a["test"]) == 1


def test_product_generator_splits_are_idempotent_and_have_disjoint_ids() -> None:
    generator = ProductGenerator(_generator_config(), seed=9)

    first = generator.generate_splits()
    second = generator.generate_splits()
    first_ids = {
        product.product_id
        for products in first.values()
        for product in products
    }

    assert first == second
    assert len(first_ids) == sum(len(products) for products in first.values())


def test_product_generator_creates_historical_demand() -> None:
    product = ProductGenerator(_generator_config(), seed=1).generate_splits()["train"][0]

    assert len(product.historical_demand) == 12
    assert all(isinstance(value, int) for value in product.historical_demand)
    assert all(value >= 0 for value in product.historical_demand)


def test_observable_features_do_not_expose_true_demand_parameters() -> None:
    product = Product(
        product_id="p0",
        environment_type="lost_sales_zero_lead_time",
        demand_process_type="poisson",
        demand_parameters={"lam": 9.0},
        historical_demand=[7, 8, 10],
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        fixed_order_cost=0.0,
        initial_inventory=5,
    )

    features = product.observable_features()

    assert "demand_parameters" not in features
    assert "demand_process_type" not in features
    assert "product_id" not in features
    assert "lam" not in features
    assert features["historical_demand"] == [7, 8, 10]
    assert features["historical_mean_demand"] == 25 / 3


class StatefulDemandWithoutReset:
    def __init__(self) -> None:
        self.value = 0

    def sample(self, rng, t: int) -> int:
        self.value += 1
        return self.value


def test_product_env_factory_can_use_historical_demand_override() -> None:
    product = Product(
        product_id="p0",
        environment_type="lost_sales_zero_lead_time",
        demand_process_type="poisson",
        demand_parameters={"lam": 100.0},
        historical_demand=[1, 1],
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        fixed_order_cost=0.0,
        initial_inventory=0,
    )
    env_factory = make_product_env_factory(product, horizon=2, demand=DeterministicDemand([1, 1]))
    trace = run_episode(env_factory(seed=0), BaseStockPolicy(S=1), horizon=2, seed=0)

    assert trace["demand"].tolist() == [1, 1]


def test_product_env_factory_clones_override_demand_per_environment() -> None:
    product = Product(
        product_id="p0",
        environment_type="lost_sales_zero_lead_time",
        demand_process_type="poisson",
        demand_parameters={"lam": 100.0},
        historical_demand=[1, 1],
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        fixed_order_cost=0.0,
        initial_inventory=0,
    )
    env_factory = make_product_env_factory(
        product,
        horizon=1,
        demand=StatefulDemandWithoutReset(),
    )

    first_trace = run_episode(env_factory(seed=0), BaseStockPolicy(S=1), horizon=1, seed=0)
    second_trace = run_episode(env_factory(seed=1), BaseStockPolicy(S=1), horizon=1, seed=1)

    assert first_trace["demand"].tolist() == [1]
    assert second_trace["demand"].tolist() == [1]
