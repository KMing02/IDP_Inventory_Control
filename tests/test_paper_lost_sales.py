import numpy as np
import pytest

from inventory_control.paper_lost_sales import (
    BaseStockOrderPolicy,
    evaluate_paper_lost_sales_policy,
    historical_demand_base_stock_policy,
    monotonicity_violation_rate,
    oracle_gamma_base_stock_policy,
    paper_lost_sales_step,
)
from inventory_control.paper_evaluation import PAPER_TEST_EVALUATION_SEED_START, paper_test_product_count
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator


def _paper_config() -> dict:
    return {
        "history_length_H": 32,
        "product_generation": {
            "seed": 1,
            "price_range": [10.0, 10.0],
            "purchase_cost_fraction_range": [0.5, 0.5],
            "holding_cost_fraction_range": [0.1, 0.1],
            "lost_sale_penalty_fraction_range": [0.2, 0.2],
            "mean_demand_range": [8.0, 8.0],
            "coefficient_of_variation_range": [0.5, 0.5],
            "initial_inventory_multiplier_range": [1.0, 1.0],
        },
    }


def test_gamma_product_generation_and_history_length() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")[0]

    assert product.demand_distribution_type == "Gamma"
    assert product.gamma_shape > 0
    assert product.gamma_scale > 0
    assert len(product.historical_demand) == 32
    assert all(isinstance(value, int) for value in product.historical_demand)


def test_paper_product_observable_features_hide_true_parameters() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")[0]
    features = product.observable_features()

    assert "gamma_shape" not in features
    assert "gamma_scale" not in features
    assert "true_mean_demand" not in features
    assert "true_coefficient_of_variation" not in features
    assert "historical_demand" in features


def test_paper_style_reward_calculation() -> None:
    _, reward, info = paper_lost_sales_step(
        inventory=5,
        order_quantity=2,
        demand=8,
        price=10,
        purchase_cost=4,
        holding_cost=1,
        lost_sale_penalty=3,
    )

    assert info["sales"] == 7
    assert info["lost_sales"] == 1
    assert info["ending_inventory"] == 0
    assert reward == 70 - 8 - 3
    assert "cost" not in info


def test_oracle_baseline_uses_true_gamma_parameters() -> None:
    product_low = PaperLostSalesProduct(
        product_id="p",
        price=10,
        purchase_cost=4,
        holding_cost=1,
        lost_sale_penalty=5,
        true_mean_demand=10,
        true_coefficient_of_variation=0.5,
        gamma_shape=100,
        gamma_scale=0.1,
        demand_distribution_type="Gamma",
        historical_demand=[10] * 32,
        initial_inventory=0,
    )
    product_high = PaperLostSalesProduct(
        **{**product_low.__dict__, "gamma_shape": 100, "gamma_scale": 0.3}
    )

    low_policy = oracle_gamma_base_stock_policy(product_low, n_samples=5000, seed=0)
    high_policy = oracle_gamma_base_stock_policy(product_high, n_samples=5000, seed=0)

    assert high_policy.S > low_policy.S


def test_historical_baseline_does_not_use_true_gamma_parameters() -> None:
    product_a = PaperLostSalesProduct(
        product_id="p",
        price=10,
        purchase_cost=4,
        holding_cost=1,
        lost_sale_penalty=5,
        true_mean_demand=10,
        true_coefficient_of_variation=0.5,
        gamma_shape=10,
        gamma_scale=1,
        demand_distribution_type="Gamma",
        historical_demand=[8, 9, 10, 11] * 8,
        initial_inventory=0,
    )
    product_b = PaperLostSalesProduct(
        **{**product_a.__dict__, "gamma_shape": 100, "gamma_scale": 10}
    )

    assert historical_demand_base_stock_policy(product_a).S == historical_demand_base_stock_policy(product_b).S


def test_monotonicity_diagnostic_detects_non_monotone_policy() -> None:
    def non_monotone_policy(features):
        return features[0]

    def feature_builder(inventory):
        return [inventory]

    rate = monotonicity_violation_rate(
        non_monotone_policy,
        feature_builder,
        inventory_low=np.array([1, 2, 3]),
        inventory_high=np.array([2, 3, 4]),
    )

    assert rate == 1.0


def test_paper_evaluation_uses_reward_not_cost_names() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]
    metrics = evaluate_paper_lost_sales_policy(
        product,
        BaseStockOrderPolicy(S=8, name="test"),
        horizon=5,
        n_episodes=1,
        seed=0,
    )

    assert "mean_total_reward" in metrics
    assert "mean_reward_per_period" in metrics
    assert "mean_total_cost" not in metrics
    assert "mean_cost_per_period" not in metrics


def test_oracle_gamma_baseline_has_higher_reward_than_historical_on_fixed_products() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(5, "test")
    oracle_rewards = []
    historical_rewards = []
    for idx, product in enumerate(products):
        oracle = oracle_gamma_base_stock_policy(product, n_samples=10000, seed=100 + idx)
        historical = historical_demand_base_stock_policy(product)
        oracle_rewards.append(
            evaluate_paper_lost_sales_policy(product, oracle, horizon=50, n_episodes=1, seed=200 + idx)[
                "mean_total_reward"
            ]
        )
        historical_rewards.append(
            evaluate_paper_lost_sales_policy(product, historical, horizon=50, n_episodes=1, seed=200 + idx)[
                "mean_total_reward"
            ]
        )

    assert np.mean(oracle_rewards) >= np.mean(historical_rewards)


def test_common_paper_test_product_count_uses_strictest_runtime_cap() -> None:
    config = {
        "test_products": 100,
        "runtime": {
            "baseline_eval_products": 20,
            "fcnn_test_products": 30,
        },
    }

    assert paper_test_product_count(config) == 20


def test_same_paper_seed_gives_common_random_numbers_across_policies() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]

    first = evaluate_paper_lost_sales_policy(
        product,
        BaseStockOrderPolicy(S=8, name="first"),
        horizon=5,
        n_episodes=1,
        seed=PAPER_TEST_EVALUATION_SEED_START,
    )
    second = evaluate_paper_lost_sales_policy(
        product,
        BaseStockOrderPolicy(S=8, name="second"),
        horizon=5,
        n_episodes=1,
        seed=PAPER_TEST_EVALUATION_SEED_START,
    )

    assert first["mean_total_reward"] == second["mean_total_reward"]
    assert first["service_level"] == second["service_level"]
