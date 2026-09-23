"""Direct-quantity comparison: where does the monotone constraint actually bind?

The target-stock decoder gets "order less when stock is higher" from
q = max(S - I, 0), so it holds for every model and the certified constraint only
covers the economics. Here the network emits the order itself with inventory as
an input, so the property has to come from the model. This trains three
direct-quantity policies and reports both profit and how often each one breaks
the property.

Usage:
    python scripts/run_direct_quantity_comparison.py monokan,kan,fcnn 2025,2026,2027 400 results.json
"""
import json, sys, time
import numpy as np, torch
torch.set_num_threads(6)
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.policies.neural.fcnn import (compute_paper_fcnn_feature_stats, make_paper_fcnn_model,
                                                    build_paper_lost_sales_features)
from inventory_control.policies.neural.monokan import make_paper_raw_action_monokan_model_for_products
from inventory_control.base_stock_output_experiment import _compact_settings
from inventory_control.evaluation.torch_policy import batched_torch_policy_metrics
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn

CONFIG = "configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml"
KAN_METADATA = {"hidden_layers": 1, "hidden_units": 8, "grid": 5,
                "adaptive_grid": True, "adaptive_grid_eps": 0.1, "sort_history_lags": True}


def build(name, config, products, feature_stats):
    if name == "fcnn":
        return make_paper_fcnn_model(config)
    metadata = dict(KAN_METADATA, monotone=(name == "monokan"))
    return make_paper_raw_action_monokan_model_for_products(config, products, feature_stats, metadata=metadata)


def inventory_response(model, products, feature_stats):
    """Sweep on-hand stock with everything else fixed.

    Two things matter. Sign: does the order ever rise when stock rises? Strength:
    an order-up-to rule lowers the order one-for-one with stock, so the ideal
    slope is -1 and a slope near 0 means the policy is ignoring its inventory.
    The slope is read over the range a sensible policy actually visits.
    """
    products_hit, bad_steps, steps, worst = 0, 0, 0, 0.0
    slopes = []
    with torch.no_grad():
        for product in products:
            mean_demand = max(float(np.mean(product.historical_demand)), 1.0)
            grid = np.linspace(0.0, 4.0 * mean_demand, 25)
            feats = torch.tensor([build_paper_lost_sales_features(
                inventory=float(inventory), historical_demand=product.historical_demand, price=product.price,
                purchase_cost=product.purchase_cost, holding_cost=product.holding_cost,
                lost_sale_penalty=product.lost_sale_penalty, feature_stats=feature_stats) for inventory in grid],
                dtype=torch.float32)
            orders = model(feats).numpy() * mean_demand
            rises = np.diff(orders)
            flagged = rises > 1e-4 * mean_demand
            bad_steps += int(flagged.sum()); steps += len(rises)
            if flagged.any():
                products_hit += 1
                worst = max(worst, float(rises.max() / mean_demand))
            near = grid <= 1.5 * mean_demand
            slopes.append(float(np.polyfit(grid[near], orders[near], 1)[0]))
    return {"products_with_any_violation_pct": 100 * products_hit / len(products),
            "violating_steps_pct": 100 * bad_steps / steps,
            "worst_rise_in_mean_demand_units": worst,
            "mean_inventory_slope": float(np.mean(slopes)),
            "inventory_slope_p10": float(np.percentile(slopes, 10)),
            "inventory_slope_p90": float(np.percentile(slopes, 90))}


def run(name, seed, epochs, config, train_products, val_products, feature_stats, settings):
    torch.manual_seed(seed)
    model = build(name, config, train_products, feature_stats)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert getattr(model, "paper_action_parameterization", None) == "raw", name
    start = time.time()
    result = train_paper_lost_sales_fcnn(
        train_products, model, horizon=settings["training_horizon"],
        batch_size=min(settings["batch_size"], len(train_products)), epochs=epochs,
        learning_rate=settings["learning_rate"], seed=seed, imitation_epochs=0,
        validation_products=val_products, validation_interval=max(epochs // 8, 1), feature_stats=feature_stats,
        max_order_factor=settings["max_order_factor"], normalize_reward_loss=False,
        training_demand_mode=settings["training_demand_mode"], history_feature_mode="raw_lags",
        include_terminal_salvage=settings["include_terminal_salvage"],
        initial_inventory_mode=settings["initial_inventory_mode"], enforce_action_cap=False,
        scale_action_by_history_mean=True, device="cpu")
    if result.best_state_dict:
        model.load_state_dict(result.best_state_dict)
    model.eval()
    metrics = batched_torch_policy_metrics(val_products, model, feature_stats,
                                           settings={**settings, "scale_action_by_history_mean": True},
                                           split="validation")
    row = {"model": name, "seed": seed, "params": params, "train_seconds": round(time.time() - start),
           "reward_per_period": float(metrics["mean_reward_per_period"].mean())}
    row.update(inventory_response(model, val_products, feature_stats))
    print(json.dumps(row), flush=True)
    return row


def main():
    names = sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")]
    epochs = int(sys.argv[3])
    out_path = sys.argv[4]
    config = load_paper_lost_sales_config(CONFIG)
    settings = _compact_settings(config)
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(settings["training_products"], split="train")
    val_products = generator.generate(settings["validation_products"], split="validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    rows = [run(name, seed, epochs, config, train_products, val_products, feature_stats, settings)
            for name in names for seed in seeds]
    json.dump(rows, open(out_path, "w"), indent=1)
    print("\n== mean over seeds ==", flush=True)
    for name in names:
        subset = [r for r in rows if r["model"] == name]
        print(f"{name:8s} params={subset[0]['params']:6d} "
              f"reward={np.mean([r['reward_per_period'] for r in subset]):9.2f} "
              f"violating_products={np.mean([r['products_with_any_violation_pct'] for r in subset]):6.1f}% "
              f"slope={np.mean([r['mean_inventory_slope'] for r in subset]):7.3f} (ideal -1.0)", flush=True)


if __name__ == "__main__":
    main()
