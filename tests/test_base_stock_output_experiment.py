import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from inventory_control.base_stock_output_experiment import (
    _TorchBaseStockPolicyAdapter,
    _compact_settings,
    _restore_best_validation_state,
    run_base_stock_output_experiment,
)
from inventory_control.policies.neural.fcnn import (
    build_paper_lost_sales_features,
    compute_paper_fcnn_feature_stats,
    empirical_newsvendor_target_ratio,
    make_paper_direct_target_stock_fcnn_model_for_products,
    paper_context_feature_count,
    paper_history_feature_count,
)
from inventory_control.policies.neural.monokan import (
    make_paper_deep_grid7_direct_target_stock_kan_model,
    make_paper_deep_grid7_direct_target_stock_monokan_model,
    make_paper_direct_target_stock_kan_model,
    make_paper_direct_target_stock_monokan_model,
    make_paper_direct_target_stock_monokan_model_for_products,
)
from inventory_control.policies.neural.registry import (
    CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES,
    DIRECT_TARGET_STOCK_POLICY_NAMES,
    checkpoint_filenames,
)
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import (
    DirectBackpropResult,
    make_paper_lost_sales_training_batch,
    paper_lost_sales_fcnn_rollout,
)
from tests.test_paper_lost_sales import _paper_config


def test_base_stock_output_experiment_propagates_terminal_salvage_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import inventory_control.base_stock_output_experiment as experiment_module

    config = yaml.safe_load(Path("configs/paper_lost_sales_debug.yaml").read_text(encoding="utf-8"))
    config["idp_benchmark"]["include_terminal_salvage"] = False
    config["idp_benchmark"]["compact"].update(
        {
            "training_products": 4,
            "validation_products": 2,
            "batch_size": 2,
            "training_horizon": 3,
            "reward_epochs": 1,
            "validation_interval": 1,
            "evaluation_horizon": 3,
        }
    )
    config_path = tmp_path / "terminal_salvage.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    observed: list[bool] = []
    original_train = experiment_module.train_paper_lost_sales_fcnn

    def train_spy(*args, **kwargs):
        observed.append(kwargs["include_terminal_salvage"])
        return original_train(*args, **kwargs)

    monkeypatch.setattr(experiment_module, "train_paper_lost_sales_fcnn", train_spy)

    artifacts = run_base_stock_output_experiment(
        config_path=config_path,
        output_dir=tmp_path / "run",
        models=("PaperFCNNDirectTargetStock",),
        device="cpu",
    )

    assert observed == [False]
    assert artifacts.metadata["include_terminal_salvage"] is False


def test_base_stock_output_experiment_propagates_activation_checkpointing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import inventory_control.base_stock_output_experiment as experiment_module

    observed: list[bool] = []
    original_train = experiment_module.train_paper_lost_sales_fcnn

    def train_spy(*args, **kwargs):
        observed.append(kwargs["checkpoint_rolling_target"])
        return original_train(*args, **kwargs)

    monkeypatch.setattr(experiment_module, "train_paper_lost_sales_fcnn", train_spy)
    run_base_stock_output_experiment(
        config_path="configs/paper_lost_sales_debug.yaml",
        output_dir=tmp_path / "run",
        models=("PaperFCNNDirectTargetStock",),
        device="cpu",
        activation_checkpointing=False,
    )

    assert observed == [False]


def test_target_stock_adapter_caches_context_target_across_periods() -> None:
    class CountingTarget(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, context: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.full((context.shape[0],), 2.0, device=context.device)

    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]
    model = CountingTarget()
    historical_mean = max(sum(product.historical_demand) / len(product.historical_demand), 1.0)
    adapter = _TorchBaseStockPolicyAdapter(
        model,
        product,
        compute_paper_fcnn_feature_stats([product]),
        max_order_factor=4.0,
        enforce_action_cap=False,
    )

    first_order = adapter.act(inventory=0.0, t=0)
    second_order = adapter.act(inventory=historical_mean, t=1)

    assert model.calls == 1
    assert first_order == pytest.approx(2.0 * historical_mean)
    assert second_order == pytest.approx(historical_mean)


def test_target_stock_adapter_rolls_history_and_recomputes_target() -> None:
    class RecordingTarget(torch.nn.Module):
        paper_roll_history = True

        def __init__(self) -> None:
            super().__init__()
            self.inputs: list[torch.Tensor] = []

        def forward(self, context: torch.Tensor) -> torch.Tensor:
            self.inputs.append(context.detach().clone())
            return context[:, 2]

    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]
    product = product.__class__(**{**product.__dict__, "historical_demand": [1, 2, 3], "future_demand": [4, 99]})
    model = RecordingTarget()
    adapter = _TorchBaseStockPolicyAdapter(
        model,
        product,
        compute_paper_fcnn_feature_stats([product]),
        max_order_factor=4.0,
        enforce_action_cap=False,
    )

    low_inventory_order = adapter.act(inventory=0.0, t=0)
    high_inventory_order = adapter.act(inventory=1.0, t=0)
    adapter.observe_demand(4.0)
    adapter.act(inventory=0.0, t=1)

    assert low_inventory_order > high_inventory_order
    assert model.inputs[-3].shape[1] == 7
    assert model.inputs[-3][0, :3].tolist() == pytest.approx(model.inputs[-2][0, :3].tolist())
    assert model.inputs[-1][0, :3].tolist() != pytest.approx(model.inputs[-2][0, :3].tolist())
    assert 99.0 not in model.inputs[-1][0].tolist()


def test_direct_target_stock_registry_has_final_names_and_checkpoints() -> None:
    filenames = checkpoint_filenames()

    assert DIRECT_TARGET_STOCK_POLICY_NAMES == (
        "PaperFCNNDirectTargetStock",
        "KANDirectTargetStock",
        "MonoKANDirectTargetStock",
    )
    assert CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES == (
        "KANDirectTargetStockDeepGrid7",
        "MonoKANDirectTargetStockDeepGrid7",
    )
    assert filenames["PaperFCNNDirectTargetStock"] == "paper_fcnn_direct_target_stock.pt"
    assert filenames["MonoKANDirectTargetStockDeepGrid7"] == "monokan_direct_target_stock_deep_grid7.pt"


def test_capacity_direct_target_registry_is_fixed_and_checkpointed() -> None:
    config = {**_paper_config(), "paper_history_feature_mode": "raw_lags", "monokan": {"hidden_layers": 1, "hidden_units": 4, "grid": 3}}

    kan = make_paper_deep_grid7_direct_target_stock_kan_model(config)
    monokan = make_paper_deep_grid7_direct_target_stock_monokan_model(config)

    assert [layer.out_dim for layer in kan.layers] == [16, 16, 1]
    assert all(layer.num_grid_intervals == 7 for layer in kan.layers)
    assert kan.monotonic_vars == {}
    assert [layer.out_dim for layer in monokan.layers] == [16, 16, 1]
    assert all(layer.num_grid_intervals == 7 for layer in monokan.layers)
    assert monokan.monotonic_vars == {**{index: 1 for index in range(32)}, 32: 1, 33: -1, 34: -1, 35: 1}


def test_temporal_deep_grid7_monokan_constrains_only_lags_means_and_economics() -> None:
    config = {**_paper_config(), "paper_history_feature_mode": "raw_lags_temporal", "monokan": {"hidden_layers": 2, "hidden_units": 16, "grid": 7}}

    model = make_paper_deep_grid7_direct_target_stock_monokan_model(config)

    assert model.layers[0].in_dim == 41
    assert all(model.monotonic_vars[index] == 1 for index in range(35))
    assert 35 not in model.monotonic_vars
    assert 36 not in model.monotonic_vars
    assert model.monotonic_vars[37] == 1
    assert model.monotonic_vars[38] == -1
    assert model.monotonic_vars[39] == -1
    assert model.monotonic_vars[40] == 1


def test_capacity_medium_profile_preserves_report_ablation_protocol() -> None:
    capacity = yaml.safe_load(Path("configs/paper_lost_sales_direct_target_capacity_medium.yaml").read_text(encoding="utf-8"))

    assert _compact_settings(capacity)["training_products"] == 1_000
    assert capacity["paper_history_feature_mode"] == "raw_lags"
    assert capacity["paper_include_newsvendor_feature"] is False
    assert capacity["idp_benchmark"]["compact"]["imitation_epochs"] == 0


def test_summary_history_features_have_stable_dimension() -> None:
    features = build_paper_lost_sales_features(
        inventory=5.0,
        historical_demand=list(range(1, 33)),
        price=10.0,
        purchase_cost=4.0,
        holding_cost=1.0,
        lost_sale_penalty=2.0,
        history_feature_mode="summary",
    )

    assert paper_history_feature_count(32, "summary") == 9
    assert paper_context_feature_count(32, "summary") == 13
    assert len(features) == 1 + 9 + 4
    assert features[1] == pytest.approx(16.5 / 16.5)


def test_pure_direct_target_stock_models_use_observable_empirical_initialization() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(8, "train")
    expected = empirical_newsvendor_target_ratio(products)
    config = {
        **_paper_config(),
        "paper_history_feature_mode": "summary",
        "fcnn": {"hidden_layers": 1, "hidden_units": 4, "activation": "ELU", "max_order_factor": 4.0},
        "monokan": {"hidden_layers": 1, "hidden_units": 4, "grid": 3, "max_order_factor": 4.0, "initial_normalized_target": expected},
    }
    context = torch.randn(5, 13)
    constant_models = (
        make_paper_direct_target_stock_fcnn_model_for_products(config, products),
        make_paper_direct_target_stock_kan_model(config),
    )

    for model in constant_models:
        assert model.paper_action_parameterization == "order_up_to"
        assert model.paper_include_newsvendor_feature is False
        assert model.paper_enforce_action_cap is False
        assert model.paper_initialization_mode == "global_empirical_newsvendor_median"
        assert torch.allclose(model(context), torch.full((5,), expected), atol=1e-5)

    feature_stats = compute_paper_fcnn_feature_stats(products)
    monokan = make_paper_direct_target_stock_monokan_model_for_products(config, products, feature_stats)
    product_contexts = torch.tensor(
        [
            build_paper_lost_sales_features(
                inventory=0.0,
                historical_demand=product.historical_demand,
                price=product.price,
                purchase_cost=product.purchase_cost,
                holding_cost=product.holding_cost,
                lost_sale_penalty=product.lost_sale_penalty,
                feature_stats=feature_stats,
                history_feature_mode="summary",
            )[1:]
            for product in products
        ],
        dtype=torch.float32,
    )
    assert monokan.paper_initialization_mode == "empirical_newsvendor_open_readout"
    assert torch.median(monokan(product_contexts)).item() == pytest.approx(expected, abs=1e-5)
    assert monokan.monotonic_vars


def test_direct_target_stock_kan_constant_initialization_keeps_spline_path_trainable() -> None:
    config = {
        **_paper_config(),
        "paper_history_feature_mode": "summary",
        "monokan": {"hidden_layers": 1, "hidden_units": 4, "grid": 3, "noise_scale": 0.01, "initial_normalized_target": 1.5},
    }
    contexts = torch.randn(16, 13)
    weights = torch.linspace(-1.0, 1.0, len(contexts))

    torch.manual_seed(7)
    model = make_paper_direct_target_stock_kan_model(config)
    final_layer = model.layers[-1]

    assert torch.allclose(model(contexts), torch.full((len(contexts),), 1.5), atol=1e-5)
    assert torch.count_nonzero(final_layer.coef).item() > 0
    assert torch.count_nonzero(final_layer.scale_spline).item() == 0

    loss = torch.sum(weights * model(contexts))
    loss.backward()

    assert final_layer.scale_spline.grad is not None
    assert torch.linalg.vector_norm(final_layer.scale_spline.grad).item() > 0


def test_direct_target_stock_monokan_open_readout_preserves_backbone_gradient() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(8, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    config = {**_paper_config(), "paper_history_feature_mode": "summary", "monokan": {"hidden_layers": 1, "hidden_units": 4, "grid": 3, "noise_scale": 0.01}}
    model = make_paper_direct_target_stock_monokan_model_for_products(config, products, feature_stats)
    batch = make_paper_lost_sales_training_batch(products, horizon=4, generator=torch.Generator().manual_seed(19), feature_stats=feature_stats, history_feature_mode="summary")

    loss = -paper_lost_sales_fcnn_rollout(model, batch, include_terminal_salvage=False, enforce_action_cap=False).mean()
    loss.backward()

    first_layer_gradient = sum(float(parameter.grad.norm().item()) for parameter in model.layers[0].parameters() if parameter.grad is not None)
    assert first_layer_gradient > 0.0


def test_unconstrained_kan_does_not_apply_monokan_downstream_projection() -> None:
    config = {**_paper_config(), "paper_history_feature_mode": "summary", "monokan": {"hidden_layers": 1, "hidden_units": 4, "grid": 3, "initial_normalized_target": 1.5}}
    kan = make_paper_direct_target_stock_kan_model(config)
    monokan = make_paper_direct_target_stock_monokan_model(config)
    with torch.no_grad():
        kan.layers[-1].scale_base.fill_(-1.0)
        monokan.layers[-1].scale_base.fill_(-1.0)

    kan.apply_monotonic_constraints()
    monokan.apply_monotonic_constraints()

    assert torch.all(kan.layers[-1].scale_base == -1.0)
    assert torch.all(monokan.layers[-1].scale_base >= 0.0)


def test_base_stock_output_experiment_writes_expected_direct_target_artifacts(tmp_path: Path) -> None:
    artifacts = run_base_stock_output_experiment(
        config_path="configs/paper_lost_sales_debug.yaml",
        output_dir=tmp_path,
        models=("PaperFCNNDirectTargetStock", "MonoKANDirectTargetStockDeepGrid7"),
        device="cpu",
    )

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "training_log.csv").exists()
    assert (tmp_path / "target_stock_diagnostics.csv").exists()
    assert (tmp_path / "product_metrics.csv").exists()
    assert (tmp_path / "paper_fcnn_direct_target_stock.pt").exists()
    assert (tmp_path / "monokan_direct_target_stock_deep_grid7.pt").exists()
    assert set(artifacts.summary["policy_name"]) == {
        "PaperFCNNDirectTargetStock",
        "MonoKANDirectTargetStockDeepGrid7",
        "HistoricalDemandBaseStock",
        "NonOmniscientOptimal",
        "OmniscientOptimal",
    }
    assert artifacts.metadata["imitation_pretraining"] is False
    assert "S_t=m_0*r_theta" in artifacts.metadata["action_parameterization"]


def test_base_stock_output_experiment_restores_validation_selected_state() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(9.0)
    result = DirectBackpropResult(
        model=model,
        training_losses=[0.0, 0.0],
        mean_total_rewards=[0.0, 0.0],
        validation_rewards=[3.0, 2.0],
        validation_epochs=[20, 40],
        best_validation_reward=3.0,
        best_state_dict={"weight": torch.tensor([[4.0]])},
    )

    selected_epoch = _restore_best_validation_state(model, result)

    assert selected_epoch == 20
    assert model.weight.item() == pytest.approx(4.0)


def test_base_stock_output_experiment_records_schema(tmp_path: Path) -> None:
    run_base_stock_output_experiment(
        config_path="configs/paper_lost_sales_debug.yaml",
        output_dir=tmp_path,
        models=("KANDirectTargetStock",),
        device="cpu",
    )

    summary = pd.read_csv(tmp_path / "summary.csv")
    target_diagnostics = pd.read_csv(tmp_path / "target_stock_diagnostics.csv")
    product_metrics = pd.read_csv(tmp_path / "product_metrics.csv")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert {"policy_name", "mean_reward_per_period", "checkpoint_filename", "target_factor_mean"}.issubset(summary.columns)
    assert {"policy_name", "product_id", "target_factor", "target_stock", "historical_base_stock"}.issubset(target_diagnostics.columns)
    assert metadata["models"] == ["KANDirectTargetStock"]
    assert summary.loc[summary["policy_name"] == "KANDirectTargetStock", "imitation_epochs"].iloc[0] == 0
    assert {"run_seed", "model_seed", "product_generation_seed", "evaluation_seed", "demand_path_id", "policy_name", "product_id"}.issubset(product_metrics.columns)
    baseline = product_metrics[product_metrics["policy_name"] == "NonOmniscientOptimal"]
    learned = product_metrics[product_metrics["policy_name"] == "KANDirectTargetStock"]
    assert learned[["product_id", "evaluation_seed", "demand_path_id"]].reset_index(drop=True).equals(
        baseline[["product_id", "evaluation_seed", "demand_path_id"]].reset_index(drop=True)
    )
    checkpoint = torch.load(tmp_path / "kan_direct_target_stock.pt", map_location="cpu", weights_only=True)
    assert isinstance(checkpoint, dict)
    assert checkpoint


def test_matched_direct_target_run_shares_training_draws_and_saves_final_checkpoints(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/paper_lost_sales_debug.yaml").read_text(encoding="utf-8"))
    config["idp_benchmark"]["compact"].update(
        {
            "training_products": 4,
            "validation_products": 2,
            "batch_size": 2,
            "training_horizon": 2,
            "reward_epochs": 1,
            "validation_interval": 1,
            "evaluation_horizon": 2,
        }
    )
    config["monokan"] = {"hidden_layers": 1, "hidden_units": 2, "grid": 2, "roll_history": False}
    config_path = tmp_path / "matched.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    artifacts = run_base_stock_output_experiment(
        config_path=config_path,
        output_dir=tmp_path / "run",
        models=("PaperFCNNDirectTargetStock", "MonoKANDirectTargetStockDeepGrid7"),
        device="cpu",
        seed=2025,
        shared_training_seed=True,
    )
    learned = artifacts.summary.loc[artifacts.summary["model_seed"].notna()]

    assert learned["training_seed"].tolist() == [2025, 2025]
    assert learned["model_seed"].nunique() == 2
    assert artifacts.metadata["shared_training_seed"] is True
    for row in learned.itertuples(index=False):
        assert (tmp_path / "run" / row.checkpoint_filename).exists()
        assert (tmp_path / "run" / row.final_checkpoint_filename).exists()
        assert row.selected_checkpoint_epoch == 1
        assert row.final_checkpoint_epoch == 1


def test_final_rolling_config_resolves_without_protocol_overrides() -> None:
    config = yaml.safe_load(Path("configs/paper_lost_sales_rolling_direct_target_5k.yaml").read_text(encoding="utf-8"))

    settings = _compact_settings(config)

    assert settings["training_products"] == 5_000
    assert settings["validation_products"] == 500
    assert settings["batch_size"] == 2_500
    assert settings["reward_epochs"] == 1_000
    assert settings["training_horizon"] == 100
    assert settings["evaluation_horizon"] == 520
    assert settings["burn_in"] == 20
    assert settings["include_terminal_salvage"] is True
    assert settings["initial_inventory_mode"] == "last_historical_demand"
    assert settings["normalize_reward_loss"] is False
    assert config["fcnn"]["history_encoder"] == "causal_cnn"


def test_base_stock_output_cli_smoke(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_base_stock_output_experiment.py",
            "--config",
            "configs/paper_lost_sales_debug.yaml",
            "--output-dir",
            str(tmp_path),
            "--models",
            "PaperFCNNDirectTargetStock",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PaperFCNNDirectTargetStock" in result.stdout
    assert "saved_artifacts=" in result.stdout
    assert (tmp_path / "summary.csv").exists()
