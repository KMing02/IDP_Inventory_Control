"""Fully connected neural policy for Maggiar-style lost-sales replication."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -r requirements-neural.txt`.") from exc

from inventory_control.products import PaperLostSalesProduct


@dataclass(frozen=True)
class PaperFCNNFeatureStats:
    """Feature scales computed from training products only."""

    demand_scale: float
    price_scale: float


def compute_paper_fcnn_feature_stats(products: Sequence[PaperLostSalesProduct]) -> PaperFCNNFeatureStats:
    """Compute FCNN normalization scales from observable training data."""
    if not products:
        return PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0)
    historical_values = [float(value) for product in products for value in product.historical_demand]
    prices = [float(product.price) for product in products]
    demand_scale = max(sum(historical_values) / len(historical_values), 1.0)
    price_scale = max(sum(prices) / len(prices), 1.0)
    return PaperFCNNFeatureStats(demand_scale=demand_scale, price_scale=price_scale)


def infer_model_device(model: nn.Module) -> torch.device:
    """Return the device that should be used for model inputs."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


class FCNNPolicy(nn.Module):
    """FCNN mapping observable features to non-negative order quantity."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        activation: str = "ELU",
    ) -> None:
        super().__init__()
        activation_cls = getattr(nn, activation)
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(previous_dim, hidden_units))
            layers.append(activation_cls())
            previous_dim = hidden_units
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return non-negative continuous order quantity."""
        return torch.nn.functional.softplus(self.network(features)).squeeze(-1)


class CausalConvBlock(nn.Module):
    """One-dimensional causal convolution block for demand history."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, activation: str = "ELU") -> None:
        super().__init__()
        self.left_padding = int(dilation)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            dilation=dilation,
        )
        self.activation = getattr(nn, activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return causally filtered sequence features."""
        x = torch.nn.functional.pad(x, (self.left_padding, 0))
        return self.activation(self.conv(x))


class PaperCausalCNNPolicy(nn.Module):
    """Paper-style policy with causal CNN history encoder plus MLP head."""

    def __init__(
        self,
        history_length: int,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        cnn_channels: int = 8,
        cnn_dilations: Sequence[int] = (1, 2, 4, 8, 16),
        activation: str = "ELU",
    ) -> None:
        super().__init__()
        self.history_length = history_length
        blocks: list[nn.Module] = []
        in_channels = 1
        for dilation in cnn_dilations:
            blocks.append(CausalConvBlock(in_channels, cnn_channels, int(dilation), activation=activation))
            in_channels = cnn_channels
        self.history_encoder = nn.Sequential(*blocks)

        activation_cls = getattr(nn, activation)
        mlp_layers: list[nn.Module] = []
        previous_dim = 1 + cnn_channels + 4
        for _ in range(hidden_layers):
            mlp_layers.append(nn.Linear(previous_dim, hidden_units))
            mlp_layers.append(activation_cls())
            previous_dim = hidden_units
        mlp_layers.append(nn.Linear(previous_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return non-negative normalized order quantity."""
        inventory_feature = features[:, :1]
        history = features[:, 1 : 1 + self.history_length].unsqueeze(1)
        economics = features[:, 1 + self.history_length :]
        encoded_history = self.history_encoder(history)[:, :, -1]
        mlp_input = torch.cat([inventory_feature, encoded_history, economics], dim=1)
        return torch.nn.functional.softplus(self.mlp(mlp_input)).squeeze(-1)


class PaperContextMLPPolicy(nn.Module):
    """MLP over demand history and economics, excluding current inventory."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        activation: str = "ELU",
        output: str = "positive",
        output_upper_bound: float | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output = output
        self.output_upper_bound = float(output_upper_bound) if output_upper_bound is not None else None
        activation_cls = getattr(nn, activation)
        layers: list[nn.Module] = []
        previous_dim = self.input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(previous_dim, hidden_units))
            layers.append(activation_cls())
            previous_dim = hidden_units
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a context-dependent scalar, ignoring inventory when present."""
        context = features[:, -self.input_dim :]
        raw = self.network(context).squeeze(-1)
        if self.output == "signed":
            return torch.tanh(raw)
        if self.output_upper_bound is not None:
            return self.output_upper_bound * torch.sigmoid(raw)
        return torch.nn.functional.softplus(raw)


def make_paper_fcnn_model(config: dict, metadata: dict | None = None) -> nn.Module:
    """Build the configured paper policy model."""
    source = metadata or config["fcnn"]
    history_length = int(metadata.get("history_length_H", config["history_length_H"])) if metadata else int(config["history_length_H"])
    hidden_layers = int(source.get("hidden_layers", config["fcnn"]["hidden_layers"]))
    hidden_units = int(source.get("hidden_units", config["fcnn"]["hidden_units"]))
    activation = str(source.get("activation", config["fcnn"]["activation"]))
    history_encoder = str(source.get("history_encoder", config["fcnn"].get("history_encoder", "flat")))
    if history_encoder == "causal_cnn":
        return PaperCausalCNNPolicy(
            history_length=history_length,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            cnn_channels=int(source.get("cnn_channels", config["fcnn"].get("cnn_channels", 8))),
            cnn_dilations=tuple(source.get("cnn_dilations", config["fcnn"].get("cnn_dilations", [1, 2, 4, 8, 16]))),
            activation=activation,
        )
    return FCNNPolicy(
        input_dim=1 + history_length + 4,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation=activation,
    )


def make_paper_order_up_to_fcnn_model(config: dict, metadata: dict | None = None) -> nn.Module:
    """Build an FCNN that predicts a normalized order-up-to target."""
    source = metadata or config["fcnn"]
    history_length = int(metadata.get("history_length_H", config["history_length_H"])) if metadata else int(config["history_length_H"])
    model = PaperContextMLPPolicy(
        input_dim=history_length + 4,
        hidden_layers=int(source.get("hidden_layers", config["fcnn"]["hidden_layers"])),
        hidden_units=int(source.get("hidden_units", config["fcnn"]["hidden_units"])),
        activation=str(source.get("activation", config["fcnn"]["activation"])),
        output="positive",
        output_upper_bound=float(source.get("max_order_factor", config["fcnn"].get("max_order_factor", 4.0))),
    )
    model.paper_action_parameterization = "order_up_to"
    model.paper_scale_target_by_history_mean = bool(source.get("scale_action_by_history_mean", True))
    return model


def make_paper_residual_order_up_to_fcnn_model(config: dict, metadata: dict | None = None) -> nn.Module:
    """Build an FCNN that predicts a residual correction to historical base stock."""
    source = metadata or config["fcnn"]
    history_length = int(metadata.get("history_length_H", config["history_length_H"])) if metadata else int(config["history_length_H"])
    model = PaperContextMLPPolicy(
        input_dim=history_length + 4,
        hidden_layers=int(source.get("hidden_layers", config["fcnn"]["hidden_layers"])),
        hidden_units=int(source.get("hidden_units", config["fcnn"]["hidden_units"])),
        activation=str(source.get("activation", config["fcnn"]["activation"])),
        output="signed",
    )
    model.paper_action_parameterization = "residual_order_up_to"
    model.paper_residual_scale = float(source.get("residual_scale", 0.5))
    model.paper_residual_anchor = str(source.get("residual_anchor", "gamma"))
    return model


def build_paper_lost_sales_features(
    inventory: float,
    historical_demand: Sequence[float],
    price: float,
    purchase_cost: float,
    holding_cost: float,
    lost_sale_penalty: float,
    feature_stats: PaperFCNNFeatureStats | None = None,
) -> list[float]:
    """Build observable FCNN features without hidden Gamma parameters."""
    history = [float(value) for value in historical_demand]
    if feature_stats is None:
        demand_scale = max(sum(history) / len(history), 1.0)
        price_scale = max(float(price), 1.0)
    else:
        demand_scale = feature_stats.demand_scale
        price_scale = feature_stats.price_scale
    return [
        float(inventory) / demand_scale,
        *[value / demand_scale for value in history],
        float(price) / price_scale,
        float(purchase_cost) / price_scale,
        float(holding_cost) / price_scale,
        float(lost_sale_penalty) / price_scale,
    ]


class PaperFCNNPolicyAdapter:
    """Adapter for evaluating FCNN policies in paper-style NumPy rollouts."""

    name = "FCNN"

    def __init__(
        self,
        model: nn.Module,
        product: PaperLostSalesProduct,
        feature_stats: PaperFCNNFeatureStats | None = None,
        max_order_factor: float = 2.0,
        scale_action_by_history_mean: bool = False,
        max_order_multiplier: float | None = None,
    ) -> None:
        self.model = model
        self.product = product
        self.feature_stats = feature_stats
        self.max_order_factor = max_order_factor if max_order_multiplier is None else max_order_multiplier
        self.scale_action_by_history_mean = scale_action_by_history_mean

    def act(self, inventory: float, t: int, rng=None) -> float:
        """Return non-negative continuous order quantity."""
        features = torch.tensor(
            [
                build_paper_lost_sales_features(
                    inventory=inventory,
                    historical_demand=self.product.historical_demand,
                    price=self.product.price,
                    purchase_cost=self.product.purchase_cost,
                    holding_cost=self.product.holding_cost,
                    lost_sale_penalty=self.product.lost_sale_penalty,
                    feature_stats=self.feature_stats,
                )
            ],
            dtype=torch.float32,
            device=infer_model_device(self.model),
        )
        self.model.eval()
        with torch.no_grad():
            order = float(self.model(features).item())
        historical_mean = max(sum(self.product.historical_demand) / len(self.product.historical_demand), 1.0)
        if self.scale_action_by_history_mean:
            order *= historical_mean
        max_inventory_level = self.max_order_factor * historical_mean
        order_cap = max(max_inventory_level - float(inventory), 0.0)
        return min(order, order_cap)
