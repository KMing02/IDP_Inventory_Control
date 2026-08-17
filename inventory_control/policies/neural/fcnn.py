"""Fully connected neural policy for Maggiar-style lost-sales replication."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -r requirements-neural.txt`.") from exc

from inventory_control.paper_lost_sales import non_omniscient_gamma_base_stock_policy
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


class PaperFCNNTargetStock(nn.Module):
    """Context-only causal-CNN policy that predicts a target-stock multiple.

    The network returns the positive dimensionless quantity
    ``s_theta = softplus(z_theta)``. Current inventory is intentionally absent
    from the network input. Physical target-stock scaling and the exact
    ``q = relu(S_theta - inventory)`` decoder are shared outside the model.
    """

    paper_action_parameterization = "target_stock"
    paper_target_stock_scale = "historical_mean_demand"

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
        self.history_length = int(history_length)
        blocks: list[nn.Module] = []
        in_channels = 1
        for dilation in cnn_dilations:
            blocks.append(CausalConvBlock(in_channels, cnn_channels, int(dilation), activation=activation))
            in_channels = cnn_channels
        self.history_encoder = nn.Sequential(*blocks)

        activation_cls = getattr(nn, activation)
        layers: list[nn.Module] = []
        previous_dim = cnn_channels + 4
        for _ in range(hidden_layers):
            layers.append(nn.Linear(previous_dim, hidden_units))
            layers.append(activation_cls())
            previous_dim = hidden_units
        layers.append(nn.Linear(previous_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.target_stock_initialization_mode = "standard"
        self.target_stock_initialization_scalar: float | None = None

    def preactivation(self, context: torch.Tensor) -> torch.Tensor:
        """Return the unrestricted target-stock preactivation ``z_theta``."""
        history = context[:, : self.history_length].unsqueeze(1)
        economics = context[:, self.history_length :]
        encoded_history = self.history_encoder(history)[:, :, -1]
        return self.mlp(torch.cat([encoded_history, economics], dim=1)).squeeze(-1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return the positive normalized target-stock level ``s_theta``."""
        return torch.nn.functional.softplus(self.preactivation(context))


def target_stock_informed_scalar(products: Sequence[PaperLostSalesProduct]) -> float:
    """Return the global normalized target level minimizing training MSE.

    The scalar solves ``min_a sum_i (S_hist_i - a * mean_i)^2``, where
    ``S_hist_i`` is the historical Gamma-fit base-stock level. It is used only
    to initialize one global output value; it is not retained as a per-product
    anchor in the forward pass.
    """
    if not products:
        raise ValueError("target-stock-informed initialization requires training products")
    means = [max(float(sum(product.historical_demand) / len(product.historical_demand)), 1.0) for product in products]
    levels = [float(non_omniscient_gamma_base_stock_policy(product).S) for product in products]
    denominator = sum(mean * mean for mean in means)
    if denominator <= 0.0:  # pragma: no cover - means are clamped above
        raise ValueError("cannot initialize target stock from a zero demand scale")
    scalar = sum(mean * level for mean, level in zip(means, levels)) / denominator
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("target-stock-informed scalar must be finite and positive")
    return float(scalar)


def initialize_target_stock_fcnn(
    model: PaperFCNNTargetStock,
    products: Sequence[PaperLostSalesProduct],
) -> float:
    """Initialize the policy to ``S_i = a_star * historical_mean_i``.

    Zeroing the final readout makes ``s_theta`` identical for every context.
    The bias is mapped through softplus so the common value is exactly the
    training-only scalar returned by :func:`target_stock_informed_scalar`.
    """
    scalar = target_stock_informed_scalar(products)
    final = model.mlp[-1]
    if not isinstance(final, nn.Linear):  # pragma: no cover
        raise TypeError("PaperFCNNTargetStock final layer must be linear")
    bias = scalar + math.log(-math.expm1(-scalar))
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(bias)
    model.target_stock_initialization_mode = "target_stock_informed"
    model.target_stock_initialization_scalar = scalar
    return scalar


def make_paper_target_stock_fcnn_model(
    config: dict,
    metadata: dict | None = None,
    *,
    initialization_mode: str | None = None,
    training_products: Sequence[PaperLostSalesProduct] | None = None,
) -> PaperFCNNTargetStock:
    """Build a target-stock FCNN with standard or informed initialization."""
    if metadata is None:
        source = {**config["fcnn"], **config.get("target_stock_fcnn", {})}
    else:
        source = metadata
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    if str(source.get("history_encoder", config["fcnn"].get("history_encoder", "causal_cnn"))) != "causal_cnn":
        raise ValueError("PaperFCNNTargetStock requires the causal_cnn history encoder")
    model = PaperFCNNTargetStock(
        history_length=history_length,
        hidden_layers=int(source.get("hidden_layers", config["fcnn"]["hidden_layers"])),
        hidden_units=int(source.get("hidden_units", config["fcnn"]["hidden_units"])),
        cnn_channels=int(source.get("cnn_channels", config["fcnn"].get("cnn_channels", 8))),
        cnn_dilations=tuple(source.get("cnn_dilations", config["fcnn"].get("cnn_dilations", [1, 2, 4, 8, 16]))),
        activation=str(source.get("activation", config["fcnn"]["activation"])),
    )
    mode = str(initialization_mode or source.get("initialization_mode", "standard"))
    if mode == "standard":
        return model
    if mode == "target_stock_informed":
        if training_products is None:
            raise ValueError("target_stock_informed initialization requires training_products")
        initialize_target_stock_fcnn(model, training_products)
        return model
    raise ValueError("initialization_mode must be 'standard' or 'target_stock_informed'")


class PaperFCNNResidualOrderUpTo(nn.Module):
    """Context-only FCNN backbone for residual order-up-to control.

    The model returns only the bounded residual factor ``u_theta`` in ``(0, 2)``.
    Historical anchoring and action decoding are shared outside the architecture.
    Current inventory is intentionally absent from the network input.
    """

    paper_action_parameterization = "residual_order_up_to"
    paper_residual_center = 1.0
    paper_residual_scale = 0.5
    paper_residual_anchor = "gamma"

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
        self.history_length = int(history_length)
        blocks: list[nn.Module] = []
        in_channels = 1
        for dilation in cnn_dilations:
            blocks.append(CausalConvBlock(in_channels, cnn_channels, int(dilation), activation=activation))
            in_channels = cnn_channels
        self.history_encoder = nn.Sequential(*blocks)

        activation_cls = getattr(nn, activation)
        layers: list[nn.Module] = []
        previous_dim = cnn_channels + 4
        for _ in range(hidden_layers):
            layers.append(nn.Linear(previous_dim, hidden_units))
            layers.append(activation_cls())
            previous_dim = hidden_units
        layers.append(nn.Linear(previous_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self._calibrate_zero_context()

    def preactivation(self, context: torch.Tensor) -> torch.Tensor:
        history = context[:, : self.history_length].unsqueeze(1)
        economics = context[:, self.history_length :]
        encoded_history = self.history_encoder(history)[:, :, -1]
        return self.mlp(torch.cat([encoded_history, economics], dim=1)).squeeze(-1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return ``u_theta = 2 sigmoid(z_theta)``."""
        return 2.0 * torch.sigmoid(self.preactivation(context))

    @torch.no_grad()
    def _calibrate_zero_context(self) -> None:
        reference = torch.zeros(1, self.history_length + 4)
        offset = self.preactivation(reference).squeeze()
        final = self.mlp[-1]
        if not isinstance(final, nn.Linear):  # pragma: no cover
            raise TypeError("residual FCNN final layer must be linear")
        final.bias.sub_(offset)


def make_paper_residual_order_up_to_fcnn_model(
    config: dict,
    metadata: dict | None = None,
) -> PaperFCNNResidualOrderUpTo:
    """Build the context-only PaperFCNN residual order-up-to model."""
    source = metadata or config["fcnn"]
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    if str(source.get("history_encoder", config["fcnn"].get("history_encoder", "causal_cnn"))) != "causal_cnn":
        raise ValueError("PaperFCNNResidualOrderUpTo requires the causal_cnn history encoder")
    return PaperFCNNResidualOrderUpTo(
        history_length=history_length,
        hidden_layers=int(source.get("hidden_layers", config["fcnn"]["hidden_layers"])),
        hidden_units=int(source.get("hidden_units", config["fcnn"]["hidden_units"])),
        cnn_channels=int(source.get("cnn_channels", config["fcnn"].get("cnn_channels", 8))),
        cnn_dilations=tuple(source.get("cnn_dilations", config["fcnn"].get("cnn_dilations", [1, 2, 4, 8, 16]))),
        activation=str(source.get("activation", config["fcnn"]["activation"])),
    )


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
