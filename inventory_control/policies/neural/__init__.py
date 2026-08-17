"""Neural policy architectures for paper replication."""

from inventory_control.policies.neural.fcnn import (
    FCNNPolicy,
    PaperFCNNFeatureStats,
    PaperCausalCNNPolicy,
    PaperFCNNResidualOrderUpTo,
    PaperFCNNTargetStock,
    build_paper_lost_sales_features,
    compute_paper_fcnn_feature_stats,
    initialize_target_stock_fcnn,
    make_paper_fcnn_model,
    make_paper_residual_order_up_to_fcnn_model,
    make_paper_target_stock_fcnn_model,
    target_stock_informed_scalar,
)
from inventory_control.policies.neural.constrained_mlp import (
    ConstrainedMonotoneMLP,
    ConstrainedMLPResidualOrderUpTo,
    FirstMonotoneLinear,
    NonnegativeLinear,
    make_constrained_monotone_mlp_model,
    make_paper_residual_order_up_to_constrained_mlp_model,
    make_positive_weight_mlp_model,
)
from inventory_control.policies.neural.monokan import (
    MonoKANPolicy,
    PaperMonoKANPolicyAdapter,
    make_paper_monokan_model,
    monokan_inventory_violation_rate,
    sample_monokan_first_layer_splines,
)

__all__ = [
    "FCNNPolicy",
    "ConstrainedMonotoneMLP",
    "FirstMonotoneLinear",
    "NonnegativeLinear",
    "PaperFCNNFeatureStats",
    "PaperCausalCNNPolicy",
    "PaperFCNNResidualOrderUpTo",
    "PaperFCNNTargetStock",
    "ConstrainedMLPResidualOrderUpTo",
    "MonoKANPolicy",
    "PaperMonoKANPolicyAdapter",
    "build_paper_lost_sales_features",
    "compute_paper_fcnn_feature_stats",
    "initialize_target_stock_fcnn",
    "make_constrained_monotone_mlp_model",
    "make_paper_fcnn_model",
    "make_paper_residual_order_up_to_fcnn_model",
    "make_paper_target_stock_fcnn_model",
    "make_paper_residual_order_up_to_constrained_mlp_model",
    "make_paper_monokan_model",
    "make_positive_weight_mlp_model",
    "monokan_inventory_violation_rate",
    "sample_monokan_first_layer_splines",
    "target_stock_informed_scalar",
]
