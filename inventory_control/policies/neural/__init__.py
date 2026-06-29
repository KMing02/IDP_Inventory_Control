"""Neural policy architectures for paper replication."""

from inventory_control.policies.neural.fcnn import (
    FCNNPolicy,
    PaperFCNNFeatureStats,
    PaperCausalCNNPolicy,
    build_paper_lost_sales_features,
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.policies.neural.constrained_mlp import (
    ConstrainedMonotoneMLP,
    FirstMonotoneLinear,
    NonnegativeLinear,
    make_constrained_monotone_mlp_model,
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
    "MonoKANPolicy",
    "PaperMonoKANPolicyAdapter",
    "build_paper_lost_sales_features",
    "compute_paper_fcnn_feature_stats",
    "make_constrained_monotone_mlp_model",
    "make_paper_fcnn_model",
    "make_paper_monokan_model",
    "make_positive_weight_mlp_model",
    "monokan_inventory_violation_rate",
    "sample_monokan_first_layer_splines",
]
