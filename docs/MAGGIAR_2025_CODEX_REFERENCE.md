# Maggiar Et Al. Reference Mapping

The project uses the synthetic lost-sales setup from Maggiar et al. (2025), "Structure-Informed Deep Reinforcement Learning for Inventory Management", as the reference environment for the final IDP report.

## Retained Reference Elements

- No-lead-time lost-sales inventory dynamics.
- Gamma demand with hidden product parameters.
- Observable historical demand and product economics.
- DirectBackprop-style differentiable training through the simulator.
- A paper-style rolling raw-action FCNN anchor.

## Product Distribution

The retained paper config samples products from the Table 1-style synthetic lost-sales distribution:

```text
selling price p         ~ Exponential(mean=100)
purchase cost c         = p * U, U ~ Uniform(0, 1)
holding cost h          ~ Exponential(mean=5)
lost-sale penalty k     = 10 * U, U ~ Uniform(0, 1)
mean demand mu          ~ Exponential(mean=100)
demand CV nu            ~ Uniform(0, 1)
```

The hidden `mu` and `nu` define Gamma demand parameters. Policies receive historical demand and economics, not hidden demand parameters or future demand.

## Repository Entry Points

- `configs/paper_lost_sales.yaml`: base paper-style configuration.
- `configs/paper_lost_sales_raw_fcnn_replication_5k.yaml`: retained raw-action FCNN anchor.
- `configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml`: final matched target-stock development comparison.
- `scripts/train_paper_lost_sales_fcnn.py`: standalone paper-style raw-action FCNN training.
- `scripts/evaluate_paper_lost_sales_fcnn.py`: standalone paper-style raw-action FCNN evaluation.
- `scripts/run_paper_lost_sales_baselines.py`: paper-style base-stock baselines.
- `scripts/run_rolling_direct_target_matched.py`: final matched direct-target protocol.
