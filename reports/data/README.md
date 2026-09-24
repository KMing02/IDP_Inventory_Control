# Bundled validation-only plotting data

These two compact CSV files are copied without modification from the completed
validation-only experiment outputs:

- `direct_q_checkpoint_seed_metrics.csv` comes from
  `outputs/idp_followup/direct_q_120_update_diagnostic/`;
- `target_stock_extended_checkpoint_seed_metrics.csv` comes from
  `outputs/idp_followup/target_stock_training_dynamics/extended_checkpoint_seed_metrics.csv`.

They contain the seed-level fixed-checkpoint values used by
`reports/plot_target_stock_reward_dynamics.py`. They are included so that this
code-based visualization can be regenerated from a clean Git checkout without
distributing the full experiment output directories or any model checkpoints.

Both files contain validation-only results. They contain no protected-test
products, predictions, or metrics.
