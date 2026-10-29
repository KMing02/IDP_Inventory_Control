# Observed validation environment

Recorded on 21 October 2026. These versions were read from the existing local environment used to validate the submitted code. They are not guessed requirements and are not a claim about the environment used for every historical experiment.

| Component | Observed version |
| --- | --- |
| Python | 3.12.13 |
| OS / architecture | macOS 26.2 / Apple arm64 |
| NumPy | 2.5.1 |
| pandas | 3.0.5 |
| Matplotlib | 3.11.1 |
| PyYAML | 6.0.3 |
| SciPy | 1.18.0 |
| PyTorch | 2.13.0 |
| pytest | 9.1.1 |
| setuptools | 83.0.0 |

The existing environment does not have pip installed; it is a pre-existing managed virtual environment. A new environment can use pip, as described in README.txt.

This is an environment record, not a complete dependency lockfile. No clean installation with these exact versions has been validated on another operating system or machine. Historical training environments were not reconstructed. Retain the original requirements.txt and pyproject.toml; this record does not change dependency resolution or experiment behavior.

Validation command: `python -m pytest -q`, run from the extracted restored submission root using the observed environment on 21 October 2026. Result: 151 passed in 28.76 seconds. This was not a clean dependency installation.

Final submission cleanup validation, 22 October 2026: the new ZIP was extracted into another fresh directory and the same existing environment ran `python -m pytest -q`: 151 passed in 14.58 seconds. The saved final-policy summary reproduced regret 0.80% and reward 4285.53. The packaged report source compiled successfully with the bundled Tectonic engine and cached packages. This is not a new training run or a clean Python-environment installation.

Repository report sync validation, 29 October 2026: an isolated checkout based on `soft-quantile-monokan` commit
`d5677f7222f9683adef3e32ca9964b52d5f117a8` ran the complete suite: **151 passed in 17.65 seconds**.
The existing Python environment was reused; the checkout was explicitly placed on `PYTHONPATH` for the saved-result
CLI because that environment's editable installation points at another checkout. The archived three-seed CSVs
reproduced regret **0.80%** and reward **4285.53**. The matching standalone source compiled successfully with the
desktop LaTeX compiler. No training, fresh dependency installation or historical-environment reconstruction was run.
