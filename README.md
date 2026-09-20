# PRA-Net: Physics-Residual Attention Network

Source code for PRA-Net, a compact architecture for joint detection and
multi-label localization of adversarial false data injection attacks (FDIAs)
against power-system state estimation. PRA-Net fuses a classical
weighted-least-squares bad-data residual with residual-biased self-attention,
and jointly trains a Dirichlet evidential detection head with a directly
supervised per-bus localization head in a single forward pass.

This repository contains the full pipeline needed to regenerate the
simulated data, train and evaluate PRA-Net, and reproduce the benchmark,
ablation, robustness, and adversarial-diagnostic experiments from scratch.

## Environment

```
pip install -r requirements.txt
```

Python 3.13, PyTorch (CUDA optional), pandapower (standard IEEE case30 /
case118 topologies).

## 1. Data generation

Simulates AC power-flow measurements driven by real historical utility load
records, constructs bad-data-detection-passing FDIAs via the classical
DC-linearized unobservable-attack model, and builds a pool of adversarial
FDIAs via a Carlini & Wagner attack.

```
python src/sim/generate_dataset.py --system all --out data/processed --seed 0
```

## 2. Backbone reproduction (detection baselines + adversarial training)

```
python src/train/run_reproduction.py --system case30  --seed 0 --epochs 25 --cw_iter 150
python src/train/run_reproduction.py --system case118 --seed 0 --epochs 25 --cw_iter 150
```

## 3. Benchmark suite (classical / boosting / deep / graph baselines)

```
python src/train/run_benchmarks.py --system case30  --seed 0 --epochs 25
python src/train/run_benchmarks.py --system case118 --seed 0 --epochs 25
```

## 4. Candidate architecture screening and tuning

```
python src/train/run_candidates.py --stage smoke  --system case30
python src/train/run_candidates.py --stage screen --system case30 --epochs 20
python src/train/run_tune_final.py --candidate PRA-Net --system case30 --n_trials 15 --write_final
```

Candidate architectures live under `src/candidates/`; `pra_net.py` is the
proposed method. `FINAL_MODEL_CONFIG.yaml` holds the frozen hyperparameters
used for all reported results.

## 5. Final frozen-model evaluation (PRA-Net, multi-seed)

```
python src/train/run_final_test.py --system case30  --epochs 25 --seeds 0 1 2 3 4
python src/train/run_final_test.py --system case118 --epochs 25 --seeds 0 1 2 3 4
```

Reports clean detection, multi-label localization, adversarial-FDIA
detection/localization on canonical test cases, and efficiency (parameter
count, inference latency).

## 6. Ablation, robustness, and adversarial diagnostics

```
python src/train/run_ablation.py               --system case30 --epochs 25 --seeds 0 1 2 3 4 --split test
python src/train/run_robustness.py              --system case30 --epochs 25 --seed 0
python src/train/run_adversarial_diagnostics.py --system case30 --epochs 25 --seed 0
```

`run_adversarial_diagnostics.py` implements the gradient-norm trajectory
comparison, an adaptive attack targeting the model's epistemic-uncertainty
signal directly, and a black-box transfer attack from an independently
trained surrogate.

## 7. Statistics and evaluation utilities

```
python src/eval/build_wtl.py
python src/eval/aggregate_adversarial.py
```

`src/eval/statistics.py` provides the paired significance-testing utilities
(t-tests, effect sizes) used throughout the evaluation pipeline.

## Repository layout

```
src/sim/        power-system simulation, FDIA construction, dataset generation
src/attacks/    Carlini & Wagner adversarial-FDIA crafting
src/models/     backbones, multihead ensemble + last-layer Laplace UQ, evidential head,
                VAE-based localization, graph feature utilities
src/candidates/ candidate hybrid architectures, including PRA-Net (src/candidates/pra_net.py)
src/baselines/  classical / boosting / deep benchmark suite
src/train/      training and evaluation drivers for every stage above
src/eval/       statistics, win/tie/loss aggregation, evaluation utilities
```

## License

MIT License. See `LICENSE`.
