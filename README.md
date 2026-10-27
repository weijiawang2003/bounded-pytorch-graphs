# Bounded PyTorch Feature-Token Graphs

A standalone component export from **FinTable**, a local research workstation. This repository demonstrates a constrained architecture format, explicit PyTorch feature attention, deterministic CPU execution, and tested checkpoint/inference behavior. It runs on synthetic arrays without accounts, market data, or external services.

The two runtime modules and their existing test file are byte-for-byte copies of the source project. Packaging and the synthetic example were added for this portfolio export. File hashes and original relative paths are recorded in [provenance.json](provenance.json). This documents source lineage; it is not an independent authorship or ownership verification.

## What it demonstrates

- **Closed JSON architecture parser:** attention, feed-forward, gated feed-forward, token mixer, and normalization blocks; residual connections; CLS or mean pooling. Unknown fields and unsupported combinations are rejected.
- **Pre-allocation resource checks:** exact trainable parameter counts plus conservative activation-memory and arithmetic-work estimates. Estimates are planning proxies, not measured peak memory or FLOPs.
- **Explicit feature-token attention:** input `[rows, ordered_features]` becomes `[rows, feature_tokens, width]`; each scalar feature has learned weights and a bias.
- **Deterministic CPU execution:** a shared reentrant lock protects native compute and process-wide thread-pool changes; seeded sections preserve the caller's CPU random-generator state.
- **Bounded inference and serialization:** shape and finite-value checks, chunked predictions, restoration of training mode, and architecture/state-dictionary reload checks.

## Execution boundaries

The parser rejects executable architecture fields, but it is not a process sandbox. The shared guard is cooperative and serializes participating native sections within one process; it is not a cross-process scheduler. Callers must hold `cpu_execution()` for the entire forward/backward/optimizer step. Memory and arithmetic calculations are estimates, not hard memory or wall-clock quotas; whole-training work is checked only when both `training_rows` and `epochs` are supplied to `architecture_complexity()`. Repeatability was verified on CPU in the recorded environment, not across devices or library versions.

The attention axis is **features within a row**. This component does not implement a temporal Transformer, a multimodal model, distributed training, the FinTable queue, or the full training/evaluation service.

## Run locally

Python 3.10 or newer is declared; the verified environment below uses Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/synthetic_regression.py
```

The example creates local Gaussian arrays, trains a small attention graph on CPU, writes a JSON report, and verifies predictions after reloading a locally created state dictionary. It needs no API key and performs no downloads after dependency installation. Generated `.pt` weights are excluded from Git.

## Verified example

Checked on 2026-10-26 with Python 3.12.7, NumPy 1.26.4, PyTorch 2.7.1, and threadpoolctl 3.5.0:

| Check | Observed result |
|---|---:|
| Existing focused runtime tests | 51 passed |
| Synthetic train/test rows | 512 / 128 |
| Input features / parameters | 5 / 4,945 |
| Adam steps | 100 |
| Initial test MSE | 0.83013 |
| Final test MSE | 0.02334 |
| Training-mean baseline test MSE | 0.84014 |
| Checkpoint reload maximum absolute error | 0.0 |

The task is `y = 0.75*x0 - 0.5*x1 + 0.25*sin(x2)`. These numbers demonstrate runtime plumbing on one fixed synthetic task. They are not financial results, a competitive model benchmark, or evidence of real-world predictive value. Different numerical libraries/platforms may produce small numerical differences. The machine-readable report is [artifacts/synthetic_demo.json](artifacts/synthetic_demo.json).

The tests exercise parser rejection and resource budgets, parameter accounting, real forward/backward gradients, Q/K/V attention, RNG and thread-pool restoration, finite-output handling, fresh-process serialization, and concurrent SciPy/scikit-learn/PyTorch work under the shared guard. They do not test FinTable's excluded services or real-data performance.

## Source layout

```text
fintable/neural_graph.py        Original constrained graph runtime
fintable/native_compute.py      Original native computation ownership guard
tests/test_neural_graph.py      Original focused runtime checks
examples/synthetic_regression.py Added standalone CPU demonstration
provenance.json                 Source paths and exact SHA-256 hashes
artifacts/synthetic_demo.json    Reproducible toy-run report
```

The returned `architecture_schema()` retains an original `training_owner` reference to FinTable's `model_lab`; that service is outside this export. Here the example owns training. Only load checkpoints that you created or trust; the demonstration uses PyTorch `weights_only=True` when loading its state dictionary. Existing joblib tests only reload artifacts created within the test itself.

## Scope and rights

FinTable's market datasets, research signals, agent outputs, local settings, credentials, process state, and private experiment history are excluded. No return, Sharpe ratio, alpha discovery, or complete-system benchmark is claimed here. The source did not contain an open-source license; this portfolio snapshot is provided for inspection with all rights reserved. Dependencies retain their respective licenses.
