"""CPU-only synthetic demo for the exported FinTable graph runtime.

The task is generated locally. Its metrics test runtime plumbing; they are not
financial results, competitive benchmarks, or evidence of real-world utility.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fintable import neural_graph as graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic_demo.json"))
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.steps <= 120:
        parser.error("--steps must be in [1, 120]")

    generator = np.random.default_rng(1030)
    values = generator.normal(size=(640, 5)).astype(np.float32)
    targets = (0.75 * values[:, 0] - 0.5 * values[:, 1] + 0.25 * np.sin(values[:, 2])).astype(np.float32)
    train_x, test_x = values[:512], values[512:]
    train_y, test_y = targets[:512], targets[512:]
    architecture = graph.default_architectures()[0]
    complexity = graph.architecture_complexity(architecture, 5, batch_size=512, training_rows=512, epochs=args.steps)
    model = graph.build_module(architecture, n_features=5, seed=1030)
    initial_test_mse = float(np.mean((model.predict(test_x) - test_y) ** 2))
    with graph.cpu_execution(threads=1, seed=1030):
        torch = graph._load_runtime()
        x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(args.steps):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()

    predictions = model.predict(test_x, batch_size=32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = args.output.with_suffix(".pt")
    torch.save({"architecture": model.architecture_, "n_features": model.n_features_in_, "state_dict": model.state_dict()}, weights_path)
    record = torch.load(weights_path, map_location="cpu", weights_only=True)
    restored = graph.build_module(record["architecture"], record["n_features"], seed=0)
    restored.load_state_dict(record["state_dict"])
    reload_predictions = restored.predict(test_x, batch_size=32)
    report = {
        "task": "synthetic independent-Gaussian regression; y=0.75*x0-0.5*x1+0.25*sin(x2)",
        "purpose": "runtime and serialization demonstration; not a real-data benchmark",
        "seed": 1030,
        "device": "cpu",
        "dtype": "float32",
        "train_rows": len(train_x),
        "test_rows": len(test_x),
        "features": 5,
        "optimizer_steps": args.steps,
        "attention_axis": "features",
        "temporal_attention": False,
        "parameter_count": complexity["parameter_count"],
        "initial_test_mse": initial_test_mse,
        "final_train_mse": float(np.mean((model.predict(train_x) - train_y) ** 2)),
        "final_test_mse": float(np.mean((predictions - test_y) ** 2)),
        "train_mean_baseline_test_mse": float(np.mean((float(train_y.mean()) - test_y) ** 2)),
        "checkpoint_reload_max_abs_error": float(np.max(np.abs(predictions - reload_predictions))),
        "all_predictions_finite": bool(np.isfinite(predictions).all()),
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["numpy", "torch", "threadpoolctl"]}},
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
