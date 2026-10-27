"""Real CPU tensor checks only; no data service, Pro or production database."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from fintable import neural_graph as graph


@pytest.fixture
def torch():
    return graph._load_runtime()


def simple_arch(**updates):
    spec = {"width": 8, "use_cls": False, "pool": "mean",
            "blocks": [{"kind": "feed_forward", "expansion": 1}], "head": {"hidden_width": 0}}
    spec.update(updates)
    return spec


def test_planning_metadata_does_not_import_torch_or_allocate_native_pool():
    code = """
import sys
from fintable import neural_graph as graph
assert 'torch' not in sys.modules
for architecture in graph.default_architectures():
    graph.parse_architecture(architecture, 50)
    graph.architecture_complexity(architecture, 50, 256, 304436, 20)
graph.architecture_schema()
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True, timeout=15)


def test_normalization_is_immutable_idempotent_and_graphs_have_different_mechanisms():
    original = simple_arch()
    before = copy.deepcopy(original)
    normalized = graph.parse_architecture(original, 5)
    assert original == before
    assert graph.parse_architecture(json.dumps(normalized), 5) == normalized
    baseline, ablation = graph.default_architectures()
    assert [block["kind"] for block in baseline["blocks"]] == ["attention", "feed_forward"] * 2
    assert [block["kind"] for block in ablation["blocks"]] == ["token_mixer", "feed_forward"] * 2
    assert baseline["width"] == ablation["width"]
    assert graph.architecture_schema()["semantics"].startswith("Feature-token")
    assert graph.architecture_schema()["pool"] == ["cls", "mean"]


@pytest.mark.parametrize("changes", [
    {"source": "import os"}, {"version": True}, {"version": 2}, {"width": True}, {"width": 128},
    {"tokenizer": "eval"}, {"use_cls": 1}, {"pool": "temporal"},
    {"use_cls": False, "pool": "cls"}, {"blocks": []},
    {"blocks": [{"kind": "feed_forward"}] * 9},
    {"blocks": [{"kind": "arbitrary_python", "code": "x"}]},
    {"blocks": [{"kind": "attention", "heads": 3}]},
    {"blocks": [{"kind": "attention", "heads": True}]},
    {"blocks": [{"kind": "feed_forward", "expansion": 5}]},
    {"blocks": [{"kind": "feed_forward", "activation": "__import__"}]},
    {"blocks": [{"kind": "feed_forward", "residual": "false"}]},
    {"blocks": [{"kind": "gated_ffn", "pre_norm": 0}]},
    {"blocks": [{"kind": "gated_ffn", "dropout": float("nan")}]},
    {"blocks": [{"kind": "gated_ffn", "dropout": float("inf")}]},
    {"blocks": [{"kind": "token_mixer", "hidden_tokens": 129}]},
    {"blocks": [{"kind": "normalization", "residual": True}]},
    {"head": {"hidden_width": 0, "activation": "gelu"}},
    {"head": {"hidden_width": 3, "dropout": -.1}},
    {"head": {"hidden_width": 3, "dropout": 10**400}},
    {"head": {"hidden_width": 8, "python": "lambda"}},
    {"use_cls": True, "pool": "cls", "blocks": [{"kind": "normalization"}]},
])
def test_parser_rejects_unimplemented_or_invalid_graphs(changes):
    with pytest.raises(graph.UnsupportedArchitecture):
        graph.parse_architecture(simple_arch(**changes), 5)


@pytest.mark.parametrize("features", [0, 65, True, 3.0])
def test_feature_shape_limits(features):
    with pytest.raises(graph.UnsupportedArchitecture):
        graph.parse_architecture(simple_arch(), features)


@pytest.mark.parametrize("seed", [None, True, -1, 2**32, 42.0])
def test_builder_requires_reproducible_integer_seed_before_runtime_load(seed, monkeypatch):
    monkeypatch.setattr(graph, "_load_runtime", lambda: pytest.fail("Invalid seed must not load runtime"))
    with pytest.raises(graph.UnsupportedArchitecture, match="seed"):
        graph.build_module(simple_arch(), 5, seed)


def test_parameter_activation_and_work_limits_fail_before_runtime_load(monkeypatch):
    monkeypatch.setattr(graph, "_load_runtime", lambda: pytest.fail("Reject before loading runtime/allocation"))
    huge = simple_arch(width=64, blocks=[{"kind": "gated_ffn", "expansion": 4}] * 8)
    with pytest.raises(graph.UnsupportedArchitecture, match="parameter budget"):
        graph.build_module(huge, 64, 42)
    attention = simple_arch(width=16, blocks=[{"kind": "attention", "heads": 8}] * 8)
    with pytest.raises(graph.UnsupportedArchitecture, match="activation memory"):
        graph.architecture_complexity(attention, 64, 1024)
    with pytest.raises(graph.UnsupportedArchitecture, match="arithmetic work"):
        graph.architecture_complexity(graph.default_architectures()[0], 64, 64, 1000000, 120)
    with pytest.raises(graph.UnsupportedArchitecture, match="both"):
        graph.architecture_complexity(simple_arch(), 5, 64, training_rows=1000)


@pytest.mark.parametrize("architecture", [
    *graph.default_architectures(),
    simple_arch(blocks=[{"kind": "normalization"}, {"kind": "gated_ffn", "expansion": 3, "activation": "relu"},
                        {"kind": "feed_forward", "residual": False, "pre_norm": False},
                        {"kind": "token_mixer", "hidden_tokens": 4}]),
    simple_arch(final_normalization=False, head={"hidden_width": 5, "activation": "silu", "dropout": .1}),
])
def test_exact_parameter_accounting_real_forward_and_backward(torch, architecture):
    module = graph.build_module(architecture, 5, 42)
    complexity = graph.architecture_complexity(architecture, 5, 16, 100, 2)
    assert complexity["parameter_count"] == sum(parameter.numel() for parameter in module.parameters())
    assert complexity["token_shape"] == [None, 5 + int(module.architecture_["use_cls"]), module.architecture_["width"]]
    assert complexity["temporal_attention"] is False
    assert complexity["training_multiply_adds"] == 3 * complexity["forward_multiply_adds_per_row"] * 100 * 2
    with graph.cpu_execution(seed=71):
        values = torch.randn(16, 5)
        predictions = module(values)
        assert predictions.shape == (16,)
        loss = ((predictions - values[:, 0]) ** 2).mean()
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
        assert module.feature_weights.grad.abs().sum().item() > 0


def test_manual_qkv_is_scaled_dot_product_attention_and_differentiable(torch):
    module = graph.build_module(simple_arch(blocks=[{"kind": "attention", "heads": 2, "pre_norm": False, "residual": False}]), 4, 7)
    operation = module.blocks[0].operation
    with graph.cpu_execution(seed=13):
        x = torch.randn(3, 4, 8, requires_grad=True)
        projected = operation.qkv(x).reshape(3, 4, 3, 2, 4)
        q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
        expected = operation.projection(((q @ k.transpose(-2, -1) / 2).softmax(-1) @ v).transpose(1, 2).reshape(3, 4, 8))
        assert torch.equal(operation(x), expected)
        operation(x).sum().backward()
        assert torch.isfinite(x.grad).all() and operation.qkv.weight.grad.abs().sum().item() > 0


def test_fixed_seed_preserves_ambient_cpu_rng_and_explicit_dtype_device(torch):
    state = torch.random.get_rng_state().clone()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        # Even a caller's meta default device cannot relocate application data.
        with torch.device("meta"):
            first = graph.build_module(graph.default_architectures()[0], 5, 42)
        second = graph.build_module(graph.default_architectures()[0], 5, 42)
        third = graph.build_module(graph.default_architectures()[0], 5, 43)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in first.parameters())
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())
    assert not torch.equal(first.feature_weights, third.feature_weights)


def test_training_step_with_dropout_is_reproducible_under_guard(torch):
    spec = simple_arch(blocks=[{"kind": "attention", "heads": 2, "dropout": .1}, {"kind": "gated_ffn", "dropout": .1}])
    models = [graph.build_module(spec, 5, 42) for _ in range(2)]
    for module in models:
        with graph.cpu_execution(seed=12):
            optimizer = torch.optim.Adam(module.parameters(), lr=.001)
            x = torch.arange(100, dtype=torch.float32).reshape(20, 5) / 100
            optimizer.zero_grad()
            ((module(x) - x[:, 0]) ** 2).mean().backward()
            optimizer.step()
    assert all(torch.equal(value, models[1].state_dict()[key]) for key, value in models[0].state_dict().items())


def test_nested_pool_and_rng_restoration_after_exception(torch):
    previous_threads = torch.get_num_threads()
    state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="fixture failure"):
        with graph.cpu_execution(threads=1, seed=5):
            assert torch.get_num_threads() == 1
            with graph.cpu_execution(threads=2):
                assert torch.get_num_threads() == 1
            torch.randn(3)
            raise RuntimeError("fixture failure")
    assert torch.get_num_threads() == previous_threads
    assert torch.equal(torch.random.get_rng_state(), state)
    assert graph._LOCAL.depth == 0
    with graph.cpu_execution():
        assert torch.get_num_threads() == 1


def test_prediction_chunking_mode_restoration_and_state_dict_joblib_reload(torch, tmp_path):
    module = graph.build_module(graph.default_architectures()[0], 5, 42)
    values = np.arange(185, dtype=np.float32).reshape(37, 5) / 10
    assert module.training
    expected = module.predict(values, batch_size=16)
    assert module.training and expected.shape == (37,)
    assert module.predict(np.empty((0, 5))).shape == (0,)
    artifact = tmp_path / "module.joblib"
    joblib.dump(module, artifact)
    restored = joblib.load(artifact)
    assert np.array_equal(restored.predict(values, 16), expected)
    weights = tmp_path / "weights.pt"
    torch.save({"architecture": module.architecture_, "n_features": 5, "state_dict": module.state_dict()}, weights)
    record = torch.load(weights, map_location="cpu", weights_only=True)
    recreated = graph.build_module(record["architecture"], record["n_features"], 0)
    recreated.load_state_dict(record["state_dict"])
    assert np.array_equal(recreated.predict(values, 16), expected)
    np.save(tmp_path / "x.npy", values)
    np.save(tmp_path / "expected.npy", expected)
    code = """
import sys, joblib, numpy as np
from pathlib import Path
from fintable import neural_graph
assert 'torch' not in sys.modules
folder=Path(sys.argv[1])
module=joblib.load(folder/'module.joblib')
assert np.array_equal(module.predict(np.load(folder/'x.npy'),16),np.load(folder/'expected.npy'))
print('fresh_process_reload_maxerror=0')
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=Path(__file__).parents[1], check=True, capture_output=True, text=True, timeout=30)
    assert "fresh_process_reload_maxerror=0" in result.stdout


def test_invalid_inference_and_prediction_failure_restore_training_mode(torch, monkeypatch):
    module = graph.build_module(simple_arch(), 5, 42)
    for values in (np.zeros((2, 4)), np.full((2, 5), np.nan), [["bad"] * 5], np.zeros(5)):
        with pytest.raises(ValueError):
            module.predict(values)
    with pytest.raises(ValueError, match="float32"):
        module(torch.zeros((2, 5), dtype=torch.float64))
    with pytest.raises(ValueError, match="frozen_feature_count"):
        module(torch.zeros((2, 4)))
    previous_threads = torch.get_num_threads()
    def failed(_):
        raise RuntimeError("fixture forward error")
    monkeypatch.setattr(module, "forward", failed)
    with pytest.raises(RuntimeError, match="fixture forward error"):
        module.predict(np.zeros((2, 5), dtype=np.float32))
    assert module.training and torch.get_num_threads() == previous_threads


def test_finite_inputs_cannot_silently_return_nonfinite_numerical_predictions(torch):
    module = graph.build_module(graph.default_architectures()[0], 5, 42)
    with pytest.raises(ValueError, match="nonfinite predictions"):
        module.predict(np.full((2, 5), 1e30, dtype=np.float32))
    assert module.training


def test_real_scipy_sklearn_threadpoolctl_torch_interleaving_in_30s_fence():
    code = """
import os, numpy as np, threading
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_info
from fintable.native_compute import native_compute
from fintable import neural_graph as graph
# sklearn itself sets this variable in its package initializer. Disable that
# vendor default before Torch loads; this check must not depend on a bypass.
os.environ.pop('KMP_DUPLICATE_LIB_OK',None)
assert not os.environ.get('KMP_DUPLICATE_LIB_OK'), 'No duplicate-OpenMP bypass is allowed'
values=np.random.default_rng(5).normal(size=(128,5)).astype(np.float32)
with native_compute(limits=1):
    before=Ridge(alpha=1).fit(values, values[:,0]).predict(values)
    eigen=eigh(np.eye(16),check_finite=True)[0]
module=graph.build_module(graph.default_architectures()[0],5,42)
with graph.cpu_execution(seed=7):
    torch=graph._load_runtime()
    optimizer=torch.optim.Adam(module.parameters(),lr=.001)
    x=torch.from_numpy(values)
    for _ in range(3):
        optimizer.zero_grad(); loss=((module(x)-x[:,0])**2).mean(); loss.backward(); optimizer.step()
    with native_compute(limits=2):
        assert np.isfinite(eigh(np.eye(16),check_finite=True)[0]).all()
assert np.isfinite(module.predict(values)).all()
with native_compute(limits=1):
    after=Ridge(alpha=1).fit(values, values[:,0]).predict(values)
    assert np.array_equal(before,after)
assert np.isfinite(eigen).all()
# Start a fresh graph build/Adam lane concurrently with a real sklearn/SciPy
# lane. The shared owner serializes native sections while both workers progress.
barrier=threading.Barrier(2)
def torch_lane():
    barrier.wait(timeout=5)
    model=graph.build_module(graph.default_architectures()[1],5,19)
    for _ in range(3):
        with graph.cpu_execution(seed=5):
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            optimizer.zero_grad(); loss=((model(x)-x[:,0])**2).mean(); loss.backward(); optimizer.step()
        assert np.isfinite(model.predict(values)).all()
    return True
def scipy_lane():
    barrier.wait(timeout=5)
    for _ in range(3):
        with native_compute(limits=2):
            assert np.isfinite(eigh(np.eye(16),check_finite=True)[0]).all()
            assert np.isfinite(Ridge(alpha=1).fit(values,values[:,0]).predict(values)).all()
    return True
with ThreadPoolExecutor(max_workers=2) as executor:
    futures=[executor.submit(torch_lane),executor.submit(scipy_lane)]
    assert all(future.result(timeout=15) for future in futures)
print('native_interleave_ok',len(threadpool_info()))
"""
    # No KMP/OpenMP workaround, no native child survives a failed fence.
    environment = {key: value for key, value in os.environ.items() if key != "KMP_DUPLICATE_LIB_OK"}
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1], env=environment, check=True, capture_output=True, text=True, timeout=30)
    assert "native_interleave_ok" in result.stdout
