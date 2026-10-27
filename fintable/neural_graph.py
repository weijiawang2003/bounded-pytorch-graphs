"""Closed, composable CPU networks over scalar feature tokens.

Each row is one observation, not a temporal window: [N, F] becomes [N, T, D].
Feature identity is learned by scalar-feature affine tokenization. An architecture
is bounded JSON data, never Python, an import path, or an executable expression.
Training/splitting/preprocessing/early stopping belong to model_lab. Its entire
forward/backward/optimizer step must hold cpu_execution(), including backward.
"""
from __future__ import annotations

import copy
import json
import math
import threading
from contextlib import contextmanager

import numpy as np

from .native_compute import native_compute

LIMITS = {
    "max_features": 64, "max_blocks": 8, "max_width": 64,
    "max_parameters": 250000, "max_batch_size": 1024,
    "max_activation_bytes": 256 * 1024**2,
    "max_training_rows": 1000000, "max_epochs": 120,
    "max_training_multiply_adds": 20000000000000,
    "max_predict_rows": 1000000,
}
_LOCAL = threading.local()
_TORCH = None
_RUNTIME_CLASSES = {}
_KINDS = {"attention", "feed_forward", "gated_ffn", "token_mixer", "normalization"}
_ACTIVATIONS = {"gelu", "relu", "silu"}


class UnsupportedArchitecture(ValueError):
    """A graph is outside the implemented shape, operation or resource limits."""


def _integer(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise UnsupportedArchitecture(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not low <= value <= high or not math.isfinite(value):
        raise UnsupportedArchitecture(f"{name} must be a finite number in [{low}, {high}]")
    return float(value)


def _boolean(value, name):
    if not isinstance(value, bool):
        raise UnsupportedArchitecture(f"{name} must be boolean")
    return value


def _keys(value, allowed, name):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value) or set(value) - allowed:
        raise UnsupportedArchitecture(f"{name} contains an unimplemented field or is not an object")


def _choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise UnsupportedArchitecture(f"{name} must be one of {sorted(choices)}")
    return value


def parse_architecture(spec, n_features=None):
    """Return canonical structure; optional feature count checks actual shape limits.

    Complexity is separate so architecture identity never depends on batch size,
    training length, or a particular data split. Defaults are explicit in output.
    """
    if isinstance(spec, str):
        if len(spec) > 16000:
            raise UnsupportedArchitecture("Architecture JSON exceeds 16000 characters")
        try:
            spec = json.loads(spec)
        except (TypeError, ValueError) as exc:
            raise UnsupportedArchitecture("Architecture must be closed JSON data") from exc
    _keys(spec, {"version", "tokenizer", "width", "use_cls", "blocks", "pool", "final_normalization", "head"}, "architecture")
    version = _integer(spec.get("version", 1), 1, 1, "version")
    tokenizer = _choice(spec.get("tokenizer", "scalar_feature"), {"scalar_feature"}, "tokenizer")
    width = _integer(spec.get("width", 16), 8, LIMITS["max_width"], "width")
    use_cls = _boolean(spec.get("use_cls", True), "use_cls")
    pool = _choice(spec.get("pool", "cls" if use_cls else "mean"), {"cls", "mean"}, "pool")
    if pool == "cls" and not use_cls:
        raise UnsupportedArchitecture("CLS pooling requires a CLS token")
    blocks = spec.get("blocks")
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= LIMITS["max_blocks"]:
        raise UnsupportedArchitecture("Architecture needs between one and eight blocks")
    normalized = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise UnsupportedArchitecture("Each block must be an object")
        kind = _choice(block.get("kind"), _KINDS, f"blocks[{index}].kind")
        if kind == "normalization":
            _keys(block, {"kind"}, "normalization")
            normalized.append({"kind": kind})
            continue
        fields = {"kind", "residual", "pre_norm", "dropout"}
        fields |= {"heads"} if kind == "attention" else ({"hidden_tokens", "activation"} if kind == "token_mixer" else {"expansion", "activation"})
        _keys(block, fields, kind)
        item = {"kind": kind,
                "residual": _boolean(block.get("residual", True), "residual"),
                "pre_norm": _boolean(block.get("pre_norm", True), "pre_norm"),
                "dropout": _number(block.get("dropout", 0.0), 0.0, 0.3, "dropout")}
        if kind == "attention":
            heads = _integer(block.get("heads", 2), 1, 8, "heads")
            if width % heads:
                raise UnsupportedArchitecture("Attention width must be divisible by heads")
            item["heads"] = heads
        elif kind == "token_mixer":
            item["hidden_tokens"] = _integer(block.get("hidden_tokens", 16), 4, 128, "hidden_tokens")
            # The implemented mixer always uses GELU. Accept its schema spelling
            # without adding a key to the established canonical/cache identity.
            _choice(block.get("activation", "gelu"), {"gelu"}, "token_mixer.activation")
        else:
            item["expansion"] = _integer(block.get("expansion", 2), 1, 4, "expansion")
            item["activation"] = _choice(block.get("activation", "gelu" if kind == "feed_forward" else "silu"), _ACTIVATIONS, "activation")
        normalized.append(item)
    if pool == "cls" and not any(block["kind"] in {"attention", "token_mixer"} for block in normalized):
        raise UnsupportedArchitecture("CLS output has no path from feature tokens; add attention/mixer or mean-pool features")
    head = spec.get("head", {})
    _keys(head, {"hidden_width", "activation", "dropout"}, "head")
    hidden = _integer(head.get("hidden_width", 16), 0, 64, "head.hidden_width")
    activation = _choice(head.get("activation", "gelu" if hidden else "identity"), _ACTIVATIONS if hidden else {"identity"}, "head.activation")
    result = {"version": version, "tokenizer": tokenizer, "width": width, "use_cls": use_cls,
              "blocks": normalized, "pool": pool,
              "final_normalization": _boolean(spec.get("final_normalization", True), "final_normalization"),
              "head": {"hidden_width": hidden, "activation": activation,
                       "dropout": _number(head.get("dropout", 0.0), 0.0, 0.3, "head.dropout")}}
    if n_features is not None:
        _complexity(result, n_features, 1, None, None)
    return result


def _complexity(arch, n_features, batch_size, training_rows, epochs):
    features = _integer(n_features, 1, LIMITS["max_features"], "n_features")
    batch = _integer(batch_size, 1, LIMITS["max_batch_size"], "batch_size")
    tokens, width = features + int(arch["use_cls"]), arch["width"]
    parameters = 2 * features * width + int(arch["use_cls"]) * width
    operations = features * width
    # Conservative retained tensors for training, including gradients, Q/K/V,
    # attention probabilities/dropout buffers and residual/pre-norm activations.
    saved = 4 * tokens * width
    for block in arch["blocks"]:
        kind = block["kind"]
        if kind == "normalization":
            parameters += 2 * width
            operations += 8 * tokens * width
            saved += 3 * tokens * width
            continue
        if block["pre_norm"]:
            parameters += 2 * width
            operations += 8 * tokens * width
        if kind == "attention":
            parameters += 4 * width**2 + 4 * width
            operations += 4 * tokens * width**2 + 2 * tokens**2 * width + 5 * block["heads"] * tokens**2
            saved += 10 * tokens * width + 4 * block["heads"] * tokens**2
        elif kind in {"feed_forward", "gated_ffn"}:
            hidden = width * block["expansion"]
            branches = 3 if kind == "gated_ffn" else 2
            parameters += branches * width * hidden + (2 if kind == "gated_ffn" else 1) * hidden + width
            operations += branches * tokens * width * hidden + 8 * tokens * hidden
            saved += 5 * tokens * width + (5 if kind == "gated_ffn" else 3) * tokens * hidden
        else:
            hidden = block["hidden_tokens"]
            parameters += 2 * tokens * hidden + hidden + tokens
            operations += 2 * width * tokens * hidden + 8 * width * hidden
            saved += 5 * tokens * width + 3 * width * hidden
    if arch["final_normalization"]:
        parameters += 2 * width
        operations += 8 * tokens * width
        saved += 3 * tokens * width
    hidden = arch["head"]["hidden_width"]
    parameters += width * hidden + 2 * hidden + 1 if hidden else width + 1
    operations += width * hidden + hidden + 8 * hidden if hidden else width
    saved += 4 * (width + hidden + 1)
    activation_bytes = 4 * batch * saved
    # Parameters, parameter gradients, Adam moments and a checkpoint copy.
    estimated_working_bytes = activation_bytes + parameters * 4 * 6 + batch * features * 4
    if parameters > LIMITS["max_parameters"]:
        raise UnsupportedArchitecture("Graph exceeds parameter budget")
    if activation_bytes > LIMITS["max_activation_bytes"]:
        raise UnsupportedArchitecture("Graph exceeds activation memory budget at this batch size")
    if (training_rows is None) != (epochs is None):
        raise UnsupportedArchitecture("Training work check requires both training_rows and epochs")
    work = None
    if training_rows is not None:
        rows = _integer(training_rows, 1, LIMITS["max_training_rows"], "training_rows")
        passes = _integer(epochs, 1, LIMITS["max_epochs"], "epochs")
        work = 3 * operations * rows * passes
        if work > LIMITS["max_training_multiply_adds"]:
            raise UnsupportedArchitecture("Graph exceeds training arithmetic work budget")
    return {"parameter_count": parameters, "token_count": tokens, "width": width,
            "input_shape": [None, features], "token_shape": [None, tokens, width],
            "estimated_activation_bytes": activation_bytes, "estimated_working_bytes": estimated_working_bytes,
            "forward_multiply_adds_per_row": operations, "training_multiply_adds": work,
            "work_estimate_note": "Arithmetic proxy including three forward equivalents per training row/epoch; not measured FLOPs or wall time",
            "temporal_attention": False, "attention_axis": "features", "limits": LIMITS.copy()}


def architecture_complexity(arch, n_features, batch_size=256, training_rows=None, epochs=None):
    """Validate allocation and arithmetic budgets before any training allocation."""
    return _complexity(parse_architecture(arch), n_features, batch_size, training_rows, epochs)


@contextmanager
def cpu_execution(threads=1, seed=None):
    """Guard *all* native work, RNG and process-wide torch pool restoration.

    No interop pool is created/resized. A nested execution inherits outer thread
    ownership. Seed only the CPU generator, preserving ambient RNG afterwards.
    """
    _integer(threads, 1, 8, "threads")
    if seed is not None:
        _integer(seed, 0, 2**32 - 1, "seed")
    with native_compute(limits=threads):
        torch = _load_runtime()
        depth = getattr(_LOCAL, "depth", 0)
        old_threads = torch.get_num_threads() if not depth else None
        _LOCAL.depth = depth + 1
        try:
            if not depth and old_threads != threads:
                torch.set_num_threads(threads)
            if seed is None:
                yield
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    yield
        finally:
            try:
                if not depth and torch.get_num_threads() != old_threads:
                    torch.set_num_threads(old_threads)
            finally:
                _LOCAL.depth = depth


def _load_runtime():
    """Load Torch only under the existing process-wide native owner."""
    global _TORCH
    with native_compute(limits=1):
        if _TORCH is None:
            import torch
            from torch import nn
            _define_runtime(torch, nn)
            _TORCH = torch
        return _TORCH


def __getattr__(name):
    # Ordinary module/class import during joblib unpickling resolves the same
    # statically implemented class; no generated source or import path executes.
    if name in {"FeatureTokenGraph", "_Attention", "_FeedForward", "_TokenMixer", "_GraphBlock"}:
        _load_runtime()
        return _RUNTIME_CLASSES[name]
    raise AttributeError(name)


def _define_runtime(torch, nn):
    def _activation(name):
        return {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "identity": nn.Identity}[name]()


    class _Attention(nn.Module):
        """Explicit scaled dot-product feature attention, without fused/thread pools."""
        def __init__(self, width, heads, dropout):
            super().__init__()
            self.heads, self.head_width = heads, width // heads
            self.qkv = nn.Linear(width, 3 * width, device="cpu", dtype=torch.float32)
            self.projection = nn.Linear(width, width, device="cpu", dtype=torch.float32)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            batch, tokens, width = x.shape
            qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_width)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width), dim=-1)
            mixed = torch.matmul(self.dropout(weights), v).transpose(1, 2).reshape(batch, tokens, width)
            return self.projection(mixed)


    class _FeedForward(nn.Module):
        def __init__(self, width, expansion, activation, gated):
            super().__init__()
            hidden = width * expansion
            self.value = nn.Linear(width, hidden, device="cpu", dtype=torch.float32)
            self.gate = nn.Linear(width, hidden, device="cpu", dtype=torch.float32) if gated else None
            self.output = nn.Linear(hidden, width, device="cpu", dtype=torch.float32)
            self.activation = _activation(activation)

        def forward(self, x):
            value = self.value(x)
            activated = value * self.activation(self.gate(x)) if self.gate is not None else self.activation(value)
            return self.output(activated)


    class _TokenMixer(nn.Module):
        def __init__(self, tokens, hidden):
            super().__init__()
            self.mix = nn.Sequential(nn.Linear(tokens, hidden, device="cpu", dtype=torch.float32), nn.GELU(),
                                     nn.Linear(hidden, tokens, device="cpu", dtype=torch.float32))

        def forward(self, x):
            return self.mix(x.transpose(1, 2)).transpose(1, 2)


    class _GraphBlock(nn.Module):
        def __init__(self, spec, width, tokens):
            super().__init__()
            self.residual = spec["residual"]
            self.norm = nn.LayerNorm(width, device="cpu", dtype=torch.float32) if spec["pre_norm"] else nn.Identity()
            if spec["kind"] == "attention":
                self.operation = _Attention(width, spec["heads"], spec["dropout"])
            elif spec["kind"] == "token_mixer":
                self.operation = _TokenMixer(tokens, spec["hidden_tokens"])
            else:
                self.operation = _FeedForward(width, spec["expansion"], spec["activation"], spec["kind"] == "gated_ffn")
            self.dropout = nn.Dropout(spec["dropout"])

        def forward(self, x):
            branch = self.dropout(self.operation(self.norm(x)))
            return x + branch if self.residual else branch


    class FeatureTokenGraph(nn.Module):
        """CPU float32 graph; columns are ordered features, never time steps.

        Do not share a mutable training instance with another fit/predict task.
        Save the architecture and state_dict or joblib this top-level class. Loading
        is restricted to application-owned artifacts; never unpickle agent content.
        """
        def __init__(self, architecture, n_features):
            super().__init__()
            self.architecture_ = parse_architecture(architecture, n_features)
            self.n_features_in_ = n_features
            width = self.architecture_["width"]
            tokens = n_features + int(self.architecture_["use_cls"])
            self.feature_weights = nn.Parameter(torch.empty(n_features, width, device="cpu", dtype=torch.float32))
            self.feature_bias = nn.Parameter(torch.empty(n_features, width, device="cpu", dtype=torch.float32))
            nn.init.normal_(self.feature_weights, std=.02)
            nn.init.normal_(self.feature_bias, std=.02)
            self.cls = nn.Parameter(torch.zeros(1, 1, width, device="cpu", dtype=torch.float32)) if self.architecture_["use_cls"] else None
            self.blocks = nn.ModuleList([
                nn.LayerNorm(width, device="cpu", dtype=torch.float32) if block["kind"] == "normalization" else _GraphBlock(block, width, tokens)
                for block in self.architecture_["blocks"]])
            self.final_norm = nn.LayerNorm(width, device="cpu", dtype=torch.float32) if self.architecture_["final_normalization"] else nn.Identity()
            head = self.architecture_["head"]
            self.head = nn.Sequential(nn.Linear(width, head["hidden_width"], device="cpu", dtype=torch.float32),
                                      _activation(head["activation"]), nn.Dropout(head["dropout"]),
                                      nn.Linear(head["hidden_width"], 1, device="cpu", dtype=torch.float32)) if head["hidden_width"] else nn.Sequential(
                                          nn.Dropout(head["dropout"]), nn.Linear(width, 1, device="cpu", dtype=torch.float32))
            expected = _complexity(self.architecture_, n_features, 1, None, None)["parameter_count"]
            if sum(parameter.numel() for parameter in self.parameters()) != expected:
                raise RuntimeError("Implementation parameter count differs from preregistered complexity")

        def forward(self, x):
            if not isinstance(x, torch.Tensor) or x.device.type != "cpu" or x.dtype != torch.float32:
                raise ValueError("FeatureTokenGraph requires a CPU float32 tensor")
            if x.ndim != 2 or x.shape[1] != self.n_features_in_:
                raise ValueError("FeatureTokenGraph expects [rows, frozen_feature_count]")
            if not torch.isfinite(x).all():
                raise ValueError("Neural graph input contains nonfinite values; use fitted preprocessing")
            if x.shape[0]:
                _complexity(self.architecture_, self.n_features_in_, x.shape[0], None, None)
            tokens = x.unsqueeze(-1) * self.feature_weights.unsqueeze(0) + self.feature_bias.unsqueeze(0)
            if self.cls is not None:
                tokens = torch.cat((self.cls.expand(x.shape[0], -1, -1), tokens), dim=1)
            for block in self.blocks:
                tokens = block(tokens)
            tokens = self.final_norm(tokens)
            pooled = tokens[:, 0] if self.architecture_["pool"] == "cls" else tokens[:, int(self.cls is not None):].mean(dim=1)
            prediction = self.head(pooled).squeeze(-1)
            if not torch.isfinite(prediction).all():
                raise ValueError("Neural graph produced nonfinite predictions; reject the numerical fit")
            return prediction

        def predict(self, values, batch_size=256):
            return predict_cpu(self, values, batch_size=batch_size)

    for cls in (_Attention, _FeedForward, _TokenMixer, _GraphBlock, FeatureTokenGraph):
        cls.__qualname__ = cls.__name__
        _RUNTIME_CLASSES[cls.__name__] = cls
        globals()[cls.__name__] = cls


def build_module(arch, n_features, seed):
    """Allocate a deterministic CPU graph without changing ambient RNG or dtype."""
    _integer(seed, 0, 2**32 - 1, "seed")
    normalized = parse_architecture(arch, n_features)
    with cpu_execution(threads=1, seed=seed):
        torch = _load_runtime()
        with torch.device("cpu"):
            return _RUNTIME_CLASSES["FeatureTokenGraph"](normalized, n_features)


def predict_cpu(module, values, batch_size=256):
    """Inference only; no fitting, preprocessing, labels or split selection."""
    torch = _load_runtime()
    if not isinstance(module, _RUNTIME_CLASSES["FeatureTokenGraph"]):
        raise ValueError("predict_cpu accepts an application-owned FeatureTokenGraph")
    architecture_complexity(module.architecture_, module.n_features_in_, batch_size)
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != module.n_features_in_ or array.dtype.kind not in "fiu":
        raise ValueError("Prediction requires a numeric [rows, frozen_feature_count] matrix")
    if len(array) > LIMITS["max_predict_rows"] or not np.isfinite(array).all():
        raise ValueError("Prediction row limit exceeded or input contains nonfinite values")
    if not len(array):
        return np.empty(0, dtype=np.float32)
    with cpu_execution(threads=1):
        training = module.training
        module.eval()
        try:
            with torch.inference_mode():
                predictions = [module(torch.from_numpy(np.array(array[start:start + batch_size], dtype=np.float32, copy=True))).numpy().copy()
                               for start in range(0, len(array), batch_size)]
            result = np.concatenate(predictions)
            if not np.isfinite(result).all():
                raise ValueError("Neural graph produced nonfinite predictions")
            return result
        finally:
            module.train(training)


def default_architectures():
    """Preregistered structural control pair, not chosen from validation returns."""
    baseline = {"version": 1, "width": 16, "use_cls": True, "pool": "cls", "blocks": [
        {"kind": "attention", "heads": 2}, {"kind": "feed_forward", "expansion": 2},
        {"kind": "attention", "heads": 2}, {"kind": "feed_forward", "expansion": 2}],
        "head": {"hidden_width": 16}}
    ablation = copy.deepcopy(baseline)
    ablation["blocks"] = [{"kind": "token_mixer", "hidden_tokens": 16} if block["kind"] == "attention" else block for block in ablation["blocks"]]
    return [parse_architecture(baseline), parse_architecture(ablation)]


def architecture_schema():
    """Machine-readable implementation contract for research proposal evidence."""
    return {"version": 1, "name": "feature_token_graph", "input": "CPU float32 [rows, ordered_features]",
            "semantics": "Feature-token attention over one row; no temporal windows or sequence model",
            "tokenizer": "x_j * learned_feature_weight_j + learned_feature_bias_j; optional learned CLS",
            "width": {"min": 8, "max": 64}, "blocks": {
                "attention": {"heads": {"min": 1, "max": 8, "must_divide_width": True}},
                "feed_forward": {"expansion": {"min": 1, "max": 4}, "activation": sorted(_ACTIVATIONS)},
                "gated_ffn": {"expansion": {"min": 1, "max": 4}, "activation": sorted(_ACTIVATIONS)},
                "token_mixer": {"hidden_tokens": {"min": 4, "max": 128}, "activation": "gelu"},
                "normalization": {"operation": "LayerNorm over token width", "parameters": []}},
            "branch_options": {"applies_to": sorted(_KINDS - {"normalization"}), "residual": [True, False], "pre_norm": [True, False], "dropout": {"min": 0, "max": .3}},
            "topology": "Ordered shape-preserving blocks, each optionally branching residual addition; no arbitrary code",
            "pool": ["cls", "mean"], "mean_pool_excludes_cls": True, "head": {"hidden_width": {"min": 0, "max": 64}, "activation": sorted(_ACTIVATIONS), "linear_head_activation": "identity", "dropout": {"min": 0, "max": .3}},
            "limits": LIMITS.copy(), "training_owner": "model_lab train-only preprocessing/purged early stopping",
            "defaults": default_architectures()}
