"""Execute hash-pinned encoder arithmetic with a tiny NumPy tensor test double.

This tests the actual extracted public augmentation/fusion program, not neural
quality or CUDA. Peak counting is deliberately stubbed: it is tested for real
in the staged GPU preflight. No notebook or whole predictor module is executed.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from public_d4_preflight import encoder_only_function, named_definitions

PINS = {
    "public-predictor-original.py": "ddca518b0e838b2123f30cfcb31819fb159b4e68fc20ef6b3a42003cc2d528ed",
    "public-predictor-d4-corrected.py": "ef6fc4f8335ec799f7229dc880b2247fb776394821c382a2f113906f6b9b0c6a",
    "public-postprocess-d4-corrected.py": "f5122587381f17a451547327abff3948734e455a2ae8cfc9d09e0e506c1e974d",
}


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    def clone(self):
        return Tensor(self.data.copy())

    def flip(self, dims):
        return Tensor(np.flip(self.data, dims))

    def transpose(self, a, b):
        return Tensor(self.data.swapaxes(a, b))

    def float(self):
        # FP64 in this independent arithmetic check; actual CUDA uses its
        # original public dtype and is separately measured by the GPU runner.
        return Tensor(self.data.astype(np.float64))

    def abs(self):
        return Tensor(np.abs(self.data))

    def mean(self):
        return Tensor(np.mean(self.data))

    def std(self, unbiased=False):
        return Tensor(np.std(self.data, ddof=1 if unbiased else 0))

    def clamp_min(self, bound):
        return Tensor(np.maximum(self.data, bound))

    def clamp(self, lower, upper):
        return Tensor(np.clip(self.data, lower, upper))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def to(self, **kwargs):
        return self

    def __float__(self):
        return float(self.data)

    def __getitem__(self, item):
        return Tensor(self.data[item])

    def __add__(self, other):
        return Tensor(self.data + unwrap(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Tensor(self.data - unwrap(other))

    def __rsub__(self, other):
        return Tensor(unwrap(other) - self.data)

    def __mul__(self, other):
        return Tensor(self.data * unwrap(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return Tensor(self.data / unwrap(other))


def unwrap(value):
    return value.data if isinstance(value, Tensor) else value


TORCH = SimpleNamespace(
    rot90=lambda x, k, dims: Tensor(np.rot90(x.data, k, axes=dims)),
    no_grad=nullcontext, from_numpy=Tensor, float32=np.float32,
    sigmoid=lambda x: Tensor(1 / (1 + np.exp(-x.data))),
)


def orient(value, view, *, corrected, inverse=False):
    """Independent explicit NumPy permutations, separate from public code."""
    if view == 0:
        return value
    if view < 4:
        return np.flip(value, {1: -1, 2: -2, 3: (-1, -2)}[view])
    if view in (4, 5):
        turns = {4: 1, 5: 3}[view] * (-1 if inverse else 1)
        return np.rot90(value, turns, axes=(-2, -1))
    if view == 6:
        return value.swapaxes(-1, -2)
    turns = 2 if corrected else 1
    if inverse:
        return np.rot90(value.swapaxes(-1, -2), -turns, axes=(-2, -1))
    return np.rot90(value, turns, axes=(-2, -1)).swapaxes(-1, -2)


class ToyEncoder:
    def __init__(self, variant):
        self.variant = variant
        self.inputs = []

    def arithmetic(self, value):
        directional = (self.variant + 1) * np.roll(value, 1, axis=-1) - np.roll(value, 1, axis=-2)
        features = np.stack([directional, value * value / 9 + np.roll(value, -1, axis=-2)], axis=2)
        logits = features[:, :, 0] * 0.4 - features[:, :, 1] * (0.2 + self.variant / 10)
        return features, logits

    def encode(self, tensor):
        value = tensor.data
        self.inputs.append((value.shape, hashlib.sha256(value.tobytes()).hexdigest()))
        features, logits = self.arithmetic(value)
        return Tensor(features), [Tensor(logits[:, frame:frame+1]) for frame in range(value.shape[1])]


def read_pinned(root: Path, name: str) -> str:
    payload = (root / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != PINS[name]:
        raise ValueError(f"Pinned source changed: {name}")
    return payload.decode("utf-8")


def encoder_check(root: Path) -> list[dict]:
    result = []
    # Both square and rectangular image pairs, and a three-frame window.
    for shape in ((1, 2, 3, 8, 8), (1, 2, 3, 6, 10), (1, 3, 2, 5, 5)):
        image = np.sin(np.arange(np.prod(shape)).reshape(shape) / 19) + 2
        for arm, name, corrected in (
            ("original", "public-predictor-original.py", False),
            ("corrected", "public-predictor-d4-corrected.py", True),
        ):
            source = read_pinned(root, name)
            environment = dict(torch=TORCH,
                os=SimpleNamespace(environ={"BIOHUB_EDGE_FEATURE_TTA": "1",
                    "BIOHUB_SECONDARY_EDGE_FEATURE_TTA": "1",
                    "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT": "0.75"}),
                cfg=SimpleNamespace(det_tta=True, det_threshold=0.965),
                pool_k=(3, 3, 3), _detect_cells_pooled=lambda *args: [0, 1, 2])
            exec(compile(encoder_only_function(source), "pinned-encoder-arithmetic", "exec"), environment)
            models = [ToyEncoder(0), ToyEncoder(1)]
            primary, secondary, logits = environment["public_encoder_pass"](Tensor(image), *models)
            means = []
            for model in models:
                features, detections = [], []
                for view in range(8):
                    f, d = model.arithmetic(orient(image, view, corrected=corrected))
                    features.append(orient(f, view, corrected=corrected, inverse=True))
                    detections.append(orient(d, view, corrected=corrected, inverse=True))
                means.append((np.mean(features, axis=0), np.mean(detections, axis=0)))
            native_secondary = models[1].arithmetic(image)[0]
            expected_secondary = 0.25 * native_secondary + 0.75 * means[1][0]
            error = max(float(np.max(np.abs(primary.data - means[0][0]))),
                        float(np.max(np.abs(secondary.data - expected_secondary))))
            for frame, actual in enumerate(logits):
                a, b = means[0][1][:, frame:frame+1], means[1][1][:, frame:frame+1]
                ratio = np.clip(max(a.std(), 1e-4) / max(b.std(), 1e-4), 0.5, 2)
                expected = 0.2 * a + 0.8 * ((b-b.mean()) * ratio + a.mean())
                error = max(error, float(np.max(np.abs(actual.data-expected))))
            counts = [dict(calls=len(m.inputs), unique=len(set(m.inputs))) for m in models]
            if error > 1e-12 or any(r != dict(calls=8, unique=8 if corrected else 7) for r in counts):
                raise ValueError("Actual public encoder differs from independent arithmetic")
            result.append(dict(arm=arm, shape=list(shape), counts=counts, maximum_absolute_error=error))
    return result


def deepcenter_check(root: Path) -> list[dict]:
    source = read_pinned(root, "public-postprocess-d4-corrected.py")
    result = []
    for yx in ((24, 24), (24, 32)):
        raw = (100 + 20 * np.sin(np.arange(8*yx[0]*yx[1]).reshape(8, *yx) / 23)).astype(np.float32)
        cfg = SimpleNamespace(pool_factor=2)
        for arm in ("original", "corrected"):
            env = dict(torch=TORCH, np=np, os=SimpleNamespace(environ={"BIOHUB_DEEPCENTER_TTA": "1"}),
                DEEPCENTER_SCORE_CACHE_MAX_FRAMES=8, read_test_frame=lambda *args: raw)
            code = named_definitions(source, ("_dc_pool_frame_xy", "_dc_normalize_dynamic_range",
                                               "_dc_cache_trim", "deepcenter_heatmap_for_frame"))
            if arm == "original":
                pairs = [("torch_mod.rot90(tensor, 2, dims=(-2, -1))", "torch_mod.rot90(tensor, 1, dims=(-2, -1))"),
                         ("torch_mod.rot90(model(at).transpose(-1, -2), -2, dims=(-2, -1))",
                          "torch_mod.rot90(model(at).transpose(-1, -2), -1, dims=(-2, -1))")]
                for old, new in pairs:
                    if code.count(old) != 1:
                        raise ValueError("DeepCenter source reconstruction drift")
                    code = code.replace(old, new, 1)
            exec(compile(code, "pinned-deepcenter-arithmetic", "exec"), env)
            inputs = []

            def arithmetic(x):
                return np.roll(x, 1, -1) * 0.3 - np.roll(x, 1, -2) * 0.7

            def model(value):
                inputs.append((value.shape, value.data.tobytes()))
                return Tensor(arithmetic(value.data))

            bundle = dict(model=model, cfg=cfg, device="test-double", torch=TORCH)
            cache = {}
            actual = env["deepcenter_heatmap_for_frame"]("toy", 0, bundle, {}, cache)
            # A cache hit must not add model calls.
            cached = env["deepcenter_heatmap_for_frame"]("toy", 0, bundle, {}, cache)
            image = env["_dc_normalize_dynamic_range"](env["_dc_pool_frame_xy"](raw, 2), cfg)[None, None]
            n_views = 8 if yx[0] == yx[1] else 4
            mean = sum(orient(arithmetic(orient(image, view, corrected=arm == "corrected")), view,
                              corrected=arm == "corrected", inverse=True) for view in range(n_views)) / n_views
            expected = (1/(1+np.exp(-mean)))[0, 0].astype(np.float32)
            error = float(np.max(np.abs(actual-expected)))
            unique = 8 if arm == "corrected" else 7
            if (actual is not cached or len(inputs) != n_views
                    or len(set(inputs)) != (unique if n_views == 8 else 4) or error > 1e-6):
                raise ValueError("Actual DeepCenter differs from independent arithmetic")
            result.append(dict(arm=arm, yx=list(yx), calls=len(inputs), unique=len(set(inputs)),
                               maximum_absolute_error=error, cache_hit_no_new_calls=True))
    return result
