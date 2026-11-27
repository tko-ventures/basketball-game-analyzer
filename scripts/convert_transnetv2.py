"""Convert the official TransNet V2 TensorFlow weights to the vendored PyTorch model, checked against TensorFlow.

The name mapping and tensor layout follow upstream ``inference-pytorch/convert_weights.py`` (soCzech/TransNetV2, MIT).
Needs ``tensorflow`` and ``torch``, so it runs in a throwaway environment, not the project's:

    uv venv /tmp/tfenv --python 3.12 && uv pip install --python /tmp/tfenv/bin/python tensorflow==2.18.* torch numpy
    /tmp/tfenv/bin/python -I scripts/convert_transnetv2.py <TransNetV2 checkout>/inference/transnetv2-weights \
        --commit <upstream commit> --out weights/pretrained/transnetv2

The weights are written as ``transnetv2-<sha256 first 8>.pth`` with a ``source.json`` naming the upstream commit, the
TensorFlow and torch versions, and the agreement with TensorFlow on random clips. Weights are never committed.

"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
MODEL_PY = Path(__file__).resolve().parents[1] / "src/bga/ingest/_transnetv2/model.py"

#: TensorFlow scope names and their PyTorch module names (upstream ``remap_name``).
NAMES = {
    "conv_spatial": "layers.0",
    "conv_temporal": "layers.1",
    "kernel:0": "weight",
    "gamma:0": "weight",
    "bias:0": "bias",
    "beta:0": "bias",
    "dense": "fc1",
    "dense_1": "cls_layer1",
    "dense_2": "cls_layer2",
    "dense_3": "frame_sim_layer.projection",
    "dense_4": "frame_sim_layer.fc",
    "dense_5": "color_hist_layer.fc",
    "FrameSimilarity": "",
    "ColorHistograms": "",
    "moving_mean:0": "running_mean",
    "moving_variance:0": "running_var",
}


def remap_name(name):
    parts = []
    for a in name.replace("TransNet/", "").split("/"):
        if a.startswith(("SDDCNN", "DDCNN")):
            head, idx = a.split("_")
            a = f"{head}.{int(idx) - 1}"
        parts.append(NAMES.get(a, a))
    return ".".join(p for p in parts if p)


def remap_tensor(v):
    x = v.numpy()
    if x.ndim == 5:
        x = np.transpose(np.transpose(x, [0, 1, 2, 4, 3]), [3, 4, 0, 1, 2])
    elif x.ndim == 2:
        x = x.T
    return torch.from_numpy(np.ascontiguousarray(x)).clone()


def load_model_class():
    spec = importlib.util.spec_from_file_location("transnetv2_model", MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TransNetV2


def convert(tf_dir):
    tf_model = tf.saved_model.load(str(tf_dir))
    state = {remap_name(v.name): remap_tensor(v) for v in tf_model.variables}
    model = load_model_class()()
    shapes = {k: tuple(v.shape) for k, v in [*model.named_parameters(), *model.named_buffers()]}
    for k in shapes:
        if k not in state and k.endswith("num_batches_tracked"):
            state[k] = torch.tensor(1, dtype=torch.long)
    missing, extra = set(shapes) - set(state), set(state) - set(shapes)
    wrong = [k for k in shapes.keys() & state.keys() if tuple(state[k].shape) != shapes[k]]
    if missing or extra or wrong:
        raise SystemExit(f"conversion failed: missing {sorted(missing)}, extra {sorted(extra)}, wrong shape {wrong}")
    model.load_state_dict(state)
    return model.eval(), tf_model, state


def agreement(model, tf_model, clips=10, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(clips):
        x = rng.integers(0, 256, size=(2, 100, 27, 48, 3), dtype=np.uint8)
        with torch.no_grad():
            single, many = model(torch.from_numpy(x))
        tf_single, tf_many = tf_model(tf.cast(x, tf.float32))
        worst = max(
            worst,
            float(np.abs(torch.sigmoid(single).numpy() - tf.sigmoid(tf_single).numpy()).max()),
            float(np.abs(torch.sigmoid(many["many_hot"]).numpy() - tf.sigmoid(tf_many["many_hot"]).numpy()).max()),
        )
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tf_weights", type=Path, help="the SavedModel directory inference/transnetv2-weights")
    ap.add_argument("--commit", required=True, help="upstream soCzech/TransNetV2 commit of the weights")
    ap.add_argument("--out", type=Path, default=Path("weights/pretrained/transnetv2"))
    a = ap.parse_args(argv)
    model, tf_model, state = convert(a.tf_weights)
    worst = agreement(model, tf_model)
    if worst > 1e-3:
        raise SystemExit(f"PyTorch and TensorFlow disagree: max probability difference {worst:.2e}")
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / "transnetv2.pth.tmp"
    torch.save(state, tmp)
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    path = a.out / f"transnetv2-{sha[:8]}.pth"
    tmp.rename(path)
    source = {
        "upstream": "https://github.com/soCzech/TransNetV2",
        "commit": a.commit,
        "file": path.name,
        "sha256": sha,
        "tensorflow": tf.__version__,
        "torch": torch.__version__,
        "max_prob_diff_vs_tensorflow": worst,
        "licence": "MIT",
    }
    (a.out / "source.json").write_text(json.dumps(source, indent=1))
    print(json.dumps(source, indent=1))


if __name__ == "__main__":
    main()
