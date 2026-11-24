"""Regenerate the tiny embedded-metadata ONNX fixtures for TestSemSeg.

Creates (in this directory):
- embedded_meta.onnx   valid tiny graph carrying a semseg.manifest metadata
                       entry (the C++ file scanner + merge tests use it; no
                       sidecar is shipped next to it on purpose)
- embedded_broken.onnx same graph, but the embedded manifest is not valid JSON
- embedded_absent.onnx same graph with NO metadata entry (no-key negatives)

Both are a few hundred bytes, safe to commit (the real Open3D-ML models are
38-41 MB and stay uncommitted). Usage:
    python make_embedded_fixture.py  (needs the `onnx` package, e.g. the
    stable_py conda env used by make_fixture.py/run_reference.py)
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

MANIFEST_KEY = "semseg.manifest"

# matches what TestSemSeg asserts: in_channels 4 == features last dim,
# num_classes 4 == logits last dim, distinct-from-default values everywhere
EMBEDDED_MANIFEST = """{
  "model_name": "GenericTestModel",
  "in_channels": 4,
  "num_classes": 4,
  "num_points": 123,
  "preprocessing": {
    "voxel_size": 0.05,
    "shift_min": true
  },
  "labels": {
    "values": [10, 20, 30, 40],
    "names": ["alpha", "beta", "gamma", "delta"]
  }
}"""


def build_graph():
    points = helper.make_tensor_value_info("points", TensorProto.FLOAT, ["batch", "num_points", 3])
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", "num_points", 4])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "num_points", 4])
    node = helper.make_node("Identity", ["features"], ["logits"], name="logits_identity")
    graph = helper.make_graph([node], "semseg_embedded_fixture", [points, features], [logits])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def set_metadata(model, key, value):
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value


def main():
    here = Path(__file__).resolve().parent / "data"
    here.mkdir(exist_ok=True)
    model = build_graph()

    stamped = onnx.ModelProto(); stamped.CopyFrom(model)
    set_metadata(stamped, MANIFEST_KEY, EMBEDDED_MANIFEST)
    meta_path = here / "embedded_meta.onnx"
    onnx.save(stamped, str(meta_path))
    print(f"wrote {meta_path} ({meta_path.stat().st_size} bytes)")

    broken = onnx.ModelProto(); broken.CopyFrom(model)
    set_metadata(broken, MANIFEST_KEY, "{not json")
    broken_path = here / "embedded_broken.onnx"
    onnx.save(broken, str(broken_path))
    print(f"wrote {broken_path} ({broken_path.stat().st_size} bytes)")

    absent_path = here / "embedded_absent.onnx"
    onnx.save(model, str(absent_path))
    print(f"wrote {absent_path} ({absent_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
