"""Generates the golden fixtures for TestSemSeg (tensor parity, output decoding).

Run from the Open3D-ML repo (which provides the reference pipeline and the
converted model):

    python <CloudCompare>/plugins/core/Standard/qSemSeg/test/make_fixture.py \
        --open3dml-src /path/to/Open3D-ML/src \
        --out-dir      <CloudCompare>/plugins/core/Standard/qSemSeg/test/data

The script drives the REAL reference pipeline (onnx_adapters.FKAConvONNXAdapter,
FKAConv._compute_hierarchy) on deterministic synthetic clouds and dumps raw
binary arrays consumed by the C++ test. RNG streams differ between numpy and
C++, so every randomness-dependent value is recorded and fed to the C++ side:
the patch index list, the per-stage FPS start indices, and the patch center.

Files written to --out-dir (all raw little-endian, no headers):

  main_xyz.f32           raw cloud, N*3 (the C++ test re-runs preprocessCloud on it)
  main_sub_points.f32    Python-ordered subsampled cloud, M*3 (grid barycenters)
  main_proj_target.f32   subsampled coords each original point projects to, N*3
  main_idxs.i64          patch indices (Python-sampled, shuffled), k entries
  main_center.f32        patch center point (3)
  main_fps_start.i64     FPS start index per hierarchy stage (4 entries)
  main_points_chw.f32    golden 'points'  patch tensor, 3*k (channels-first)
  main_features_chw.f32  golden 'features' patch tensor, C*k (feature_mode=xyz)
  main_sp<i>.f32         golden support points stage i, S_i*3 (point-major)
  main_idx<i>.i64        golden hierarchy indices stage i, row-major S*K / S*1
  mini_*                 same set for a 10-point cloud (m < num_points: exercises
                         index padding and the no-FPS support path; no fps_start)
  decode_logits.f32      synthetic logits, 2 batches * 64 rows * C
  decode_probs.f32       golden per-batch float32 softmax probs (float64 softmax)
  decode_acc.f32         golden test_probs after both batches (smooth=0.95)
  decode_labels.i32      golden argmax labels after accumulation
  fixture_meta.json      human-readable record of sizes/parameters (not parsed by C++)
"""

import argparse
import os
import json
import sys
from pathlib import Path


def dump_f32(path, arr):
    arr = arr.astype('<f4', copy=False)
    with open(path, 'wb') as f:
        f.write(arr.tobytes())


def dump_i64(path, arr):
    arr = arr.astype('<i8', copy=False)
    with open(path, 'wb') as f:
        f.write(arr.tobytes())


def dump_i32(path, arr):
    arr = arr.astype('<i4', copy=False)
    with open(path, 'wb') as f:
        f.write(arr.tobytes())


def build_scene(rng, n_ground=60000, n_objects=20000, extent=12.0):
    """Urban-ish synthetic scene: z=0 ground plane, boxes, noise. Coordinates
    start away from the origin (absolute-like), no exact duplicate distances."""
    pts = []
    # ground: uniform xy, tiny z jitter
    g = rng.random((n_ground, 3)).astype(np.float32)
    g[:, 0] = (g[:, 0] - 0.5) * extent + 100.0
    g[:, 1] = (g[:, 1] - 0.5) * extent + 200.0
    g[:, 2] = (rng.random(n_ground) - 0.5) * 0.02
    pts.append(g)
    # boxes (building-like pillars)
    boxes = [(2.0, 3.0, 2.0, 1.5, 1.5, 4.0),
             (7.0, 6.0, 2.0, 2.0, 2.0, 6.0),
             (5.0, 8.5, 2.0, 1.0, 1.0, 3.0),
             (9.0, 2.0, 2.0, 1.2, 1.2, 5.0)]
    per_box = n_objects // len(boxes)
    for (cx, cy, cz, sx, sy, sz) in boxes:
        b = rng.random((per_box, 3)).astype(np.float32)
        b[:, 0] = cx + (b[:, 0] - 0.5) * sx + 100.0
        b[:, 1] = cy + (b[:, 1] - 0.5) * sy + 200.0
        b[:, 2] = cz + b[:, 2] * sz
        pts.append(b)
    cloud = np.concatenate(pts, axis=0)
    # jitter to kill exact distance ties (KDTree tie-breaking differences)
    cloud += (rng.random(cloud.shape).astype(np.float32) - 0.5) * 1e-4
    return cloud


class FpsRecorder:
    """Wraps FKAConv._farthest_point_sample to record the random start index of
    each stage, so the C++ side can reproduce the exact same support points.
    The rng is proxied so the recorded value is the draw the original used
    (single consumption — the model rng stream stays unchanged)."""

    def __init__(self, original):
        self.original = original
        self.starts = []

    def __call__(self, points, n_samples, rng):
        recorder = self

        class _Proxy:
            def integers(self, *a, **k):
                v = rng.integers(*a, **k)
                recorder.starts.append(int(v))
                return v

        return self.original(points, n_samples, _Proxy())


def dump_case(out_dir, prefix, points, num_points, center_idx, model, adapter, meta):
    rng = np.random.default_rng(1234)

    pre = adapter.preprocess_cloud(points, None, None, 0.08, rng)
    sub_points = pre['sub_points']

    # patch indices (shuffled) from a fixed center, as the regular sampler does
    idxs, center_point = adapter.sample_patch(
        pre, num_points, np.random.default_rng(77), center_idx=center_idx, tolerance=1.0)
    idxs = np.asarray(idxs, dtype=np.int64)

    # the eval augmentation subtracts the per-dim MEAN of the patch points,
    # computed by numpy in float32 (the model's augmenter recenter). Record
    # the exact float32 values: they are the reference subtrahends the C++
    # parity test replays via buildPatch's fixedMean override.
    sampled = sub_points[idxs].copy()
    mean_f32 = sampled.mean(0).astype(np.float32)

    # record the FPS start index of every hierarchy stage, then build the patch
    recorder = FpsRecorder(fkaconv_mod._farthest_point_sample)
    fkaconv_mod._farthest_point_sample = recorder
    try:
        inputs = adapter.build_patch_inputs(pre, idxs, np.random.default_rng(7), 3)
    finally:
        fkaconv_mod._farthest_point_sample = recorder.original

    dump_f32(out_dir / f'{prefix}_xyz.f32', points)
    dump_f32(out_dir / f'{prefix}_sub_points.f32', sub_points)
    dump_f32(out_dir / f'{prefix}_proj_target.f32', sub_points[pre['proj_inds']])
    dump_i64(out_dir / f'{prefix}_idxs.i64', idxs)
    dump_f32(out_dir / f'{prefix}_center.f32', np.asarray(center_point, dtype=np.float32).reshape(3))
    dump_f32(out_dir / f'{prefix}_recenter_mean.f32', mean_f32)
    dump_i64(out_dir / f'{prefix}_fps_start.i64', np.asarray(recorder.starts, dtype=np.int64))
    # points/features arrive as [1, C, k] (channels-first) — dump as-is
    dump_f32(out_dir / f'{prefix}_points_chw.f32', inputs['points'].reshape(-1))
    dump_f32(out_dir / f'{prefix}_features_chw.f32', inputs['features'].reshape(-1))
    for i in range(4):
        # sp arrives as [1, 3, S] (channels-first); dump POINT-MAJOR (S, 3),
        # the layout the C++ hierarchy stores
        dump_f32(out_dir / f'{prefix}_sp{i}.f32', inputs[f'sp{i}'].reshape(3, -1).T)
    for i in range(14):
        dump_i64(out_dir / f'{prefix}_idx{i}.i64', inputs[f'idx{i}'].reshape(-1))

    meta[prefix] = {
        'num_input_points': int(points.shape[0]),
        'num_sub_points': int(sub_points.shape[0]),
        'num_points': num_points,
        'center_idx': center_idx,
        'fps_starts': recorder.starts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--open3dml-src', dest='open3dml_src', default=os.environ.get('OPEN3DML_SRC', ''), help='path to the Open3D-ML src/ directory (or set OPEN3DML_SRC)')
    parser.add_argument('--out-dir', default=str(Path(__file__).resolve().parent / 'data'))
    parser.add_argument('--num-points', type=int, default=4096,
                        help='patch size for the main case (the real model uses 25536; '
                             'tensor construction is size-independent)')
    args = parser.parse_args()

    sys.path.insert(0, args.open3dml_src)
    sys.path.insert(0, str(Path(args.open3dml_src) / 'onnx'))

    global np, fkaconv_mod
    import numpy as np
    from onnx_common import build_adapter, build_model, load_config
    from models import fkaconv as fkaconv_mod

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config('fkaconv_parislille3d.yml')
    cfg.model['in_channels'] = 3
    cfg.model['feature_mode'] = 'xyz'
    model = build_model(cfg, seed=42)
    adapter = build_adapter(model, in_channels=3)

    meta = {'voxel_size': 0.08, 'feature_mode': 'xyz', 'in_channels': 3}

    # main case: scene with far more subsampled points than the patch size
    rng_scene = np.random.default_rng(1234)
    scene = build_scene(rng_scene)
    dump_case(out_dir, 'main', scene, args.num_points, center_idx=0, model=model, adapter=adapter, meta=meta)

    # mini case: 16 distinct points -> every support stage is a plain copy
    # (no FPS except the last), stage shapes match the patch size, and the
    # k-NN neighborhoods contain no duplicate points (duplicate points would
    # tie at distance 0 and make the neighbor ORDER implementation-defined
    # between sklearn and nanoflann). The m < num_points padding path of
    # samplePatch is covered by a dedicated C++ unit test instead.
    rng_mini = np.random.default_rng(5)
    mini = rng_mini.random((16, 3)).astype(np.float32)
    mini[:, 0] *= 3.0
    mini[:, 1] = mini[:, 1] * 3.0 + 50.0
    mini[:, 2] = mini[:, 2] * 3.0 + 100.0
    mini += (rng_mini.random(mini.shape).astype(np.float32) - 0.5) * 1e-3
    dump_case(out_dir, 'mini', mini, num_points=16, center_idx=3, model=model, adapter=adapter, meta=meta)

    # output-decoding case: synthetic logits, two batches over the same 64 rows
    # (exercises repeated accumulation), smooth=0.95 exactly as infer_onnx.py
    rng_dec = np.random.default_rng(99)
    num_classes = 9
    logits = rng_dec.standard_normal((2, 64, num_classes)).astype(np.float32) * 3.0
    # keep classes well separated so argmax is stable against 1-ulp noise
    logits[np.arange(2)[:, None, None], np.arange(64)[None, :, None], rng_dec.integers(0, num_classes, 64)[None, :, None]] += 6.0

    def softmax64(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    smooth = 0.95
    probs = softmax64(logits.astype(np.float64)).astype(np.float32)
    idxs = np.arange(64, dtype=np.int64)
    test_probs = np.zeros((64, num_classes), dtype=np.float32)
    for b in range(2):
        test_probs[idxs] = smooth * test_probs[idxs] + (1 - smooth) * probs[b]
    labels = np.argmax(test_probs, axis=1).astype(np.int32)

    dump_f32(out_dir / 'decode_logits.f32', logits)
    dump_f32(out_dir / 'decode_probs.f32', probs)
    dump_f32(out_dir / 'decode_acc.f32', test_probs)
    dump_i32(out_dir / 'decode_labels.i32', labels)
    dump_i64(out_dir / 'decode_idxs.i64', idxs)
    meta['decode'] = {'batches': 2, 'rows': 64, 'num_classes': num_classes, 'smooth': smooth}

    with open(out_dir / 'fixture_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'fixtures written to {out_dir}')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
