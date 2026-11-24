"""Generate the SphereFormer parity fixtures (test/data/sphere_*).

Run with the stable_py env; Open3D-ML's src/onnx must be importable (hard-coded
paths below, same convention as run_reference.py).

Fixtures:
1. sphere_small  - 2048-pt synthetic lattice with 16 exact duplicate rows and
   PADDED caps (padding rows + phantom windows exercised). All 67 structural
   tensors come from the REAL Python builders (_sphere_patch_tensors with a
   fake attribute chain carrying the real per-level window/quant constants -
   verified == model attributes when the manifests were stamped). Consumed by
   TestSemSeg's table-parity stage: every tensor must match the C++ adapter
   BIT-EXACT (int32) / exactly (float32), incl. the mid-range recenter shift.
2. sphere_lille - a REAL patch (32768 pts) from dijon_9.ply, subsampled with
   the production grid_subsampling(0.05) and selected by the production
   regular-sampler loop until the patch fits the REAL 32k caps; golden logits
   from the REAL 32k ONNX model (onnxruntime CPU).
   Consumed by the gated TestSemSegSphereParity stage (C++ buildPatch ->
   assembleBatch -> session.run -> logits vs golden, argmax equality).
3. Rounding minisuite: sphere_rint (np.round vs C rint on exact ties),
   sphere_fmod (numpy % vs C fmodf on the adapter's non-negative domain),
   sphere_c2s (cart2sphere edge points: axis-aligned, theta wrap, z=0).

Everything is deterministic (seed 42). A debug bundle of all reference tensors
is written to the session temp dir, NOT to test/data (keeps the repo lean).
"""

import json
import sys
import types
from pathlib import Path

import numpy as np

OPEN3DML = Path(os.environ.get("OPEN3DML_ROOT", ""))
sys.path.insert(0, str(OPEN3DML / "src"))
sys.path.insert(0, str(OPEN3DML / "src" / "onnx"))

from onnx_adapters import (_cart2sphere_np, _sphere_patch_tensors,  # noqa: E402
                           _sphere_scatter_mean, _sphere_windows_sorted)

DATA = Path(__file__).resolve().parent / "data"
DEBUG = Path(os.environ.get("SPHERE_DEBUG_DIR", str(Path(tempfile.gettempdir()) / "qsemseg_sphere_debug")))
MODEL_32K = OPEN3DML / "logs" / "onnx" / "sphereformer_parislille3d_32k.onnx"
SIDE = json.loads((OPEN3DML / "logs" / "onnx" / "sphereformer_parislille3d_32k.onnx.meta.json")
                  .read_text(encoding="utf-8"))

QUANT = float(SIDE["preprocessing"]["voxel_size"])  # 0.05 = model.quant_size[0]
WINDOW = SIDE["adapter"]["window"]                  # verified == model attrs
CAPS_32K = SIDE["adapter"]["caps"]                  # the real export capacities


# ---------------------------------------------------------------- fake model
class _Attn:
    def __init__(self, level):
        self.window_size = np.asarray(WINDOW["car_window"][3 * level:3 * level + 3], np.float32)
        self.quant_size = np.asarray(WINDOW["car_quant"][3 * level:3 * level + 3], np.float32)
        self.window_size_sphere = np.asarray(WINDOW["sph_window"][3 * level:3 * level + 3], np.float32)
        self.quant_size_sphere = np.asarray(WINDOW["sph_quant"][3 * level:3 * level + 3], np.float32)


class _UB:
    def __init__(self, level, child=None):
        self.transformer_block = types.SimpleNamespace(attn=_Attn(level))
        self.u = child


_chain = _UB(4)
for _l in (3, 2, 1, 0):
    _chain = _UB(_l, _chain)
MODEL = types.SimpleNamespace(unet=_chain, quant_size=np.asarray([QUANT] * 3, np.float32))

_ATTS = [MODEL.unet.transformer_block.attn,
         MODEL.unet.u.transformer_block.attn,
         MODEL.unet.u.u.transformer_block.attn,
         MODEL.unet.u.u.u.transformer_block.attn,
         MODEL.unet.u.u.u.u.transformer_block.attn]


# ----------------------------------------------------------------- lattices
def recenter(pts_raw):
    """The reference build_patch_inputs mid-range recenter (np.round path)."""
    mn, mx = pts_raw.min(0), pts_raw.max(0)
    shift = (np.round((mn + mx) / 2.0 / QUANT) * QUANT).astype(np.float32)
    return (pts_raw - shift).astype(np.float32), shift


def sheet_cells(n, w, d, seed, z_span):
    """n unique integer (x, y) cells in [0,w)x[0,d) with deterministic
    roughness z in [-z_span, z_span] -> n unique voxel triples."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(w * d, size=n, replace=False)
    x = (idx // d).astype(np.int64)
    y = (idx % d).astype(np.int64)
    z = (x * 7 + y * 13) % (2 * z_span + 1) - z_span
    return np.stack([x, y, z], axis=1).astype(np.int64)


def cells_to_points(cells, seed, dup_count=0):
    """points = c*quant + u, u in [0, 0.4*quant) (the planned lattice recipe).
    dup_count rows are APPENDED as exact copies of earlier rows
    (deterministic) to exercise the duplicate-voxel paths."""
    rng = np.random.default_rng(seed + 1)
    pts = (cells.astype(np.float32) + rng.uniform(0.0, 0.4, size=cells.shape).astype(np.float32)) * np.float32(QUANT)
    if dup_count:
        src = rng.integers(0, len(pts), size=dup_count)
        pts = np.concatenate([pts, pts[src]], axis=0)
    return pts.astype(np.float32)


def level_coords(sampled):
    """The builder's level coordinate hierarchy (voxel descent + scatter-mean).

    Returns (levels, row_counts) where row_counts[l] = the number of rows at
    level l (n0 at level 0, unique parents deeper)."""
    levels = [sampled]
    row_counts = [len(sampled)]
    vox = np.floor((sampled - sampled.min(0)) / np.float32(QUANT)).astype(np.int64)
    for _ in range(4):
        uniqp, invp = np.unique(vox // 2, axis=0, return_inverse=True)
        levels.append(_sphere_scatter_mean(levels[-1], invp, len(uniqp)))
        row_counts.append(len(uniqp))
        vox = uniqp
    return levels, row_counts


def requirements(pts_raw, n0):
    """What the cap checks see, per level/branch (wl, sum(counts^2), n_real)."""
    sampled, _ = recenter(pts_raw)
    levels, row_counts = level_coords(sampled)
    req = {"C": [int(c) for c in row_counts], "W": {}, "M": {}}
    for l in range(5):
        for b in ("car", "sph"):
            x_br = _cart2sphere_np(levels[l].astype(np.float32)) if b == "sph" else levels[l]
            wsize = np.asarray(_ATTS[l].window_size_sphere if b == "sph" else _ATTS[l].window_size, np.float32)
            _, _, counts = _sphere_windows_sorted(x_br, wsize)
            c64 = counts.astype(np.int64)
            req["W"][f"{b}{l}"] = int(c64.shape[0])  # wl (n_pad added by the caller)
            req["M"][f"{b}{l}"] = int((c64 ** 2).sum())
    return req


def builder_caps(req, caps, n0):
    """caps dict for _sphere_patch_tensors from the plugin manifest format."""
    return {"C": [n0] + list(caps["C"][1:]),
            "W": caps["W"],
            "M": caps["M"]}


def check_caps(req, caps, n0):
    """True when every builder cap check passes for these caps."""
    c = builder_caps(req, caps, n0)
    for l in range(1, 5):
        if req["C"][l] > c["C"][l]:
            return False
    for l in range(5):
        n_real = n0 if l == 0 else req["C"][l]
        cap = c["C"][l]
        n_pad = cap - n_real
        for b in ("car", "sph"):
            if req["M"][f"{b}{l}"] > c["M"][f"{b}{l}"]:
                return False
            if req["W"][f"{b}{l}"] > c["W"][f"{b}{l}"]:
                return False
            if req["W"][f"{b}{l}"] + n_pad > c["W"][f"{b}{l}"]:
                return False
    return True


def dump_f32(path, arr):
    np.ascontiguousarray(arr, dtype=np.float32).tofile(path)


def dump_i32(path, arr):
    np.ascontiguousarray(arr, dtype=np.int32).tofile(path)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ minisuite
    rng = np.random.default_rng(7)
    # rint: random values + every exact k+0.5 tie in a range + large magnitudes
    rint_in = np.concatenate([
        rng.uniform(-1000.0, 1000.0, 4000).astype(np.float32),
        (np.arange(-64, 64, dtype=np.float32) + np.float32(0.5)),
        rng.uniform(-0.02, 0.02, 512).astype(np.float32),
    ]).astype(np.float32)
    rint_out = np.round(rint_in).astype(np.float32)
    dump_f32(DATA / "sphere_rint_in.f32", rint_in)
    dump_f32(DATA / "sphere_rint_out.f32", rint_out)

    # fmod: the adapter's domain - dividend >= 0, divisor > 0 (numpy % ==
    # fmodf there). Exact multiples, sub-quant values, window-scale values.
    a = np.concatenate([
        rng.uniform(0.0, 500.0, 4000).astype(np.float32),
        np.arange(0, 64, dtype=np.float32),
        np.array([0.0, 1e-6, 1e6, 40.0, 80.0, 160.0, 0.05, 0.1, 0.15, 0.3], np.float32),
    ]).astype(np.float32)
    b = np.concatenate([
        rng.choice(np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.8, 2.0, 3.0, 4.5, 6.75, 10.125, 80.0], np.float32), a.size),
    ]).astype(np.float32)
    fmod_in = np.stack([a, b], axis=1).astype(np.float32)
    fmod_out = np.mod(fmod_in[:, 0], fmod_in[:, 1]).astype(np.float32)
    dump_f32(DATA / "sphere_fmod_in.f32", fmod_in)
    dump_f32(DATA / "sphere_fmod_out.f32", fmod_out)

    # cart2sphere: axis-aligned / axis-plane points (theta wrap, z=0), plus
    # random shell + random box points
    c2s = [
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],
                  [1, 1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 1], [-1, -1, -1],
                  [1e-6, 1e-6, 1e-6], [1e6, 1e6, 1e6], [0, 0, 5.0], [3.0, 0, -2.0]], np.float32)]
    c2s.append(rng.uniform(-20.0, 20.0, (8192, 3)).astype(np.float32))
    ang = rng.uniform(0, 2 * np.pi, 8192)
    sph = np.stack([np.cos(ang) * rng.uniform(0.1, 30, 8192),
                    np.sin(ang) * rng.uniform(0.1, 30, 8192),
                    rng.uniform(-30, 30, 8192)], axis=1).astype(np.float32)
    c2s.append(sph)
    c2s_in = np.concatenate(c2s).astype(np.float32)
    c2s_out = _cart2sphere_np(c2s_in.copy()).astype(np.float32)
    dump_f32(DATA / "sphere_c2s_in.f32", c2s_in)
    dump_f32(DATA / "sphere_c2s_out.f32", c2s_out)
    print(f"minisuite: rint {rint_in.size}, fmod {fmod_in.shape[0]}, c2s {c2s_in.shape[0]}")

    # ---------------------------------------------------------- small fixture
    n_small = 2048
    cells = sheet_cells(n_small - 16, 96, 64, seed=11, z_span=3)
    pts_small = cells_to_points(cells, seed=11, dup_count=16)
    req_small = requirements(pts_small, n_small)
    # padded caps: C gets ~12% headroom; W must hold wl + n_pad (the builder
    # checks the padding term), so W is padded by the same ABSOLUTE amount as
    # its level's C plus a margin; M only sees real windows (+12% margin)
    pad_c = [req_small["C"][l] + max(8, req_small["C"][l] // 8) for l in range(1, 5)]
    caps_small = {"C": [None] + pad_c, "W": {}, "M": {}}
    for l in range(5):
        n_pad = 0 if l == 0 else pad_c[l - 1] - req_small["C"][l]
        for b in ("car", "sph"):
            wl = req_small["W"][f"{b}{l}"]
            caps_small["W"][f"{b}{l}"] = wl + n_pad + max(8, wl // 8)
            caps_small["M"][f"{b}{l}"] = req_small["M"][f"{b}{l}"] + max(64, req_small["M"][f"{b}{l}"] // 8)
    assert check_caps(req_small, caps_small, n_small)
    sampled_small, shift_small = recenter(pts_small)
    tensors_small = _sphere_patch_tensors(sampled_small, sampled_small.copy(), MODEL,
                                          builder_caps(req_small, caps_small, n_small))
    print(f"small: {n_small} pts, caps C={caps_small['C']}, "
          f"W car0={caps_small['W']['car0']}, M car0={caps_small['M']['car0']}")

    index = {}
    for name, arr in tensors_small.items():
        f = DATA / f"sphere_small_{name}.{'f32' if arr.dtype == np.float32 else 'i32'}"
        (dump_f32 if arr.dtype == np.float32 else dump_i32)(f, arr)
        index[name] = {"file": f.name, "dtype": str(arr.dtype), "shape": list(arr.shape)}
    meta_small = {
        "num_points": n_small,
        "in_channels": 3,
        "num_classes": 9,
        "voxel_size": QUANT,
        "labels": {
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "names": ["ground", "building", "pole-road_sign-traffic_light",
                      "bollard-small_pole", "trash_can", "barrier",
                      "pedestrian", "car", "natural-vegetation"],
            "description": "ParisLille3D label values (fixture copy of the production table).",
        },
        "adapter": {"id": "SphereFormer", "window": WINDOW, "caps": caps_small},
        "shift": [float(v) for v in shift_small],
        "inputs": [{"name": n, **index[n]} for n in tensors_small],
    }
    dump_f32(DATA / "sphere_small_points.f32", pts_small)
    dump_f32(DATA / "sphere_small_shift.f32", shift_small)
    (DATA / "sphere_small_meta.json").write_text(json.dumps(meta_small, indent=2) + "\n", encoding="utf-8")
    print(f"small fixture: {len(tensors_small)} tensors written")

    # --------------------------------------------------------- lille fixture
    # A synthetic lattice could not fit the REAL 32k caps (level-1 parent
    # survival of a flat sheet is too high vs the real statistics). Use the
    # REAL dijon_9 cloud instead, preprocessed exactly like production
    # (grid_subsampling at 0.05), and walk the production regular-sampler
    # patch sequence until a patch fits the caps (each attempt is cheap
    # numpy; the possibility bump on failure mirrors the degenerate path).
    from sklearn.neighbors import KDTree
    from datasets.utils.data_processing import DataProcessing
    from utils.pcd_utils import PCD

    import os as _os
    _cloud = _os.environ.get("SPHERE_FIXTURE_CLOUD", "")
    if not _cloud:
        raise SystemExit("set SPHERE_FIXTURE_CLOUD to the dijon_9.ply path (e.g. /path/to/dijon_9.ply)")
    pcd = PCD(_cloud)
    data = pcd.get_data_dict(None)
    points = np.asarray(data["point"][:, 0:3], dtype=np.float32).copy()
    print(f"dijon_9: {points.shape[0]} original points")
    sub_points = DataProcessing.grid_subsampling(points, grid_size=np.float32(QUANT))
    print(f"subsampled: {sub_points.shape[0]} points")
    tree = KDTree(sub_points)

    caps_real = {"C": [None] + list(CAPS_32K["C"][1:]), "W": CAPS_32K["W"], "M": CAPS_32K["M"]}
    m_sub = sub_points.shape[0]
    rng = np.random.default_rng(42)
    possibilities = rng.random(m_sub) * 1e-3
    chosen = None
    for attempt in range(200):
        center_idx = int(np.argmin(possibilities))
        idxs = tree.query(sub_points[center_idx:center_idx + 1], k=32768)[1][0]
        rng.shuffle(idxs)
        pts_patch = sub_points[idxs].astype(np.float32)
        req = requirements(pts_patch, 32768)
        if check_caps(req, caps_real, 32768):
            chosen = (attempt, idxs.astype(np.int64), pts_patch, req)
            print(f"patch {attempt}: FITS the real caps")
            break
        possibilities[center_idx] += 0.001  # degenerate path: next argmin
        if attempt % 20 == 0:
            worst = max(((f"C{l}", req["C"][l], caps_real["C"][l]) for l in range(1, 5)),
                        key=lambda t: t[1] / t[2])
            print(f"patch {attempt}: caps exceeded (worst {worst[0]} {worst[1]}/{worst[2]}), retrying")
    assert chosen, "no dijon_9 patch fits the real 32k caps in 200 attempts"
    attempt, idxs_lille, pts_lille, req_lille = chosen

    sampled_l, shift_l = recenter(pts_lille)
    tensors_lille = _sphere_patch_tensors(sampled_l, sampled_l.copy(), MODEL,
                                          builder_caps(req_lille, caps_real, 32768))
    print(f"lille: patch {attempt} (center idx {idxs_lille[0]} after shuffle); tensors {len(tensors_lille)}")

    # golden logits from the REAL 32k model (CPU)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(MODEL_32K), providers=["CPUExecutionProvider"])
    graph_inputs = {i.name: i for i in sess.get_inputs()}
    feeds = {}
    want_dtype = {"tensor(float)": np.float32, "tensor(int32)": np.int32}
    for name, arr in tensors_lille.items():
        assert name in graph_inputs, f"graph has no input '{name}'"
        want = graph_inputs[name]
        assert list(arr.shape) == list(want.shape), (name, arr.shape, want.shape)
        assert arr.dtype == want_dtype[want.type], (name, arr.dtype, want.type)
        feeds[name] = arr
    logits = sess.run(None, feeds)[0]
    print(f"logits: shape={logits.shape} dtype={logits.dtype} "
          f"finite={np.isfinite(logits).all()}")

    dump_f32(DATA / "sphere_lille_points.f32", pts_lille)
    dump_f32(DATA / "sphere_lille_shift.f32", shift_l)
    dump_f32(DATA / "sphere_lille_logits.f32", logits)
    meta_lille = {
        "num_points": 32768,
        "in_channels": 3,
        "num_classes": 9,
        "voxel_size": QUANT,
        "source": {"cloud": "see SPHERE_FIXTURE_CLOUD",
                   "subsampled": "grid_subsampling 0.05 (DataProcessing)",
                   "patch": f"regular-sampler attempt {attempt}, kNN k=32768, seed 42"},
        "adapter": {"id": "SphereFormer", "window": WINDOW, "caps": caps_real},
        "shift": [float(v) for v in shift_l],
        "logits_shape": list(logits.shape),
        "inputs": [{"name": n, "dtype": str(a.dtype), "shape": list(a.shape)} for n, a in tensors_lille.items()],
    }
    (DATA / "sphere_lille_meta.json").write_text(json.dumps(meta_lille, indent=2) + "\n", encoding="utf-8")

    # debug bundle (NOT committed): every reference tensor for diffing
    bundle = DEBUG / "sphere_lille_tensors.bin"
    offsets = {}
    blob = bytearray()
    for name, arr in tensors_lille.items():
        raw = np.ascontiguousarray(arr).tobytes()
        offsets[name] = {"offset": len(blob), "bytes": len(raw),
                         "dtype": str(arr.dtype), "shape": list(arr.shape)}
        blob += raw
    bundle.write_bytes(bytes(blob))
    (DEBUG / "sphere_lille_tensors.json").write_text(json.dumps(offsets, indent=2), encoding="utf-8")
    print(f"debug bundle: {bundle} ({len(blob)} bytes)")

    # also dump the small tensors for C++ diffing
    bundle_s = DEBUG / "sphere_small_tensors.bin"
    offsets_s = {}
    blob_s = bytearray()
    for name, arr in tensors_small.items():
        raw = np.ascontiguousarray(arr).tobytes()
        offsets_s[name] = {"offset": len(blob_s), "bytes": len(raw),
                           "dtype": str(arr.dtype), "shape": list(arr.shape)}
        blob_s += raw
    bundle_s.write_bytes(bytes(blob_s))
    (DEBUG / "sphere_small_tensors.json").write_text(json.dumps(offsets_s, indent=2), encoding="utf-8")

    print("DONE")


if __name__ == "__main__":
    main()
