# qSemSeg — ONNX semantic segmentation plugin

Applies a pretrained semantic segmentation model (ONNX format, e.g. **FKAConv**
exported from a training pipeline such as Open3D-ML/LightConvPoint) to the
point cloud currently selected in CloudCompare.

Menu entry: **Plugins → qSemSeg → Apply Semantic Segmentation Model**
(enabled when a point cloud is selected).

> **Repository role:** this repository is the home of the plugin source.
> CloudCompare includes it as a git submodule at
> `plugins/core/Standard/qSemSeg`, so a normal
> `git clone --recurse-submodules` checkout builds it in place (enable it with
> `-DPLUGIN_STANDARD_QSEMSEG=ON`). The plugin can also be built against any
> CloudCompare source tree by copying this directory to that location.

The predicted labels are added to the cloud as a new scalar field named
`SemSeg.label` (versioned on reruns: `SemSeg.label (2)`, ...) together with a
`SemSeg.confidence` field, and displayed immediately.

**Determinism:** running the same model on the same cloud with the same
parameters always produces the exact same labels — by design. The per-point
coverage weights start at a small **seeded** random jitter (the *Random seed*
field, default 42 — same as the reference pipeline's
`rng.random(n) * 1e-3`), patch centers are then picked greedily as the
least-covered point, and nothing is cached between runs. The same seed
replays the identical patch sequence, so labels are fully reproducible; a
different seed re-orders the coverage walk (the first patch center is
effectively random) and can flip a few borderline points near class
boundaries, while the overall segmentation stays stable. The model itself
contributes no randomness — its weights are fixed in the `.onnx` file.

## Build

```shell
cmake .. -DPLUGIN_STANDARD_QSEMSEG=ON
cmake --build . --parallel
```

The plugin has **no build-time dependency on ONNX Runtime**: the ONNX Runtime
shared library is loaded dynamically at runtime (`LoadLibrary`/`dlopen` +
`OrtGetApiBase`, C API vendored in `third_party/onnxruntime_c_api.h`), so the
plugin always compiles, on every platform and compiler.

The Ubuntu CI job builds the plugin and its test suite against Qt 6.4: wrap
`std::string` values in `QString::fromStdString(...)` before passing them to
`QString::arg(...)` (Qt 6.11 converts implicitly, Qt 6.4 does not).

Optionally, deploy the ONNX Runtime shared libraries next to the apps at
install time:

```shell
cmake .. -DPLUGIN_STANDARD_QSEMSEG=ON -DONNXRUNTIME_DLL_DIR=<ort>/lib
```

## Runtime deployment (Windows + GPUs)

The plugin needs the ONNX Runtime **CPU** provider to run at all; a GPU is an
optional accelerator. Two GPU runtimes exist — pick one:

- **CUDA** (NVIDIA only): download the ONNX Runtime **GPU** package, e.g.
`onnxruntime-win-x64-gpu-1.23.2.zip` from
<https://github.com/microsoft/onnxruntime/releases>, and copy next to
`CloudCompare.exe`:

```
onnxruntime.dll
onnxruntime_providers_shared.dll
onnxruntime_providers_cuda.dll
```

For GPU execution the NVIDIA user-mode runtime DLLs must sit next to the exe
as well (`cudart64_12.dll`, `cudnn64_9.dll`, `cublas64_12.dll`, `cufft64_11.dll`,
`curand64_10.dll`, `cusolver64_11.dll`, `cusparse64_12.dll`,
`nvjitlink64_12.dll`, `nvrtc64_120_0.dll`, ... — available from the pip wheels
`nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, ...). The target PC only needs
an NVIDIA driver, not the CUDA toolkit.

- **DirectML** (any DirectX 12 GPU, Windows): see the next-but-one section.

When the selected GPU provider cannot be initialized (no GPU, missing DLLs,
broken driver, ...), *Auto* warns in the console and falls back to the CPU
execution provider; the pinned choices abort with the reason instead.

### Bundling the CPU runtime (what CI ships)

The **CPU** runtime needs no manual download on machines that get their
CloudCompare build from this repository's CI: both the Windows job and the
Ubuntu jobs download the official ONNX Runtime CPU release (version pinned in
`.github/workflows/build.yml`, SHA-256-verified), extract it, and point the
plugin's install-time hook at it:

```cmake
# plugins/core/Standard/qSemSeg/CMakeLists.txt
-DONNXRUNTIME_DLL_DIR=<folder with onnxruntime*.dll / libonnxruntime*.so*>
```

`cmake --install` then copies the runtime next to `CloudCompare.exe` /
`ccViewer` (Windows) or `<prefix>/bin` (Linux), and the Windows job additionally
runs `qSemSegProbe.exe` against the committed tiny test model (see Tests) as an
end-to-end smoke of the bundled runtime. Packagers can use the same hook for
their own builds; a CUDA runtime stays an optional manual download (above).

### DirectML (any DX12 GPU, Windows)

Official ONNX Runtime releases carry **either** the CUDA provider **or** the
DirectML provider - a single `onnxruntime.dll` cannot offer both. To use
DirectML, download the DML package, e.g.
`onnxruntime-win-x64-dml-1.23.0.zip` from
<https://github.com/microsoft/onnxruntime/releases>, and copy next to
`CloudCompare.exe`:

```
onnxruntime.dll
onnxruntime_providers_shared.dll
DirectML.dll
```

No vendor runtime DLLs are needed and no CUDA-capable GPU is required: DirectML
runs on any DirectX 12 GPU (NVIDIA, AMD, Intel). The plugin asks the loaded
runtime which providers it offers at run time, so the same plugin binary works
with either runtime - a DML runtime simply never sees a CUDA candidate and vice
versa. The same `ONNXRUNTIME_DLL_DIR` install hook bundles the DML files when it
points at the extracted DML package.

DirectML sessions are created with `ep.dml.disable_graph_fusion=1`: the default
DirectML graph-fusion pass rejects large exported graphs at session creation
(`DmlGraphFusionHelper.cpp ... E_INVALIDARG`, reproduced with the SphereFormer
export), while the unfused path is numerically equivalent and fast (measured on
the real 32k patch: max |logit diff| 5e-4 vs the CPU EP, identical argmax;
2.3-2.6x faster than the CPU EP there). FKAConv models run on DirectML with
output identical to the CPU EP.

### Choosing where the model runs

The dialog has a single *Execution* control with four choices:

- **Auto (recommended)** - resolves the provider with these exact rules:
  1. If the manifest declares `execution.provider = "cpu"`, the CPU is used
     and no GPU is probed.
  2. Otherwise the GPU candidate list is: `["cuda"]` for a `cuda` hint,
     `["dml"]` for a `dml` hint, and `[CUDA, DirectML]` for `auto` / `gpu` /
     no hint (CUDA first: longest-supported in this plugin).
  3. Each candidate is accepted only when (a) the deployed ONNX Runtime lists
     that provider, (b) the per-model failure cache does not already name it,
     and (c) its probe verdict for this runtime version is "ok" - or a fresh
     probe now passes. The probe is the `qSemSegProbe` executable in its own
     process: ONNX Runtime load, session creation with that provider,
     manifest/IO validation and one warm-up inference on zero-filled inputs.
     A native crash in a broken driver stack therefore kills the probe only,
     never CloudCompare. A clean probe failure is retried once before the
     verdict is cached, so a transient GPU-memory failure does not disable
     the provider until the ONNX Runtime changes.
  4. The first candidate that passes is used, with the CPU appended so ONNX
     Runtime can still fall back at session creation. If the session lands on
     the CPU anyway, the console says why.
  5. If no candidate passes, the run uses the CPU and the console prints one
     line per skipped candidate naming the reason (provider not deployed,
     cached failure, probe error).
  Verdicts are cached per provider and ONNX Runtime identity (QSettings
  `qSemSeg/gpuProbe_<provider>`: the version plus the loaded library path, so
  swapping the CUDA package for the DirectML package re-probes) and re-checked
  after a runtime update.
- **CPU only** - never touches the GPU (no probe is run).
- **CUDA only** - NVIDIA CUDA; aborts with the probe/session error when that
  provider is not deployed or fails the probe (never silently runs on the CPU).
- **DirectML only** - DirectML; same abort behavior.

The dialog reflects the deployed runtime: ONNX Runtime is loaded *before* the
dialog opens, and a pinned choice this runtime cannot provide is greyed out and
labelled "not in this ONNX Runtime build" (its tooltip names the package to
deploy), with a "This ONNX Runtime build provides: ..." line under the combo. A
saved choice that is no longer available falls back to *Auto* instead of
aborting after Apply. One `onnxruntime.dll` carries the CUDA or the DirectML
family, never both, so a given deployment only offers the matching pin.

**Broken NVIDIA driver note:** some recent NVIDIA driver releases (e.g. the
621.xx series shipping CUDA UMD 13.5) crash inside `cudnnCreate` during CUDA
session initialization. With *Auto* this is detected by the out-of-process
probe (the probe process exits abnormally; CloudCompare is unaffected) and the
next candidate / the CPU is used automatically — no manual switch needed. The
cached verdict then explains every later run ("the GPU probe previously failed
on this machine with ONNX Runtime ...").

Even after a successful GPU probe, the in-process session runs one warm-up
inference on zero-filled inputs before the real run, so ORT-level failures
(driver DLLs present but broken, EP/graph incompatibilities) surface as a
catchable console error before any data is processed.

**Model-declared execution.** A manifest (sidecar or embedded metadata) may
pin the execution policy with `"execution": { "provider": "cpu" }` — useful for
exports whose GPU kernels are known to produce NaN/Inf or to be far slower
than the CPU (the SphereFormer exports are ~70x slower on CUDA and return
NaN/Inf on real data). It may also name a GPU family (`"cuda"` or `"dml"`) to
say which provider its kernels were validated on. In *Auto* the hint
**restricts the GPU candidates to the named family** — the other GPU family is
never tried with that model (an unvalidated path is not used; the run goes to
the CPU instead) — and it never overrides the probe. The dialog always wins:
the hint only shapes *Auto*, it can never force the CPU when the user pinned
*CUDA only* or *DirectML only*.

Even without any hint a GPU run is not blindly trusted: if the active provider
returns NaN/Inf for real data (the all-zero warm-up can pass while real patches
do not), *Auto* recreates the session on the CPU, reruns the current cloud and
remembers the model per provider (QSettings
`qSemSeg/gpuModelFail_<provider>_<fingerprint>`, keyed by the model file and
the ONNX Runtime identity) so later runs go straight to the next candidate /
the CPU. Delete the `qSemSeg/gpuModelFail_*` value(s) to re-check the GPU after
a driver update. While a cloud runs, the progress
dialog shows a ticking elapsed clock, the patch counter and the covered
fraction — a single forward pass on a slow provider can take minutes, so the
label (not just the bar) is the liveness signal.

The plugin locates `onnxruntime.dll` by trying, in order: the application
directory, its `plugins` subdirectory, the sibling `CloudCompare` deployment
folders, and only then the OS search path. A library that is too old for the
vendored API header is recorded and skipped (the search continues), and a
failed search reports every candidate tried with its version plus the runtime
download URL.

### Performance notes

With the default *Patch batch size* of "auto", low GPU utilization (~30%) and a
modest speedup over CPU are **normal**: each forward pass is small (one to a
few patches of ~25k points), the greedy sliding-window loop is serial, and most
wall-clock time goes to CPU-side patch building — the per-patch FKAConv
hierarchy (farthest-point sampling) runs on the CPU, so the GPU only works
during the conv forward between those CPU steps.

The *Patch batch size* groups several patches into one forward pass (the
model's batch axis is dynamic). The default value 0 means **auto**: at run
start the plugin times zero-filled forward passes for increasing batch sizes
(1–32 on GPU, 1–8 on CPU, one warm-up run per size) and keeps the size with the
best per-patch time — a larger size only wins when it is at least 10% faster
per patch, and the selected value is printed to the console. This does not
change the patch sequence — coverage updates are applied before the forward
pass and never depend on its output — so the labels stay equivalent (at most
tiny float-level batch effects). Set an explicit value (1–64) to skip the
calibration.

CPU-side patch building is parallelized where it is exact-output-preserving:
the k-nearest-neighbor batches of the FKAConv hierarchy query the shared
KD-tree from a pool of worker threads (disjoint query ranges, identical
results), and the two identical neighbor-table calls (`idx0`/`idx1`) are
computed once and reused. The farthest-point-sampling loop stays serial by
design — a thread round-trip per FPS iteration would cost more than the
arithmetic it would save.

### Memory and the patch point count

`num_points` (points per patch) is a **dynamic** model input, so ONNX Runtime
accepts any value — but the model's per-patch working buffers grow with it,
multiplied by the patch batch size (FKAConv's kernel block allocates several
`[batch, 16, num_points, 16]` float32 buffers per forward pass). The trained
value (shown on the model card, e.g. 25536) is a few hundred MB of peak
memory; raising the patch size multiplies every one of those buffers, and
raising it far enough makes a single buffer exceed the available RAM/VRAM —
ONNX Runtime then fails a node with `Failed to allocate memory for requested
buffer of size N` (N in bytes).

When that happens during the **batch-size calibration**, a candidate size
simply fails and the plugin continues with the largest working size — the
console shows the `[E:onnxruntime]` log line from the failed attempt plus a
`Batch-size calibration note` naming the failed sizes, the chosen size and
the original error. When it happens during a **run** (warm-up or inference),
the run aborts and the error message carries a hint: lower the points-per-
patch toward the model's trained value and/or set the patch batch size to 1.
The Advanced expander shows an inline note whenever the patch size exceeds
the model default.

### The dialog and the model card

Picking a model file shows a **model card** above the parameters — everything
about the run is derived from the model, and the card tells you what was
found:

- model name, class count and class names (when the manifest/embedded
  metadata provides them),
- patch size (points per patch) and voxel size,
- required features (coordinates / intensity / colors) and which intensity
  scalar field of the selected cloud would be used,
- knowledge source for every line: sidecar manifest, embedded metadata, or
  "default (assumed)" — assumptions are highlighted,
- the warning `loadModelSpec` produced, if any,
- ONNX Runtime status (loaded version + provider plan) once known.

Details (voxel size, patch points, batch size, seed) are collapsed into an
**Advanced** expander, pre-filled from the model; values that differ from the
model default get an inline note. The card refreshes when the model path loses
focus (`editingFinished`, not per keystroke — the card reads the potentially
40 MB model file) and when the selected cloud changes.

The dialog also has a **Validate model** button (contributor on-ramp): it runs
every compatibility check for the selected model — manifest parsing, model
architecture resolution, the graph IO contract (when ONNX Runtime is loadable)
and the selected cloud's feature availability — without inferring, and shows a
pass/fail report. See "Adding a new model" below.

### Multiple clouds

Every **selected** point cloud is processed one after another (the dialog opens
once; the session, warm-up and batch calibration run once). The progress bar
spans all clouds. Per-cloud problems — an empty cloud, a feature mismatch, an
inference error — are reported as console warnings while the run continues with
the remaining clouds; after the run, a summary line reports how many clouds
were processed and each failure is shown as an error (one closable popup per
failed cloud). Cancel stops the whole run; clouds finished before it keep
their results.

## Outputs

A model package is a directory (or pair of files) containing:

- `model.onnx` — the exported model. Inputs are discovered from the session.
- `model.onnx.meta.json` — manifest written next to the model (by Open3D-ML's
  `convert_onnx.py`). The dialog pre-fills the voxel size and patch point count
  from it; the pipeline reads everything else from it.

### Self-describing models (embedded metadata)

Since `convert_onnx.py` stamps the manifest into the ONNX file itself, a
**single `.onnx` file is fully self-describing**: the manifest JSON is stored
in the model's `metadata_props` under the key `semseg.manifest` (same schema as
the sidecar file — `convert_onnx.py` embeds the complete manifest, and the
standalone `src/onnx/stamp_manifest.py` can add it to an already-exported
model).

The plugin reads the embedded manifest in two ways:

- **file scan** (before any session exists) — a small protobuf wire-format
  parser walks the ModelProto's `metadata_props` records directly in the file,
  so the dialog's model card can show the classes, patch size, voxel size and
  feature requirements of a not-yet-loaded model. ORT-independent, so it is
  also exercised in the no-runtime CI tests.
- **session metadata** — via the ONNX Runtime model-metadata API right after
  session creation, as an idempotent verification of the file scan.

**Precedence order for model knowledge** (strict):

1. session IO contract (always wins)
2. sidecar `model.onnx.meta.json`
3. embedded `semseg.manifest` metadata
4. built-in defaults — every field resolved here is reported in the console
   with an explicit "assuming ..." warning naming the default.

Each field is tracked individually: embedded metadata only fills fields that
no higher-precedence source provided, a sidecar overrides embedded values, and
fields the user set in the dialog are never clobbered. A manifest-less model
without embedded metadata still works, but warns loudly about every assumed
default.

### Manifest schema (fields the plugin consumes)

```jsonc
{
  "schema_version": "1.0",               // optional; absent = "1.x" (current schema).
                                         // An unknown MAJOR version is a HARD error:
                                         // the manifest is ignored and the plugin refuses to run.
  "model_name": "FKAConv",               // selects the registered adapter:
                                         // "FKAConv" | "PointNet". Unknown names are a HARD error.
  "in_channels": 3,
  "num_classes": 9,
  "num_points": 25536,
  "output": {                            // optional (default: logits)
    "kind": "logits"                     // "logits" (softmaxed by the plugin) | "probabilities"
                                         // (already normalized, used as-is). Unknown values are a HARD error.
  },
  "execution": {                         // optional (default: auto)
    "provider": "auto"                   // auto | cpu | cuda | dml | gpu.
                                         // "cpu": Auto uses the CPU and skips the
                                         // GPU probes (known-bad or far slower GPU
                                         // kernels). "cuda" / "dml": restrict Auto
                                         // to that GPU family (the other family is
                                         // not tried; CPU fallback). "gpu": CUDA,
                                         // then DirectML. An explicit dialog
                                         // choice always wins. Unknown values are
                                         // a HARD error.
  },
  "adapter": {
    "id": "PointNet",                    // optional: pins the adapter (code path)
                                         // feeding the model (registry key),
                                         // independent of a descriptive
                                         // model_name. Unknown ids are a HARD error.
    "stage_npoints": [256, 64, 16, 4],   // FKAConv hierarchy support sizes
    "num_neighbors": 16,                 // K of the KNN hierarchy
    "feature_mode": "xyz",               // xyz | i | rgb | i_rgb | feat_xyz
    "window": { ... }, "caps": { ... }   // SphereFormer only: per-level
                                         // window/quant constants + the graph's
                                         // fixed buffer sizes (see the
                                         // SphereFormer section)
  },
  "preprocessing": {
    "voxel_size": 0.08,                  // <= 0 disables subsampling
    "subsampling": {                     // optional (default: voxel)
      "type": "voxel",                   // voxel | none | random | fps
      "target": 5000                     // required for random/fps: subset size
    },
    "shift_min": false,                  // PointNet: subtract the cloud min before subsampling
    "augment": {                         // eval-time per-patch augmentation
      "recenter": { "dim": [0, 1, 2] },  // subtract the per-dim mean of the patch points
      "normalize": { ... }               // subtract the mean again, divide by the largest range
    },
    "feature_columns": []                // extra feature columns in training order, e.g. ["intensity","rgb"]
  },
  "labels": {                            // class index -> dataset label value
    "values": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "names": ["ground", "building", "..."]
  }
}
```

**Hard manifest errors** (`schema_version` with an unknown major version,
unknown `output.kind`, unknown `execution.provider`) make the model
un-runnable: the dialog disables Apply, the console lists the errors and the
run aborts. They are never downgraded to warnings because guessing the schema,
the output semantics or the execution policy would silently produce wrong
labels.

### Label mapping

The model outputs **class indices** (`0..num_classes-1`, the dataset's
non-ignored labels in sorted order), while the dataset itself may use
different label values — e.g. Paris-Lille3D reserves `0` for *unclassified*
(the ignored label), so the 9 model classes map to dataset labels `1..9`.
When the manifest carries a `labels.values` table, the plugin maps every
predicted class index through it **before** writing the scalar field, so
`SemSeg.label` holds true dataset labels that survive round-trips through
the dataset's own readers/writers. Without the table the raw indices are
written (documented fallback).

### Feature sources

- `xyz` / `feat_xyz`: coordinates of the **recentered** patch points — for
  FKAConv with `feature_mode=xyz`, the `features` input is exactly the
  recentered `points` input (the Python reference builds features from the
  *augmented* points, never from absolute coordinates).
- `i`/`intensity`: the cloud's currently displayed scalar field, rescaled to
  `[0, 1]` by dividing by 255 when the values are in the 0..255 range
  (matching the reference `DataProcessing` normalization).
- `rgb`: cloud colors, already 0..1 in CloudCompare.
- `feature_columns` in the manifest overrides the default column order; the
  plugin fails loudly when a required column is missing from the cloud.

### Compatibility validation

`loadModelSpec` + `validateSpecAgainstIO` + `validateCloudFeatures` cross-check
the manifest against (a) the ONNX session's actual input/output names, dtypes
and shapes, and (b) the selected cloud's available scalar fields/colors. Any
mismatch - missing input tensor, unexpected dtype, unsupported feature mode,
missing intensity/color column - aborts the run with an explicit error message
instead of silently producing wrong labels.

**Fixed-shape models**: a model exported with a *static* `num_points` axis
(concrete value instead of a dynamic placeholder) only accepts exactly that
many points per patch. The session contract is authoritative:
`validateSpecAgainstIO` records the value in `spec.staticNumPoints` and the
run uses it for patch sampling (the requested patch size / tolerance are
ignored for such models). When the static axis disagrees with the manifest
`num_points`, a console warning names both values - the graph value wins,
never silently.

### Model adapters

The manifest's `model_name` selects a registered **model adapter** — the
component that knows how one model architecture is fed. The sliding-window
math (patch sampling, coverage walk, accumulation, projection) is
architecture-independent and shared; the adapter owns everything the MODEL
decides:

- **FKAConv** — 4-stage hierarchy + 14 `idx*` inputs + `points`/`features`
  tensors; requires `adapter.stage_npoints` in the manifest; keeps the cloud
  minimum (`shift_min` defaults to *false*).
- **PointNet** — generic single-tensor architecture: every input must be a
  float32 `points` or `features` tensor (CHW), the output must be
  `[batch, num_points, num_classes]`; no hierarchy inputs.
  `shift_min` defaults to *true*. Most PointNet-style community models run
  with zero C++ code: declare `model_name` freely (e.g. `"Open3D-ML-Small"`)
  and pin the architecture with `"adapter": {"id": "PointNet"}` in the
  manifest — the registry resolves `adapter.id` BEFORE `model_name`.
- **SphereFormer** — fixed-shape, fixed-batch exports of Open3D-ML's
  `SphereFormerSemSeg`: 67 inputs per patch (`feat` + 5 levels of
  SubMConv/window-attention tables + stride-2 down/inverse gathers), all
  index tables precomputed per patch by the adapter. Requires the
  `adapter.window` and `adapter.caps` manifest blocks (see below). The
  export bakes `batch = 1` and `num_points`; exceeding a capacity is a loud
  "re-export with larger caps" error, never a silent truncation.

An unknown `model_name` / `adapter.id` is a **loud validation error** naming
the supported architectures — the plugin never silently falls back to another
architecture's code path (a guessed patch format would produce wrong labels,
not a crash). Contributed models with a new graph layout need a new adapter
(see `ModelAdapter.h`).

#### SphereFormer exports (fixed shape, batch 1)

The graph moves ALL structure out of ONNX: the adapter computes the 66 int32
index tables (voxel grid, window partitions, SubMConv in-row gathers,
stride-2 down/inverse gathers) plus the float32 `feat` tensor per patch,
exactly as Open3D-ML's `_sphere_patch_tensors` + `spconv_plugin` table
builders do. Two manifest blocks carry the constants the graph cannot
re-derive:

```jsonc
"adapter": {
  "id": "SphereFormer",
  "window": {                        // 5 levels x 3 axes (model attributes)
    "car_window": [...], "car_quant": [...],
    "sph_window": [...], "sph_quant": [...]
  },
  "caps": {                          // the graph's fixed buffer sizes
    "C": [null, 28059, 16844, 7660, 3259],      // rows, levels 1..4
    "W": { "car0": 17658, ..., "sph4": 11509 }, // windows per branch/level
    "M": { "car0": 3201060, ..., "sph4": 24878 } // attention pairs
  }
}
```

Both blocks are REQUIRED — guessing a window partition would silently feed
the attention a layout the export cannot represent — and **Validate model**
checks them against the graph's declared input shapes. Notes:

- The patch point count is imposed by the export's `feat` row count; a
  differing dialog value is ignored with a warning (the graph wins).
- The batch axis is baked to 1, so the batch size is forced to 1 (auto
  calibration is skipped with a console note). One patch per forward.
- The 32k export needs several GB of session memory on CPU; allocation
  failures surface through the usual memory hint.
- Preprocessing: `shift_min: false` (origin-anchored voxel subsampling) and
  `augment.recenter: false` (the adapter does its own mid-range recentering,
  snapped to `model.quant_size`).

### Adding a new model

The fast path needs no C++: any model whose graph takes float32 `points` [+
`features`] tensors and produces a `[batch, num_points, num_classes]` output
runs through the **PointNet** adapter.

1. **Export the ONNX model** with dynamic `batch` and `num_points` axes (a
   static `num_points` axis also works — the patch size is then imposed by
   the graph, see Compatibility validation above).
2. **Describe the model** to the plugin — either way works:
   - **recommended**: stamp the manifest INTO the model (Open3D-ML's
     `convert_onnx.py` does it; standalone: `src/onnx/stamp_manifest.py`) so
     a single `.onnx` file is fully self-describing, or
   - ship a **sidecar manifest** next to it named
     `<model-file-name>.onnx.meta.json`.
   Start from `test/template.meta.json` (a complete PointNet example);
   the field reference is the manifest schema section above.
3. **Check it**: the dialog's **Validate model** button runs every
   compatibility check without inferring — manifest parsing, model
   architecture resolution, the graph IO contract (when ONNX Runtime is
   loadable) and the selected cloud's feature availability — and shows a
   pass/fail report. When the runtime is not loadable, the session checks
   are skipped with a note and the other checks still run.

A model architecture whose graph needs different inputs (e.g. hierarchy
indices like FKAConv's) needs a new adapter: subclass `ModelAdapter`,
implement `applyDefaults` / `validate` / `buildPatch` / `assembleBatch` and
register it in `registry()` (`ModelAdapter.cpp`); `Validate model` then
covers it automatically.

## Outputs

- **`SemSeg.label`** — the predicted per-point labels (mapped to dataset label
  values when the manifest provides the table). The name is **versioned**:
  rerunning the plugin creates `SemSeg.label (2)`, `(3)`, ... and never deletes
  a previous result. The new field is displayed immediately.
- **`SemSeg.confidence`** — the accumulated probability of the winning class
  (row maximum of the smoothed `test_probs`, converging toward 1 as the
  sliding-window coverage grows). Low values flag points covered by few
  patches or sitting near class boundaries.
- **per-class palette** — on a *colorless* cloud whose model provides class
  names, the points are colored with a deterministic per-class palette (fixed
  saturation/value, hues spaced by the golden ratio; the original colors of a
  colored cloud are never overwritten). CloudCompare's show-colors toggle
  switches between the palette and the displayed scalar field.
- **named color scale** — when the model provides class names, the label
  scalar field gets a discrete per-class color scale (one solid band per
  class in the palette color, ticked with the dataset label + class name)
  instead of CloudCompare's default numeric ramp; the 3D view's color-ramp
  legend shows the class names. Requires at least 2 named classes.
- **class distribution** — the console prints a per-class table (dataset label,
  class name, point count, percentage) for every class that actually got
  predicted.

## Pipeline

1. Voxel-grid subsampling of the cloud (barycenter per voxel) with a
   nearest-neighbor projection map back to the original points. Voxel keys are
   computed in **double precision** (`floor((double)p / (double)voxelSize)`),
   matching the reference implementation exactly — single-precision keys put
   boundary points into the wrong voxel and change the subsampled point count.
2. Sliding-window inference: patch center = least-covered point
   (`argmin(possibilities)`), k-NN patch of `num_points`, coverage update
   `(1 - d/max)^2`, batched ONNX forward passes, probability accumulation
   `0.95 * old + 0.05 * softmax(logits)`. Stops when `min(possibilities) > 0.5`.
3. Per-patch inputs: eval augmentation from the manifest (`recenter` /
   `normalize`), then `points` / `features`, FPS support points (`sp0..sp3`)
   and the KNN index hierarchy (`idx0..idx13`, int64). The recenter mean is
   accumulated in double precision and rounded to float32 (≤ ~1e-4 difference
   from numpy's own float32 accumulation, which is implementation-defined;
   see Tests).
4. Final labels = `argmax(test_probs)` mapped back to the original cloud
   through `proj_inds`, then through the manifest's label table.

## Tests

`test/` holds a parity test suite (`TestSemSeg.cpp`, registered with CTest as
`TestSemSeg` plus the run-parity test `TestSemSegRunParity`, built when
`BUILD_TESTING=ON`):

1. **Tensor parity** — rebuilds the full FKAConv ONNX input set (`points`,
   `features`, `sp0..sp3`, `idx0..idx13`) from fixture clouds dumped by
   `test/make_fixture.py` (which runs the real Open3D-ML reference pipeline)
   and compares element-by-element against the reference dumps.
2. **Patch sampler** — regular path and the `m < num_points` padding path.
3. **Output decoding** — float64 softmax, `0.95*old + 0.05*new` accumulation,
   argmax and the argmax confidence (`maxProbabilities`) against reference
   values.
4. **Point-to-label mapping** — class index → dataset label table.
5. **Manifest + validation** — parsing and the compatibility checks.
6. **Embedded metadata** — the protobuf file scan (valid/absent/broken-JSON
   fixtures in `test/data/embedded_*.onnx`, generated by
   `test/make_embedded_fixture.py`), the merge precedence (sidecar wins over
   embedded values, mismatching label tables are refused), idempotence, and
   the defaults-only `loadModelSpec` flow — all without ONNX Runtime.
7. **Guards** — memory pre-flight, the NaN/Inf logits check and the
   execution-provider helpers (pure functions: GPU classification, display
   names, and `autoGpuCandidates` — the exact Auto candidate order for every
   `execution.provider` value).
8. **Validation report** — `buildModelValidationReport` on the committed
   fixtures (pass with warnings, unknown-architecture fail, broken-metadata
   warn)
   without ONNX Runtime; the session + IO-contract path is covered by a
   gated stage (needs ORT + model).
9. **Embedded-metadata session merge** (needs an ORT path; CI runs it against
   the tiny model) — the session-metadata API path on a real session.
10. **Warm-up probe + batch calibration** (needs ORT + model; CI runs it
    against the tiny model) — one real forward pass through the session's IO
    contract, the runtime-provider capability query (`GetAvailableProviders`:
    the CPU and the active provider must be listed, DirectML/CUDA must count
    as GPU providers), then the auto batch-size sweep.
11. **SphereFormer adapter** — the rounding minisuite (`np.round` →
    `std::rint`, numpy `%` → `std::fmod` on the non-negative domain, and
    `cart2sphere` with axis/wrap/z=0 edge cases: all bit-exact), the full
    67-tensor table parity against the real Python builders on a committed
    2048-pt fixture (`test/data/sphere_small_*`, generated by
    `test/make_sphere_fixture.py`: exact duplicate rows and padded caps so
    the phantom-window / identity-tail / padding paths are exercised), the
    mid-range recenter shift, `assembleBatch` payload checks and the
    cap/static-shape/grid-extent guards (all without ONNX Runtime).

### The tiny CI model (run parity on CI hardware)

The real parity model (~38 MB) is too large to commit. `test/make_tiny_model.py`
exports a miniature with the **same graph geometry** (20 inputs: `points`,
`features`, `sp0..sp3`, `idx0..idx13`; dynamic
batch/`num_points` axes, 9 classes, `stage_npoints [256,64,16,4]`,
`num_neighbors 16`, `feature_mode xyz`) and only a shrunk `base_width`
(64 → 4, ~0.7 MB). It consumes the SAME `main_*` fixture inputs, so its only
golden is a logits dump generated by the reference pipeline:

```shell
python test/make_tiny_model.py
python test/run_reference.py --onnx test/data/tiny_fkaconv.onnx --prefix tiny
```

Both outputs (`tiny_fkaconv.onnx` + `tiny_logits.f32`) are committed; the tiny
model also carries its manifest as embedded ONNX metadata (no sidecar).

When an ONNX Runtime library is configured
(`-DQSEMSEG_TEST_ORT_LIBRARY=...`), CTest gains **`TestSemSegRunParity`**:

```shell
TestSemSeg.exe run <test/data dir> <tiny_fkaconv.onnx> <onnxruntime.dll> tiny
```

The last argument selects the golden prefix (default `main`). The Ubuntu CI
jobs set this variable (plus `QSEMSEG_TEST_MODEL=<tiny model>` for the gated
stages), so CI proves ORT loading + run parity + the gated stages on plain
Linux with zero user hardware. Check counts with the tiny model as the gated
model run one lower than the real model's (365 vs 366 after the
execution-provider round) because the report-session stage's negative feature
check only applies to `irgb` models.

### SphereFormer run parity (local, needs the real export)

The 32k SphereFormer export (~392 MB) itself is never committed; the fixture
(points + shift + golden logits for one real patch, ~1.2 MB) is.
`test/make_sphere_fixture.py` computes the `sphere_lille_*`
fixture from the real `dijon_9.ply` (grid subsampling at the model's voxel
size, then the production regular-sampler loop until the patch fits the real
caps) and runs the same graph in Python for the golden logits. Configure

```shell
-DQSEMSEG_TEST_SPHERE_MODEL=C:/.../sphereformer_parislille3d_32k.onnx
```

and CTest gains **`TestSemSegSphereParity`**, which drives the full plugin
path (sidecar manifest, IO validation, 67 tensors, one real forward) and
compares logits + argmax:

```shell
TestSemSeg.exe sphere <test/data dir> <sphereformer_32k.onnx> <onnxruntime.dll> [provider]
```

`provider` defaults to `CPUExecutionProvider`; with the DirectML runtime pass
`DmlExecutionProvider` to run the same graph through DirectML (verified: 0
differing logits beyond 1e-3 and identical argmax against the CPU golden).

The reference pipeline computes some values with implementation-defined
numerics (numpy float32 accumulation order, open3d's hash-defined subsample
order). The test therefore:

- pins the reference's exact recenter mean and FPS start indices via
  `buildPatch`'s test-only `fixedMean` / `fpsStarts` overrides, so everything
  downstream must match **bit-for-bit** (1e-6 tolerance);
- compares the plugin's own double-precision mean only at accumulation-noise
  tolerance (3e-4);
- compares voxel barycenter *sets* by nearest-neighbor pairing, never by
  position (the subsample output order is hash-implementation-defined).

Run the whole suite:

```shell
ctest --test-dir build -R TestSemSeg
```

There is also an optional end-to-end run-parity mode (not part of the main
ctest invocation) that runs a real ONNX model on the plugin-built inputs and
compares logits and argmax labels against the reference (`<prefix>_logits.f32`,
dumped by `test/run_reference.py`, which runs the actual reference pipeline +
ORT):

```shell
TestSemSeg.exe run <test/data dir> <model.onnx> [onnxruntime.dll] [goldenPrefix=main] [provider]
```

`goldenPrefix` selects the reference dump: `main` (the real Open3D-ML parity
model — the default, local runs) or `tiny` (the committed miniature used by the
CTest `TestSemSegRunParity` test above). `provider` defaults to
`CPUExecutionProvider`; pass `DmlExecutionProvider` together with a DirectML
runtime to prove the model runs correctly through DirectML (the logits are
compared against the CPU-produced golden; DirectML matched it on the real
FKAConv parislille3d model and on the tiny model with 0 differing elements
beyond 1e-3 and identical argmax).

The explicit ORT DLL path is often needed on Windows because
`C:\Windows\System32\onnxruntime.dll` (an older copy) shadows the PATH.
