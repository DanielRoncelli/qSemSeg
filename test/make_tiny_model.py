"""Exports a tiny FKAConv ONNX model for the CI run-parity test (T18).

The real parity model (Open3D-ML logs/fkaconv_parislille3d.onnx, ~38 MB) is too
large to commit to the CloudCompare repository. This script exports a miniature
that keeps the EXACT graph geometry of the real model - same 20 inputs
(points/features/sp0..sp3/idx0..idx13), same dynamic batch/num_points axes,
num_classes=9, stage_npoints=[256,64,16,4], num_neighbors=16, feature_mode=xyz -
and shrinks ONLY base_width (64 -> 4), which scales the conv channel widths and
therefore the parameter count (~1/256 of the weights -> ~150 KB file).

The committed run-parity fixtures (test/data/main_*) are defined by the stage
geometry alone, so the tiny model consumes the SAME fixture inputs; its golden
is a logits dump generated with the tiny weights (test/data/tiny_logits.f32,
written by `run_reference.py --onnx <out> --prefix tiny`).

Run with the fixture Python env (needs torch + onnx + onnxruntime + omegaconf):
    python test/make_tiny_model.py
    python test/run_reference.py --onnx test/data/tiny_fkaconv.onnx --prefix tiny

Both outputs (tiny_fkaconv.onnx + tiny_logits.f32) are committed; the test
invocation is wired in test/CMakeLists.txt (TestSemSegRunParity, gated on
QSEMSEG_TEST_ORT_LIBRARY).
"""

import argparse
import os
import logging
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--open3dml-src', dest='open3dml_src',
                        default=os.environ.get('OPEN3DML_SRC', ''))
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'data' / 'tiny_fkaconv.onnx'))
    parser.add_argument('--base-width', dest='base_width', type=int, default=4,
                        help='FKAConv base_width (the ONLY geometry knob that differs from the real model)')
    parser.add_argument('--num-points', dest='num_points', type=int, default=4096,
                        help='export/tracing patch size (validateRun feeds 4096-point patches)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    sys.path.insert(0, args.open3dml_src)
    sys.path.insert(0, str(Path(args.open3dml_src) / 'onnx'))

    import torch
    from convert_onnx import dataset_labels
    from graph_simplify import probe_conditions, simplify
    from onnx_common import (build_adapter, build_model, load_config,
                             package_versions, stamp_manifest)

    cfg = load_config('fkaconv_parislille3d.yml')
    # same geometry as the real model (the parity fixtures and the C++
    # validateRun spec are tied to it); ONLY the channel width shrinks
    cfg.model['in_channels'] = 3
    cfg.model['feature_mode'] = 'xyz'
    cfg.model['base_width'] = args.base_width
    cfg.model['num_points'] = args.num_points

    model = build_model(cfg, seed=args.seed)
    model.eval()

    adapter = build_adapter(model, in_channels=3)
    adapter.eval()

    # sanity: one forward pass at the export shape
    dummy = adapter.make_dummy(args.batch_size, args.num_points, 3, seed=args.seed)
    with torch.no_grad():
        logits = adapter(*dummy)
    num_classes = int(cfg.model.get('num_classes', 2))
    expected = (args.batch_size, args.num_points, num_classes)
    if tuple(logits.shape) != expected:
        raise SystemExit(f'adapter forward produced {tuple(logits.shape)}, expected {expected}')
    log.info(f'adapter output shape: {tuple(logits.shape)}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            adapter,
            dummy,
            str(out_path),
            export_params=True,
            input_names=adapter.input_names(),
            output_names=adapter.output_names,
            dynamic_axes=adapter.dynamic_axes(),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    log.info(f'Exported ONNX model: {out_path}')

    import onnx
    onnx_model = onnx.load(str(out_path), load_external_data=False)
    onnx.checker.check_model(onnx_model)
    log.info('ONNX checker: model is valid')

    # fold the traced scatter_reduce If guards (zero-size checks whose two
    # branches make onnxruntime's shape inference fail); conditions are
    # measured by running the graph once with the onnx reference evaluator
    feeds = adapter.make_random_inputs(args.batch_size, args.num_points, 3, seed=args.seed)
    conds = probe_conditions(onnx_model, feeds, str(out_path))
    log.info(f'If guard conditions: {conds}')
    simplify(str(out_path), cond_values=conds)
    log.info('Graph simplified (If guards folded)')

    # eval-time per-patch augmentation, exactly as build_patch_inputs applies it
    # (the config's augment is a plain-JSON-serializable ConfigDict; only the
    # recenter/normalize keys matter at eval time)
    augment_cfg = cfg.model.get('augment', None) or {}
    eval_augment = {k: augment_cfg[k] for k in ('recenter', 'normalize') if k in augment_cfg}

    meta = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'model_name': cfg.model['name'],
        'checkpoint': 'tiny CI test model (random init, no checkpoint)',
        'config': f'fkaconv_parislille3d.yml with base_width={args.base_width} (tiny CI model)',
        'onnx_file': out_path.name,
        'inputs': adapter.describe_inputs(args.num_points, 3),
        'output': {
            'name': 'logits',
            'dtype': 'float32',
            'shape': ['batch', 'num_points', 'num_classes'],
            'layout': '[B, N, num_classes] per-point logits; argmax over last axis = label',
        },
        'in_channels': 3,
        'num_classes': num_classes,
        'ignored_label_inds': list(cfg.model.get('ignored_label_inds', []) or []),
        'num_points': args.num_points,
        'dynamic_axes': {'batch': True, 'num_points': True},
        'opset': args.opset,
        'versions': package_versions(),
        'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'adapter': adapter.extra_meta(),
        'preprocessing': {
            'description': 'Tiny CI test model; same preprocessing as the real FKAConv export (see README.md).',
            'voxel_size': cfg.model.get('grid_size', None),
            'num_points': args.num_points,
            'channels': 3,
            'shift_min': bool(getattr(adapter, 'shift_min_before_subsampling', False)),
            'augment': eval_augment,
            'feature_columns': [],
            **adapter.preprocessing_spec(),
        },
    }
    labels = dataset_labels(cfg, num_classes)
    if labels:
        meta['labels'] = labels

    # embed the manifest in the model file itself (no sidecar is committed -
    # the single .onnx is self-describing, consumed via the T7 embedded path)
    stamp_manifest(out_path, meta)

    size_kb = out_path.stat().st_size / 1024.0
    log.info(f'Done: {out_path} ({size_kb:.0f} KB). Now generate the golden logits:')
    log.info(f'  python run_reference.py --onnx {out_path} --prefix tiny')


if __name__ == '__main__':
    main()
