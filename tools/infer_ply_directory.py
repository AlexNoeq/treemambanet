#!/usr/bin/env python3
"""Run ForestMamba WHU-Stree inference and save instance IDs as int16 NPYs.

The model's sliding-window test path expects an ``eval_ann_info`` field even
when no labels are available. This script supplies empty placeholders and
replaces the model's PLY writer so the output contains only per-point instance
IDs, without writing placeholder ground truth.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from mmengine.config import Config
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import register_all_modules
from plyfile import PlyData


DEFAULT_CONFIG = PROJECT_ROOT / 'configs/WHU-Stree/forestmamba_whu_stree.py'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'work_dirs/forestmamba_whu_stree/epoch_950.pth'
DEFAULT_INPUT_DIR = Path('/home/wyj225/225/fgl/data/test')
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / 'work_dirs/inference/forestmamba_whu_stree_epoch950')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Infer tree instances from XYZ+intensity PLY files.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Recompute output NPY files that already exist.')
    return parser.parse_args()


def field_name(names, requested, aliases=()):
    by_lower = {name.lower(): name for name in names}
    for name in (requested, *aliases):
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    raise ValueError(
        f'PLY is missing {requested!r}; found vertex fields: {list(names)}')


def make_instance_writer(num_points, output_path):
    """Return a saver that stores only per-point int16 instance IDs."""
    def save_ply_withscore(points, semantic_pred, instance_pred, scores,
                           filename, semantic_gt=None, instance_gt=None):
        labels = np.asarray(instance_pred).reshape(-1)
        if labels.size != num_points:
            raise ValueError(
                f'Prediction has {labels.size} labels; input has {num_points}')
        limits = np.iinfo(np.int16)
        if labels.size and (labels.min() < limits.min
                            or labels.max() > limits.max):
            raise ValueError(
                f'Instance IDs do not fit int16: range '
                f'[{labels.min()}, {labels.max()}]')
        np.save(output_path, labels.astype(np.int16, copy=False))

    return save_ply_withscore


def load_xyz_for_model(ply_path):
    ply = PlyData.read(str(ply_path))
    vertices = ply['vertex'].data
    names = vertices.dtype.names or ()
    xyz_names = tuple(field_name(names, axis) for axis in ('x', 'y', 'z'))
    intensity_name = field_name(names, 'intensity', ('reflectance',))
    if len(vertices) == 0:
        raise ValueError(f'{ply_path} contains no vertices')

    xyz = np.empty((len(vertices), 3), dtype=np.float32)
    for axis, name in enumerate(xyz_names):
        values = vertices[name]
        if not np.isfinite(values).all():
            raise ValueError(f'{ply_path} has non-finite values in {name}')
        xyz[:, axis] = values
    if not np.isfinite(vertices[intensity_name]).all():
        raise ValueError(
            f'{ply_path} has non-finite values in {intensity_name}')

    # Match tools/prepare_whu_stree.py: center XY and set minimum Z to zero.
    xyz[:, 0] -= np.mean(vertices[xyz_names[0]], dtype=np.float64)
    xyz[:, 1] -= np.mean(vertices[xyz_names[1]], dtype=np.float64)
    xyz[:, 2] -= np.min(vertices[xyz_names[2]])
    return ply, vertices, xyz_names, intensity_name, xyz


def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    target_state = model.state_dict()
    metadata = getattr(state_dict, '_metadata', None)
    if metadata is None:
        state_dict._metadata = {}
        metadata = state_dict._metadata

    # These checkpoint tensors already use spconv 2's layout. The checkpoint
    # was saved without state-dict version metadata, so mmdet3d's compatibility
    # loader otherwise mistakes them for spconv 1 tensors and transposes them.
    for module_name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if (module.__class__.__module__.startswith('spconv.')
                and isinstance(weight, torch.Tensor) and weight.ndim == 5):
            key = f'{module_name}.weight' if module_name else 'weight'
            if key in state_dict and state_dict[key].shape != target_state[key].shape:
                raise RuntimeError(
                    f'{key} checkpoint shape {tuple(state_dict[key].shape)} '
                    f'does not match model shape {tuple(target_state[key].shape)}')
            metadata[module_name] = {'version': 2}

    model.load_state_dict(state_dict, strict=True)
    print('Loaded checkpoint with spconv 2 kernel layout metadata.')


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError(
            'CUDA is unavailable in this environment. ForestMamba sparse '
            'convolutions require the project CUDA runtime; run this command '
            'in the configured GPU environment.')
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)

    ply_files = sorted(args.input_dir.glob('*.ply'))
    if not ply_files:
        raise FileNotFoundError(f'No .ply files in {args.input_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    register_all_modules()
    import oneformer3d  # noqa: F401  Register this repository's model classes.

    cfg = Config.fromfile(str(args.config))
    model = MODELS.build(cfg.model)
    load_model_weights(model, args.checkpoint)
    model.to(args.device)
    model.eval()
    # XAwarequery uses this directory for its merged per-point predictions.
    model._output_path = str(args.output_dir.resolve())

    errors = []
    for index, ply_path in enumerate(ply_files, start=1):
        if not ply_path.stem.endswith('_test'):
            raise ValueError(f'Expected a *_test.ply filename: {ply_path.name}')
        output_path = args.output_dir / f'{ply_path.stem[:-5]}.npy'
        if output_path.exists() and not args.overwrite:
            print(
                f'[{index}/{len(ply_files)}] skip existing {output_path}',
                flush=True)
            continue
        try:
            _, vertices, _, _, xyz = load_xyz_for_model(ply_path)
            # Model's test branch uses this path marker to select sliding-window
            # inference. The requested input directory is named "test".
            if 'test' not in str(ply_path).lower():
                raise ValueError(
                    f'{ply_path} must be under a path containing "test" so '
                    'the model selects its sliding-window inference path')

            data_sample = Det3DDataSample()
            data_sample.set_metainfo({'lidar_path': str(ply_path)})
            # The existing test branch reads these keys unconditionally. The
            # custom writer discards them and emits only instance IDs.
            data_sample.eval_ann_info = {
                'pts_semantic_mask': None,
                'pts_instance_mask': None,
            }
            model.save_ply_withscore = make_instance_writer(
                len(vertices), output_path)

            batch = {
                'inputs': {'points': [torch.from_numpy(xyz)]},
                'data_samples': [data_sample],
            }
            with torch.inference_mode():
                batch = model.data_preprocessor(batch, training=False)
                model(**batch, mode='predict')

            print(
                f'[{index}/{len(ply_files)}] {ply_path.name}: '
                f'{len(vertices):,} instance labels -> {output_path}',
                flush=True)
        except Exception as exc:
            errors.append((ply_path, exc))
            print(
                f'[{index}/{len(ply_files)}] ERROR {ply_path.name}: {exc}',
                flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if errors:
        print(f'Finished with {len(errors)} failed file(s).')
        raise SystemExit(1)
    completed = len(list(args.output_dir.glob('*.npy')))
    print(
        f'Inference complete: {completed} instance NPY files in '
        f'{args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
