#!/usr/bin/env python3
"""Convert WHU3D ALS/MLS H5 files to binary PLY with tree/non-tree labels.

Input datasets: coords (N, 3), semantics (N,), instances (N,).
Output properties: x/y/z (double), semantic_label (int), treeID (int).
Tree class -> semantic_label=2, preserving the original instance ID.
Use --tree-class 200301 for ALS (default) or --tree-class 100300 for the
instance-annotated trees in the published MLS H5 files. The author's pywhu3d
MLS config calls 100300 'tree1' and lists 102000 ('tree2') as a stuff class
without instance annotations. Older web label tables use different codes.
All other classes -> semantic_label=1, treeID=0.
Tree points without instance annotations retain their original IDs (including 0).
Use --renumber-tree-ids for consecutive positive tree IDs 1..N per scene
(unannotated tree points become 0). Use --preserve-features to also retain
the original intensity and number_returns fields.
Only scenes with at least 11 distinct positive tree instance IDs are retained
by default. Set --min-tree-instances 0 to retain every scene, including those
without annotated trees. Existing PLYs for excluded scenes are deleted; source H5s are kept.
Use --filter-only to clean existing outputs without converting any H5 files.

Dependencies: numpy, h5py. No plyfile dependency is needed.
"""

import argparse
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


DEFAULT_INPUT = Path('/media/wyj225/fgl/公开数据集/whu3d/als/h5')
DEFAULT_OUTPUT = Path('/media/wyj225/fgl/公开数据集/whu3d/als/ply')
TREE_CLASS = 200301
VERTEX_DTYPE = np.dtype([
    ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
    ('semantic_label', '<i4'), ('treeID', '<i4'),
])


def tree_instance_ids(source, chunk_size=500_000, tree_class=TREE_CLASS):
    """Return sorted distinct positive instance IDs belonging to the tree class."""
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    ids = set()
    with h5py.File(source, 'r') as src:
        semantics = src['semantics']
        instances = src['instances']
        if semantics.ndim != 1 or instances.shape != semantics.shape:
            raise ValueError(f'{source}: semantics/instances must have matching shape (N,)')
        for labels in (semantics, instances):
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f'{source}: {labels.name} must use integer labels')
        for start in range(0, len(semantics), chunk_size):
            stop = start + chunk_size
            classes = semantics[start:stop]
            labels = instances[start:stop]
            ids.update(np.unique(labels[(classes == tree_class) & (labels > 0)]).tolist())
    return np.array(sorted(ids), dtype=np.int64)


def count_tree_instances(source, chunk_size=500_000, tree_class=TREE_CLASS):
    return len(tree_instance_ids(source, chunk_size, tree_class))


def convert_file(source, destination, chunk_size=500_000, tree_class=TREE_CLASS,
                 renumber_tree_ids=False, preserve_features=False):
    """Write one file atomically, preserving point order and coordinates."""
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    temporary = None
    tree_count = 0
    tree_without_positive_id = 0
    tree_ids = set()
    try:
        with h5py.File(source, 'r') as src:
            for name in ('coords', 'semantics', 'instances'):
                if name not in src or not isinstance(src[name], h5py.Dataset):
                    raise ValueError(f'{source}: missing dataset {name!r}')
            coords = src['coords']
            semantics = src['semantics']
            instances = src['instances']
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(f'{source}: expected coords (N, 3), got {coords.shape}')
            count = coords.shape[0]
            for labels in (semantics, instances):
                if labels.shape != (count,):
                    raise ValueError(
                        f'{source}: {labels.name} has shape {labels.shape}; '
                        f'expected ({count},)')
                if not np.issubdtype(labels.dtype, np.integer):
                    raise ValueError(f'{source}: {labels.name} must use integer labels')

            sorted_ids = (tree_instance_ids(source, chunk_size, tree_class)
                          if renumber_tree_ids else None)
            if sorted_ids is not None and len(sorted_ids) > np.iinfo(np.int32).max:
                raise ValueError(f'{source}: number of tree instances exceeds int32 range')
            feature_names = ('intensity', 'number_returns') if preserve_features else ()
            for name in feature_names:
                if name not in src or src[name].shape != (count,):
                    raise ValueError(f'{source}: expected feature {name} with shape ({count},)')
            vertex_dtype = np.dtype(
                VERTEX_DTYPE.descr + [(name, '<f8') for name in feature_names])
            instance_comment = (
                'comment treeID: tree instances remapped per scene to 1..N; '
                'non-tree and missing tree instance IDs=0\n'
                if renumber_tree_ids else
                'comment treeID: original tree instance ID; non-tree=0\n'
                'comment tree points with missing instance annotations retain source IDs\n')
            header = (
                'ply\n'
                'format binary_little_endian 1.0\n'
                f'comment semantic_label: 1=non-tree, 2=tree (source class {tree_class})\n'
                + instance_comment +
                f'element vertex {count}\n'
                'property double x\n'
                'property double y\n'
                'property double z\n'
                'property int semantic_label\n'
                'property int treeID\n'
                + ''.join(f'property double {name}\n' for name in feature_names)
                + 'end_header\n'
            )
            with tempfile.NamedTemporaryFile(
                    mode='wb', dir=destination.parent,
                    prefix=f'.{destination.name}.', suffix='.tmp',
                    delete=False) as out:
                temporary = Path(out.name)
                out.write(header.encode('ascii'))
                for start in range(0, count, chunk_size):
                    stop = min(start + chunk_size, count)
                    xyz = coords[start:stop]
                    is_tree = semantics[start:stop] == tree_class
                    original_instances = instances[start:stop]
                    selected_ids = original_instances[is_tree]
                    if selected_ids.size:
                        tree_count += int(selected_ids.size)
                        tree_without_positive_id += int(np.count_nonzero(selected_ids <= 0))
                        tree_ids.update(np.unique(selected_ids[selected_ids > 0]).tolist())
                        if renumber_tree_ids:
                            valid = selected_ids > 0
                            normalized = np.zeros(len(selected_ids), dtype=np.int64)
                            normalized[valid] = np.searchsorted(sorted_ids, selected_ids[valid]) + 1
                            selected_ids = normalized
                        limits = np.iinfo(np.int32)
                        if selected_ids.min() < limits.min or selected_ids.max() > limits.max:
                            raise ValueError(
                                f'{source}: tree instance IDs exceed PLY int32 range; '
                                'cannot preserve them without overflow')

                    vertices = np.empty(stop - start, dtype=vertex_dtype)
                    for axis, name in enumerate(('x', 'y', 'z')):
                        vertices[name] = xyz[:, axis]
                    vertices['semantic_label'] = np.where(is_tree, 2, 1)
                    vertices['treeID'] = 0
                    vertices['treeID'][is_tree] = selected_ids
                    for name in feature_names:
                        vertices[name] = src[name][start:stop]
                    vertices.tofile(out)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {
        'points': count,
        'tree_points': tree_count,
        'non_tree_points': count - tree_count,
        'tree_instances': len(tree_ids),
        'tree_without_positive_id': tree_without_positive_id,
    }


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--tree-class', type=int, default=TREE_CLASS,
                        help='source tree semantic label: ALS=200301 (default), '
                             'published MLS H5 instance trees=100300; '
                             '102000 is tree2 without instance annotations')
    parser.add_argument('--chunk-size', type=positive_int, default=500_000,
                        help='points processed per chunk (default: 500000)')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace existing PLY files; otherwise skip them')
    parser.add_argument('--renumber-tree-ids', action='store_true',
                        help='map positive tree instance IDs per scene to consecutive 1..N')
    parser.add_argument('--preserve-features', action='store_true',
                        help='also preserve original intensity and number_returns')
    parser.add_argument('--min-tree-instances', type=nonnegative_int, default=11,
                        help='minimum distinct positive tree IDs per scene '
                             '(default: 11; use 0 to retain all scenes)')
    parser.add_argument('--filter-only', action='store_true',
                        help='only remove existing PLYs for excluded scenes; '
                             'do not convert any H5 files')
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f'input directory does not exist: {args.input_dir}')
    sources = sorted(p for p in args.input_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in ('.h5', '.hdf5'))
    if not sources:
        parser.error(f'no H5 files found in {args.input_dir}')
    stems = [p.stem for p in sources]
    if len(stems) != len(set(stems)):
        parser.error('multiple input files would produce the same PLY filename')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted = skipped = failed = excluded = removed = 0
    print(f'Found {len(sources)} H5 files. Output: {args.output_dir}', flush=True)
    print(f'Retain scenes with >= {args.min_tree_instances} distinct positive tree IDs.',
          flush=True)
    for index, source in enumerate(sources, 1):
        destination = args.output_dir / f'{source.stem}.ply'
        prefix = f'[{index}/{len(sources)}] {source.name}'
        try:
            instance_count = count_tree_instances(source, args.chunk_size, args.tree_class)
            if instance_count < args.min_tree_instances:
                action = 'no PLY to remove'
                if destination.exists():
                    destination.unlink()
                    removed += 1
                    action = f'removed {destination.name}'
                excluded += 1
                print(
                    f'{prefix}: EXCLUDED, tree instances (positive IDs)={instance_count} '
                    f'< {args.min_tree_instances}; {action}', flush=True)
                continue
            if args.filter_only:
                skipped += 1
                print(f'{prefix}: RETAINED, tree instances (positive IDs)={instance_count}',
                      flush=True)
                continue
            if destination.exists() and not args.overwrite:
                skipped += 1
                print(f'{prefix}: skipped (PLY exists; use --overwrite to replace)', flush=True)
                continue
            stats = convert_file(source, destination, args.chunk_size, args.tree_class,
                                 args.renumber_tree_ids, args.preserve_features)
        except Exception as exc:
            failed += 1
            print(f'{prefix}: ERROR: {exc}', flush=True)
            continue
        converted += 1
        print(
            f'{prefix} -> {destination.name}: points={stats["points"]:,}, '
            f'non-tree={stats["non_tree_points"]:,}, tree={stats["tree_points"]:,}, '
            f'tree instances (positive IDs)={stats["tree_instances"]:,}', flush=True)
        if stats['tree_without_positive_id']:
            print(
                f'  WARNING: {stats["tree_without_positive_id"]:,} tree points have '
                'source instance IDs <= 0; '
                + ('output IDs set to 0.' if args.renumber_tree_ids else
                   'preserved as supplied (no IDs invented).'),
                flush=True)
    print(f'Done: converted={converted}, skipped={skipped}, excluded={excluded}, '
          f'removed={removed}, failed={failed}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
