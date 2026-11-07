#!/usr/bin/env python3
"""Normalize the reduced WHU-Urban3D dataset to tree/non-tree PLYs.

Read the paired STPLS3D TXT files (x y z r g b semantic instance). The scene
list and output filenames follow {als,mls}/for-instance/**/*.ply, avoiding
duplicate extra TXT crops. The TXT files preserve IDs above 255 that were
truncated in four ALS uint8 PLYs, and avoid the malformed MLS 8018 PLY.
Source semantic: 1=background, 2=tree. Output semantic_label has the same
values. Non-tree treeID becomes 0. Each scene's distinct tree IDs (including
a valid source tree ID 0) are sorted and remapped to consecutive IDs 1..N.
The same source instance always maps to the same output ID across chunks.
No points are dropped from retained scenes.

Outputs: <output-root>/{als,mls}/ply, preserving relative paths and splits.
The default output root is the input root. Scenes with <=10 distinct tree
instances are excluded by default. Existing outputs for excluded scenes are
removed; the source PLYs are never modified.

Run --inspect-only for a read-only inventory. Only numpy is required.
"""

import argparse
import os
from pathlib import Path
import re
import tempfile

import numpy as np

DEFAULT_ROOT = Path('/media/wyj225/fgl/公开数据集/whu-urban3d')
OUTPUT_DTYPE = np.dtype([
    ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
    ('semantic_label', '<i4'), ('treeID', '<i4'),
])


def load_vertices(path):
    dtype = np.dtype([
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('semantic_seg', 'i8'), ('treeID', 'i8'),
    ])
    return np.loadtxt(path, dtype=dtype, usecols=(0, 1, 2, 6, 7), ndmin=1)


def inspect_vertices(vertices, chunk_size):
    ids = set()
    tree_points = 0
    for name in ('semantic_seg', 'treeID'):
        if not np.issubdtype(vertices.dtype[name], np.integer):
            raise ValueError(f'{name} must use integer labels')
    for start in range(0, len(vertices), chunk_size):
        chunk = vertices[start:start + chunk_size]
        sem = chunk['semantic_seg']
        if np.any((sem != 1) & (sem != 2)):
            raise ValueError(f'expected semantic_seg values 1/2; got {np.unique(sem)}')
        tree = sem == 2
        tree_points += int(np.count_nonzero(tree))
        ids.update(np.unique(chunk['treeID'][tree]).tolist())
    if ids and min(ids) < 0:
        raise ValueError('negative tree instance IDs require explicit annotation handling')
    if len(ids) > np.iinfo(np.int32).max:
        raise ValueError('number of tree instances exceeds the output PLY int32 range')
    return dict(points=len(vertices), tree_points=tree_points,
                tree_instances=len(ids), tree_ids=np.array(sorted(ids), dtype=np.int64),
                min_tree_id=min(ids) if ids else None)


def write_normalized(vertices, destination, stats, chunk_size):
    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        'comment semantic_label: 1=non-tree, 2=tree\n'
        'comment treeID: non-tree=0; tree instances remapped per scene to 1..N\n'
        'comment tree IDs ordered by ascending source instance ID\n'
        f'element vertex {len(vertices)}\n'
        'property double x\n'
        'property double y\n'
        'property double z\n'
        'property int semantic_label\n'
        'property int treeID\n'
        'end_header\n'
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='wb', dir=destination.parent, prefix=f'.{destination.name}.',
                suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(header.encode('ascii'))
            for start in range(0, len(vertices), chunk_size):
                chunk = vertices[start:start + chunk_size]
                output = np.empty(len(chunk), dtype=OUTPUT_DTYPE)
                for axis in ('x', 'y', 'z'):
                    output[axis] = chunk[axis]
                tree = chunk['semantic_seg'] == 2
                output['semantic_label'] = np.where(tree, 2, 1)
                output['treeID'] = 0
                output['treeID'][tree] = (
                    np.searchsorted(stats['tree_ids'], chunk['treeID'][tree]) + 1)
                output.tofile(stream)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-root', type=Path,
                        help='default: same as --input-root')
    parser.add_argument('--subset', choices=('als', 'mls', 'both'), default='both')
    parser.add_argument('--min-tree-instances', type=positive_int, default=11)
    parser.add_argument('--chunk-size', type=positive_int, default=500_000)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--inspect-only', action='store_true',
                        help='report counts without writing or deleting files')
    args = parser.parse_args()
    output_root = args.output_root or args.input_root
    subsets = ('als', 'mls') if args.subset == 'both' else (args.subset,)
    failed = 0
    for subset in subsets:
        input_dir = args.input_root / subset / 'for-instance'
        output_dir = output_root / subset / 'ply'
        files = sorted(input_dir.rglob('*.ply'))
        if not files:
            parser.error(f'no source PLY files in {input_dir}')
        txt_index = {}
        for path in (args.input_root / subset / 'stpls3d').rglob('*.txt'):
            txt_index.setdefault(path.stem, []).append(path)
        totals = dict(kept=0, excluded=0, tree_instances=0,
                      kept_instances=0, written=0, skipped=0, removed=0, failed=0)
        print(f'{subset}: {len(files)} scenes; output={output_dir}', flush=True)
        for index, source in enumerate(files, 1):
            relative = source.relative_to(input_dir)
            destination = output_dir / relative
            prefix = f'[{subset} {index}/{len(files)}] {relative}'
            try:
                if source.resolve() == destination.resolve():
                    raise ValueError('output must not replace the source PLY')
                scene = source.stem
                if subset == 'mls':
                    scene = re.sub(r'_(train|test|val)$', '', scene)
                candidates = txt_index.get(scene, [])
                if len(candidates) != 1:
                    raise ValueError(f'expected one paired STPLS3D TXT for {scene}; '
                                     f'found {len(candidates)}')
                vertices = load_vertices(candidates[0])
                stats = inspect_vertices(vertices, args.chunk_size)
                totals['tree_instances'] += stats['tree_instances']
                keep = stats['tree_instances'] >= args.min_tree_instances
                totals['kept' if keep else 'excluded'] += 1
                if keep:
                    totals['kept_instances'] += stats['tree_instances']
                action = 'KEEP' if keep else 'EXCLUDE'
                if not args.inspect_only:
                    if not keep and destination.exists():
                        destination.unlink()
                        totals['removed'] += 1
                        action += ' (removed existing output)'
                    elif keep and destination.exists() and not args.overwrite:
                        totals['skipped'] += 1
                        action += ' (output exists)'
                    elif keep:
                        write_normalized(vertices, destination, stats, args.chunk_size)
                        totals['written'] += 1
                        action += ' (written)'
                print(f'{prefix}: {action}; points={stats["points"]:,}, '
                      f'tree points={stats["tree_points"]:,}, '
                      f'tree instances={stats["tree_instances"]}, '
                      f'tree IDs=1..{stats["tree_instances"]}', flush=True)
                del vertices
            except Exception as exc:
                totals['failed'] += 1
                failed += 1
                print(f'{prefix}: ERROR: {exc}', flush=True)
        print(f'{subset} SUMMARY: {totals}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
