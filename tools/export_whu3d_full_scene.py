#!/usr/bin/env python3
"""Export one original WHU3D H5 scene to PLY without filtering or remapping.

Preserves every point, XYZ, intensity, number_returns, original semantic label,
and original instance ID. PLY semantic_label is the original H5 semantics;
PLY treeID is the original H5 instances for ALL categories.
Optionally use --tree-class to produce tree/non-tree labels while retaining
all points and features; --renumber-tree-ids maps positive tree IDs to 1..N.
Use --labels-only to omit intensity/number_returns and write only XYZ and labels.
"""

import argparse
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


VERTEX_DTYPE = np.dtype([
    ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
    ('intensity', '<f8'), ('number_returns', '<f8'),
    ('semantic_label', '<i4'), ('treeID', '<i4'),
])


def export_scene(source, destination, tree_class=None, renumber_tree_ids=False,
                 labels_only=False):
    temporary = None
    try:
        with h5py.File(source, 'r') as src:
            count = src['coords'].shape[0]
            if src['coords'].shape != (count, 3):
                raise ValueError('coords must have shape (N, 3)')
            feature_names = () if labels_only else ('intensity', 'number_returns')
            vertex_dtype = (np.dtype([(name, VERTEX_DTYPE[name]) for name in
                                      ('x', 'y', 'z', 'semantic_label', 'treeID')])
                            if labels_only else VERTEX_DTYPE)
            for name in feature_names + ('semantics', 'instances'):
                if src[name].shape != (count,):
                    raise ValueError(f'{name} must have shape (N,)')
            for name in ('semantics', 'instances'):
                if not np.issubdtype(src[name].dtype, np.integer):
                    raise ValueError(f'{name} must contain integer labels')
            tree_ids = None
            if renumber_tree_ids:
                ids = set()
                for start in range(0, count, 500_000):
                    semantics = src['semantics'][start:start + 500_000]
                    instances = src['instances'][start:start + 500_000]
                    ids.update(np.unique(instances[
                        (semantics == tree_class) & (instances > 0)]).tolist())
                tree_ids = np.array(sorted(ids), dtype=np.int64)
                if len(tree_ids) > np.iinfo(np.int32).max:
                    raise ValueError('number of tree instances exceeds PLY int32 range')
            label_comment = (
                'comment Complete original scene; no filtering or remapping\n'
                'comment semantic_label is original H5 semantics\n'
                'comment treeID is original H5 instances for all categories\n'
                if tree_class is None else
                f'comment All original points; source tree class={tree_class}\n'
                'comment semantic_label: 1=non-tree, 2=tree; non-tree treeID=0\n'
                + ('comment Positive tree IDs remapped per scene to 1..N; missing IDs=0\n'
                   if renumber_tree_ids else 'comment Tree instance IDs preserved\n'))
            header = (
                'ply\n'
                'format binary_little_endian 1.0\n'
                + label_comment +
                f'element vertex {count}\n'
                'property double x\n'
                'property double y\n'
                'property double z\n'
                + ''.join(f'property double {name}\n' for name in feature_names) +
                'property int semantic_label\n'
                'property int treeID\n'
                'end_header\n'
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                    mode='wb', dir=destination.parent,
                    prefix=f'.{destination.name}.', suffix='.tmp',
                    delete=False) as out:
                temporary = Path(out.name)
                out.write(header.encode('ascii'))
                for start in range(0, count, 500_000):
                    stop = min(start + 500_000, count)
                    vertices = np.empty(stop - start, dtype=vertex_dtype)
                    xyz = src['coords'][start:stop]
                    for axis, name in enumerate(('x', 'y', 'z')):
                        vertices[name] = xyz[:, axis]
                    for name in feature_names:
                        vertices[name] = src[name][start:stop]
                    for original, field in (('semantics', 'semantic_label'),
                                            ('instances', 'treeID')):
                        values = src[original][start:stop]
                        if tree_class is not None:
                            if original == 'semantics':
                                tree = values == tree_class
                                values = np.where(tree, 2, 1)
                            else:
                                selected = values[tree]
                                if renumber_tree_ids:
                                    valid = selected > 0
                                    normalized = np.zeros(len(selected), dtype=np.int64)
                                    normalized[valid] = (
                                        np.searchsorted(tree_ids, selected[valid]) + 1)
                                    selected = normalized
                                values = np.zeros(len(values), dtype=np.int64)
                                values[tree] = selected
                        bounds = np.iinfo(np.int32)
                        if values.size and (values.min() < bounds.min or
                                            values.max() > bounds.max):
                            raise ValueError(f'{original} exceeds PLY int32 range')
                        vertices[field] = values
                    vertices.tofile(out)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Exported {count:,} points: {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_h5', type=Path)
    parser.add_argument('output_ply', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--tree-class', type=int,
                        help='optional source tree class for tree/non-tree conversion')
    parser.add_argument('--renumber-tree-ids', action='store_true',
                        help='map positive tree instance IDs to consecutive 1..N')
    parser.add_argument('--labels-only', action='store_true',
                        help='only write x/y/z/semantic_label/treeID')
    args = parser.parse_args()
    if args.output_ply.exists() and not args.overwrite:
        parser.error('output exists; use --overwrite to replace it')
    if args.renumber_tree_ids and args.tree_class is None:
        parser.error('--renumber-tree-ids requires --tree-class')
    export_scene(args.input_h5, args.output_ply, args.tree_class,
                 args.renumber_tree_ids, args.labels_only)


if __name__ == '__main__':
    main()
