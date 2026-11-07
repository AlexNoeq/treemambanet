#!/usr/bin/env python3
"""Convert WHU-Stree training PLYs and test PLY/NPY pairs for ForestMamba.

The raw PLYs have x/y/z/intensity/tree/label vertex properties. The converter
also accepts the equivalent semantic_label and treeID property names. Test
instance labels come from test/reference_data/<scene>.npy in point order.
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np


PLY_TYPES = {
    'char': 'i1',
    'int8': 'i1',
    'uchar': 'u1',
    'uint8': 'u1',
    'short': 'i2',
    'int16': 'i2',
    'ushort': 'u2',
    'uint16': 'u2',
    'int': 'i4',
    'int32': 'i4',
    'uint': 'u4',
    'uint32': 'u4',
    'float': 'f4',
    'float32': 'f4',
    'double': 'f8',
    'float64': 'f8',
}


def read_ply_header(stream, path):
    if stream.readline().strip() != b'ply':
        raise ValueError(f'{path} is not a PLY file')

    format_line = stream.readline().decode('ascii').strip().split()
    if len(format_line) < 2 or not format_line[0] == 'format':
        raise ValueError(f'{path} has a malformed PLY format line')
    ply_format = format_line[1]
    if ply_format not in ('binary_little_endian', 'binary_big_endian'):
        raise ValueError(
            f'{path} uses {ply_format}; only binary PLY is supported')
    endian = '<' if ply_format == 'binary_little_endian' else '>'

    current_element = None
    vertex_count = None
    vertex_properties = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError(f'{path} has no end_header')
        parts = line.decode('ascii').strip().split()
        if not parts:
            continue
        if parts[0] == 'element':
            current_element = parts[1]
            if current_element == 'vertex':
                vertex_count = int(parts[2])
        elif parts[0] == 'property' and current_element == 'vertex':
            if len(parts) != 3:
                raise ValueError(
                    f'{path} has an unsupported list property in vertex data')
            ply_type, field_name = parts[1], parts[2]
            if ply_type not in PLY_TYPES:
                raise ValueError(
                    f'{path} uses unsupported PLY type {ply_type}')
            vertex_properties.append(
                (field_name, np.dtype(endian + PLY_TYPES[ply_type])))
        elif parts[0] == 'end_header':
            break

    if vertex_count is None or not vertex_properties:
        raise ValueError(f'{path} has no vertex data')
    names = [name for name, _ in vertex_properties]
    if len(set(names)) != len(names):
        raise ValueError(f'{path} has duplicate vertex property names')
    dtype = np.dtype(vertex_properties)
    return vertex_count, dtype, stream.tell(), names


def load_reference(path, point_count):
    labels = np.load(path, mmap_mode='r', allow_pickle=False)
    if labels.shape != (point_count,):
        raise ValueError(
            f'{path} has shape {labels.shape}; expected ({point_count},) '
            'to match the test PLY vertex order')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'{path} must contain integer instance IDs')
    return labels


def iter_ply_chunks(path, chunk_size, reference_path=None):
    with path.open('rb') as stream:
        count, dtype, data_offset, names = read_ply_header(stream, path)
        reference = (
            load_reference(reference_path, count)
            if reference_path is not None else None)
        stream.seek(data_offset)
        remaining = count
        while remaining:
            take = remaining if chunk_size is None else min(remaining, chunk_size)
            points = np.fromfile(stream, dtype=dtype, count=take)
            if len(points) != take:
                raise ValueError(
                    f'{path} ended early: expected {take} more vertices')
            if reference is not None:
                # NPY labels are aligned by index; never sort or sample here.
                start = count - remaining
                raw_ids = reference[start:start + take]
                if np.any(raw_ids < -1):
                    raise ValueError(
                        f'{reference_path} contains instance IDs below -1')
                labeled = np.empty(take, dtype=[
                    ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                    ('intensity', 'f4'), ('tree', 'i8'), ('label', 'i8')])
                for axis in ('x', 'y', 'z'):
                    name = find_field(names, (axis,), path, axis)
                    labeled[axis] = points[name]
                intensity_name = next(
                    (name for name in names
                     if name.lower() in ('intensity', 'reflectance')), None)
                labeled['intensity'] = (
                    points[intensity_name] if intensity_name else 0.0)
                labeled['tree'] = raw_ids
                labeled['label'] = np.where(raw_ids > 0, 1, -1)
                points = labeled
            remaining -= take
            yield points, list(points.dtype.names)


def find_field(names, aliases, path, purpose):
    name_map = {name.lower(): name for name in names}
    for alias in aliases:
        if alias.lower() in name_map:
            return name_map[alias.lower()]
    raise ValueError(
        f'{path} has no {purpose} field; available fields: {names}')


def integer_labels(values, path, field_name):
    if not np.isfinite(values).all():
        raise ValueError(f'{path} contains non-finite values in {field_name}')
    rounded = np.rint(values)
    if not np.all(np.abs(values - rounded) < 1e-4):
        raise ValueError(f'{path} contains non-integer values in {field_name}')
    return rounded.astype(np.int64, copy=False)


def inspect_scene(path, chunk_size, reference_path=None):
    point_count = 0
    sum_xy = np.zeros(2, dtype=np.float64)
    min_z = np.inf
    ground_count = 0
    tree_count = 0
    tree_ids = set()
    tree_bounds = {}
    field_names = None

    for points, names in iter_ply_chunks(path, chunk_size, reference_path):
        if field_names is None:
            field_names = names
            x_name = find_field(names, ('x',), path, 'x coordinate')
            y_name = find_field(names, ('y',), path, 'y coordinate')
            z_name = find_field(names, ('z',), path, 'z coordinate')
            intensity_name = find_field(
                names, ('intensity', 'reflectance'), path, 'intensity')
            semantic_name = find_field(
                names, ('semantic_label', 'semantic_seg', 'semantic', 'label'),
                path, 'semantic label')
            instance_name = find_field(
                names, ('treeID', 'tree_id', 'tree', 'instance_id', 'instance'),
                path, 'tree instance ID')

        x = points[x_name].astype(np.float64, copy=False)
        y = points[y_name].astype(np.float64, copy=False)
        z = points[z_name].astype(np.float64, copy=False)
        if not (np.isfinite(x).all() and np.isfinite(y).all()
                and np.isfinite(z).all()
                and np.isfinite(points[intensity_name]).all()):
            raise ValueError(f'{path} contains non-finite point data')

        labels = integer_labels(points[semantic_name], path, semantic_name)
        raw_ids = integer_labels(points[instance_name], path, instance_name)
        tree_mask = labels != -1
        if np.any(raw_ids[tree_mask] <= 0):
            bad_count = int(np.count_nonzero(raw_ids[tree_mask] <= 0))
            raise ValueError(
                f'{path} has {bad_count} tree-labeled points with a '
                'non-positive tree ID')

        point_count += len(points)
        sum_xy[0] += x.sum(dtype=np.float64)
        sum_xy[1] += y.sum(dtype=np.float64)
        min_z = min(min_z, float(z.min()))
        ground_count += int(np.count_nonzero(~tree_mask))
        tree_count += int(np.count_nonzero(tree_mask))

        if tree_mask.any():
            ids, inverse = np.unique(raw_ids[tree_mask], return_inverse=True)
            tree_ids.update(int(tree_id) for tree_id in ids)
            tree_xyz = np.column_stack(
                (x[tree_mask], y[tree_mask], z[tree_mask]))
            local_min = np.full((len(ids), 3), np.inf, dtype=np.float64)
            local_max = np.full((len(ids), 3), -np.inf, dtype=np.float64)
            np.minimum.at(local_min, inverse, tree_xyz)
            np.maximum.at(local_max, inverse, tree_xyz)
            local_counts = np.bincount(inverse, minlength=len(ids))
            for index, tree_id in enumerate(ids):
                tree_id = int(tree_id)
                if tree_id in tree_bounds:
                    old_min, old_max, old_count = tree_bounds[tree_id]
                    tree_bounds[tree_id] = (
                        np.minimum(old_min, local_min[index]),
                        np.maximum(old_max, local_max[index]),
                        old_count + int(local_counts[index]))
                else:
                    tree_bounds[tree_id] = (
                        local_min[index], local_max[index],
                        int(local_counts[index]))

    if point_count == 0:
        raise ValueError(f'{path} contains no vertices')
    if tree_count == 0:
        raise ValueError(f'{path} contains no tree points')
    offsets = np.array(
        [sum_xy[0] / point_count, sum_xy[1] / point_count, min_z],
        dtype=np.float64)
    return {
        'point_count': point_count,
        'ground_count': ground_count,
        'tree_count': tree_count,
        'offsets': offsets,
        'tree_ids': np.array(sorted(tree_ids), dtype=np.int64),
        'tree_bounds': tree_bounds,
        'fields': {
            'xyz': (x_name, y_name, z_name),
            'intensity': intensity_name,
            'semantic': semantic_name,
            'instance': instance_name,
        },
    }


def make_instances(scene_info):
    offsets = scene_info['offsets']
    instances = []
    for tree_id, (minimum, maximum, count) in sorted(
            scene_info['tree_bounds'].items()):
        minimum = minimum - offsets
        maximum = maximum - offsets
        center = (minimum + maximum) * 0.5
        dimensions = maximum - minimum
        instances.append({
            'bbox_3d': [
                float(value) for value in np.concatenate((center, dimensions))
            ],
            'bbox_label_3d': 1,
            'num_lidar_pts': int(count),
        })
    return instances


def write_scene(source, sample_id, output_root, scene_info, chunk_size,
                reference_path=None):
    points_dir = output_root / 'points'
    semantic_dir = output_root / 'semantic_mask'
    instance_dir = output_root / 'instance_mask'
    points_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    xyz_names = scene_info['fields']['xyz']
    intensity_name = scene_info['fields']['intensity']
    semantic_name = scene_info['fields']['semantic']
    instance_name = scene_info['fields']['instance']
    offsets = scene_info['offsets']
    tree_ids = scene_info['tree_ids']

    point_path = points_dir / f'{sample_id}.bin'
    semantic_path = semantic_dir / f'{sample_id}.bin'
    instance_path = instance_dir / f'{sample_id}.bin'
    with (point_path.open('wb') as point_file,
          semantic_path.open('wb') as semantic_file,
          instance_path.open('wb') as instance_file):
        for points, _ in iter_ply_chunks(source, chunk_size, reference_path):
            raw_semantic = integer_labels(
                points[semantic_name], source, semantic_name)
            raw_instance = integer_labels(
                points[instance_name], source, instance_name)
            tree_mask = raw_semantic != -1

            xyz = np.empty((len(points), 3), dtype=np.float32)
            for axis, field_name in enumerate(xyz_names):
                xyz[:, axis] = points[field_name]
            xyz -= offsets.astype(np.float32)
            model_points = np.empty((len(points), 4), dtype=np.float32)
            model_points[:, :3] = xyz
            # WHU intensity is commonly stored on a 0..65535 scale.
            model_points[:, 3] = points[intensity_name] / 65535.0

            semantic = np.where(tree_mask, 1, 0).astype(np.int64)
            instance = np.full(len(points), -1, dtype=np.int64)
            if tree_mask.any():
                positions = np.searchsorted(tree_ids, raw_instance[tree_mask])
                instance[tree_mask] = positions.astype(np.int64) + 1

            model_points.tofile(point_file)
            semantic.tofile(semantic_file)
            instance.tofile(instance_file)

    return {
        'sample_idx': sample_id,
        'token': sample_id,
        'lidar_points': {
            'num_pts_feats': 4,
            'lidar_path': point_path.name,
        },
        'pts_semantic_mask_path': semantic_path.name,
        'pts_instance_mask_path': instance_path.name,
        'axis_align_matrix': np.eye(4, dtype=np.float32).tolist(),
        'instances': make_instances(scene_info),
    }


def dump_infos(output_root, split, data_list):
    info = {
        'metainfo': {
            'categories': {'ground': 0, 'tree': 1},
            'dataset': 'WHU-Stree',
            'info_version': '1.1',
        },
        'data_list': data_list,
    }
    info_path = output_root / f'whu_stree_oneformer3d_infos_{split}.pkl'
    with info_path.open('wb') as stream:
        pickle.dump(info, stream, protocol=pickle.HIGHEST_PROTOCOL)

    list_path = output_root / 'meta_data' / f'{split}_list.txt'
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        ''.join(f'{item["sample_idx"]}\n' for item in data_list),
        encoding='utf-8')
    print(f'Wrote {len(data_list)} {split} infos to {info_path}')


def scene_key(value):
    name = re.sub(r'_(train|test)$', '', value)
    if not re.fullmatch(r'\d+_\d+(?:-\d+)*', name):
        raise ValueError(f'Invalid WHU-Stree scene name: {value}')
    return tuple(int(part) for part in re.split(r'[_-]', name))


def collect_sources(source_root, split, train_start, train_end,
                    test_points_dir, test_reference_dir):
    split_dir = source_root / (test_points_dir if split == 'test' else split)
    if not split_dir.is_dir():
        raise FileNotFoundError(f'Missing split directory: {split_dir}')
    sources = sorted(split_dir.glob('*.ply'))
    if not sources:
        raise FileNotFoundError(f'No PLY files found in {split_dir}')
    if split == 'train':
        first, last = scene_key(train_start), scene_key(train_end)
        sources = [source for source in sources
                   if first <= scene_key(source.stem) <= last]
        if not sources:
            raise ValueError(f'No training scenes in {train_start}..{train_end}')

    pairs = []
    for source in sources:
        reference = None
        if split == 'test':
            scene = re.sub(r'_test$', '', source.stem)
            reference = source_root / test_reference_dir / f'{scene}.npy'
            if not reference.is_file():
                raise FileNotFoundError(f'Missing test labels: {reference}')
        with source.open('rb') as stream:
            count, dtype, offset, _ = read_ply_header(stream, source)
        if source.stat().st_size < offset + count * dtype.itemsize:
            raise ValueError(f'{source} has truncated vertex data')
        if reference is not None:
            load_reference(reference, count)
        pairs.append((source, reference, count))
    return pairs


def prepare_split(output_root, split, chunk_size, sources):

    data_list = []
    for index, (source, reference, _) in enumerate(sources, start=1):
        sample_id = f'{split}_{source.stem}'
        print(f'[{split} {index}/{len(sources)}] scanning {source.name}')
        scene_info = inspect_scene(source, chunk_size, reference)
        item = write_scene(
            source, sample_id, output_root, scene_info, chunk_size, reference)
        data_list.append(item)
        print(
            f'  {scene_info["point_count"]:,} points: '
            f'{scene_info["ground_count"]:,} ground, '
            f'{scene_info["tree_count"]:,} tree, '
            f'{len(scene_info["tree_ids"]):,} tree instances')

    dump_infos(output_root, split, data_list)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare WHU-Stree data for ForestMamba')
    parser.add_argument(
        '--source-root',
        type=Path,
        default=Path('/media/wyj225/Win11Pro X64/fgl'),
        help='Directory containing train/ and test/ folders')
    parser.add_argument(
        '--train-start', default='00_1', help='First training scene (inclusive)')
    parser.add_argument(
        '--train-end', default='10_3', help='Last training scene (inclusive)')
    parser.add_argument(
        '--splits', nargs='+', choices=('train', 'val', 'test'),
        default=['train', 'test'])
    parser.add_argument(
        '--test-points-dir', type=Path, default=Path('test/whu-sy-test'),
        help='Test PLY directory, relative to SOURCE_ROOT or absolute')
    parser.add_argument(
        '--test-reference-dir', type=Path, default=Path('test/reference_data'),
        help='Test NPY label directory, relative to SOURCE_ROOT or absolute')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate inputs and list selected scenes without writing files')
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Output directory (default: SOURCE_ROOT/processed)')
    read_mode = parser.add_mutually_exclusive_group()
    read_mode.add_argument(
        '--whole-scene', action='store_true',
        help='Load and process each complete PLY without chunking')
    read_mode.add_argument(
        '--chunk-size',
        type=int,
        default=500_000,
        help='Vertices processed at a time to keep memory use bounded')
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error('--chunk-size must be greater than zero')
    if scene_key(args.train_start) > scene_key(args.train_end):
        parser.error('--train-start must not be after --train-end')
    chunk_size = None if args.whole_scene else args.chunk_size

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root
    if output_root is None:
        output_root = source_root / 'processed'
    output_root = output_root.expanduser().resolve()

    # Validate every PLY/NPY pair before producing any output.
    splits = {}
    for split in args.splits:
        sources = collect_sources(
            source_root, split, args.train_start, args.train_end,
            args.test_points_dir, args.test_reference_dir)
        splits[split] = sources
        print(f'{split}: {len(sources)} scenes')
        for source, reference, count in sources:
            suffix = f' + {reference.name}' if reference else ''
            print(f'  {source.name}{suffix}: {count:,} points')
    total_points = sum(count for sources in splits.values()
                       for _, _, count in sources)
    print(f'Output: {output_root}; binary data: '
          f'{total_points * 32 / 2**30:.2f} GiB')
    print('Read mode: whole scene' if args.whole_scene
          else f'Read mode: {chunk_size:,} points per chunk')
    if args.dry_run:
        return

    for split, sources in splits.items():
        prepare_split(output_root, split, chunk_size, sources)
    print(f'WHU-Stree preprocessing complete: {output_root}')


if __name__ == '__main__':
    main()
