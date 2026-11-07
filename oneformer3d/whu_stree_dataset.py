"""Dataset registration for WHU-Stree point clouds."""

from mmdet3d.datasets.scannet_dataset import ScanNetDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class WHUStreeDataset(ScanNetDataset):
    """ScanNet-style dataset metadata for two-class WHU-Stree labels.

    Semantic masks and dataset metadata use only 0 (ground) and 1 (tree).
    """

    METAINFO = {
        'classes': ('ground', 'tree'),
        'palette': [[128, 128, 128], [34, 139, 34]],
        'seg_valid_class_ids': (0, 1),
        'seg_all_class_ids': (0, 1),
    }
