# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .dark_zurich import DarkZurichDataset
from .gta import GTADataset
from .uda_dataset import UDADataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'DarkZurichDataset',
    'GTADataset',
    'UDADataset',
]
