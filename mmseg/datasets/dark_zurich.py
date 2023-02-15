# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.17.0

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class DarkZurichDataset(CityscapesDataset):
    """DarkZurichDataset dataset."""

    def __init__(self, **kwargs):
        super(DarkZurichDataset, self).__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)
