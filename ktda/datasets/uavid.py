# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class UavidDataset(BaseSegDataset):

    METAINFO = dict(
        classes=("Background-Clutter", "Building", "Road", "Tree", "Low-vegetation","Moving-Car","Static-Car","Human"),
        palette=[
            [0, 0, 0],
            [128, 0, 0],
            [128, 64, 128],
            [0, 128, 0],
            [128, 128, 0],
            [64, 0, 128],
            [192, 0, 192],
            [64, 64, 0]
        ],
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)