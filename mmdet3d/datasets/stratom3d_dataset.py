# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class Stratom3DDataset(Det3DDataset):
    r"""Stratom 3D Dataset.

    This class serves as the API for experiments on Stratom's 3D data.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True).
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes.
            Defaults to [0, -40, -3, 70.4, 40, 0.0].
    """

    # TODO: use full classes of kitti
    METAINFO = {"classes": ("Dunnage"), "palette": [(92, 247, 245)]}

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        pipeline: List[Union[dict, Callable]] = [],
        modality: dict = dict(use_lidar=True),
        box_type_3d: str = "LiDAR",
        filter_empty_gt: bool = True,
        test_mode: bool = False,
        pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
        **kwargs
    ) -> None:

        self.pcd_limit_range = pcd_limit_range
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs
        )
        assert self.modality is not None
        assert box_type_3d.lower() in ("lidar", "camera")

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality["use_lidar"]:
            if "plane" in info:
                # convert ground plane to velodyne coordinates
                plane = np.array(info["plane"])
                lidar2cam = np.array(
                    info["images"]["CAM2"]["lidar2cam"], dtype=np.float32
                )
                reverse = np.linalg.inv(lidar2cam)

                (plane_norm_cam, plane_off_cam) = (plane[:3], -plane[:3] * plane[3])
                plane_norm_lidar = (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
                plane_off_lidar = (
                    reverse[:3, :3] @ plane_off_cam[:, None][:, 0] + reverse[:3, 3]
                )
                plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4,))
                plane_lidar[:3] = plane_norm_lidar
                plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
            else:
                plane_lidar = None

            info["plane"] = plane_lidar

        info = super().parse_data_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info["gt_bboxes_3d"] = np.zeros((0, 7), dtype=np.float32)
            ann_info["gt_labels_3d"] = np.zeros(0, dtype=np.int64)

        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info["gt_bboxes_3d"])
        ann_info["gt_bboxes_3d"] = gt_bboxes_3d
        return ann_info
