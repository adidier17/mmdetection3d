import json
import math
import mmengine
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple, List, Union, Optional
from os import PathLike


def get_info(
    root_dir: PathLike,
    sensor_name: str,
    frame_ids: list,
    annotations_dir: Path,
    num_points_feats: int,
    relative_path: bool,
):
    # Create the info file
    info = []
    # TODO: Do I need the sample_idx key?
    for frame_id in frame_ids:
        lidar_path = f"{sensor_name}/{frame_id}.bin"
        if not relative_path:
            lidar_path = Path(root_dir) / lidar_path
        frame_info = {
            "frame_id": frame_id,
            "lidar_points": {
                "num_pts_feats": num_points_feats,
                "lidar_path": str(lidar_path),
            },
        }
        with open(annotations_dir / f"{frame_id}.json", "r") as f:
            annotations = json.load(f)["instances"]

        for idx in range(len(annotations)):
            if "bbox_label_3d" not in annotations[idx]:
                annotations[idx]["bbox_label_3d"] = 0
        frame_info["instances"] = annotations
        info.append(frame_info)
        info_dict = {
            "data_list": info,
            "metainfo": {"categories": {"Dunnage": 0}, "dataset": "stratom3d"},
        }

    return info_dict


def get_splits(
    sensor_files: List[PathLike],
    frame_ids: List[str],
    splits_file: Optional[PathLike] = None,
) -> Tuple[List[str], List[str], List[str]]:
    if splits_file is not None:
        with open(splits_file, "r") as f:
            splits = json.load(f)
        train_ids = splits["train"]
        val_ids = splits["val"]
        if "test" in splits:
            test_ids = splits["test"]
        else:
            test_ids = []
    else:
        scene_ids = [int(f.stem.split("_")[0]) for f in sensor_files]
        scene_ids = list(set(scene_ids))
        scene_num = len(scene_ids)
        train_num = math.floor(scene_num * 0.7)
        val_num = math.floor(scene_num * 0.15)
        # take all frame ids from i_j ids where i <= train_num
        train_ids = [_id for _id in frame_ids if int(_id.split("_")[0]) <= train_num]
        val_ids = [
            _id
            for _id in frame_ids
            if int(_id.split("_")[0]) > train_num
            and int(_id.split("_")[0]) <= train_num + val_num
        ]
        test_ids = [
            _id for _id in frame_ids if int(_id.split("_")[0]) > train_num + val_num
        ]
    return train_ids, val_ids, test_ids


def create_stratom_info_file(
    data_root,
    sensor_name="_combined_ouster_points",
    num_points_feats=3,
    pkl_prefix="stratom3d",
    splits_file=None,
    save_path=None,
    relative_path=True,
):
    """
    Create a Stratom3D info file from the given data root.
    """
    # Check the data for validity
    sensor_dir = Path(data_root) / sensor_name
    annotations_dir = Path(data_root) / "annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not sensor_dir.exists():
        raise FileNotFoundError(f"Sensor directory not found: {sensor_dir}")
    sensor_files = list(sensor_dir.glob("*.bin"))
    annotation_files = list(annotations_dir.glob("*.json"))
    sensor_frame_ids = set([f.stem for f in sensor_files])
    annotation_frame_ids = set([f.stem for f in annotation_files])
    assert (
        sensor_frame_ids == annotation_frame_ids
    ), f"Sensor frames not in annotations: {sensor_frame_ids - annotation_frame_ids} \n Annotations with no sensor frames: {annotation_frame_ids - sensor_frame_ids}"

    if save_path is None:
        train_save_path = Path(data_root) / f"{pkl_prefix}_infos_train.pkl"
        val_save_path = Path(data_root) / f"{pkl_prefix}_infos_val.pkl"
        test_save_path = Path(data_root) / f"{pkl_prefix}_infos_test.pkl"
    else:
        train_save_path = Path(save_path) / f"{pkl_prefix}_infos_train.pkl"
        val_save_path = Path(save_path) / f"{pkl_prefix}_infos_val.pkl"
        test_save_path = Path(save_path) / f"{pkl_prefix}_infos_test.pkl"

    train_ids, val_ids, test_ids = get_splits(
        sensor_files, sensor_frame_ids, splits_file
    )
    train_info = get_info(
        data_root,
        sensor_name,
        train_ids,
        annotations_dir,
        num_points_feats,
        relative_path,
    )
    mmengine.dump(train_info, train_save_path)
    print(f"Train info saved to {train_save_path}")

    val_info = get_info(
        data_root,
        sensor_name,
        val_ids,
        annotations_dir,
        num_points_feats,
        relative_path,
    )
    mmengine.dump(val_info, val_save_path)
    print(f"Val info saved to {val_save_path}")
    if test_ids:
        test_info = get_info(
            data_root,
            sensor_name,
            test_ids,
            annotations_dir,
            num_points_feats,
            relative_path,
        )
        mmengine.dump(test_info, test_save_path)
        print(f"Test info saved to {test_save_path}")
