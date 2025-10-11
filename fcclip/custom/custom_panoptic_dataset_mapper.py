"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py
"""

import copy
import logging

import numpy as np
import torch
import os

from PIL import Image
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances

from fcclip import COCOPanopticNewBaselineDatasetMapper

__all__ = ["CustomDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class CustomDatasetMapper(COCOPanopticNewBaselineDatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(self, *args, **kwargs):
        sam_storage_path = kwargs.pop('sam_storage_path', None)
        sam_storage = kwargs.pop('sam_storage', False)
        super().__init__(*args, **kwargs)
        self.sam_storage_path = sam_storage_path
        self.sam_storage = sam_storage

    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = super().from_config(cfg, is_train=True)
        ret['sam_storage_path'] = cfg.INPUT.SAM.STORAGE_PATH
        ret['sam_storage'] = cfg.INPUT.SAM.STORAGE
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image_ori"] = image
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            # We are not use the pan_set_gt (segmentation mask)
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
            # For SAM STORAGE
            mask_pan_seg_gt = None
            if self.sam_storage:
                sam_mask_path = os.path.join(self.sam_storage_path, dataset_dict["image_id"])
                if os.path.exists(sam_mask_path+'.png'):
                    sam_mask = np.array(Image.open(sam_mask_path+'.png'))
                else:
                    sam_mask = np.load(sam_mask_path+'.npy')
                mask_pan_seg_gt = transforms.apply_segmentation(sam_mask)

            instances = Instances(image_shape)
            classes = []
            points = []
            sam_masks = []
            sam_mask_cnt = 0
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    points.append(segment_info['point'])
                    if self.sam_storage:
                        sam_masks.append(mask_pan_seg_gt == sam_mask_cnt)
                    sam_mask_cnt += 1

            classes = np.array(classes)
            # For points
            points = np.array(points)[:, ::-1].astype(float) + 0.5
            points = transforms.apply_coords(points)

            points[:, 0] /= image_shape[0]
            points[:, 1] /= image_shape[1]

            inside = (points >= np.array([0, 0])) & (points <= np.array(image_shape[::-1]))
            points = np.delete(points, (~inside).nonzero()[0], axis=0)
            classes = np.delete(classes, (~inside).nonzero()[0], axis=0)
            if self.sam_storage:
                sam_masks = np.delete(sam_masks, (~inside).nonzero()[0], axis=0)
            matrix_points = points[None, :].repeat(points.shape[0], 0)
            matrix_labels = np.eye(points.shape[0])

            # store to the instances
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_points_orig = points
            # change to the matrix
            instances.gt_points = matrix_points
            instances.gt_points_labels = matrix_labels

            if self.sam_storage:
                instances.sam_masks = sam_masks
            dataset_dict["instances"] = instances

            # TODO: check point coordinates transformed well.

        return dataset_dict
