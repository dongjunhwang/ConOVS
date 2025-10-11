# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data.datasets.lvis import register_lvis_instances

from .register_coco_instance import _get_coco_instances_meta
from .lvis_v1_category_image_count import LVIS_CATEGORY_IMAGE_COUNT as LVIS_V1_CATEGORY_IMAGE_COUNT

from . import openseg_classes

"""
This file contains functions to parse LVIS-format annotations into dicts in the
"Detectron2 format".
"""

LVIS_V1_CATEGORIES = openseg_classes.get_lvis_v1_categories_with_prompt_eng()

_PREDEFINED_SPLITS_LVIS = {
    "openvocab_lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
    # "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
}


def get_lvis_instances_meta(dataset_name):
    """
    Load LVIS metadata.

    Args:
        dataset_name (str): LVIS dataset name without the split name (e.g., "lvis_v0.5").

    Returns:
        dict: LVIS metadata with keys: thing_classes
    """
    if "cocofied" in dataset_name:
        return _get_coco_instances_meta()
    elif "v1" in dataset_name:
        return _get_lvis_instances_meta_v1()
    raise ValueError("No built-in metadata for dataset {}".format(dataset_name))


def _get_lvis_instances_meta_v1():
    assert len(LVIS_V1_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {
        "thing_classes": thing_classes,
        "class_image_count": LVIS_V1_CATEGORY_IMAGE_COUNT,
    }
    return meta


def register_all_lvis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVIS.items():
        register_lvis_instances(
            key,
            get_lvis_instances_meta(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_lvis(_root)
