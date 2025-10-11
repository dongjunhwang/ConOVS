import os
import json
import copy
import torch
import numpy as np
import random
import argparse
from tqdm.auto import tqdm

from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Instances, Boxes

from fcclip.data.datasets.register_cityscapes_panoptic import load_cityscapes_panoptic
from fcclip.data.datasets import openseg_classes

root = os.getenv("DETECTRON2_DATASETS", "datasets")
CITYSCAPES_CATEGORIES, CITYSCAPES_CATEGORIES_ORIGINAL = openseg_classes.get_cityscapes_categories_with_prompt_eng()

# rename to avoid conflict
_RAW_CITYSCAPES_PANOPTIC_SPLITS = {
    "openvocab_cityscapes_fine_panoptic_train": (
        "cityscapes/leftImg8bit/train",
        "cityscapes/gtFine/cityscapes_panoptic_train",
        "cityscapes/gtFine/cityscapes_panoptic_train.json",
    )
    # "cityscapes_fine_panoptic_test": not supported yet
}


def load_cityscapes_inst():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Custom
    thing_classes_orig = [k["name"] for k in CITYSCAPES_CATEGORIES_ORIGINAL]
    thing_colors_orig = [k["color"] for k in CITYSCAPES_CATEGORIES_ORIGINAL]
    stuff_classes_orig = [k["name"] for k in CITYSCAPES_CATEGORIES_ORIGINAL]
    stuff_colors_orig = [k["color"] for k in CITYSCAPES_CATEGORIES_ORIGINAL]

    meta["thing_classes_orig"] = thing_classes_orig
    meta["thing_colors_orig"] = thing_colors_orig
    meta["stuff_classes_orig"] = stuff_classes_orig
    meta["stuff_colors_orig"] = stuff_colors_orig

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    storage = []

    for key, (image_dir, gt_dir, gt_json) in _RAW_CITYSCAPES_PANOPTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)
        storage.append(load_cityscapes_panoptic(image_dir, gt_dir, gt_json, meta))

    return storage


def load_instance(dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format='RGB')
    utils.check_image_size(dataset_dict, image)
    image_shape = image.shape[:2]  # h, w

    # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    # Therefore it's important to use torch.Tensor.

    if "pan_seg_file_name" in dataset_dict:
        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
        segments_info = dataset_dict["segments_info"]

        # apply the same transformation to panoptic segmentation
#         pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        instances = Instances(image_shape)
        classes = []
        masks = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_boxes = masks.get_bounding_boxes()

        dataset_dict["instances"] = instances

    return dataset_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cityscapes')
    args = parser.parse_args()

    storage, load_json_path, save_json_path = None, None, None
    if args.dataset == 'cityscapes':
        storage = load_cityscapes_inst()
        load_json_path = 'datasets/cityscapes/gtFine/cityscapes_panoptic_train.json'
        save_json_path = 'utils/cityscapes_panoptic_train_point.json'

    with open(load_json_path, 'r') as f:
        save_json_file = json.load(f)

    random.seed(1234)

    for img_idx, info in tqdm(enumerate(storage[0])):
        extend_storage = load_instance(info)
        inst_info = extend_storage['instances'].get_fields()
        # point_coords = [y0, x0]
        mask_idx = 0
        segment_len = len(save_json_file['annotations'][img_idx]['segments_info'])
        for segment_idx in range(segment_len):
            if not save_json_file['annotations'][img_idx]['segments_info'][segment_idx]['iscrowd']:
                gt_coords = inst_info['gt_masks'][mask_idx].nonzero()
                point_coords = gt_coords[random.randint(0, len(gt_coords) - 1)]
                save_json_file['annotations'][img_idx]['segments_info'][segment_idx]['point'] = point_coords.tolist()
                mask_idx += 1

    with open(save_json_path, 'w') as f:
        json.dump(save_json_file, f)