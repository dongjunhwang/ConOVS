import torch
import logging
import os
from copy import deepcopy
from fcclip import FCCLIP
from torch import nn

from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY

from detectron2.utils.registry import Registry
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.utils.comm import get_world_size
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_backbone, build_sem_seg_head
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import nested_tensor_from_tensor_list, is_dist_avail_and_initialized

from fcclip.modeling.transformer_decoder.fcclip_transformer_decoder import get_classification_logits
from fcclip.modeling.criterion import (
    SetCriterion, sigmoid_ce_loss_jit, dice_loss_jit, calculate_uncertainty
)
from fcclip.modeling.matcher import HungarianMatcher


class SetCriterionLwF(SetCriterion):
    def __init__(self, *args, **kwargs):
        self.temperature = kwargs.pop("temperature", None)
        self.alpha = kwargs.pop("alpha", None)
        super().__init__(*args, **kwargs)

    def loss_labels(self, outputs, outputs_old, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        src_logits_old = outputs_old["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        # To do, change to sigmoid?
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')

        src_logits = src_logits / self.temperature
        src_logits_old = torch.softmax(src_logits_old / self.temperature, dim=-1)
        loss_ce_old = F.cross_entropy(src_logits.transpose(1, 2), src_logits_old.transpose(1, 2),
                                      self.empty_weight, reduction='none') * (self.temperature ** 2)
        losses = {"loss_ce": self.alpha * loss_ce.mean() + (1-self.alpha) * loss_ce_old.mean()}
        return losses

    def loss_masks(self, outputs, outputs_old, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        src_masks_old = outputs_old["pred_masks"]
        src_masks_old = src_masks_old[src_idx]

        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        src_masks_old = src_masks_old[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)
            # get old logits
            point_logits_old = point_sample(
                src_masks_old,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
        loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

        point_logits = point_logits / self.temperature
        point_logits_old = torch.sigmoid(point_logits_old) / self.temperature
        loss_mask_old = sigmoid_ce_loss_jit(point_logits, point_logits_old, num_masks)
        loss_dice_old = dice_loss_jit(point_logits, point_logits_old, num_masks)

        losses = {
            "loss_mask": self.alpha * loss_mask.sum() + (1-self.alpha) * loss_mask_old.sum(),
            "loss_dice": self.alpha * loss_dice.sum() + (1-self.alpha) * loss_dice_old.sum(),
        }

        return losses

    def get_loss(self, loss, outputs, outputs_old, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, outputs_old, targets, indices, num_masks)

    def forward(self, outputs, outputs_old, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, outputs_old, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, (aux_outputs, aux_outputs_old) in enumerate(zip(outputs["aux_outputs"],
                                                                   outputs_old["aux_outputs"])):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, aux_outputs_old, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


@META_ARCH_REGISTRY.register()
class FCCLIPLwF(FCCLIP):
    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sem_seg_head_old = None
        self.deepcopy_flag = True

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterionLwF(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            temperature=cfg.MODEL.LWF.TEMPERATURE,
            alpha=cfg.MODEL.LWF.ALPHA
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                    or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                    or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK,
        }

    def forward(self, batched_inputs):
        if self.deepcopy_flag:
            self.deepcopy_flag = False
            self.sem_seg_head_old = deepcopy(self.sem_seg_head)
            self.sem_seg_head_old.requires_grad_(False)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        text_classifier, num_templates = self.get_text_classifier()
        # Append void class weight
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates
        outputs = self.sem_seg_head(features)
        with torch.no_grad():
            outputs_old = self.sem_seg_head_old(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, outputs_old, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            # We ensemble the pred logits of in-vocab and out-vocab
            clip_feature = features["clip_vis_dense"]
            mask_for_pooling = F.interpolate(mask_pred_results, size=clip_feature.shape[-2:],
                                             mode='bilinear', align_corners=False)
            if "convnext" in self.backbone.model_name.lower():
                pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
                pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
            elif "rn" in self.backbone.model_name.lower():
                pooled_clip_feature = self.backbone.visual_prediction_forward(clip_feature, mask_for_pooling)
            else:
                raise NotImplementedError

            out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier,
                                                              self.backbone.clip_model.logit_scale, num_templates)
            in_vocab_cls_results = mask_cls_results[..., :-1]  # remove void
            out_vocab_cls_results = out_vocab_cls_results[..., :-1]  # remove void

            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
            category_overlapping_mask = self.category_overlapping_mask.to(self.device)

            if self.ensemble_on_valid_mask:
                # Only include out_vocab cls results on masks with valid pixels
                # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(-1).sum(-1) > 0
                valid_masking = valid_masking.to(in_vocab_cls_results.dtype).unsqueeze(-1)
                alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
                beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
                alpha = alpha * valid_masking
                beta = beta * valid_masking
            else:
                alpha = self.geometric_ensemble_alpha
                beta = self.geometric_ensemble_beta

            cls_logits_seen = (
                    (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs ** alpha).log()
                    * category_overlapping_mask
            )
            cls_logits_unseen = (
                    (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs ** beta).log()
                    * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            # This is used to filtering void predictions.
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results
