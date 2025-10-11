import torch
import time
from fcclip import FCCLIP

from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess


from fcclip.modeling.transformer_decoder.fcclip_transformer_decoder import get_classification_logits
from fcclip.modeling import FCCLIPHead
from fcclip.utils.time_calculator import TimeCalculator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


@SEM_SEG_HEADS_REGISTRY.register()
class FCCLIPHeadECLIPSE(FCCLIPHead):
    def forward(self, features, mask=None, training=False, no_concat=False, origin=False):
        return self.layers(features, mask, training, no_concat, origin)

    def layers(self, features, mask=None, training=False, no_concat=False, origin=False):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask,
                                         text_classifier=features["text_classifier"],
                                         num_templates=features["num_templates"],
                                         training=training, no_concat=no_concat, origin=False)
        else:
            raise NotImplementedError
        return predictions


@META_ARCH_REGISTRY.register()
class FCCLIPECLIPSE(FCCLIP):
    @configurable
    def __init__(self, *args, **kwargs):
        num_prompts = kwargs.pop('num_prompts', None)
        self.no_concat = kwargs.pop('no_concat', None)
        super().__init__(*args, **kwargs)
        self.tuned_only_prompt()
        if not self.no_concat:
            self.num_queries *= (num_prompts + 1)
        self.time_cal = TimeCalculator()

    @classmethod
    def from_config(cls, cfg):
        config_dict = super().from_config(cfg)
        config_dict['num_prompts'] = cfg.MODEL.ECLIPSE.NUM_PROMPTS
        config_dict['no_concat'] = cfg.MODEL.ECLIPSE.NO_CONCAT
        return config_dict

    def tuned_only_prompt(self):
        for param in self.sem_seg_head.parameters():
            param.requires_grad = False

        # Tuning the last prompt only
        for param in self.sem_seg_head.predictor.prompt_embed[-1].parameters():
            param.requires_grad = True
        for param in self.sem_seg_head.predictor.prompt_feat[-1].parameters():
            param.requires_grad = True

    def return_calculate_head_time(self):
        output = self.time_cal.return_avg_sec()
        return f"{output}"

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        text_classifier, num_templates = self.get_text_classifier()
        # Append void class weight
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates

        self.time_cal.time_start()
        outputs = self.sem_seg_head(features, training=self.training, no_concat=self.no_concat)
        self.time_cal.time_end()

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

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