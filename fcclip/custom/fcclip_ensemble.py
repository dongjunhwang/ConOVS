import time
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fcclip import FCCLIP

from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess

from sklearn.mixture import GaussianMixture
from fcclip.modeling.transformer_decoder.fcclip_transformer_decoder import get_classification_logits
from copy import deepcopy

from fcclip.custom.task_vectors import TaskVectorEnsemble
from fcclip.fcclip import VILD_PROMPT

from fcclip.utils.time_calculator import TimeCalculator

PLOT_COLOR = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
              'white', 'gray', 'grey', 'maroon', 'olive', 'purple', 'teal', 'navy']


@META_ARCH_REGISTRY.register()
class FCCLIPConOVS(FCCLIP):
    @configurable
    def __init__(self, *args, **kwargs):
        self.output_dir = kwargs.pop("output_dir", None)
        self.domain_selection = kwargs.pop("domain_selection", None)
        self.extract_prototype = kwargs.pop("extract_prototype", None)
        self.prototype_name = kwargs.pop("prototype_name", None)
        self.prototypes_path = kwargs.pop("prototypes_path", None)
        self.softmax_temperature = kwargs.pop("softmax_temperature", None)
        self.text_prototype = kwargs.pop("text_prototype", None)
        self.text_prototype_type = kwargs.pop("text_prototype_type", None)
        self.method_analysis = kwargs.pop("method_analysis", None)
        self.one_hot = kwargs.pop("one_hot", None)
        self.weight_constant = kwargs.pop("weight_constant", None)
        self.calculate_head_time = kwargs.pop("calculate_head_time", None)
        self.no_image_prototype = kwargs.pop("no_image_prototype", None)
        self.prompt_selection = kwargs.pop("prompt_selection", None)
        finetuned_models_weight = kwargs.pop("finetuned_models_weight", None)
        pretrained_model_weight = kwargs.pop("pretrained_model_weight", None)

        super().__init__(*args, **kwargs)

        self.sem_seg_head_ensemble = deepcopy(self.sem_seg_head)
        self.prototype_text = None
        self.text_prototype_distance = None
        self.weight_list = []
        if self.extract_prototype:
            self.prototype_list = []
        else:
            with torch.no_grad():
                self.prototype_tensor = []
                self.prototype_tensor_text = []
                for p in self.prototypes_path:
                    with open(p + '.pkl', 'rb') as f:
                        self.prototype_tensor.append(pickle.load(f))
                    with open(p + '_text.pkl', 'rb') as f:
                        self.prototype_tensor_text.append(pickle.load(f))

                # Make All Task Vectors
                self.task_vectors = []
                for f_model_weight in finetuned_models_weight:
                    psdict = torch.load(pretrained_model_weight)['model']
                    fsdict = torch.load(f_model_weight)['model']
                    self.task_vectors.append(TaskVectorEnsemble(pretrained_state_dict=psdict,
                                                                finetuned_state_dict=fsdict))

        self.time_cal_adaptive = TimeCalculator()
        self.time_cal_original = TimeCalculator()

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = \
            self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.prototype_list = []
        self.test_text_classifier = None
        self.prototype_text = None
        return

    @classmethod
    def from_config(cls, cfg):
        config_dict = super().from_config(cfg)
        config_dict['output_dir'] = cfg.OUTPUT_DIR
        config_dict['domain_selection'] = cfg.MODEL.ENSEMBLE.DOMAIN_SELECTION
        config_dict['extract_prototype'] = cfg.MODEL.ENSEMBLE.EXTRACT_PROTOTYPE
        config_dict['prototypes_path'] = cfg.MODEL.ENSEMBLE.PROTOTYPES_PATH
        config_dict['finetuned_models_weight'] = cfg.MODEL.ENSEMBLE.FINETUNED_MODELS_WEIGHT
        config_dict['pretrained_model_weight'] = cfg.MODEL.WEIGHTS
        config_dict['softmax_temperature'] = cfg.MODEL.ENSEMBLE.SOFTMAX_TEMPERATURE
        config_dict['prototype_name'] = cfg.MODEL.ENSEMBLE.PROTOTYPE_NAME
        config_dict['text_prototype'] = cfg.MODEL.ENSEMBLE.TEXT_PROTOTYPE
        config_dict['text_prototype_type'] = cfg.MODEL.ENSEMBLE.TEXT_PROTOTYPE_TYPE
        config_dict['method_analysis'] = cfg.MODEL.ENSEMBLE.METHOD_ANALYSIS
        config_dict['one_hot'] = cfg.MODEL.ENSEMBLE.ONE_HOT
        config_dict['weight_constant'] = cfg.MODEL.ENSEMBLE.WEIGHT_CONSTANT
        config_dict['calculate_head_time'] = cfg.SOLVER.CALCULATE_HEAD_TIME
        config_dict['no_image_prototype'] = cfg.MODEL.ENSEMBLE.NO_IMAGE_PROTOTYPE
        config_dict['prompt_selection'] = cfg.MODEL.ENSEMBLE.PROMPT_SELECTION
        return config_dict

    def weight_kde_dist(self):
        weights = np.array(self.weight_list)
        plt.figure(figsize=(6, 4))
        for i in range(weights.shape[-1]):
            dataset_name = self.prototypes_path[i].split('/')[-1].split('_')[0]
            sns.kdeplot(weights[:, i], fill=True, color=PLOT_COLOR[i], label=dataset_name)
            # sns.histplot(weights[:, i], bins=30, kde=False, color=PLOT_COLOR[i], label=dataset_name, element='step')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.xlim(0, 1)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'weight_dist_plot.png'), dpi=300)
        np.save(os.path.join(self.output_dir, 'weight_list.npy'), weights)

    def extract_prototype_func(self, batched_inputs):
        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            features = self.backbone(images.tensor)
        return features

    def get_text_prototype(self):
        if self.prototype_text is None:
            text_classifier = []
            # this is needed to avoid oom, which may happen when num of class is large
            bs = 128
            for idx in range(0, len(self.test_class_names), bs):
                text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
            text_classifier = torch.cat(text_classifier, dim=0)

            # average across templates and normalization.
            text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT),
                                                      text_classifier.shape[-1])
            text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
            self.prototype_text = text_classifier
        return self.prototype_text

    def save_gaussian_dist(self, features, dataset_idx, suffix='_gmm.pkl'):
        gmm = GaussianMixture(n_components=1, random_state=0).fit(features)
        with open(self.prototypes_path[dataset_idx] + suffix, 'wb') as f:
            pickle.dump(gmm, f)

    def save_prototype(self, dataset_idx):
        # Image Embedding
        features = torch.cat(self.prototype_list, dim=0)
        # Text Embedding
        text_features = self.get_text_prototype()
        text_features = text_features.reshape(text_features.shape[0] * text_features.shape[1], -1)

        # Save Image MVN Distribution
        self.save_gaussian_dist(features.numpy(), dataset_idx, suffix='_gmm.pkl')
        # Save Text MVN Distribution
        self.save_gaussian_dist(text_features.detach().cpu().numpy(), dataset_idx, suffix='_gmm_text.pkl')

    def calculate_text_distance(self):
        if self.text_prototype_distance is None:
            features = self.get_text_prototype()
            features = features.mean(dim=1)
            self.text_prototype_distance = self.prototype_distance(features, self.softmax_temperature,
                                                                   self.prototype_tensor_text)
            self.text_prototype_distance = self.text_prototype_distance.mean(dim=0)
        return self.text_prototype_distance

    def prototype_distance(self, features, softmax_temp, prototype_tensor):
        batch_list = []
        for feature in features:
            score_list = []
            for pkl in prototype_tensor:
                score_list.append(pkl.score(feature[None, :].detach().cpu()))
            batch_list.append(score_list)
        cosine_sim = F.softmax(torch.tensor(batch_list)*softmax_temp, dim=-1)

        return cosine_sim

    def optimized_task_arithmetics(self, features):
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.shape[:-2])
        cosine_sim = self.prototype_distance(pooled_features,
                                             self.softmax_temperature,
                                             self.prototype_tensor)
        cosine_distance = cosine_sim[0]
        text_distance = self.calculate_text_distance()
        if len(self.prototypes_path) <= 3:
            cosine_distance = torch.max(cosine_distance, text_distance)
        cosine_distance = cosine_distance[1:]
        # Manipulate Task Vectors
        task_vector_list = []
        for idx in range(len(self.task_vectors)):
            task_vector_list.append(self.task_vectors[idx] * cosine_distance[idx])
        sum_task_vector = sum(task_vector_list)
        self.sem_seg_head_ensemble = sum_task_vector.apply_to(self.sem_seg_head, self.sem_seg_head_ensemble)

    def forward(self, batched_inputs):
        if self.extract_prototype and not self.training:
            with torch.no_grad():
                # Image Feature Embedding
                features = self.extract_prototype_func(batched_inputs)
                pooled_features = F.adaptive_avg_pool2d(features['clip_vis_dense'], (1, 1))\
                    .view(features['clip_vis_dense'].shape[:-2])
                self.prototype_list.append(pooled_features.detach().cpu())
        elif not self.training:
            return self.forward_ensemble(batched_inputs)

        return super().forward(batched_inputs)

    def return_calculate_head_time(self):
        adapt = self.time_cal_adaptive.return_avg_sec()
        orig = self.time_cal_original.return_avg_sec()
        return f"{adapt}, {orig}"

    def forward_ensemble(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        text_classifier, num_templates = self.get_text_classifier()
        # Append void class weight
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates

        self.optimized_task_arithmetics(features['clip_vis_dense'])
        outputs = self.sem_seg_head_ensemble(features)

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