import random
import logging
import torch
import time

from torch import nn
from torch.nn import functional as F
import torch.utils.data as torchdata

from detectron2.utils.comm import get_world_size
from detectron2.config import configurable
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.samplers import InferenceSampler
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess

from train_net import Trainer
from fcclip import (
    COCOPanopticNewBaselineDatasetMapper,
    FCCLIP
)

from fcclip.modeling.criterion import SetCriterion
from fcclip.modeling.transformer_decoder.fcclip_transformer_decoder import get_classification_logits
from fcclip.modeling.matcher import HungarianMatcher

from ..utils.misc import is_dist_avail_and_initialized


VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]



class AMPTrainerER(AMPTrainer):
    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        num_dataset = len(self.model.dataset_names)+1
        batch_num = len(data) // num_dataset

        with autocast():
            for idx in range(num_dataset):
                loss_dict_temp = self.model(data[batch_num * idx:batch_num * (idx + 1)], data_idx=idx)
                if idx == 0:
                    loss_dict = loss_dict_temp
                else:
                    for key in loss_dict.keys():
                        loss_dict[key] = loss_dict[key] + loss_dict_temp[key]

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


def collect_class_ids(ann, class_to_samples, buffer_size):
    class_list = [info['category_id'] for info in ann]
    for class_id in class_list:
        if class_id not in class_to_samples:
            class_to_samples[class_id] = []
        if len(class_to_samples[class_id]) < buffer_size:
            return class_id, class_to_samples
    return None, class_to_samples


def create_coco_buffer(old_dataset_name, buffer_size=100, all_samples=False):
    coco_dicts = DatasetCatalog.get(old_dataset_name)
    if all_samples:
        return coco_dicts
    class_to_samples = {}

    for sample in coco_dicts:
        class_id, class_to_samples = collect_class_ids(sample['segments_info'],
                                                       class_to_samples, buffer_size)
        if class_id is not None:
            class_to_samples[class_id].append(sample)
            if all(len(samples) >= buffer_size for samples in class_to_samples.values()):
                break

    old_data_buffer = []
    for k, v in class_to_samples.items():
        old_data_buffer = old_data_buffer + v
    return old_data_buffer

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

class TrainerER(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._trainer = (AMPTrainerER if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            self.model, self.data_loader, self.optimizer
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_names):
        if cfg.MODEL.ER.DIVIDE_ABLATE:
            dataset_list = []
            for dataset_name, divide_ratio in zip(dataset_names, cfg.MODEL.ER.DIVIDE_RATIO):
                dataloader = super().build_test_loader(cfg, dataset_name)
                subset_size = int(len(dataloader.dataset) * divide_ratio)
                indices = list(range(subset_size))[0:subset_size]
                dataset = torchdata.Subset(dataloader.dataset, indices)
                dataset_list.append(dataset)

            combined_dataset = torchdata.ConcatDataset(dataset_list)

            return torchdata.DataLoader(
                combined_dataset,
                batch_size=dataloader.batch_size,
                sampler=InferenceSampler(len(combined_dataset)),
                drop_last=False,
                num_workers=dataloader.num_workers,
                collate_fn=trivial_batch_collator
            )
        else:
            return super().build_test_loader(cfg, dataset_names)

    @classmethod
    def build_train_loader(cls, cfg):
        logger = logging.getLogger("detectron2.trainer")

        # Change the new data loader batch_size
        dataset_num = len(cfg.DATASETS.ER_OLD_DATASETS)

        cfg.defrost()
        original_batch_size = cfg.SOLVER.IMS_PER_BATCH
        cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH // (dataset_num+1)
        new_data_loader = super().build_train_loader(cfg)
        cfg.SOLVER.IMS_PER_BATCH = original_batch_size
        cfg.freeze()

        batch_size = cfg.SOLVER.IMS_PER_BATCH
        new_data_iter = iter(new_data_loader)
        old_data_buffer = create_coco_buffer(cfg.DATASETS.ER_OLD_DATASETS[0],
                                             buffer_size=cfg.MODEL.ER.BUFFER_SIZE,
                                             all_samples=cfg.MODEL.ER.ALL_SAMPLES or cfg.MODEL.ER.JOINT_TRAINING_ALL)
        logger.info(f"Old Dataset Buffer Length: {len(old_data_buffer)}")
        coco_mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)

        if cfg.MODEL.ER.DIVIDE_ABLATE:
            divide_ratio = cfg.MODEL.ER.DIVIDE_RATIO
            new_data_buffer = create_coco_buffer(cfg.DATASETS.TRAIN[0],
                                                 buffer_size=cfg.MODEL.ER.BUFFER_SIZE,
                                                 all_samples=cfg.MODEL.ER.ALL_SAMPLES or cfg.MODEL.ER.JOINT_TRAINING_ALL)
            new_data_buffer = new_data_buffer[0:round(len(new_data_buffer) * divide_ratio[0])]
            old_data_buffer = old_data_buffer[0:round(len(old_data_buffer) * divide_ratio[1])]
            while True:
                batch = []
                for _ in range(batch_size // 2):
                    sample = random.choice(new_data_buffer)
                    processed_sample = coco_mapper(sample)
                    batch.append(processed_sample)

                for _ in range(batch_size // 2):
                    sample = random.choice(old_data_buffer)
                    processed_sample = coco_mapper(sample)
                    batch.append(processed_sample)
                yield batch

        if not cfg.MODEL.ER.JOINT_TRAINING_ALL:
            while True:
                batch = []
                for _ in range(batch_size // 2 // new_data_loader.batch_size):
                    try:
                        batch = batch + next(new_data_iter)
                    except StopIteration:
                        new_data_iter = iter(new_data_loader)
                        batch = batch + next(new_data_iter)

                for _ in range(batch_size // 2):
                    sample = random.choice(old_data_buffer)
                    processed_sample = coco_mapper(sample)
                    batch.append(processed_sample)
                yield batch
        else:
            # Above 3 datasets
            dataset_num = len(cfg.DATASETS.ER_OLD_DATASETS)
            data_buffer_list = []
            mapper_list = []
            for idx in range(1, dataset_num):
                temp_data_buffer = create_coco_buffer(cfg.DATASETS.ER_OLD_DATASETS[idx],
                                                      buffer_size=cfg.MODEL.ER.BUFFER_SIZE,
                                                      all_samples=cfg.MODEL.ER.ALL_SAMPLES or cfg.MODEL.ER.JOINT_TRAINING_ALL)
                data_buffer_list.append(temp_data_buffer)
                logger.info(f"{idx}-th Dataset Buffer Length: {len(temp_data_buffer)}")
                mapper_list.append(COCOPanopticNewBaselineDatasetMapper(cfg, True))

            while True:
                batch = []
                total_dataset_num = dataset_num + 1 # add the new dataset
                for _ in range(batch_size // total_dataset_num):
                    try:
                        batch = batch + next(new_data_iter)
                    except StopIteration:
                        new_data_iter = iter(new_data_loader)
                        batch = batch + next(new_data_iter)

                for _ in range(batch_size // total_dataset_num):
                    sample = random.choice(old_data_buffer)
                    processed_sample = coco_mapper(sample)
                    batch.append(processed_sample)

                for idx in range(1, dataset_num):
                    for _ in range(batch_size // total_dataset_num):
                        sample = random.choice(data_buffer_list[idx-1])
                        processed_sample = mapper_list[idx-1](sample)
                        batch.append(processed_sample)

                yield batch


class SetCriterionER(SetCriterion):
    def __init__(self, *args, **kwargs):
        self.num_classes_list = kwargs.pop("num_classes_list", None)
        super().__init__(*args, **kwargs)
        self.num_classes_list = [self.num_classes] + self.num_classes_list
        self.empty_weight_list = [self.empty_weight]
        for idx in range(1, len(self.num_classes_list)):
            temp_empty_weight = torch.ones(self.num_classes_list[idx] + 1)
            temp_empty_weight[-1] = self.eos_coef
            self.empty_weight_list.append(temp_empty_weight)

    def loss_masks(self, outputs, targets, indices, num_masks, data_idx=0):
        return super().loss_masks(outputs, targets, indices, num_masks)

    def loss_labels(self, outputs, targets, indices, num_masks, data_idx=0):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        num_classes = self.num_classes_list[data_idx]
        empty_weight = self.empty_weight_list[data_idx]
        if src_logits.device != empty_weight.device:
            empty_weight = empty_weight.to(src_logits.device)

        target_classes = torch.full(
            src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight, reduction='none')
        losses = {"loss_ce": loss_ce.mean()}
        if self.analysis:
            losses['target_classes_o'] = target_classes_o
            losses['loss_ce_inst'] = loss_ce[idx]
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks, data_idx=0):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, data_idx)

    def forward(self, outputs, targets, data_idx=0):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
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
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, data_idx))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, data_idx)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


@META_ARCH_REGISTRY.register()
class FCCLIPER(FCCLIP):
    @configurable
    def __init__(self, *args, **kwargs):
        self.dataset_names = kwargs.pop("dataset_names", None)
        super().__init__(*args, **kwargs)
        self.train_text_classifier_list, self.train_num_templates_list, self.train_class_names_list = [],[],[]
        for dataset_name in self.dataset_names:
            train_metadata = MetadataCatalog.get(dataset_name)
            _, temp_train_num_templates, temp_train_class_names = \
                self.prepare_class_names_from_metadata(train_metadata, train_metadata)
            self.train_num_templates_list.append(temp_train_num_templates)
            self.train_class_names_list.append(temp_train_class_names)
            self.train_text_classifier_list.append(None)

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

        criterion = SetCriterionER(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            num_classes_list=cfg.MODEL.ER.NUM_CLASSES_LIST
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
            # Custom
            "dataset_names": cfg.DATASETS.ER_OLD_DATASETS
        }

    def get_old_text_classifier(self, data_idx):
        if self.train_text_classifier_list[data_idx] is None:
            text_classifier = []
            # this is needed to avoid oom, which may happen when num of class is large
            bs = 128
            for idx in range(0, len(self.train_class_names_list[data_idx]), bs):
                text_classifier.append(self.backbone.get_text_classifier(self.train_class_names_list[data_idx][idx:idx+bs], self.device).detach())
            text_classifier = torch.cat(text_classifier, dim=0)

            # average across templates and normalization.
            text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
            text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
            text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
            self.train_text_classifier_list[data_idx] = text_classifier
        return self.train_text_classifier_list[data_idx], self.train_num_templates_list[data_idx]

    def forward(self, batched_inputs, data_idx=0):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        if data_idx == 0:
            text_classifier, num_templates = self.get_text_classifier()
        else:
            text_classifier, num_templates = self.get_old_text_classifier(data_idx=data_idx-1)

        # Append void class weight
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, data_idx=data_idx)

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