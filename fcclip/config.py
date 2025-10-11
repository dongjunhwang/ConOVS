# -*- coding: utf-8 -*-
"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/config.py
"""
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_fcclip_config(cfg):
    # FC-CLIP model config
    cfg.MODEL.FC_CLIP = CN()
    cfg.MODEL.FC_CLIP.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS = "laion2b_s29b_b131k_ft_soup"
    cfg.MODEL.FC_CLIP.EMBED_DIM = 768
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA = 0.4
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA = 0.8
    cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK = False


def add_custom_config(cfg):
    cfg.INPUT.SAM = CN()
    cfg.INPUT.SAM.STORAGE = False
    cfg.INPUT.SAM.STORAGE_PATH = ""
    cfg.INPUT.SAM.DIRECT = False
    cfg.MODEL.PROMPT_TUNING = False

    # Prompt
    cfg.MODEL.PROMPT_TUNING = CN()
    cfg.MODEL.PROMPT_TUNING.ADDITIONAL_PROMPT = False
    cfg.MODEL.PROMPT_TUNING.NUM_QUERIES = 0
    cfg.MODEL.PROMPT_TUNING.FREEZE_PARAM_NAMES = ['mask_embed', 'class_embed', 'query_feat', 'query_embed']
    cfg.MODEL.PROMPT_TUNING.TASK_ARITHMETIC = False
    cfg.MODEL.PROMPT_TUNING.TASK_ARITHMETIC_LAMBDA = 0.25
    cfg.MODEL.PROMPT_TUNING.ANALYSIS = False
    cfg.MODEL.PROMPT_TUNING.ALL_TRAIN = False
    cfg.MODEL.PROMPT_TUNING.L2_REG = False

    # Continual
    cfg.MODEL.CONTINUAL = CN()
    cfg.MODEL.CONTINUAL.MASKING_LOSS = False
    cfg.MODEL.CONTINUAL.MASKING_LOSS_REVERSE = False
    cfg.MODEL.CONTINUAL.FINETUNE_TASK_ARITH = False
    cfg.MODEL.CONTINUAL.FINETUNE_TASK_ARITH_PREV = False

    # Progressive
    cfg.MODEL.CONTINUAL.PROGRESS = CN()
    cfg.MODEL.CONTINUAL.PROGRESS.FLAG = False
    cfg.MODEL.CONTINUAL.PROGRESS.CHANGE_ITER = 1000
    cfg.MODEL.CONTINUAL.PROGRESS.FREEZE_PARAM = ['predictor']
    cfg.MODEL.CONTINUAL.PROGRESS.SCALING_COEF = 0.8

    # Context
    cfg.MODEL.CONTINUAL.CONTEXT = CN()
    cfg.MODEL.CONTINUAL.CONTEXT.CONTEXT_TEMPLATE = ""

    # Ensemble
    cfg.MODEL.ENSEMBLE = CN()
    cfg.MODEL.ENSEMBLE.DOMAIN_SELECTION = False
    cfg.MODEL.ENSEMBLE.EXTRACT_PROTOTYPE = False
    cfg.MODEL.ENSEMBLE.TEXT_PROTOTYPE = False
    cfg.MODEL.ENSEMBLE.TEXT_PROTOTYPE_TYPE = 'division'
    cfg.MODEL.ENSEMBLE.PROTOTYPES_PATH = []
    cfg.MODEL.ENSEMBLE.FINETUNED_MODELS_WEIGHT = []
    cfg.MODEL.ENSEMBLE.SOFTMAX_TEMPERATURE = 0.01
    cfg.MODEL.ENSEMBLE.PROTOTYPE_NAME = 'kmeans'
    cfg.MODEL.ENSEMBLE.METHOD_ANALYSIS = False
    cfg.MODEL.ENSEMBLE.ONE_HOT = False
    cfg.MODEL.ENSEMBLE.WEIGHT_CONSTANT = False
    cfg.MODEL.ENSEMBLE.NO_IMAGE_PROTOTYPE = False
    cfg.MODEL.ENSEMBLE.PROMPT_SELECTION = False

    # Previous Methods
    # LwF
    cfg.MODEL.LWF = CN()
    cfg.MODEL.LWF.TEMPERATURE = 1.0
    cfg.MODEL.LWF.ALPHA = 0.5
    cfg.SOLVER.TRAINER = "Trainer"
    # EWC
    cfg.MODEL.EWC = CN()
    cfg.MODEL.EWC.IMPORTANCE_LAMBDA = 100.0
    # Save and choose best
    cfg.SOLVER.BEST_CHECKPOINT = False
    # ER
    cfg.DATASETS.ER_OLD_DATASETS = ("openvocab_coco_2017_train_panoptic_with_sem_seg",)
    cfg.MODEL.ER = CN()
    cfg.MODEL.ER.BUFFER_SIZE = 100
    cfg.MODEL.ER.NUM_CLASSES_LIST = [133]
    cfg.MODEL.ER.ALL_SAMPLES = False
    cfg.MODEL.ER.JOINT_TRAINING_ALL = False
    cfg.MODEL.ER.DIVIDE_RATIO = [1.0, 1.0]
    cfg.MODEL.ER.DIVIDE_ABLATE = False
    # ECLIPSE
    cfg.MODEL.ECLIPSE = CN()
    cfg.MODEL.ECLIPSE.NUM_PROMPTS = 1
    cfg.MODEL.ECLIPSE.PROMPT_DEEP = False
    cfg.MODEL.ECLIPSE.NO_CONCAT = False
    # Calculate Time
    cfg.SOLVER.CALCULATE_HEAD_TIME = False
    cfg.VISUALIZE = False

