python train_net.py --num-gpus 1 \
    --config-file \
    configs/ade20k/previous/fcclip_convnext_large_eval_ade20k_eclipse.yaml \
    MODEL.WEIGHTS "./checkpoints/eclipse/eclipse_ade20k.pth" \
    MODEL.ECLIPSE.NUM_PROMPTS 1 \
    MODEL.ECLIPSE.PROMPT_DEEP True \
    DATASETS.TEST "('openvocab_coco_2017_val_panoptic_with_sem_seg','openvocab_cityscapes_fine_panoptic_val','openvocab_ade20k_panoptic_val',)"