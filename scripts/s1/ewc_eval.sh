python train_net.py --num-gpus 1 \
    --config-file \
    configs/cityscapes/custom/fcclip_convnext_large_eval_cityscapes.yaml \
    MODEL.WEIGHTS "./checkpoints/ewc/ewc_cityscapes.pth" \
    DATASETS.TEST "('openvocab_coco_2017_val_panoptic_with_sem_seg','openvocab_cityscapes_fine_panoptic_val','openvocab_ade20k_panoptic_val',)"