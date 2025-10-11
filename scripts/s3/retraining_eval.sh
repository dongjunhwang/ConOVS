python train_net.py --num-gpus 1 \
    --config-file \
    configs/ade20k/custom/fcclip_convnext_large_eval_ade20k.yaml \
    MODEL.WEIGHTS "./checkpoints/retraining/retraining_cityscapes_ade20k.pth" \
    DATASETS.TEST "('openvocab_coco_2017_val_panoptic_with_sem_seg','openvocab_cityscapes_fine_panoptic_val','openvocab_ade20k_panoptic_val','openvocab_lvis_v1_val','openvocab_bdd100k_val_panoptic','openvocab_mapillary_vistas_panoptic_val','openvocab_pascal20_sem_seg_val','openvocab_pascal21_sem_seg_val','openvocab_pascal_ctx459_sem_seg_val','openvocab_pascal_ctx59_sem_seg_val','openvocab_ade20k_full_sem_seg_val')"