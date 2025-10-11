python train_net.py --eval-only --num-gpus 1 \
    --config-file ./configs/ade20k/custom/fcclip_convnext_large_eval_ade20k_conovs.yaml \
    SOLVER.IMS_PER_BATCH 1 \
    MODEL.ENSEMBLE.FINETUNED_MODELS_WEIGHT "['./checkpoints/finetuning/cityscapes.pth','./checkpoints/finetuning/ade20k.pth']" \
    MODEL.ENSEMBLE.PROTOTYPES_PATH "['./mvn_dist/coco_gmm','./mvn_dist/cityscapes_gmm','./mvn_dist/ade20k_gmm']" \
    DATASETS.TEST "('openvocab_coco_2017_val_panoptic_with_sem_seg','openvocab_cityscapes_fine_panoptic_val','openvocab_ade20k_panoptic_val','openvocab_lvis_v1_val','openvocab_bdd100k_val_panoptic','openvocab_mapillary_vistas_panoptic_val','openvocab_pascal20_sem_seg_val','openvocab_pascal21_sem_seg_val','openvocab_pascal_ctx459_sem_seg_val','openvocab_pascal_ctx59_sem_seg_val','openvocab_ade20k_full_sem_seg_val')"