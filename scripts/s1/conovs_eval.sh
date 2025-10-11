python train_net.py --eval-only --num-gpus 1 \
    --config-file ./configs/cityscapes/custom/fcclip_convnext_large_eval_cityscapes_conovs.yaml \
    SOLVER.IMS_PER_BATCH 1 \
    MODEL.ENSEMBLE.FINETUNED_MODELS_WEIGHT "['./checkpoints/finetuning/cityscapes.pth']" \
    MODEL.ENSEMBLE.PROTOTYPES_PATH "['./mvn_dist/coco_gmm','./mvn_dist/cityscapes_gmm']" \
    DATASETS.TEST "('openvocab_coco_2017_val_panoptic_with_sem_seg','openvocab_cityscapes_fine_panoptic_val','openvocab_ade20k_panoptic_val',)"