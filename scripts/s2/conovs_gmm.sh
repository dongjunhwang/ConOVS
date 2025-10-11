python train_net.py \
    --eval-only \
    --num-gpus 1 \
    --config-file configs/ade20k/custom/fcclip_convnext_large_eval_ade20k_conovs.yaml \
    MODEL.ENSEMBLE.EXTRACT_PROTOTYPE True \
    MODEL.ENSEMBLE.FINETUNED_MODELS_WEIGHT "['./checkpoints/finetuning/ade20k.pth']" \
    MODEL.ENSEMBLE.PROTOTYPES_PATH "['./mvn_dist/coco_gmm','./mvn_dist/ade20k_gmm']" \
    DATASETS.TEST "('openvocab_coco_2017_train_panoptic_with_sem_seg','openvocab_ade20k_panoptic_train')"