python train_net.py --num-gpus 1 \
  --config-file \
  configs/cityscapes/custom/fcclip_convnext_large_eval_cityscapes.yaml \
  MODEL.WEIGHTS ./checkpoints/fcclip_cocopan.pth \
  SOLVER.STEPS "(7000, 8500)" \
  SOLVER.MAX_ITER 10000 \
  TEST.EVAL_PERIOD 2500 \
  SOLVER.CHECKPOINT_PERIOD 2500 \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.BASE_LR 0.00005