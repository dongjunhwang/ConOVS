python train_net.py --num-gpus 1 \
  --config-file \
  configs/ade20k/previous/fcclip_convnext_large_eval_ade20k_eclipse.yaml \
  MODEL.WEIGHTS ./checkpoints/fcclip_cocopan.pth \
  SOLVER.STEPS "(30000, 35000)" \
  SOLVER.MAX_ITER 40000 \
  TEST.EVAL_PERIOD 10000 \
  SOLVER.CHECKPOINT_PERIOD 10000 \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.BASE_LR 0.005 \
  MODEL.ECLIPSE.NUM_PROMPTS 1 \
  MODEL.ECLIPSE.PROMPT_DEEP True