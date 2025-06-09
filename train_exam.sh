CUDA_VISIBLE_DEVICES=1 python vla-scripts/train.py \
  --pretrained_checkpoint /data1/model_weight/pretrain_weight/openvla/openvla-7b \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir /data1/workspace/wxl/data/ \
  --run_root_dir /data1/workspace/huqiong/openvla-oft \
  --run_id parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state_0520 \
  --image_aug True \
  --wandb_project ruijia  \
  --wandb_entity openvla  \
  --save_interval 100 \
  --is_resume False

