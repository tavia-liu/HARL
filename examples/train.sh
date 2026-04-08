CUDA_VISIBLE_DEVICES=1 python train.py --algo mappo --env maniskill --task TwoRobotPickCube-v1 --exp_name test --seed 1 \
--n_rollout_threads 512 --episode_length 50 --actor_num_mini_batch 32 --critic_num_mini_batch 32 \
--lr 0.0003 --critic_lr 0.0003 --gamma 0.8 --gae_lambda 0.9 \
--ppo_epoch 4 --critic_epoch 4 --entropy_coef 0.0 --value_loss_coef 0.5 \
--max_grad_norm 0.5 --use_clipped_value_loss False \
--use_valuenorm False --use_feature_normalization False \
--hidden_sizes '[256,256,256]' --activation_func tanh \
--share_param False

# KL early stop not handled in PPO update
# GAE last state value different from maniskill ppo
    # mappo: masked
    # ppo: not masked