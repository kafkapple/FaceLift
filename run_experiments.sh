#!/bin/bash
cd /home/joon/dev/FaceLift

# E2.4: 4-View Alpha Mask + Random
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_4_4v_alpha_random.yaml > logs/d7t_e2_4_4v_alpha_random.log 2>&1 &
echo 'Started E2_4 on GPU 5'

# E3.3: 5-View Alpha Mask + Random
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E3_3_5v_alpha_random.yaml > logs/d7t_e3_3_5v_alpha_random.log 2>&1 &
echo 'Started E3_3 on GPU 6'

# E3.4: 6-View Alpha Mask + Random
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E3_4_6v_alpha_random.yaml > logs/d7t_e3_4_6v_alpha_random.log 2>&1 &
echo 'Started E3_4 on GPU 7'

# E4.3: 5-View Alpha Loss + Random
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E4_3_5v_alpha_loss_random.yaml > logs/d7t_e4_3_5v_alpha_loss_random.log 2>&1 &
echo 'Started E4_3 on GPU 0'

# E4.4: 6-View Alpha Loss + Random
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E4_4_6v_alpha_loss_random.yaml > logs/d7t_e4_4_6v_alpha_loss_random.log 2>&1 &
echo 'Started E4_4 on GPU 1'

echo 'All experiments started!'
