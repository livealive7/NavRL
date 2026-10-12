#!/bin/bash
source ~/Volume/IsaacLab/.venv/bin/activate
cd ~/Volume/NavRL/isaac-lab-port
python scripts/train.py --resume ../quick-demos/ckpts/navrl_checkpoint.pt --num_envs 1024 --num_static 80 --num_dynamic 10 --max_iterations 15000 --save_interval 1000 --lr 1e-4 --run_name exp_C_long > logs/exp_C_long.out 2>&1
python scripts/eval_sweep.py --run logs/navrl/exp_C_long --num_envs 256 --episodes 2 --num_static 80 --num_dynamic 10 > logs/exp_C_long_eval.out 2>&1
echo "done $(date)" > logs/exp_C_long.status
