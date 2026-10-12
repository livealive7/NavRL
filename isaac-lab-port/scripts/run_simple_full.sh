#!/bin/bash
# From scratch, original recipe (lr 5e-4), 1024 envs, 80 static / 0 dynamic obstacles, to the original frame budget (12e8 frames = 36621 iterations).
source ~/Volume/IsaacLab/.venv/bin/activate
cd ~/Volume/NavRL/isaac-lab-port
python scripts/train.py --num_envs 1024 --num_static 80 --num_dynamic 0 --max_iterations 36621 --save_interval 1000 --run_name simple_80_0_full > logs/simple_80_0_full.out 2>&1
python scripts/eval_sweep.py --run logs/navrl/simple_80_0_full --num_envs 256 --episodes 2 --num_static 80 --no_dynamic > logs/simple_80_0_full_eval.out 2>&1
echo "done $(date)" > logs/simple_80_0_full.status
