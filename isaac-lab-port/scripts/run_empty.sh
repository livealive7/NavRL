#!/bin/bash
# Sanity run: from scratch, original recipe (lr 5e-4), 1024 envs, EMPTY map (0 static, 0 dynamic), 10000 iterations.
source ~/Volume/IsaacLab/.venv/bin/activate
cd ~/Volume/NavRL/isaac-lab-port
python scripts/train.py --num_envs 1024 --num_static 0 --num_dynamic 0 --max_iterations 10000 --save_interval 1000 --run_name empty_0_0 > logs/empty_0_0.out 2>&1
python scripts/eval_sweep.py --run logs/navrl/empty_0_0 --num_envs 256 --episodes 2 --num_static 0 --no_dynamic > logs/empty_0_0_eval.out 2>&1
echo "done $(date)" > logs/empty_0_0.status
