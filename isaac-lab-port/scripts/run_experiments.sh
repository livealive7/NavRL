#!/bin/bash
# Sequentially: train 4 variants from the official checkpoint, then sweep-evaluate each (eval line, 256 envs, 80/10).
source ~/Volume/IsaacLab/.venv/bin/activate
cd ~/Volume/NavRL/isaac-lab-port
COMMON="--resume ../quick-demos/ckpts/navrl_checkpoint.pt --num_envs 1024 --num_static 80 --num_dynamic 10 --max_iterations 4000 --save_interval 500"
run() {  # name, extra args
  python scripts/train.py $COMMON $2 --run_name $1 > logs/$1.out 2>&1
  python scripts/eval_sweep.py --run logs/navrl/$1 --num_envs 256 --episodes 2 --num_static 80 --num_dynamic 10 > logs/$1_eval.out 2>&1
  echo "done $1 $(date)" >> logs/experiments.status
}
: > logs/experiments.status
run exp_A_base ""
run exp_B_desync "--randomize_episode_start"
run exp_C_lr1e4 "--lr 1e-4"
run exp_D_both "--randomize_episode_start --lr 1e-4"
echo "ALL DONE $(date)" >> logs/experiments.status
