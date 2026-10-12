#!/bin/bash
cd /home/peterson/Volume/NavRL/isaac-lab-port
for it in 8000 14500 12000 7500; do
  for mode in line random; do
    extra=""; [ $mode = random ] && extra="--random_start"
    echo "=== it $it $mode"
    python scripts/eval_policy.py --checkpoint logs/navrl/cur_s20/checkpoint_$it.pt --num_envs 1024 --episodes 3 --num_static 20 --no_dynamic $extra 2>&1 | grep -E "^\[eval\]"
  done
done
echo ALLDONE
