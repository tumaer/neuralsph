#!/bin/bash
# run this script with:
# nohup bash run_ldc2d.sh 2>&1 &

GPU=7
RLT_DIR="rlt/ldc2d"

run_basic() {
    ./venv/bin/python main.py --mode=infer --gpu=$GPU  --test \
    --model_dir=ckp/ldc_2d_seeds_gns/gns_2D_LDC_2708_10kevery100_20231022-080213/best \
    "$@"
}

run() {
    run_basic --eval_n_trajs_infer=10 --batch_size_infer=10 --metrics_stride_infer=20 \
    --n_rollout_steps=500 "$@"
}

run_long() {
    run_basic --eval_n_trajs_infer=-1 --batch_size_infer=4 --metrics_stride_infer=10 \
    --n_rollout_steps=400 "$@"
}

# run -rvp=None --rollout_dir=${RLT_DIR}/test0

# run -rl=1 -ra=0.005 -rrt=0.98 --rollout_dir=${RLT_DIR}/test10
# run -rl=1 -ra=0.01 -rrt=0.98 --rollout_dir=${RLT_DIR}/test11
# run -rl=1 -ra=0.015 -rrt=0.98 --rollout_dir=${RLT_DIR}/test12
# run -rl=1 -ra=0.02 -rrt=0.98 --rollout_dir=${RLT_DIR}/test13
# run -rl=1 -ra=0.03 -rrt=0.98 --rollout_dir=${RLT_DIR}/test14
# run -rl=1 -ra=0.04 -rrt=0.98 --rollout_dir=${RLT_DIR}/test15
# run -rl=1 -ra=0.05 -rrt=0.98 --rollout_dir=${RLT_DIR}/test16
# run -rl=1 -ra=0.07 -rrt=0.98 --rollout_dir=${RLT_DIR}/test17
# run -rl=1 -ra=0.1 -rrt=0.98 --rollout_dir=${RLT_DIR}/test18

# run -rl=2 -ra=0.01 -rrt=0.98 --rollout_dir=${RLT_DIR}/test20
# run -rl=5 -ra=0.01 -rrt=0.98 --rollout_dir=${RLT_DIR}/test21
# run -rl=10 -ra=0.01 -rrt=0.98 --rollout_dir=${RLT_DIR}/test22
# run -rl=5 -ra=0.02 -rrt=0.98 --rollout_dir=${RLT_DIR}/test23
# run -rl=5 -ra=0.02 -rrt=0.98 --rollout_dir=${RLT_DIR}/test24  # no bp. All later ones also have no background pressure
# run -rl=2 -ra=0.02 -rrt=0.98 --rollout_dir=${RLT_DIR}/test27
# run -rl=3 -ra=0.02 -rrt=0.98 --rollout_dir=${RLT_DIR}/test28
# run -rl=2 -ra=0.1 -rrt=0.98 --rollout_dir=${RLT_DIR}/test25
# run -rl=3 -ra=0.1 -rrt=0.98 --rollout_dir=${RLT_DIR}/test26
# run -rl=5 -ra=0.1 -rrt=0.98 --rollout_dir=${RLT_DIR}/test32
# run -rl=2 -ra=0.04 -rrt=0.98 --rollout_dir=${RLT_DIR}/test29
# run -rl=3 -ra=0.04 -rrt=0.98 --rollout_dir=${RLT_DIR}/test30
# run -rl=5 -ra=0.04 -rrt=0.98 --rollout_dir=${RLT_DIR}/test31

# run -rl=5 -ra=0.02 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test40
# run -rl=5 -ra=0.02 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test41
# run -rl=5 -ra=0.02 --redist_visc=0.5 --rollout_dir=${RLT_DIR}/test42
# run -rl=5 -ra=0.02 --redist_visc=1.0 --rollout_dir=${RLT_DIR}/test43
# run -rl=5 -ra=0.02 --redist_visc=2.0 --rollout_dir=${RLT_DIR}/test44
# run -rl=5 -ra=0.02 --redist_visc=5.0 --rollout_dir=${RLT_DIR}/test45
# run -rl=5 -ra=0.02 --redist_visc=10. --rollout_dir=${RLT_DIR}/test46


# run_long -rvp=None --rollout_dir=${RLT_DIR}/test100
# run_long -rl=3 -ra=0.03 --rollout_dir=${RLT_DIR}/test101
# run_long -rl=5 -ra=0.03 --rollout_dir=${RLT_DIR}/test102