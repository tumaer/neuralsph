#!/bin/bash
# run this script with:
# nohup bash run_dam2d.sh 2>&1 &

GPU=1
RLT_DIR="rlt/dam2d"

run_basic() {
    ./venv_012/bin/python main.py --mode=infer --gpu=$GPU --test \
    --model_dir=ckp/gns_dam2d_20240112-202605/best "$@"
}

run20() {
    run_basic --eval_n_trajs_infer=30 --batch_size_infer=5 --metrics_stride_infer=1 \
    --n_rollout_steps=20 "$@"
}

run() {
    run_basic --eval_n_trajs_infer=1 --batch_size_infer=1 --metrics_stride_infer=1 \
    --n_rollout_steps=50 "$@"
}

run_longer() {
    run_basic --eval_n_trajs_infer=1 --batch_size_infer=1 --metrics_stride_infer=100 \
    --n_rollout_steps=80 "$@"
}

run_long() {
    run_basic --eval_n_trajs_infer=-1 --batch_size_infer=10 --metrics_stride_infer=20 \
    --n_rollout_steps=395 "$@"
}

####################
run_gext_basic() {
    ./venv_gext/bin/python main.py --mode=infer --gpu=$GPU --test \
    --model_dir=/home/atoshev/code/lb_dam/lagrangebench/ckp/gns_dam2d_20240125-165801/best \
    "$@"
}

run_gext20() {
    run_gext_basic --eval_n_trajs_infer=15 --batch_size_infer=5 --metrics_stride_infer=2 \
    --n_rollout_steps=20 "$@"
}

run_gext() {
    run_gext_basic --eval_n_trajs_infer=1 --batch_size_infer=1 \
    --metrics_stride_infer=1 --n_rollout_steps=50 "$@"
}

run_gext_longer() {
    run_gext_basic \
    --eval_n_trajs_infer=1 --batch_size_infer=1 --metrics_stride_infer=100 \
    --n_rollout_steps=80 "$@"
}

run_gext_long() {
    run_gext_basic \
    --eval_n_trajs_infer=-1 --batch_size_infer=5 --metrics_stride_infer=1000 \
    --n_rollout_steps=395 "$@"
}

# run20 -rvp=None --rollout_dir=${RLT_DIR}/test-1  # setup valid
# run_long -rvp=None --rollout_dir=${RLT_DIR}/test100

# run_gext20 -rvp=None --rollout_dir=${RLT_DIR}/test-2  # setup valid
# run_gext_long -rvp=None --rollout_dir=${RLT_DIR}/test101

# run_longer -rvp=None --rollout_dir=${RLT_DIR}/test3000 # no --test
# run_longer -rvp=None --rollout_dir=${RLT_DIR}/test3001  # --test
# run_longer -rvp=None --rollout_dir=${RLT_DIR}/test3002  # no --test
# run_longer -rvp=None --rollout_dir=${RLT_DIR}/test3003  # --test
# run_longer -rvp=None --rollout_dir=${RLT_DIR}/test3004  # no --test
# run_gext_longer -rvp=None --rollout_dir=${RLT_DIR}/test3500
# run_gext_longer -rl=1 -ra=0.03 --rollout_dir=${RLT_DIR}/test3535 # best
# run_gext_longer -rl=1 -ra=0.04 --rollout_dir=${RLT_DIR}/test3534
# run_gext_longer -rl=5 -ra=0.01 --rollout_dir=${RLT_DIR}/test3566

# run_gext_long -rl=1 -ra=0.02 --rollout_dir=${RLT_DIR}/test102
# run_gext_long -rl=1 -ra=0.03 --rollout_dir=${RLT_DIR}/test103 # best
# run_gext_long -rl=1 -ra=0.04 --rollout_dir=${RLT_DIR}/test104

# run_gext_long -rl=3 -ra=0.01 --rollout_dir=${RLT_DIR}/test111
# run_gext_long -rl=3 -ra=0.02 --rollout_dir=${RLT_DIR}/test112
# run_gext_long -rl=3 -ra=0.03 --rollout_dir=${RLT_DIR}/test113
# run_gext_long -rl=3 -ra=0.04 --rollout_dir=${RLT_DIR}/test114
# run_gext_long -rl=5 -ra=0.03 --rollout_dir=${RLT_DIR}/test115
# run_gext_long -rl=2 -ra=0.03 --rollout_dir=${RLT_DIR}/test116

# run_gext_long -rl=1 -ra=0.02 --redist_visc=0.05 --rollout_dir=${RLT_DIR}/test120
# run_gext_long -rl=1 -ra=0.02 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test121
# run_gext_long -rl=1 -ra=0.02 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test122
# run_gext_long -rl=1 -ra=0.02 --redist_visc=0.3 --rollout_dir=${RLT_DIR}/test123
