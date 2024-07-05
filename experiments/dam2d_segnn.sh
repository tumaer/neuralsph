#!/bin/bash
# run with:
# nohup bash dam2d_segnn.sh 2>&1 &

GPU=3
RLT_DIR="rlt/dam2d_segnn"

### pretrained checkpoint
run_basic() {
    python main.py gpu=$GPU load_ckp=ckp/pretrained/segnn_dam2d/best eval.test=True mode=infer "$@"
}

run() {
    run_basic eval.infer.n_trajs=-1 eval.infer.batch_size=20 eval.n_rollout_steps=20 \
    eval.infer.metrics_stride=100 "$@"
}

run_400() {
    run_basic eval.infer.n_trajs=-1 eval.infer.batch_size=6 eval.n_rollout_steps=395 \
    eval.infer.metrics=['mse','e_kin','sinkhorn','chamfer','rho_mae','dirichlet'] \
    eval.infer.metrics_stride=10 "$@"
}

### with external force treatment
run_ext_basic() {
    python main.py gpu=$GPU eval.test=True r.is_subtract_ext_force=True mode=infer \
    load_ckp=ckp/redist/segnn_dam2d_20240126-023816/best "$@"
}

run_ext() {
    run_ext_basic eval.infer.n_trajs=-1 eval.infer.batch_size=20 eval.n_rollout_steps=20 \
    eval.infer.metrics_stride=100 "$@"
}

run_ext_400() {
    run_ext_basic eval.infer.n_trajs=-1 eval.infer.batch_size=6 eval.n_rollout_steps=395 \
    eval.infer.metrics=['mse','e_kin','sinkhorn','chamfer','rho_mae','dirichlet'] \
    eval.infer.metrics_stride=10 "$@"
}

########################################################################################
# sanity check by reproducing the original checkpoint performance
run r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_val
run_ext r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ext_val

run_400 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_0
run_ext_400 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ext_0

run_400 r.loops=3 r.acc=0.03 eval.rollout_dir=${RLT_DIR}/test_3_003
run_ext_400 r.loops=3 r.acc=0.03 eval.rollout_dir=${RLT_DIR}/test_ext_3_003
