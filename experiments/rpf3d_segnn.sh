#!/bin/bash
# run with:
# nohup bash experiments/rpf3d_segnn.sh 2>&1 &

GPU=6
RLT_DIR="rlt/rpf3d_segnn"

### pretrained checkpoint
run_basic() {
    python main.py gpu=$GPU load_ckp=ckp/pretrained/segnn_rpf3d/best eval.test=Tru mode=infere "$@"
}

run() {
    run_basic eval.infer.n_trajs=-1 eval.infer.batch_size=10 eval.n_rollout_steps=20 \
    eval.infer.metrics_stride=100 "$@"
}

run_400() {
    run_basic eval.infer.n_trajs=-1 eval.infer.batch_size=6 eval.n_rollout_steps=400 \
    eval.infer.metrics=['mse','e_kin','sinkhorn','chamfer','rho_mae','dirichlet'] \
    eval.infer.metrics_stride=10 "$@"
}

### with external force treatment (nonsmooth)
run_ext_basic() {
    python main.py gpu=$GPU eval.test=True r.is_subtract_ext_force=True mode=infer \
    load_ckp=ckp/redist/segnn_rpf3d_20240326-014157/best r.is_smooth_force=False "$@"
}

run_ext() {
    run_ext_basic eval.infer.n_trajs=-1 eval.infer.batch_size=10 eval.n_rollout_steps=20 \
    eval.infer.metrics_stride=100 "$@"
}

run_ext_400() {
    run_ext_basic eval.infer.n_trajs=-1 eval.infer.batch_size=6 eval.n_rollout_steps=400 \
    eval.infer.metrics=['mse','e_kin','sinkhorn','chamfer','rho_mae','dirichlet'] \
    eval.infer.metrics_stride=10 "$@"
}

### with external force treatment (smoothed)
run_ex2_basic() {
    python main.py gpu=$GPU eval.test=True r.is_subtract_ext_force=True mode=infer \
    load_ckp=ckp/redist/segnn_rpf3d_20240129-021751/best "$@"
}

run_ex2() {
    run_ex2_basic eval.infer.n_trajs=-1 eval.infer.batch_size=10 eval.n_rollout_steps=20 \
    eval.infer.metrics_stride=100 "$@"
}

run_ex2_400() {
    run_ex2_basic eval.infer.n_trajs=-1 eval.infer.batch_size=6 eval.n_rollout_steps=400 \
    eval.infer.metrics=['mse','e_kin','sinkhorn','chamfer','rho_mae','dirichlet'] \
    eval.infer.metrics_stride=10 "$@"
}


########################################################################################
# sanity check by reproducing the original checkpoint performance
run r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_val
run_ext r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ext_val
run_ex2 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ex2_val

run_400 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_0
run_ext_400 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ext_0
run_ex2_400 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ex2_0

### to get all configurations from the paper
run_400 r.loops=1 r.acc=0.005 eval.rollout_dir=${RLT_DIR}/test_1_0005
run_ex2_400 r.loops=1 r.acc=0.005 eval.rollout_dir=${RLT_DIR}/test_ex2_1_0005

### timing runs
run_timing() {
    python main.py gpu=$GPU load_ckp=ckp/pretrained/segnn_rpf3d/best mode=infer \
    eval.infer.n_trajs=1 eval.infer.batch_size=1 eval.infer.out_type=none "$@"
}
run_timing r.variant_p=None
run_timing r.loops=1 r.acc=0.001
run_timing r.loops=3 r.acc=0.001
run_timing r.loops=5 r.acc=0.001
