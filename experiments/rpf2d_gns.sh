#!/bin/bash
# run with:
# nohup bash rpf2d_gns.sh 2>&1 &

GPU=3
RLT_DIR="rlt/rpf2d_gns"

### pretrained checkpoint
run_basic() {
    python main.py gpu=$GPU load_ckp=ckp/pretrained/gns_rpf2d/best eval.test=True mode=infer "$@"
}

run() {
    run_basic eval.infer.n_trajs=-1 eval.infer.batch_size=20 eval.n_rollout_steps=20 \
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
    load_ckp=ckp/redist/gns_rpf2d_20240125-165807/best r.is_smooth_force=False "$@"
}

run_ext() {
    run_ext_basic eval.infer.n_trajs=-1 eval.infer.batch_size=20 eval.n_rollout_steps=20 \
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
    load_ckp=ckp/redist/gns_rpf2d_20240128-160131/best "$@"
}

run_ex2() {
    run_ex2_basic eval.infer.n_trajs=-1 eval.infer.batch_size=20 eval.n_rollout_steps=20 \
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

run_400 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_0

### The following two are not the numbers from the paper, as they don't have _g
run_400 r.loops=1 r.acc=0.001 eval.rollout_dir=${RLT_DIR}/test_1_0001
run_400 r.loops=1 r.acc=0.002 eval.rollout_dir=${RLT_DIR}/test_1_0002
run_400 r.loops=1 r.acc=0.005 eval.rollout_dir=${RLT_DIR}/test_1_0005
run_400 r.loops=1 r.acc=0.01 eval.rollout_dir=${RLT_DIR}/test_1_001
run_400 r.loops=1 r.acc=0.02 eval.rollout_dir=${RLT_DIR}/test_1_002
run_400 r.loops=1 r.acc=0.03 eval.rollout_dir=${RLT_DIR}/test_1_003
run_400 r.loops=1 r.acc=0.05 eval.rollout_dir=${RLT_DIR}/test_1_005

run_400 r.loops=3 r.acc=0.02 eval.rollout_dir=${RLT_DIR}/test_3_002
run_400 r.loops=5 r.acc=0.02 eval.rollout_dir=${RLT_DIR}/test_5_002
run_400 r.loops=7 r.acc=0.02 eval.rollout_dir=${RLT_DIR}/test_7_002
run_400 r.loops=10 r.acc=0.02 eval.rollout_dir=${RLT_DIR}/test_10_002



### Compare variants by external force treatment
run r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_val
run_ext r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ext_val
run_ex2 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ex2_val

run_ext_400 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ext_0
run_ex2_400 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_ex2_0

### to get all configurations from the paper
run_ex2_400 r.loops=3 r.acc=0.02 eval.rollout_dir=${RLT_DIR}/test_ex2_3_002
run_ex2_400 r.loops=3 r.acc=0.02 r.visc=0.2 eval.rollout_dir=${RLT_DIR}/test_ex2_3_002_02


### timing runs
run_timing() {
    python main.py gpu=$GPU load_ckp=ckp/pretrained/gns_rpf2d/best mode=infer \
    eval.infer.n_trajs=1 eval.infer.batch_size=1 eval.infer.out_type=none "$@"
}
run_timing r.variant_p=None
run_timing r.loops=1 r.acc=0.001
run_timing r.loops=3 r.acc=0.001
run_timing r.loops=5 r.acc=0.001
