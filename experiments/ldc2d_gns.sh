#!/bin/bash
# run with:
# nohup bash experiments/ldc2d_gns.sh 2>&1 &

GPU=4
RLT_DIR="rlt/ldc2d_gns"

run_basic() {
    python main.py gpu=$GPU load_ckp=ckp/pretrained/gns_ldc2d/best eval.test=True mode=infer "$@"
}

run() {
    run_basic eval.infer.n_trajs=-1 eval.infer.batch_size=20 eval.n_rollout_steps=20 \
    eval.infer.metrics_stride=100 "$@"
}

run_400() {
    run_basic eval.infer.n_trajs=-1 eval.infer.batch_size=4 eval.n_rollout_steps=400 \
    eval.infer.metrics=['mse','e_kin','sinkhorn','chamfer','rho_mae','dirichlet'] \
    eval.infer.metrics_stride=10 "$@"
}

# sanity check by reproducing the original checkpoint performance
run r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_val

# long rollouts
run_400 r.variant_p=None eval.rollout_dir=${RLT_DIR}/test_0

run_400 r.loops=1 r.acc=0.001 eval.rollout_dir=${RLT_DIR}/test_1_0001
run_400 r.loops=1 r.acc=0.002 eval.rollout_dir=${RLT_DIR}/test_1_0002
run_400 r.loops=1 r.acc=0.005 eval.rollout_dir=${RLT_DIR}/test_1_0005
run_400 r.loops=1 r.acc=0.01 eval.rollout_dir=${RLT_DIR}/test_1_001
run_400 r.loops=1 r.acc=0.02 eval.rollout_dir=${RLT_DIR}/test_1_002
run_400 r.loops=1 r.acc=0.03 eval.rollout_dir=${RLT_DIR}/test_1_003
run_400 r.loops=1 r.acc=0.05 eval.rollout_dir=${RLT_DIR}/test_1_005

run_400 r.loops=3 r.acc=0.03 eval.rollout_dir=${RLT_DIR}/test_3_003
run_400 r.loops=5 r.acc=0.03 eval.rollout_dir=${RLT_DIR}/test_5_003
run_400 r.loops=7 r.acc=0.03 eval.rollout_dir=${RLT_DIR}/test_7_003
run_400 r.loops=10 r.acc=0.03 eval.rollout_dir=${RLT_DIR}/test_10_003

run_400 r.loops=5 r.acc=0.03 r.visc=0.1 eval.rollout_dir=${RLT_DIR}/test_5_003_01
run_400 r.loops=5 r.acc=0.03 r.visc=0.2 eval.rollout_dir=${RLT_DIR}/test_5_003_02
run_400 r.loops=5 r.acc=0.03 r.visc=0.5 eval.rollout_dir=${RLT_DIR}/test_5_003_05
run_400 r.loops=5 r.acc=0.03 r.visc=1.0 eval.rollout_dir=${RLT_DIR}/test_5_003_10
