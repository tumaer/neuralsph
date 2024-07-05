#!/bin/bash
# nohup bash experiments/rlx3.sh >> experiments/rlx3.out 2>&1 &

GPU=3

### pretrained checkpoint
python main.py config=experiments/rlx.yaml gpu=$GPU r.acc=0.01

RLT_DIR="rlt/rlx_ldc2d/gns_ldc2d_20240328-220053"

run_basic() {
    python main.py gpu=$GPU load_ckp=ckp/rlx_ldc2d/gns_ldc2d_20240328-220053 \
    eval.test=True mode=infer "$@"
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

run_400 r.variant_p=None r.loops=0 r.acc=0.0 eval.rollout_dir=${RLT_DIR}/test_0
run_400 r.variant_p=standard r.loops=1 eval.rollout_dir=${RLT_DIR}/test_1