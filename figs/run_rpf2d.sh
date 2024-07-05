#!/bin/bash
# run this script with:
# nohup bash run_rpf2d.sh 2>&1 &

GPU=2
RLT_DIR="rlt/rpf2d"

run_basic() {
    ./venv/bin/python main.py --mode=infer --gpu=$GPU  --test \
    --model_dir=ckp/rpf_2d_seeds_gns/gns_2D_RPF_3200_20kevery100_20231022-223939/best \
    "$@"
}

run() {
    run_basic --eval_n_trajs_infer=-1 --batch_size_infer=10 --metrics_stride_infer=100 \
    --n_rollout_steps=20 "$@"
}

run_400() {
    run_basic --eval_n_trajs_infer=-1 --batch_size_infer=6 --metrics_stride_infer=10 \
    --n_rollout_steps=400 "$@"
}

run_long() {
    run_basic --eval_n_trajs_infer=10 --batch_size_infer=5 --metrics_stride_infer=40 \
    --n_rollout_steps=990 "$@"
}


###################################
run_gext_basic() {
    ./venv_gext/bin/python main.py --mode=infer --gpu=$GPU --test \
    --model_dir=/home/atoshev/code/lb_dam/lagrangebench/ckp/gns_rpf2d_20240125-165807/best \
    "$@"
}

run_gext() {
    run_gext_basic --eval_n_trajs_infer=-1 --batch_size_infer=4 --metrics_stride_infer=100 \
    --n_rollout_steps=20 "$@"
}

# run_gext_400() {
#     run_gext_basic --eval_n_trajs_infer=-1 --batch_size_infer=6 --metrics_stride_infer=10 \
#     --n_rollout_steps=400 "$@"
# }

run_gext_400() {
    run_gext_basic --eval_n_trajs_infer=5 --batch_size_infer=5 --metrics_stride_infer=10 \
    --n_rollout_steps=400 "$@"
}

run_gext_long() {
    run_gext_basic --eval_n_trajs_infer=10 --batch_size_infer=5 --metrics_stride_infer=40 \
    --n_rollout_steps=990 "$@"
}


###################################
run_gext2_basic() {
    ./venv_gext/bin/python main.py --mode=infer --gpu=$GPU --test \
    --model_dir=/home/atoshev/code/lb_dam/lagrangebench/ckp/gns_rpf2d_20240128-160131/best \
    "$@"
}

run_gext2_400() {
    run_gext2_basic --eval_n_trajs_infer=-1 --batch_size_infer=5 --metrics_stride_infer=10 \
    --n_rollout_steps=400 "$@"
}

# run -rvp=None --rollout_dir=${RLT_DIR}/test0

# run -rl=1 -ra=0.01 --rollout_dir=${RLT_DIR}/test10
# run -rl=1 -ra=0.015 --rollout_dir=${RLT_DIR}/test11
# run -rl=1 -ra=0.02 --rollout_dir=${RLT_DIR}/test12
# run -rl=1 -ra=0.003 --rollout_dir=${RLT_DIR}/test13
# run -rl=1 -ra=0.005 --rollout_dir=${RLT_DIR}/test14

# run -rl=1 -ra=0.02 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test20
# run -rl=1 -ra=0.02 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test21
# run -rl=1 -ra=0.02 --redist_visc=0.5 --rollout_dir=${RLT_DIR}/test22
# run -rl=1 -ra=0.02 --redist_visc=1.0 --rollout_dir=${RLT_DIR}/test23

# run -rl=3 -ra=0.01 --rollout_dir=${RLT_DIR}/test30
# run -rl=5 -ra=0.01 --rollout_dir=${RLT_DIR}/test31
# run -rl=3 -ra=0.005 --rollout_dir=${RLT_DIR}/test32

# run -rl=1 -ra=0.005 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test40
# run -rl=1 -ra=0.005 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test41
# run -rl=1 -ra=0.005 --redist_visc=0.5 --rollout_dir=${RLT_DIR}/test42


# run_long -rvp=None --rollout_dir=${RLT_DIR}/test100
# run_long -rl=1 -ra=0.01 --rollout_dir=${RLT_DIR}/test110
# run_long -rl=1 -ra=0.02 --rollout_dir=${RLT_DIR}/test112
# run_long -rl=1 -ra=0.005 --rollout_dir=${RLT_DIR}/test114
# run_long -rl=1 -ra=0.005 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test140
# run_long -rl=1 -ra=0.005 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test141
# run_long -rl=1 -ra=0.005 --redist_visc=0.5 --rollout_dir=${RLT_DIR}/test142
# run_long -rl=1 -ra=0.005 --redist_visc=1.0 --rollout_dir=${RLT_DIR}/test143
# run_long -rl=1 -ra=0.01 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test150
# run_long -rl=1 -ra=0.01 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test151
# run_long -rl=1 -ra=0.01 --redist_visc=0.5 --rollout_dir=${RLT_DIR}/test152
# run_long -rl=1 -ra=0.01 --redist_visc=1.0 --rollout_dir=${RLT_DIR}/test153



# train 2d rpf without external forces in target. 
# check /home/atoshev/code/lb_dam/lagrangebench
# nohup taskset -c 64-79 python main.py --gpu=4 -c=configs/dam_2d/gns.yaml >> nohup_dam.out 2>&1 &
# nohup taskset -c 80-95 python main.py --gpu=5 -c=configs/rpf_2d/gns.yaml >> nohup_rpf.out 2>&1 &

# nohup taskset -c 64-79 python main.py --gpu=6 -c=configs/dam_2d/segnn.yaml >> nohup_dam_segnn.out 2>&1 &
# nohup taskset -c 80-95 python main.py --gpu=7 -c=configs/rpf_2d/segnn.yaml >> nohup_rpf_segnn.out 2>&1 &



# run -rvp=None --rollout_dir=${RLT_DIR}/test200
# run_gext -rvp=None --rollout_dir=${RLT_DIR}/test202

# run_400 -rvp=None --rollout_dir=${RLT_DIR}/test400

# run_400 -rl=1 -ra=0.01 --rollout_dir=${RLT_DIR}/test402
# run_400 -rl=3 -ra=0.01 --rollout_dir=${RLT_DIR}/test403
# run_400 -rl=1 -ra=0.005 --rollout_dir=${RLT_DIR}/test404
# run_400 -rl=3 -ra=0.005 --rollout_dir=${RLT_DIR}/test405
# run_400 -rl=1 -ra=0.002 --rollout_dir=${RLT_DIR}/test406


# run_gext_400 -rvp=None --rollout_dir=${RLT_DIR}/test401

# run_gext_400 -ra=0.01 --rollout_dir=${RLT_DIR}/test410
# run_gext_400 -ra=0.02 --rollout_dir=${RLT_DIR}/test412
# run_gext_400 -ra=0.003 --rollout_dir=${RLT_DIR}/test413  # unstable
# run_gext_400 -rl=3 -ra=0.02 --rollout_dir=${RLT_DIR}/test414  # best in class
# run_gext_400 -rl=5 -ra=0.02 --rollout_dir=${RLT_DIR}/test415  # unstable

# run_gext_400 -rl=3 -ra=0.005 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test420
# run_gext_400 -rl=3 -ra=0.005 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test421 # best in class - unstable
# run_gext_400 -rl=3 -ra=0.005 --redist_visc=0.5 --rollout_dir=${RLT_DIR}/test422
# run_gext_400 -rl=3 -ra=0.005 --redist_visc=1.0 --rollout_dir=${RLT_DIR}/test423
# run_gext_400 -rl=3 -ra=0.01 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test424 # best in class
# run_gext_400 -rl=3 -ra=0.01 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test425
# run_gext_400 -rl=3 -ra=0.01 --redist_visc=0.5 --rollout_dir=${RLT_DIR}/test426
# run_gext_400 -rl=3 -ra=0.01 --redist_visc=1.0 --rollout_dir=${RLT_DIR}/test427
# run_gext_400 -rl=3 -ra=0.02 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test428
# run_gext_400 -rl=3 -ra=0.02 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test429 # best in class
# run_gext_400 -rl=3 -ra=0.02 --redist_visc=0.5 --rollout_dir=${RLT_DIR}/test430
# run_gext_400 -rl=3 -ra=0.02 --redist_visc=1.0 --rollout_dir=${RLT_DIR}/test431
# run_gext_400 -rl=1 -ra=0.002 --rollout_dir=${RLT_DIR}/test432
# run_gext_400 -rl=1 -ra=0.005 --rollout_dir=${RLT_DIR}/test433



# run_gext_400 -rvp=None --rollout_dir=${RLT_DIR}/test501  # tried the above, but with smoothed force field



# nohup taskset -c 80-95 python main.py --gpu=7 -c=configs/rpf_2d/gns.yaml >> nohup_rpf_smooth.out 2>&1 &
# run_gext2_400 -rvp=None --rollout_dir=${RLT_DIR}/test502

# run_gext2_400 -ra=0.005 --rollout_dir=${RLT_DIR}/test509
# run_gext2_400 -ra=0.01 --rollout_dir=${RLT_DIR}/test510
# run_gext2_400 -ra=0.02 --rollout_dir=${RLT_DIR}/test511
# run_gext2_400 -ra=0.03 --rollout_dir=${RLT_DIR}/test512
# run_gext2_400 -ra=0.04 --rollout_dir=${RLT_DIR}/test513

# run_gext2_400 -rl=3 -ra=0.01 --rollout_dir=${RLT_DIR}/test520
# run_gext2_400 -rl=3 -ra=0.02 --rollout_dir=${RLT_DIR}/test521
# run_gext2_400 -rl=5 -ra=0.01 --rollout_dir=${RLT_DIR}/test522
# run_gext2_400 -rl=5 -ra=0.02 --rollout_dir=${RLT_DIR}/test523


# run_gext2_400 -rl=3 -ra=0.02 --redist_visc=0.1 --rollout_dir=${RLT_DIR}/test530
# run_gext2_400 -rl=3 -ra=0.02 --redist_visc=0.2 --rollout_dir=${RLT_DIR}/test531
# run_gext2_400 -rl=3 -ra=0.02 --redist_visc=0.5 --rollout_dir=${RLT_DIR}/test532



# nohup taskset -c 80-95 python main.py --gpu=5 -c=configs/rpf_3d/gns.yaml >> nohup_rpf3d_gns.out 2>&1 &
# nohup taskset -c 80-95 python main.py --gpu=7 -c=configs/rpf_3d/segnn.yaml >> nohup_rpf3d_segnn.out 2>&1 &
# nohup taskset -c 80-95 python main.py --gpu=6 -c=configs/rpf_2d/segnn.yaml >> nohup_rpf_segnn_smooth.out 2>&1 &





# add sharp external force to all other rpf cases, i.e. 3d and gns/segnn
# nohup python main.py --gpu=6 -c=configs/rpf_3d/gns.yaml >> nohup_rpf3d_gns_nonsmooth.out 2>&1 &
# nohup python main.py --gpu=7 -c=configs/rpf_3d/segnn.yaml >> nohup_rpf3d_segnn_nonsmooth.out 2>&1 &
