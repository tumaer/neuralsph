"""Adapted version of the LagrangeBench runner."""

import json
import os
import os.path as osp
from datetime import datetime
from typing import Dict, Union

import haiku as hk
import jmp
import numpy as np
from jax import config
from lagrangebench import Trainer
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.runner import setup_data, setup_model
from lagrangebench.utils import NodeType

from omegaconf import DictConfig, OmegaConf

from neural_sph.rollout import infer, relax_wrapper
from neural_sph.defaults import check_cfg
from neural_sph.case import case_builder
from neural_sph.relaxed_model import RelaxedSolver

import jax.numpy as jnp
from jax_md import space


def train_or_infer(cfg: Union[Dict, DictConfig]):
    if isinstance(cfg, Dict):
        cfg = OmegaConf.create(cfg)
    # sanity check on the passed configs
    check_cfg(cfg)

    mode = cfg.mode
    load_ckp = cfg.load_ckp
    is_test = cfg.eval.test

    if cfg.dtype == "float64":
        config.update("jax_enable_x64", True)

    data_train, data_valid, data_test = setup_data(cfg)

    metadata = data_train.metadata
    # neighbors search
    bounds = np.array(metadata["bounds"])
    box = bounds[:, 1] - bounds[:, 0]

    # setup core functions
    case = case_builder(
        box=box,
        metadata=metadata,
        input_seq_length=cfg.model.input_seq_length,
        cfg_neighbors=cfg.neighbors,
        cfg_model=cfg.model,
        noise_std=cfg.train.noise_std,
        external_force_fn=data_train.external_force_fn,
        dtype=cfg.dtype,
        is_subtract_ext_force=cfg.r.is_subtract_ext_force,
        is_smooth_force=cfg.r.is_smooth_force,
    )

    _, particle_type = data_train[0]

    # setup model from configs
    if cfg.model.relaxed:
        params_redist = OmegaConf.merge(cfg.r, {"variant_p": "standard"})
        redist = relax_wrapper(data_train.dataset_path, params_redist)

        pbc = jnp.array(metadata["periodic_boundary_conditions"])
        if pbc.any():
            displacement_fn, shift_fn = space.periodic(side=jnp.array(box))
        else:
            displacement_fn, shift_fn = space.free()

        def model(x):
            return RelaxedSolver(
                redist=redist,
                particle_dimension=metadata["dim"],
                latent_size=cfg.model.latent_dim,
                blocks_per_step=cfg.model.num_mlp_layers,
                num_mp_steps=cfg.model.num_mp_steps,
                num_particle_types=NodeType.SIZE,
                particle_type_embedding_size=16,
                dt=metadata["dt"] * metadata["write_every"],
                shift_fn=shift_fn,
                displacement_fn=displacement_fn,
                normalization_stats=case.normalization_stats,
            )(x)

        MODEL = RelaxedSolver
    else:
        # setup model from configs
        model, MODEL = setup_model(
            cfg,
            metadata=metadata,
            homogeneous_particles=particle_type.max() == particle_type.min(),
            has_external_force=data_train.external_force_fn is not None,
            normalization_stats=case.normalization_stats,
        )

    model = hk.without_apply_rng(hk.transform_with_state(model))

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    if mode == "train" or mode == "all":
        print("Start training...")

        if cfg.logging.run_name is None:
            run_prefix = f"{cfg.model.name}_{data_train.name}"
            data_and_time = datetime.today().strftime("%Y%m%d-%H%M%S")
            cfg.logging.run_name = f"{run_prefix}_{data_and_time}"
        store_ckp = osp.join(cfg.logging.ckp_dir, cfg.logging.run_name)
        os.makedirs(store_ckp, exist_ok=True)
        os.makedirs(osp.join(store_ckp, "best"), exist_ok=True)
        with open(osp.join(store_ckp, "config.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f.name)
        with open(osp.join(store_ckp, "best", "config.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f.name)
        # dictionary of configs which will be stored on W&B
        wandb_config = OmegaConf.to_container(cfg)
        trainer = Trainer(
            model,
            case,
            data_train,
            data_valid,
            cfg.train,
            cfg.eval,
            cfg.logging,
            input_seq_length=cfg.model.input_seq_length,
            seed=cfg.seed,
        )
        _, _, _ = trainer.train(
            step_max=cfg.train.step_max,
            load_ckp=load_ckp,
            store_ckp=store_ckp,
            wandb_config=wandb_config,
        )
    if mode == "infer" or mode == "all":
        print("Start inference...")
        if mode == "infer":
            model_dir = load_ckp
        if mode == "all":
            model_dir = osp.join(store_ckp, "best")
            assert osp.isfile(osp.join(model_dir, "params_tree.pkl"))
            cfg.eval.rollout_dir = model_dir.replace("ckp", "rlt")
            os.makedirs(cfg.eval.rollout_dir, exist_ok=True)
            if cfg.eval.infer.n_trajs is None:
                cfg.eval.infer.n_trajs = cfg.eval.train.n_trajs

        assert model_dir, "model_dir must be specified for inference."
        metrics = infer(
            model,
            case,
            data_test if is_test else data_valid,
            load_ckp=model_dir,
            cfg_eval_infer=cfg.eval.infer,
            rollout_dir=cfg.eval.rollout_dir,
            n_rollout_steps=cfg.eval.n_rollout_steps,
            seed=cfg.seed,
            params_redist=cfg.r,
        )

        split = "test" if is_test else "valid"
        print(f"Metrics of {model_dir} on {split} split:")
        metrics_ave = averaged_metrics(metrics)
        print(metrics_ave)
        with open(osp.join(cfg.eval.rollout_dir, "metr_ave.json"), "w") as f:
            json.dump(metrics_ave, f)
