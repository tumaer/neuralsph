"""Default lagrangebench configs."""

from omegaconf import DictConfig, OmegaConf


def set_defaults(cfg: DictConfig = OmegaConf.create({})) -> DictConfig:
    """Set neural-sph default configs."""

    from lagrangebench.defaults import defaults

    cfg = OmegaConf.merge(defaults, cfg)

    ### Parameters for particle redistribution
    cfg.r = OmegaConf.create({})

    # variant of pressure term. One of ["None", "stay", "adv", "standard"]
    cfg.r.variant_p = "standard"
    # variant of viscous term. One of ["None", "standard"]
    cfg.r.variant_visc = "standard"
    # number of relaxation steps/loops
    cfg.r.loops = 1  # rl
    # acceleration prefactor
    cfg.r.acc = 0.015  # ra
    # density threshold value in rho=np.where(rho<threshold, 1, rho)
    cfg.r.rho_threshold = 0.98  # rrt
    # viscous term prefactor
    cfg.r.visc = 0.0  # redist_visc

    # whether to subtract external force from the learning target
    cfg.r.is_subtract_ext_force = False
    # whether to use the smoothed force in RPF. Only active if r.is_subtract_ext_force
    cfg.r.is_smooth_force = True

    # add relaxed model as an option on top of any other model
    cfg.model.relaxed = False

    # add the option to disable jit for degubbing purposes
    cfg.disable_jit = False

    return cfg


defaults = set_defaults()


def check_cfg(cfg: DictConfig):
    """Check if the configs are valid."""
    assert cfg.r.variant_p in [
        "None",
        "stay",
        "adv",
        "standard",
    ], f"Invalid variant_p: {cfg.r.variant_p}"
