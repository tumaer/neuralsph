import json
import os.path as osp
import pickle
import pprint

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, ops, vmap
from jax_md import space
from lagrangebench.case_setup.partition import neighbor_list


def case_setup_redist(metadata_root, rc_factor=None, is_physical=False, verbose=True):
    """Set up properties for the particle redistribution.

    Args:
        metadata_root (str): dataset directory
        rc_factor (float, optional): Cutoff radius as fraction of dx. Defaults to None.

    Returns:
        _type_: A tuple of preconfigured objects.

    We want a redistribution routine which will work equally well if the:
    1. box is rescaled in size. The accelerations should scale proportionally to dx. To
        achieve this, we scale everything with dx.
    2. the scale of the velocities (r - r_prev) varies, i.e. temporal coarsening level.
        To achieve this, we scale the velocities with the dataset velocity std.

    The assumption here is that we only work with two consecutive coordinate frames.

    We do the nondimensionalization by fixing:
    rho_ref = 1.0  - this we always require
    l_ref = dx  - we rescale all coordinates to unit dx
    u_ref = vel_std from the dataset metadata
    => t_ref = l_ref / u_ref = dx / vel_std
    => p_ref = rho_ref * u_ref**2 = vel_std**2 ?

    Also, the density over the domain is fixed to rho_ref = 1.0.
    The current code only supports particles of same size and mass.
    """
    with open(osp.join(metadata_root, "metadata.json"), "r") as f:
        metadata = json.loads(f.read())

    # nondimensionalization setup
    dx = metadata["dx"]  # l_ref
    metadata["l_ref"] = 1.0 if is_physical else dx
    _dx = dx / metadata["l_ref"]
    metadata["u_ref"] = 1.0  # not real nondimensionalization as dt=1, thus u_ref~dx
    # artificial speed of sound
    metadata["c0"] = 10 * metadata["u_ref"]
    metadata["t_ref"] = metadata["l_ref"] / metadata["u_ref"]
    metadata["p_ref"] = metadata["c0"] ** 2

    dim = metadata["dim"]
    # nondimensionalized box size
    if dim == 2:
        box_size = jnp.array([metadata["bounds"][0][1], metadata["bounds"][1][1]])
    else:
        box_size = jnp.array(
            [
                metadata["bounds"][0][1],
                metadata["bounds"][1][1],
                metadata["bounds"][2][1],
            ]
        )
    box_size /= metadata["l_ref"]
    N = metadata["num_particles_max"]
    displacement_fn, shift_fn = space.periodic(side=box_size)

    # To allocate the neighbor list, we create an arbitrary point cloud of size N.
    # Here we already nondimensionalize dx by itself.
    if dim == 2:
        pos_demo = pos_init_cartesian_2d(box_size, _dx)
    else:
        pos_demo = pos_init_cartesian_3d(box_size, _dx)
    pos_demo = pos_demo[:N]
    pos_noise = jax.random.normal(jax.random.PRNGKey(0), shape=pos_demo.shape)
    pos_demo = shift_fn(pos_demo, pos_noise * _dx)

    r_cutoff = (
        metadata["default_connectivity_radius"] if rc_factor is None else rc_factor
    )
    neighbor_fn = neighbor_list(
        displacement_fn,
        box_size,
        backend="jaxmd_vmap",
        r_cutoff=r_cutoff,
        capacity_multiplier=2.0,
        mask_self=False,
        num_particles_max=N,
        pbc=metadata["periodic_boundary_conditions"],
    )
    nbrs = neighbor_fn.allocate(pos_demo, num_particles=N)
    nbrs_update = jit(nbrs.update)

    eos = TaitEoS(p_ref=metadata["p_ref"], rho_ref=1.0, p_background=0.0, gamma=1.0)
    kernel_fn = QuinticKernel(h=_dx, dim=dim)

    state = {
        "mass": jnp.ones(N) * _dx ** metadata["dim"],
    }

    if verbose:
        pprint.pprint(metadata)

    return metadata, state, displacement_fn, shift_fn, nbrs_update, eos, kernel_fn


class QuinticKernel:
    """The quintic kernel function of Morris."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h

        self._normalized_cutoff = 3.0
        self.cutoff = self._normalized_cutoff * h
        if dim == 1:
            self._sigma = 1.0 / 120.0 * self._one_over_h
        elif dim == 2:
            self._sigma = 7.0 / 478.0 / jnp.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 3.0 / 359.0 / jnp.pi * self._one_over_h**3

    def w(self, r):
        """Evaluates the kernel at the radial distance r."""

        q = r * self._one_over_h
        q1 = jnp.maximum(0.0, 1.0 - q)
        q2 = jnp.maximum(0.0, 2.0 - q)
        q3 = jnp.maximum(0.0, 3.0 - q)

        return self._sigma * (q3**5 - 6.0 * q2**5 + 15.0 * q1**5)

    def grad_w(self, r):
        """Evaluates the 1D kernel gradient at the radial distance r."""

        return grad(self.w)(r)


class TaitEoS:
    """Equation of state

    From: "A generalized wall boundary condition for smoothed particle
    hydrodynamics", Adami et al 2012
    """

    def __init__(self, p_ref, rho_ref, p_background, gamma):
        self.p_ref = p_ref
        self.rho_ref = rho_ref
        self.p_bg = p_background
        self.gamma = gamma

    def p_fn(self, rho):
        return self.p_ref * ((rho / self.rho_ref) ** self.gamma - 1) + self.p_bg

    def rho_fn(self, p):
        p_temp = p + self.p_ref - self.p_bg
        return self.rho_ref * (p_temp / self.p_ref) ** (1 / self.gamma)


def pos_init_cartesian_2d(box_size, dx):
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), indexing="xy")
    r = (jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx
    return r

def pos_init_cartesian_3d(box_size, dx):
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), range(n[2]), indexing="xy")
    r = (jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx
    return r


def rho_computer(dataset_path, rlt_dir=None, input_seq_len=6, verbose=True):
    """Compute density from coordinates or from a rollout file.

    Args:
        dataset_path (str): directory containing a metadata.json file
        rlt_dir (str, optional): directory containing rollout files. Defaults to None.
            If rlt_dir is provided, then computation from .pkl files is used. Otherwise,
            the coordinates r have to be directly provided.
        input_seq_len (int, optional): see LagrangeBench. Defaults to 6 (6 steps).
        verbose (bool, optional): whether to print stuff in terminal. Defaults to True.

    Returns:
        Either (rho,) or (r, rho, bounds): used for different purposes
    """
    metadata, state, displacement_fn, _, nbrs_update, _, kernel_fn = case_setup_redist(
        dataset_path, rc_factor=3.0, is_physical=True, verbose=False
    )

    N = metadata["num_particles_max"]
    bounds = np.array(metadata["bounds"])
    mass = state["mass"]

    def comp_rho(r, mask_fluid):
        nbrs = nbrs_update(r, num_particles=N)
        i_s, j_s = nbrs.idx
        r_i_s, r_j_s = r[i_s], r[j_s]
        dr_i_j = vmap(displacement_fn)(r_i_s, r_j_s)
        dist = space.distance(dr_i_j)
        w_dist = vmap(kernel_fn.w)(dist)

        rho = mass * ops.segment_sum(w_dist, i_s, N)  # density summation
        rho = jnp.where(rho < 0.98, 1, rho)  # detect free surface
        rho = jnp.where(mask_fluid, rho, 1.0)  # give walls the reference quantity
        return rho

    def rho_from_pkl(dir, step_idx, rlt_idx=0):
        """Compute density from a rollout file."""
        # load the rollout file from the above plot
        rollout = pickle.load(open(f"{rlt_dir}/test{dir}/rollout_{rlt_idx}.pkl", "rb"))

        if verbose:
            for k, v in rollout.items():
                print(k, v.shape)

        r = rollout["predicted_rollout"][input_seq_len - 1 + step_idx]
        mask = rollout["particle_type"] == 0
        rho = comp_rho(r, mask)
        return r, rho, bounds

    def rho_from_r(r, tag):
        """Compute density from coordinates."""
        return comp_rho(jnp.array(r, dtype=jnp.float64), tag == 0)

    return rho_from_pkl if rlt_dir is not None else rho_from_r
