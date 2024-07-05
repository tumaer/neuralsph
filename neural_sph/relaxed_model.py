"""
GNN model with subsequent relaxation.
"""

from typing import Dict, Tuple, Callable

import haiku as hk
import jax
import jax.numpy as jnp
from lagrangebench import GNS


class RelaxedSolver(hk.Module):
    """GNN model with subsequent relaxation."""

    def __init__(
        self,
        redist: Callable,
        particle_dimension: int,
        latent_size: int,
        blocks_per_step: int,
        num_mp_steps: int,
        num_particle_types: int,
        particle_type_embedding_size: int,
        dt: float,
        shift_fn: Callable,
        displacement_fn: Callable,
        normalization_stats: Dict[str, Dict[str, jnp.ndarray]],
    ):
        """Initialize the model.

        Args:
            backbone_model: GNN model.
            redist: Redistribution function.
            particle_dimension: Dimension of the particle.
            dt: Effective physical timestep between the dataset frames.
            shift_fn: Shift function respecting periodic boundaries.
            displacement_fn: Displacement function respecting periodic boundaries.
            normalization_stats: Normalization statistics for velocity and acceleration.
        """
        super().__init__()

        # TODO: configure the model outside and pass it as one backbone model.
        self.model = GNS(
            particle_dimension=particle_dimension,
            latent_size=latent_size,
            blocks_per_step=blocks_per_step,
            num_mp_steps=num_mp_steps,
            num_particle_types=num_particle_types,
            particle_type_embedding_size=particle_type_embedding_size,
        )

        self.particle_dimension = particle_dimension
        self.effective_dt = dt
        self.normalization_stats = normalization_stats

        self.redist = redist
        self.shift_fn = shift_fn
        self.disp_fn_vmap = jax.vmap(displacement_fn)

    def to_physical(self, x, key="velocity"):
        stats = self.normalization_stats[key]
        x = x * stats["std"] + stats["mean"]
        x = x / self.effective_dt
        if key == "acceleration":
            x = x / self.effective_dt
        return x

    def to_normalized(self, x, key="velocity"):
        stats = self.normalization_stats[key]
        x = x * self.effective_dt
        if key == "acceleration":
            x = x * self.effective_dt
        x = (x - stats["mean"]) / stats["std"]
        return x

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass of the model.
        `r`, `u`, `a` live in physical space, while `a_gnn` is in normalized space.
        """

        features, tag = sample
        r0 = features["abs_pos"][:, -1]
        u0 = self.to_physical(features["vel_hist"][:, -self.particle_dimension :])

        # forward pass of the GNN model
        acc_gnn = self.model(sample)["acc"]

        # integrate acceleration as done in case_builder.integrate
        a = self.to_physical(acc_gnn, key="acceleration")
        u = u0 + self.effective_dt * a
        r = self.shift_fn(r0, self.effective_dt * u)

        # apply particle redistribution
        r = self.redist(r, r0, tag)

        # reverse engineer the resulting effective acceleration
        # as done in case_builder._compute_target to compute the acceleration
        u = self.disp_fn_vmap(r, r0) / self.effective_dt
        a = (u - u0) / self.effective_dt  # M * dt
        acc_gnn = self.to_normalized(a, "acceleration")

        return {"acc": acc_gnn}
