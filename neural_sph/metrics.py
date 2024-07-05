"""Metrics for evaluation end testing."""

import warnings
from functools import partial
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
from lagrangebench.evaluate.metrics import MetricsComputer

MetricsDict = Dict[str, Dict[str, jnp.ndarray]]


class NeuralSPHMetricsComputer(MetricsComputer):
    METRICS = ["mse", "e_kin", "sinkhorn", "chamfer", "rho_mae", "dirichlet"]
    METRICS_LAGRANGEBENCH = ["mse", "mae", "sinkhorn", "e_kin"]

    def __init__(
        self,
        active_metrics: List,
        dist_fn: Callable,
        metadata: Dict,
        input_seq_length: int,
        stride: int = 10,
        loss_ranges: Optional[List] = None,
        ot_backend: str = "ott",
        comp_rho: Callable = None,  # density computer
        rho_ref: float = 1.0,
    ):
        super().__init__(
            active_metrics=[
                m for m in active_metrics if m in self.METRICS_LAGRANGEBENCH
            ],
            dist_fn=dist_fn,
            metadata=metadata,
            input_seq_length=input_seq_length,
            stride=stride,
            loss_ranges=loss_ranges,
            ot_backend=ot_backend,
        )
        is_supported = [hasattr(self, metric) for metric in active_metrics]
        assert all(is_supported), (
            "The following metrics are not supported: "
            f"'{active_metrics[is_supported.index(False)]}'"
        )
        self._active_metrics = active_metrics
        self.comp_rho = comp_rho
        self.rho_ref = rho_ref

    def __call__(
        self, pred_rollout: jnp.ndarray, target_rollout: jnp.ndarray
    ) -> MetricsDict:
        """Compute the metrics between two rollouts.

        Args:
            pred_rollout: Predicted rollout.
            target_rollout: Target rollout.

        Returns:
            Dictionary of metrics.
        """
        # both rollouts of shape (traj_len - t_window, n_nodes, dim)
        target_rollout = jnp.asarray(target_rollout, dtype=pred_rollout.dtype)
        metrics = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = None
            for metric_name in self._active_metrics:
                metric_fn = getattr(self, metric_name)
                if metric_name in ["mse", "mae"]:
                    # full rollout loss
                    metrics[metric_name] = jax.vmap(metric_fn)(
                        pred_rollout, target_rollout
                    )
                    # shorter horizon losses
                    for i in self._loss_ranges:
                        if i < metrics[metric_name].shape[0]:
                            metrics[f"{metric_name}{i}"] = metrics[metric_name][:i]

                elif metric_name in ["e_kin"]:
                    dt = self._metadata["dt"] * self._metadata["write_every"]
                    dx = self._metadata["dx"]
                    dim = self._metadata["dim"]

                    metric_dvmap = jax.vmap(jax.vmap(metric_fn))

                    # Ekin of predicted rollout
                    velocity_rollout = self._dist_dvmap(
                        pred_rollout[1 :: self._stride],
                        pred_rollout[0 : -1 : self._stride],
                    )
                    e_kin_pred = metric_dvmap(velocity_rollout / dt).sum(1)
                    e_kin_pred = e_kin_pred * dx**dim

                    # Ekin of target rollout
                    velocity_rollout = self._dist_dvmap(
                        target_rollout[1 :: self._stride],
                        target_rollout[0 : -1 : self._stride],
                    )
                    e_kin_target = metric_dvmap(velocity_rollout / dt).sum(1)
                    e_kin_target = e_kin_target * dx**dim

                    metrics[metric_name] = {
                        "predicted": e_kin_pred,
                        "target": e_kin_target,
                        "mse": ((e_kin_pred - e_kin_target) ** 2).mean(),
                    }

                elif metric_name == "sinkhorn":
                    # vmapping over distance matrix blows up
                    metrics[metric_name] = jax.lax.scan(
                        lambda _, x: (None, self.sinkhorn(*x)),
                        None,
                        (
                            pred_rollout[0 :: self._stride],
                            target_rollout[0 :: self._stride],
                        ),
                    )[1]
                elif metric_name == "chamfer":
                    metrics[metric_name] = jax.vmap(metric_fn)(
                        pred_rollout, target_rollout
                    )
                elif metric_name == "rho_mae" or metric_name == "dirichlet":
                    if rho is None:
                        tags = jnp.zeros(pred_rollout.shape[1], dtype=jnp.int64)
                        rho, drhodr = jax.lax.scan(
                            lambda _, x: (None, self.comp_rho(x, tags)),
                            None,
                            pred_rollout[0 :: self._stride],
                        )[1]

                    if metric_name == "rho_mae":
                        metrics[metric_name] = jax.vmap(metric_fn)(rho)
                    elif metric_name == "dirichlet":
                        metrics[metric_name] = jax.vmap(metric_fn)(drhodr)
        return metrics

    @partial(jax.jit, static_argnums=(0,))
    def chamfer(self, pred: jnp.ndarray, target: jnp.ndarray) -> float:
        """Chamfer distance between two point sets."""
        shape = jax.ShapeDtypeStruct((), dtype=jnp.float32)

        def chamfer_(cloud1, cloud2):
            from scipy.spatial import KDTree

            tree = KDTree(cloud1)
            dist = tree.query(cloud2, k=1)[0].mean()
            tree = KDTree(cloud2)
            dist += tree.query(cloud1, k=1)[0].mean()
            return jnp.array(dist, dtype=jnp.float32)

        return jax.pure_callback(chamfer_, shape, pred, target)

    @partial(jax.jit, static_argnums=(0,))
    def rho_mae(self, rho: jnp.ndarray) -> float:
        """Density difference to reference density."""
        return jnp.abs(rho - self.rho_ref).mean()

    @partial(jax.jit, static_argnums=(0,))
    def dirichlet(self, grad_field: jnp.ndarray) -> float:
        """Dirichlet energy of a scalar field using the gradient of the field."""
        # compute Dirichlet energy integral
        particle_volume = self._metadata["dx"] ** self._metadata["dim"]
        dir_energy = 0.5 * jnp.sum((grad_field**2).sum(-1)) * particle_volume

        return dir_energy


if __name__ == "__main__":
    # Test the metrics
    from scipy.spatial import KDTree

    def chamfer_fn(cloud1, cloud2):
        tree = KDTree(cloud1)
        dist = tree.query(cloud2, k=1)[0].mean()
        tree = KDTree(cloud2)
        dist += tree.query(cloud1, k=1)[0].mean()
        return jnp.array(dist, dtype=jnp.float32)

    cloud1 = jnp.array([[0.5, 0.5], [0.6, 0.6]])
    cloud2 = jnp.array([[0.4, 0.4], [0.4, 0.7]])

    res = chamfer_fn(cloud1, cloud2)

    from matplotlib import pyplot as plt

    plt.scatter(cloud1[:, 0], cloud1[:, 1], c="r")
    plt.scatter(cloud2[:, 0], cloud2[:, 1], c="b")
    plt.savefig("clouds.png")
