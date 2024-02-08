import json
import os
import os.path as osp
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap, lax, jit
from jax_md.partition import space
try:
    from lagrangebench.evaluate.utils import write_vtk
except:
    from lagrangebench.utils import write_vtk

from neural_sph.utils import case_setup_redist, rho_computer

EPS = 1e-8


def average_trajs(dir):
    """Average the evolution of metrics over all rollouts."""

    files = os.listdir(dir)
    files = [f for f in files if ("metrics" in f and ".pkl" in f)]
    assert len(files) == 1, "only one metrics file should be present"
    metrics = pickle.load(open(os.path.join(dir, files[0]), "rb"))

    dict_out = {"mse": [], "sinkhorn": [], "e_kin": []}
    for _, value in metrics.items():
        dict_out["mse"].append(value["mse"])
        dict_out["sinkhorn"].append(value["sinkhorn"])
        dict_out["e_kin"].append(
            (value["e_kin"]["target"] - value["e_kin"]["predicted"]) ** 2
        )

    for key, v in dict_out.items():
        dict_out[key] = np.array(v)

    for key, v in dict_out.items():
        dict_out[key] = {"mean": np.mean(v, axis=0), "std": np.std(v, axis=0)}
    return dict_out


def plot_stats(
    dirs,
    rlt_dir,
    labels=None,
    log=True,
    limits=[None, None, None],
    limits_lower=[2e-7, 2e-5, 1e-8],
):
    """Plot the evolution of metrics averaged over all available rollouts.

    Relies on the structure `{rlt_dir}/test{str(dir)}/metrics_{...}.pkl`.
    """

    c = ["k", "r", "g", "b", "c", "m", "y", "orange", "brown", "pink", "silver"]

    # plot first row mean values, second row std values
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    for i, d in enumerate(dirs):
        dict_out = average_trajs(os.path.join(rlt_dir, "test" + str(d)))
        len_mse = len(dict_out["mse"]["mean"])

        stride_sinkhorn = np.round(len_mse / len(dict_out["sinkhorn"]["mean"])).astype(
            int
        )
        stride_e_kin = np.round(len_mse / len(dict_out["e_kin"]["mean"])).astype(int)
        x_mse = np.arange(len_mse)
        x_sinkhorn = np.arange(0, len_mse, stride_sinkhorn)
        x_e_kin = np.arange(0, len_mse - 1, stride_e_kin)

        label = labels[i] if labels is not None else d
        axs[0, 0].plot(x_mse, dict_out["mse"]["mean"] + EPS, label=label, c=c[i])
        axs[0, 2].plot(x_sinkhorn, dict_out["sinkhorn"]["mean"] + EPS, c=c[i])
        axs[0, 1].plot(x_e_kin, dict_out["e_kin"]["mean"] + EPS, c=c[i])

        axs[1, 0].plot(x_mse, dict_out["mse"]["std"] + EPS, c=c[i])
        axs[1, 2].plot(x_sinkhorn, dict_out["sinkhorn"]["std"] + EPS, c=c[i])
        axs[1, 1].plot(x_e_kin, dict_out["e_kin"]["std"] + EPS, c=c[i])

    axs[0, 0].set_title("MSE_p")
    axs[0, 1].set_title("MSE_Ekin")
    axs[0, 2].set_title("Sinkhorn")

    for i, limit in enumerate(limits):
        if log:
            axs[0, i].set_yscale("log")
            axs[1, i].set_yscale("log")
        if limit is not None:
            axs[0, i].set_ylim([limits_lower[i], limit])
            axs[1, i].set_ylim([limits_lower[i], limit])

    for ax in axs.flatten():
        ax.set_xlabel("step")
        ax.grid()

    axs[0, 0].legend(loc="lower right")
    plt.tight_layout()


def plot_scatter(
    r,
    bounds,
    color=None,
    mask=None,
    figsize=(4, 4),
    size=1,
    save_name=None,
    vmin=None,
    vmax=None,
):
    """Scatter plot the particles at a particular step."""

    if mask is not None:
        r = r[mask]
        if color is not None:
            color = color[mask]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.scatter(r[:, 0], r[:, 1], c=color, s=size, vmin=vmin, vmax=vmax)
    ax.grid()
    ax.set_aspect("equal")  # make x and y axes have the same scale
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    if color is not None:
        fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name)


def plot_hist(
    dirs,
    rlt_dir,
    metadata_root,
    labels=None,
    time_steps=[-1, 10, 20, 50],
    input_seq_len=6,
    targets=["vel_mag", "acc_mag"],
    average_over_n_rollouts=5,
    verbose=False,
    num_bins=100,
    save_name=None,
    time_steps_plot=None,
    xlims=None,  # if not none, expects [[xmin_left, xmax_left], [xmin_right, ymax_right]]
    is_lenend_everywhere=False,
):
    """Plot the historgram of properties at particular steps.

    Args:
        dirs (List[int]): list of `test{str(dir)}` directories to plot from.
        rlt_dir (str): directory containing all experiments of this type
        metadata_root (str): dataset directory.
        labels (List[str], optional): labels for the legend. Defaults to None.
        time_steps (list, optional): steps to plot. Defaults to [-1, 10, 20, 50].
        input_seq_len (int, optional): as during traiinig. Defaults to 6.
        targets (list, optional): Stats type. Defaults to ["vel_mag", "acc_mag"].
        average_over_n_rollouts (int, optional): num rollouts to use. Defaults to 5.
    """

    assert len(targets) == 2, "only two targets are supported"
    targets_choices = ["num_nbrs", "dist_nbrs", "vel_mag", "acc_mag"]
    assert targets[0] in targets_choices, f"targets must be one of {targets_choices}"
    assert targets[1] in targets_choices, f"targets must be one of {targets_choices}"

    c = ["k", "r", "g", "b", "c", "m", "y", "orange", "brown", "pink", "silver"]
    title_dict = {
        "num_nbrs": "Number of neighbors",
        "dist_nbrs": "Distance between neighbors",
        "vel_mag": "Velocity magnitudes",
        "acc_mag": "Acceleration magnitudes",
    }
    target_dict = {targets[0]: 0, targets[1]: 1}
    metadata, _, displacement_fn, _, nbrs_update, _, _ = case_setup_redist(
        metadata_root, is_physical=True, verbose=verbose
    )

    dt = metadata["dt"]
    N = metadata["num_particles_max"]

    displacement_fn_set = vmap(displacement_fn, in_axes=(0, 0))

    def v_fn(r1, r0):
        # r has a shape of (N, D)
        vels = displacement_fn_set(r1, r0) / dt / metadata["write_every"]
        return vels

    # compute averages for each step and dir over M rollouts
    metrics_dict = {}
    for step in time_steps:  # for each step along the rollout
        metrics_dict[step] = {}  # for each step
        for dir in dirs:  # for each dir in dirs
            metrics_dict[step][dir] = {targets[0]: None, targets[1]: None}

    for k, step in enumerate(time_steps):
        for j, d in enumerate(dirs):
            metrics_dict_sub = {targets[0]: [], targets[1]: []}

            for i in range(average_over_n_rollouts):
                rollout = pickle.load(
                    open(osp.join(rlt_dir, f"test{d}/rollout_{i}.pkl"), "rb")
                )

                r = rollout["predicted_rollout"][step + input_seq_len - 1]

                if "nbrs" in targets[0] or "nbrs" in targets[1]:
                    nbrs = nbrs_update(r)
                    edges = nbrs.idx

                r = r[rollout["particle_type"] == 0]  # only fluid particles

                if "num_nbrs" in targets:
                    # mask out padding edges
                    mask = edges != N
                    existing_edges = edges[:, mask[0]]

                    # plot histogram of number of neighbors per particle
                    # counts is a list of shape (N,) with integer numbers
                    _, counts = np.unique(existing_edges[1], return_counts=True)
                    metrics_dict_sub["num_nbrs"].append(counts)

                if "dist_nbrs" in targets:
                    # plot histogram of distances between connected particles
                    i_s, j_s = nbrs.idx
                    r_i_s, r_j_s = r[i_s], r[j_s]
                    dr_i_j = vmap(displacement_fn)(r_i_s, r_j_s)
                    dist = space.distance(dr_i_j)
                    dist = dist[dist > 0.000001]  # ignore self-connections
                    # # dist /= nbrs.cell_size  # normalize distance to [0, 1]
                    metrics_dict_sub["dist_nbrs"].append(dist)

                if "vel_mag" in targets or "acc_mag" in targets:
                    # plot histogram of velocity magnitudes
                    r_prev = rollout["predicted_rollout"][step + input_seq_len - 2]
                    r_prev = r_prev[rollout["particle_type"] == 0]
                    vel = v_fn(r, r_prev)
                    vel_mag = np.linalg.norm(vel, axis=-1)
                    metrics_dict_sub["vel_mag"].append(vel_mag)

                if "acc_mag" in targets:
                    # if we want the acceleration, we can do this:
                    r_next = rollout["predicted_rollout"][step + input_seq_len]
                    r_next = r_next[rollout["particle_type"] == 0]
                    vel_next = v_fn(r_next, r)
                    acc = (vel_next - vel) / dt / metadata["write_every"]
                    acc_mag = np.linalg.norm(acc, axis=-1)
                    metrics_dict_sub["acc_mag"].append(acc_mag)

            metrics_dict[step][d] = {
                targets[0]: np.concatenate(metrics_dict_sub[targets[0]], axis=0),
                targets[1]: np.concatenate(metrics_dict_sub[targets[1]], axis=0),
            }

    # plot a row for each time step
    fig, axs = plt.subplots(
        len(time_steps),
        2,
        figsize=(15, 3 * len(time_steps)),
        sharex="col",
        sharey="col",
    )
    for j, d in enumerate(dirs):
        label = labels[j] if labels is not None else d

        for i, step in enumerate(time_steps):
            if "num_nbrs" in targets:
                # plot histogram of number of neighbors per particle
                counts = metrics_dict[step][d]["num_nbrs"]
                values, counts = np.unique(counts, return_counts=True)
                bins = np.arange(values[0], values[-1] + 2) - 0.5
                idx = target_dict["num_nbrs"]
                axs[i, idx].hist(
                    values,
                    bins=bins,
                    weights=counts,
                    density=True,
                    histtype="step",
                    label=label,
                    color=c[j],
                )

            if "dist_nbrs" in targets:
                # plot histogram of distances between connected particles
                dist = metrics_dict[step][d]["dist_nbrs"]
                idx = target_dict["dist_nbrs"]
                axs[i, idx].hist(
                    dist,
                    bins=num_bins,
                    density=True,
                    histtype="step",
                    label=label,
                    color=c[j],
                )

            if "vel_mag" in targets:
                # plot histogram of velocity magnitudes
                vel_mag = metrics_dict[step][d]["vel_mag"]
                idx = target_dict["vel_mag"]
                axs[i, idx].hist(
                    vel_mag,
                    bins=num_bins,
                    density=True,
                    histtype="step",
                    label=label,
                    color=c[j],
                )

            if "acc_mag" in targets:
                # plot histogram of acceleration magnitudes
                acc_mag = metrics_dict[step][d]["acc_mag"]
                idx = target_dict["acc_mag"]
                axs[i, idx].hist(
                    acc_mag,
                    bins=num_bins,
                    density=True,
                    histtype="step",
                    label=label,
                    color=c[j],
                )

    # axs[0, 0].set_title(title_dict[targets[0]])
    # axs[0, 1].set_title(title_dict[targets[1]])
    axs[-1, 0].set_xlabel(title_dict[targets[0]])
    axs[-1, 1].set_xlabel(title_dict[targets[1]])
    if is_lenend_everywhere:
        for i in range(len(time_steps)):
            axs[i, 0].legend(loc="upper left")
            axs[i, 1].legend(loc="upper right")
    else:
        axs[0, 0].legend(loc="upper left")
    
    if xlims is not None:
        for i in range(len(time_steps)):
            axs[i, 0].set_xlim(xlims[0])
            axs[i, 1].set_xlim(xlims[1])

    if time_steps_plot is not None:
        time_steps = time_steps_plot
    for i, step in enumerate(time_steps):
        axs[i, 0].set_ylabel(f"step {step}")

    for ax in axs.flatten():
        ax.grid()

    plt.tight_layout()
    
    if save_name is not None:
        plt.savefig(save_name)


def pkl2vtk(src_path, dst_path=None, metadata_root=None):
    """Convert a rollout pickle file to a set of vtk files.

    Args:
        src_path (str): Source path to .pkl file.
        dst_path (str, optoinal): Destination directory path. Defaults to None.
            If None, then the vtk files are saved in the same directory as the pkl file.

    Example:
        pkl2vtk("rollout/test/rollout_0.pkl", "rollout/test_vtk")
        will create files rollout_0_0.vtk, rollout_0_1.vtk, etc. in the directory
        "rollout/test_vtk"
    """

    # set up destination directory
    if dst_path is None:
        dst_path = os.path.dirname(src_path)
    os.makedirs(dst_path, exist_ok=True)

    # load rollout
    with open(src_path, "rb") as f:
        rollout = pickle.load(f)

    if metadata_root is not None:
        ### Compute velocities and accelerations
        # load dataset metadata
        with open(osp.join(metadata_root, "metadata.json"), "r") as f:
            metadata = json.loads(f.read())

        bounds = np.array(metadata["bounds"])
        box = bounds[:, 1] - bounds[:, 0]

        from jax import vmap
        from jax_md import space

        if jnp.array(metadata["periodic_boundary_conditions"]).any():
            displacement_fn, _ = space.periodic(side=jnp.array(box))
        else:
            displacement_fn, _ = space.free()
        displacement_fn_set = vmap(vmap(displacement_fn, in_axes=(0, 0)))

        def v_fn(r):
            # r has a shape of (T, N, D)
            vels = displacement_fn_set(r[1:], r[:-1])
            v = jnp.concatenate([jnp.zeros_like(r[:1]), vels])
            return v

        _v = v_fn(rollout["predicted_rollout"])
        _v_ref = v_fn(rollout["ground_truth_rollout"])
        _acc = jnp.concatenate([_v[1:] - _v[:-1], jnp.zeros_like(_v[:1])])
        _acc_ref = jnp.concatenate([_v_ref[1:] - _v_ref[:-1], jnp.zeros_like(_v[:1])])

        ### Compute density
        comp_rho = rho_computer(metadata_root)
        @jit
        def body(_, r):
            rho = comp_rho(r, rollout["particle_type"])
            return None, rho
        _, _rho = lax.scan(body, None, rollout["predicted_rollout"])
        _, _rho_ref = lax.scan(body, None, rollout["ground_truth_rollout"])

    file_prefix = os.path.join(dst_path, os.path.basename(src_path).split(".")[0])
    for k in range(rollout["predicted_rollout"].shape[0]):
        # predictions
        state_vtk = {
            "r": rollout["predicted_rollout"][k],
            "tag": rollout["particle_type"],
        }
        if metadata_root is not None:
            state_vtk["v"] = _v[k]
            state_vtk["a"] = _acc[k]
            state_vtk["rho"] = _rho[k]
        write_vtk(state_vtk, f"{file_prefix}_{k}.vtk")
        # ground truth reference
        state_vtk = {
            "r": rollout["ground_truth_rollout"][k],
            "tag": rollout["particle_type"],
        }
        if metadata_root is not None:
            state_vtk["v"] = _v_ref[k]
            state_vtk["a"] = _acc_ref[k]
            state_vtk["rho"] = _rho_ref[k]
        write_vtk(state_vtk, f"{file_prefix}_ref_{k}.vtk")
