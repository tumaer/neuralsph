import json
import os
import os.path as osp
import pickle
from typing import List, Union, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap, lax, jit, tree_util
from jax_md.partition import space
from lagrangebench.evaluate.utils import write_vtk

from neural_sph.utils import case_setup_redist, rho_computer

EPS = 1e-8


def average_trajs(dir):
    """Average the evolution of metrics over all rollouts."""

    files = os.listdir(dir)
    files = [f for f in files if ("metrics" in f and ".pkl" in f)]
    assert len(files) == 1, "only one metrics file should be present"
    metrics = pickle.load(open(os.path.join(dir, files[0]), "rb"))

    keys = ["mse", "e_kin", "sinkhorn", "rho_mae", "dirichlet", "chamfer"]
    dict_out = {key: [] for key in keys}
    for _, value in metrics.items():
        dict_out["e_kin"].append(
            (value["e_kin"]["target"] - value["e_kin"]["predicted"]) ** 2
        )
        for key in keys:
            if key != "e_kin":
                dict_out[key].append(value[key])

    for key, v in dict_out.items():
        dict_out[key] = np.array(v)

    for key, v in dict_out.items():
        dict_out[key] = {
            "mean": np.mean(v, axis=0),
            "std": np.std(v, axis=0),
            "min": np.min(v, axis=0),
            "q25": np.quantile(v, 0.25, axis=0),
            "q75": np.quantile(v, 0.75, axis=0),
            "max": np.max(v, axis=0),
        }
    return dict_out


def plot_stats(
    dirs: Union[List[str], str],
    rlt_dir: Union[List[str], str],
    labels: Optional[Union[List[str], str]] = None,
    log=True,
    limits=[None, None, None],
    limits_lower=[2e-7, 2e-5, 1e-8],
    save_name=None,
    c_order=None,
):
    """Plot the evolution of metrics averaged over all available rollouts.

    Relies on the structure `{rlt_dir}/test{str(dir)}/metrics_{...}.pkl`.
    """
    if isinstance(rlt_dir, list):
        save_dir = rlt_dir[0]
        assert isinstance(dirs[0], list) and (len(dirs) == len(rlt_dir))
        assert (labels is None) or isinstance(labels[0], list)
        assert (labels is None) or (len(labels) == len(rlt_dir))
    else:
        save_dir = rlt_dir
        labels = [labels]
        rlt_dir = [rlt_dir]

    c = ["k", "r", "g", "b", "c", "m", "y", "orange", "brown", "pink", "silver"]
    if c_order is not None:
        c = [c[i] for i in c_order]
    line_styles = ["-", "--", "-.", ":", "-"]

    # plot first row mean values, second row std values
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for j, rlt_dir_i in enumerate(rlt_dir):
        labels_i = labels[j] if labels is not None else None
        dirs_i = dirs[j] if isinstance(dirs[0], list) else dirs
        ls = line_styles[j]
        for i, d in enumerate(dirs_i):
            dict_out = average_trajs(os.path.join(rlt_dir_i, "test" + str(d)))
            len_mse = len(dict_out["mse"]["mean"])

            stride_sinkhorn = np.round(
                len_mse / len(dict_out["sinkhorn"]["mean"])
            ).astype(int)
            stride_e_kin = np.round(len_mse / len(dict_out["e_kin"]["mean"])).astype(
                int
            )
            x_mse = np.arange(len_mse)
            x_sinkhorn = np.arange(0, len_mse, stride_sinkhorn)
            x_e_kin = np.arange(0, len_mse - 1, stride_e_kin)

            label = labels_i[i] if labels_i is not None else d

            keys = ["mse", "e_kin", "sinkhorn", "rho_mae", "dirichlet", "chamfer"]
            titles = [
                r"MSE$_{400}$",
                r"MSE$_{Ekin}$",
                "Sinkhorn",
                r"MAE$_{\rho}$",
                "Dirichlet",
                "Chamfer",
            ]
            x_axes = [x_mse, x_e_kin, x_sinkhorn, x_sinkhorn, x_sinkhorn, x_mse]
            for ax, key, x_axis in zip(axs.flatten(), keys, x_axes):
                x_axis = x_axis[: len(dict_out[key]["mean"])]
                # specify the line type: solid or dashed
                ax.plot(
                    x_axis,
                    dict_out[key]["mean"] + EPS,
                    ls,
                    label=label,
                    c=c[i],
                )
                ax.fill_between(
                    x_axis,
                    dict_out[key]["q25"] + EPS,
                    dict_out[key]["q75"],
                    linestyle=ls,
                    alpha=0.2,
                    color=c[i],
                )

    for i, limit in enumerate(limits):
        if log:
            axs[0, i].set_yscale("log")
            axs[1, i].set_yscale("log")
        if limit is not None:
            axs[0, i].set_ylim([limits_lower[i], limit])
            axs[1, i].set_ylim([limits_lower[i], limit])

    # for ax in axs[0]:  # not needed when sharex=True
    #     ax.set_xticklabels([])
    for ax in axs[1]:
        ax.set_xlabel("step")
    for ax, title in zip(axs.flatten(), titles):
        ax.set_title(title)
        ax.grid()

    axs[0, 0].legend(loc="lower right")
    plt.tight_layout()

    if save_name is not None:
        save_path = f"{save_dir}/{save_dir.split('/')[-1]}_{save_name}.pdf"
        plt.savefig(save_path)


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
    plt.close()


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


def preprocess_grid_search(rlt_dir, dirs):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    keys = ["mse", "e_kin", "sinkhorn", "rho_mae", "dirichlet", "chamfer"]
    titles = [
        r"MSE$_{400}$",
        r"MSE$_{Ekin}$",
        "Sinkhorn",
        r"MAE$_{\rho}$",
        "Dirichlet",
        "Chamfer",
    ]

    # Structure: {dir: {metric: {mean/std/min/q25/q75/max: value}}}
    dict_ave = {k: {} for k in dirs}
    for i, d in enumerate(dirs):
        dict_ave[d] = average_trajs(os.path.join(rlt_dir, "test" + str(d)))
    # average the properties over the temporal dimension
    dict_ave = tree_util.tree_map(lambda x: np.mean(x), dict_ave)

    # Convert to {metric: {mean/std/q25/q75: [value_per_directory, ...]}}
    dict_out = {
        k: {"mean": [], "std": [], "q25": [], "q75": []} for k in dict_ave[dirs[0]]
    }
    for d in dirs:
        for k in dict_ave[d]:
            for m in dict_out[k]:
                dict_out[k][m].append(dict_ave[d][k][m] + EPS)
    return dict_out, keys, colors, titles


def plot_grid_search(
    rlt_dir: Union[List[str], str],
    dirs: Union[List[str], str],
    x_axis: Union[List[List[float]], List[float]],
    x_label,
    x_ticklabels=None,
    log=[False, False],
    rotate_x_ticks=False,
    save_name=None,
    labels=None,
):
    if isinstance(rlt_dir, list):
        save_dir = rlt_dir[0]
        assert isinstance(dirs[0], list) and (len(dirs) == len(rlt_dir))
        assert (len(x_axis) == len(rlt_dir)) and (len(x_axis[0]) == len(dirs[0]))
        assert (len(labels) == len(rlt_dir)) or (labels is None)
    else:
        save_dir = rlt_dir
        rlt_dir = [rlt_dir]
        dirs = [dirs]
        x_axis = [x_axis]

    line_styles = ["-", "--", "-.", ":", "-"]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for j, (rlt_dir_j, dirs_j) in enumerate(zip(rlt_dir, dirs)):
        dict_out, keys, colors, titles = preprocess_grid_search(rlt_dir_j, dirs_j)
        ls = line_styles[j]
        c = colors[j]
        x_axis_j = x_axis[j]
        assert len(dirs_j) == len(x_axis_j)
        label = labels[j] if labels is not None else None

        # Plot
        for i, (ax, k) in enumerate(zip(axs.flatten(), keys)):
            # print("Debug: ", dict_out[k]["mean"], x_axis, dict_out[k]["min"])
            ax.plot(x_axis_j, dict_out[k]["mean"], ls, label=label, color=c)
            ax.fill_between(
                x_axis_j,
                dict_out[k]["q25"],
                dict_out[k]["q75"],
                alpha=0.2,
                color=c,
                linestyle=ls,
            )
            ax.set_title(titles[i])
            # ax.legend()

    for ax in axs[1]:
        ax.set_xlabel(x_label)
    for ax in axs.flatten():
        if log[0]:
            ax.set_xscale("log")
        if log[1]:
            ax.set_yscale("log")
        # set the x-axis ticks, labels, and grid lines to the values in x_axis
        ax.set_xticks([])
        ax.set_xticks(x_axis[0])
        ax.set_xlim([min(x_axis[0]), max(x_axis[0])])

    for ax in axs[1]:
        if x_ticklabels is not None:
            ax.set_xticklabels(x_ticklabels)
        else:
            ax.set_xticklabels(x_axis[0])
        if rotate_x_ticks:
            ax.tick_params(axis="x", rotation=45)
    for ax in axs[0]:
        ax.set_xticklabels([])

    for ax in axs.flatten():
        ax.grid()

    if labels is not None:
        axs[0, 0].legend(loc="upper right")

    plt.tight_layout()

    if save_name is not None:
        save_path = f"{save_dir}/{save_dir.split('/')[-1]}_{save_name}.pdf"
        plt.savefig(save_path)


def plot_grid_search_bars(
    rlt_dir,
    dirs,
    labels,
    rotate_x_ticks=False,
    save_name=None,
    width=0.5,
    log=True,
    lims=[None, None, None, None, None, None],
):
    assert len(dirs) == len(labels)
    dict_out, keys, colors, titles = preprocess_grid_search(rlt_dir, dirs)

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    j = 0
    for i, (ax, k) in enumerate(zip(axs.flatten(), keys)):
        c = colors[j]
        err = [dict_out[k]["q25"], dict_out[k]["q75"]]
        ax.bar(labels, dict_out[k]["mean"], yerr=err, color=c, alpha=0.5, width=width)
        ax.set_title(titles[i])

    for i, ax in enumerate(axs.flatten()):
        ax.grid()
        if log:
            ax.set_yscale("log")
        if lims[i] is not None:
            ax.set_ylim(None, lims[i])
        if rotate_x_ticks:
            ax.tick_params(axis="x", rotation=rotate_x_ticks)

    plt.tight_layout()

    if save_name is not None:
        save_path = f"{rlt_dir}/{rlt_dir.split('/')[-1]}_{save_name}.pdf"
        plt.savefig(save_path)
