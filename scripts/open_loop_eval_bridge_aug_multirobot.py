"""Open-loop, cross-embodiment evaluation for the `pi05_bridge_aug_widowx` checkpoint.

The OXE-AugE augmented Bridge dataset renders the *same* underlying WidowX trajectory onto
many robot embodiments (panda, sawyer, ur5e, google_robot, jaco, kinova3, kuka_iiwa, xarm7)
plus the original source view ("widowx"). This script feeds *each robot's own* state + image
into the policy and overlays the resulting predicted delta-EEF action chunks, one color per
robot, so you can compare how the model behaves across embodiments.

Model I/O for `pi05_bridge_aug_widowx` (see `LeRobotBridgeAugWidowXDataConfig` /
`BridgeAugWidowXInputs` in the repo):

    observation in:
        observation/image  -> base_0_rgb        (single third-person camera)
        observation/state  -> 7D EEF: [x, y, z, roll, pitch, yaw, gripper]
    action out:
        7D delta-EEF: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        (first 6 dims are deltas relative to the current EEF; gripper is absolute)
        horizon = model.action_horizon (10)

Per-robot state construction (the "convert" option):
    Each robot only stores `observation.<robot>.ee_pose` = [x, y, z, quat(w,x,y,z)] (no gripper),
    whose orientation lives in a different frame than the model's rpy state. We calibrate the
    constant frame offset from the *source* pair (`observation.state` rpy  <->  `observation.ee_pose`
    quat), then convert each robot's quaternion into the model's rpy convention. The gripper dim is
    taken from the shared source `observation.state` (gripper is a task command, not embodiment
    specific). The "widowx" source robot uses `observation.state` directly.

Open-loop protocol ("chunk-length stride"):
    Starting at frame 0 we feed (state_t, image_t) to the policy, get a `horizon`-step delta chunk,
    lay it on the timeline at t..t+horizon, then jump forward by `--chunk` frames and re-infer. No
    prediction is ever fed back -- observations always come from the recorded episode.

Ground truth (per robot): the delta chunk implied by that robot's own constructed state sequence
(future frames minus the current frame for the first 6 dims; absolute gripper), matching how the
training targets are built.

Run (use the openpi venv so JAX + model + lerobot import):

    cd /home/test/test12/tanner/openpi
    OPENPI_DATA_HOME=/home/test/test12/tanner/checkpoints \
    HF_LEROBOT_HOME=/home/test/test12/tanner/embodied_data/oxe-auge \
    HF_HUB_OFFLINE=1 OPENPI_VIDEO_BACKEND=pyav \
    .venv/bin/python scripts/open_loop_eval_bridge_aug_multirobot.py --episode 0

Outputs are written under::

    eval_outputs/<experiment>/<step>/ep<episode>/
        data.npz          # predictions + GT (for --replot)
        metrics.json      # full error breakdown
        metrics.csv       # per-robot summary table
        metrics.png       # bar chart + heatmap
        grid.png          # per-robot trajectory grid (default layout)
        overlay.png       # (--layout overlay|all)
        per_robot/        # (--layout per-robot|all)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib

# Resolve local caches / offline mode before importing openpi or lerobot.
os.environ.setdefault("OPENPI_DATA_HOME", "/home/test/test12/tanner/checkpoints")
os.environ.setdefault("HF_LEROBOT_HOME", "/home/test/test12/tanner/embodied_data/oxe-auge")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENPI_VIDEO_BACKEND", "pyav")
# lerobot loads episode metadata via HF `datasets`, which writes a lock/cache under HF_HOME
# (~/.cache/huggingface by default). That path is often root-owned / outside the writable workspace,
# so redirect the datasets cache to a writable location.
os.environ.setdefault("HF_DATASETS_CACHE", "/home/test/test12/tanner/.cache/hf_datasets")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("open_loop_bridge_aug")

DEFAULT_CONFIG = "pi05_bridge_aug_widowx"
DEFAULT_REPO_ID = "bridge_test_0_3475_augmented"
DEFAULT_EVAL_BASE = pathlib.Path("/home/test/test12/tanner/openpi/eval_outputs")
DEFAULT_CHECKPOINT = (
    "/home/test/test12/tanner/openpi/checkpoints/pi05_bridge_aug_widowx/"
    "pi05_bridge_aug_widowx_8gpu_20ep_260708/6659"
)

# "widowx" is the source view (observation.images.image + observation.state). The rest are the
# rendered augmented embodiments, each with its own image + ee_pose.
SOURCE_ROBOT = "widowx"
AUG_ROBOTS = ["panda", "sawyer", "ur5e", "google_robot", "jaco", "kinova3", "kuka_iiwa", "xarm7"]
ALL_ROBOTS = [SOURCE_ROBOT, *AUG_ROBOTS]

# 7D delta-EEF layout produced by the model.
ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


@dataclasses.dataclass(frozen=True)
class EvalPaths:
    """Output layout: eval_outputs/<experiment>/<step>/ep<episode>/*"""

    dir: pathlib.Path
    data: pathlib.Path
    metrics_json: pathlib.Path
    metrics_csv: pathlib.Path
    metrics_png: pathlib.Path
    grid: pathlib.Path
    overlay: pathlib.Path
    per_robot: pathlib.Path


def parse_checkpoint_paths(checkpoint: str | pathlib.Path) -> tuple[str, str]:
    """Return (experiment_name, step) from a checkpoint step directory."""
    p = pathlib.Path(checkpoint)
    return p.parent.name, p.name


def resolve_eval_dir(
    checkpoint: str | pathlib.Path,
    episode: int,
    *,
    eval_base: pathlib.Path = DEFAULT_EVAL_BASE,
    experiment: str | None = None,
) -> pathlib.Path:
    exp, step = parse_checkpoint_paths(checkpoint)
    return eval_base / (experiment or exp) / step / f"ep{episode}"


def eval_paths(eval_dir: pathlib.Path) -> EvalPaths:
    d = pathlib.Path(eval_dir)
    return EvalPaths(
        dir=d,
        data=d / "data.npz",
        metrics_json=d / "metrics.json",
        metrics_csv=d / "metrics.csv",
        metrics_png=d / "metrics.png",
        grid=d / "grid.png",
        overlay=d / "overlay.png",
        per_robot=d / "per_robot",
    )


def resolve_replot_source(path: str | pathlib.Path) -> pathlib.Path:
    """Accept an eval dir, data.npz, or legacy flat *.npz path."""
    p = pathlib.Path(path)
    if p.is_dir():
        candidate = p / "data.npz"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"no data.npz under eval dir: {p}")
    if p.suffix == ".npz":
        return p
    raise FileNotFoundError(f"expected eval dir or .npz, got: {p}")


def image_key(robot: str) -> str:
    return "observation.images.image" if robot == SOURCE_ROBOT else f"observation.images.{robot}"


def eepose_key(robot: str) -> str:
    return "observation.ee_pose" if robot == SOURCE_ROBOT else f"observation.{robot}.ee_pose"


def _chw_float_to_hwc_uint8(t) -> np.ndarray:
    """LeRobot image tensor (C,H,W float in [0,1]) -> (H,W,C) uint8 RGB."""
    arr = np.asarray(t)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _stack_col(hf_dataset, name: str) -> np.ndarray:
    """Stack a (numeric) hf_dataset column into an ndarray without decoding video."""
    return np.stack([np.asarray(v, dtype=np.float32) for v in hf_dataset[name]])


def calibrate_quat_to_rpy(source_state: np.ndarray, source_eepose: np.ndarray):
    """Learn the fixed frame offset mapping ee_pose quaternions to the model's rpy convention.

    ``source_state[:, 3:6]`` are rpy (xyz-euler) and ``source_eepose[:, 3:7]`` is the *same* source
    pose as a quaternion stored as ``[w, x, y, z]``. Empirically the two representations differ by a
    constant rotation R_off with ``R_state = R_quat @ R_off`` (R_off ~= 180deg about y). We estimate
    R_off by averaging over the episode frames and return a converter ``quat[w,x,y,z] -> rpy``.
    """
    rpy = source_state[:, 3:6].astype(np.float64)
    quat_wxyz = source_eepose[:, 3:7].astype(np.float64)
    r_quat = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]])  # -> scipy xyzw
    r_state = R.from_euler("xyz", rpy)
    r_off = (r_quat.inv() * r_state).mean()
    resid = ((r_quat * r_off).inv() * r_state).magnitude().mean()
    log.info("quat->rpy calibration: residual angle = %.4f rad (%.2f deg)", resid, np.degrees(resid))

    def convert(quat_wxyz_arr: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_wxyz_arr, dtype=np.float64).reshape(-1, 4)
        rot = R.from_quat(q[:, [1, 2, 3, 0]]) * r_off
        return rot.as_euler("xyz").astype(np.float32)

    return convert


def build_robot_states(hf_dataset, convert) -> dict[str, np.ndarray]:
    """Return {robot: state_seq (T,7)} in the model's [x,y,z,roll,pitch,yaw,gripper] convention."""
    src_state = _stack_col(hf_dataset, "observation.state")  # (T,7) x,y,z,rpy,gripper
    gripper = src_state[:, 6:7]

    states: dict[str, np.ndarray] = {SOURCE_ROBOT: src_state.astype(np.float32)}
    for r in AUG_ROBOTS:
        ee = _stack_col(hf_dataset, eepose_key(r))  # (T,7) xyz + quat(wxyz)
        xyz = ee[:, :3]
        rpy = convert(ee[:, 3:7])
        states[r] = np.concatenate([xyz, rpy, gripper], axis=1).astype(np.float32)
    return states


def load_episode(repo_id: str, episode: int, tolerance_s: float):
    """Return (dataset, start, end) frame range for one episode (episode-filtered dataset)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:  # lerobot < 0.4
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, episodes=[episode], tolerance_s=tolerance_s, video_backend="pyav")
    ep_idx = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int64)
    matches = np.flatnonzero(ep_idx == episode)
    if matches.size == 0:
        raise ValueError(f"episode {episode} not found; available: {sorted(set(ep_idx.tolist()))}")
    return ds, int(matches[0]), int(matches[-1]) + 1


def _resolve_prompt(item: dict, nli, global_idx: int, override: str | None) -> str:
    if override:
        return override
    task = item.get("task") if isinstance(item, dict) else None
    if isinstance(task, str) and task:
        return task
    if nli is not None:
        return nli[global_idx]
    return ""


def gt_delta_chunk(state_seq: np.ndarray, t: int, horizon: int, n: int) -> np.ndarray:
    """Ground-truth delta chunk at frame t: future - current for dims 0..5, absolute gripper."""
    chunk = np.empty((n, 7), dtype=np.float32)
    cur = state_seq[t]
    for k in range(n):
        nxt = state_seq[t + k]
        chunk[k, :6] = nxt[:6] - cur[:6]
        chunk[k, 6] = nxt[6]
    return chunk


def run_open_loop(policy, ds, start, end, states, robots, horizon, chunk, nli, prompt_override):
    """Per robot, infer every `chunk` frames and stitch predicted + GT delta chunks over the episode."""
    T = end - start
    preds = {r: np.full((T, 7), np.nan, dtype=np.float32) for r in robots}
    gts = {r: np.full((T, 7), np.nan, dtype=np.float32) for r in robots}

    chunk_starts = list(range(0, T, chunk))
    # Decode each chunk-start frame once (gives all robots' images), then infer per robot.
    for t in chunk_starts:
        item = ds[start + t]
        prompt = _resolve_prompt(item, nli, start + t, prompt_override)
        n = min(horizon, T - t)
        for r in robots:
            img = _chw_float_to_hwc_uint8(item[image_key(r)])
            obs = {
                "observation/image": img,
                "observation/state": np.asarray(states[r][t], dtype=np.float32),
                "prompt": prompt,
            }
            pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)  # (horizon, 7)
            preds[r][t : t + n] = pred[:n]
            gts[r][t : t + n] = gt_delta_chunk(states[r], t, horizon, n)
        log.info("frame %4d/%d: inferred %d robots", t, T, len(robots))
    return preds, gts


def _robot_colors(robots):
    cmap = matplotlib.colormaps["tab10"]
    return {r: cmap(i % 10) for i, r in enumerate(robots)}


# Alternating background per inference chunk: white / light yellow.
_CHUNK_BG = ("white", "#fff6c2")


def _shade_chunks(ax, T, chunk):
    """Shade the frame span of each inference chunk with an alternating background color."""
    for i, t0 in enumerate(range(0, T, chunk)):
        t1 = min(t0 + chunk, T)
        ax.axvspan(t0 - 0.5, t1 - 0.5, facecolor=_CHUNK_BG[i % 2], alpha=1.0, zorder=0, linewidth=0)


def _style_handles(draw_gt):
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color="black", lw=1.6, label="prediction")]
    if draw_gt:
        handles.append(Line2D([0], [0], color="black", lw=1.1, ls="--", alpha=0.6, label="ground truth"))
    return handles


def plot_overlay(preds, gts, robots, out_path, title, chunk, draw_gt):
    """All robots overlaid, one subplot per action dim (compact but can get crowded)."""
    T = next(iter(preds.values())).shape[0]
    x = np.arange(T)
    colors = _robot_colors(robots)

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(7, 1, figsize=(15, 2.2 * 7), sharex=True)
    for d, ax in enumerate(axes):
        _shade_chunks(ax, T, chunk)
        for r in robots:
            ax.plot(x, preds[r][:, d], color=colors[r], lw=1.6, alpha=0.95, zorder=3)
            if draw_gt:
                ax.plot(x, gts[r][:, d], color=colors[r], lw=1.1, ls="--", alpha=0.45, zorder=2)
        if ACTION_LABELS[d] == "gripper":
            ax.axhline(0.5, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax.set_ylabel(ACTION_LABELS[d], fontsize=9)
        ax.grid(True, alpha=0.2)

    robot_handles = [Line2D([0], [0], color=colors[r], lw=2, label=r) for r in robots]
    axes[0].legend(handles=robot_handles, loc="upper right", ncol=3, fontsize=7, title="robot (color)")
    axes[-1].legend(handles=_style_handles(draw_gt), loc="upper right", fontsize=7)
    axes[-1].set_xlabel(f"frame index (inference every {chunk} frames)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_grid(preds, gts, robots, out_path, title, chunk, draw_gt):
    """One row per robot, one column per action dim, so no two robots share a plot.

    Y-axis is shared per column so the same dimension is directly comparable across robots.
    """
    T = next(iter(preds.values())).shape[0]
    x = np.arange(T)
    colors = _robot_colors(robots)
    nrows, ncols = len(robots), 7

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.5 * ncols, 1.5 * nrows), sharex=True, sharey="col", squeeze=False
    )
    for i, r in enumerate(robots):
        for d in range(ncols):
            ax = axes[i][d]
            _shade_chunks(ax, T, chunk)
            ax.plot(x, preds[r][:, d], color=colors[r], lw=1.6, alpha=0.95, zorder=3)
            if draw_gt:
                ax.plot(x, gts[r][:, d], color="black", lw=1.0, ls="--", alpha=0.55, zorder=2)
            if ACTION_LABELS[d] == "gripper":
                ax.axhline(0.5, color="gray", lw=0.6, ls=":", alpha=0.5)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(ACTION_LABELS[d], fontsize=10)
            if d == 0:
                ax.set_ylabel(r, fontsize=9, color=colors[r], rotation=0, ha="right", va="center", labelpad=28)
        axes[i][ncols - 1].legend(handles=_style_handles(draw_gt), loc="upper right", fontsize=6)

    for d in range(ncols):
        axes[-1][d].set_xlabel(f"frame (every {chunk})", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_per_robot_files(preds, gts, robots, out_dir, title_prefix, chunk, draw_gt):
    """Save one standalone figure per robot (7 dims stacked), fully separated."""
    T = next(iter(preds.values())).shape[0]
    x = np.arange(T)
    colors = _robot_colors(robots)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in robots:
        fig, axes = plt.subplots(7, 1, figsize=(12, 1.8 * 7), sharex=True)
        for d, ax in enumerate(axes):
            _shade_chunks(ax, T, chunk)
            ax.plot(x, preds[r][:, d], color=colors[r], lw=1.8, label="prediction", zorder=3)
            if draw_gt:
                ax.plot(x, gts[r][:, d], color="black", lw=1.1, ls="--", alpha=0.6, label="ground truth", zorder=2)
            if ACTION_LABELS[d] == "gripper":
                ax.axhline(0.5, color="gray", lw=0.6, ls=":", alpha=0.5)
            ax.set_ylabel(ACTION_LABELS[d], fontsize=9)
            ax.grid(True, alpha=0.2)
            if d == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel(f"frame index (inference every {chunk} frames)")
        fig.suptitle(f"{title_prefix}  |  robot={r}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        p = out_dir / f"{r}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def save_npz(path, preds, gts, robots, chunk, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for r in robots:
        arrays[f"pred_{r}"] = preds[r]
        arrays[f"gt_{r}"] = gts[r]
    np.savez(path, robots=np.array(robots), chunk=chunk, **arrays, **{f"meta_{k}": v for k, v in meta.items()})


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    robots = [str(r) for r in z["robots"]]
    preds = {r: z[f"pred_{r}"] for r in robots}
    gts = {r: z[f"gt_{r}"] for r in robots}
    chunk = int(z["chunk"])
    meta = {k[len("meta_") :]: str(z[k]) for k in z.files if k.startswith("meta_")}
    return preds, gts, robots, chunk, meta


def compute_metrics(preds, gts, robots, chunk, *, horizon=None, meta=None):
    """Compute open-loop delta-EEF errors (MAE / RMSE) per robot, dim, and inference chunk."""
    T = next(iter(preds.values())).shape[0]
    meta = meta or {}
    metrics = {
        "config": meta.get("config"),
        "experiment": meta.get("experiment"),
        "episode": meta.get("episode"),
        "step": meta.get("step"),
        "trajectory_frames": T,
        "action_horizon": int(meta["action_horizon"]) if meta.get("action_horizon") is not None else None,
        "inference_stride": chunk,
        "num_inference_chunks": int(np.ceil(T / chunk)) if chunk else 0,
        "robots": {},
        "chunks": [],
    }

    for r in robots:
        pred, gt = preds[r], gts[r]
        valid = ~np.isnan(pred).any(axis=1)
        err = pred[valid] - gt[valid]
        per_dim_mae = np.abs(err).mean(axis=0) if err.size else np.full(7, np.nan)
        per_dim_rmse = np.sqrt((err**2).mean(axis=0)) if err.size else np.full(7, np.nan)
        worst_i = int(np.nanargmax(per_dim_mae)) if err.size else 0
        metrics["robots"][r] = {
            "mae_mean": float(np.nanmean(per_dim_mae)),
            "rmse_mean": float(np.nanmean(per_dim_rmse)),
            "mae_per_dim": {label: float(v) for label, v in zip(ACTION_LABELS, per_dim_mae)},
            "rmse_per_dim": {label: float(v) for label, v in zip(ACTION_LABELS, per_dim_rmse)},
            "worst_dim": ACTION_LABELS[worst_i],
            "worst_dim_mae": float(per_dim_mae[worst_i]),
            "covered_frames": int(valid.sum()),
        }

    for i, t0 in enumerate(range(0, T, chunk)):
        t1 = min(t0 + chunk, T)
        chunk_entry = {
            "chunk_index": i,
            "frame_start": t0,
            "frame_end": t1,
            "steps": t1 - t0,
            "robots": {},
        }
        robot_maes = []
        for r in robots:
            pred = preds[r][t0:t1]
            gt = gts[r][t0:t1]
            valid = ~np.isnan(pred).any(axis=1)
            if not valid.any():
                continue
            err = pred[valid] - gt[valid]
            mae = float(np.abs(err).mean())
            rmse = float(np.sqrt((err**2).mean()))
            chunk_entry["robots"][r] = {"mae_mean": mae, "rmse_mean": rmse}
            robot_maes.append(mae)
        chunk_entry["mae_mean_all_robots"] = float(np.mean(robot_maes)) if robot_maes else float("nan")
        metrics["chunks"].append(chunk_entry)

    return metrics


def log_metrics(metrics):
    log.info("Open-loop MAE / RMSE per robot (mean over 7 dims):")
    for r, m in metrics["robots"].items():
        log.info(
            "  %-13s MAE=%.4f RMSE=%.4f  worst=%s (%.4f)",
            r,
            m["mae_mean"],
            m["rmse_mean"],
            m["worst_dim"],
            m["worst_dim_mae"],
        )
    log.info("Per-chunk MAE (mean over robots):")
    for c in metrics["chunks"]:
        log.info(
            "  chunk %d frames [%d,%d) steps=%d  MAE=%.4f",
            c["chunk_index"],
            c["frame_start"],
            c["frame_end"],
            c["steps"],
            c["mae_mean_all_robots"],
        )


def save_metrics(paths: EvalPaths, metrics: dict, *, plot: bool = True):
    """Write metrics.json / metrics.csv under the eval dir; optionally plot metrics.png."""
    paths.dir.mkdir(parents=True, exist_ok=True)

    with paths.metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    header = ["robot", "mae_mean", "rmse_mean", "worst_dim", "worst_dim_mae", *ACTION_LABELS]
    with paths.metrics_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r, m in metrics["robots"].items():
            row = [
                r,
                f"{m['mae_mean']:.6f}",
                f"{m['rmse_mean']:.6f}",
                m["worst_dim"],
                f"{m['worst_dim_mae']:.6f}",
                *[f"{m['mae_per_dim'][label]:.6f}" for label in ACTION_LABELS],
            ]
            f.write(",".join(row) + "\n")

    log.info("Saved metrics -> %s", paths.metrics_json)
    log.info("Saved metrics -> %s", paths.metrics_csv)
    if plot:
        plot_metrics_chart(paths.metrics_csv, paths.metrics_png, metrics)
    return paths.metrics_json, paths.metrics_csv


def plot_metrics_chart(csv_path: pathlib.Path, out_path: pathlib.Path, metrics: dict | None = None):
    """Bar chart (MAE/RMSE per robot) + per-dimension MAE heatmap."""
    import csv as csv_mod

    rows = []
    with csv_path.open() as f:
        rows = list(csv_mod.DictReader(f))
    robots = [r["robot"] for r in rows]
    mae = np.array([float(r["mae_mean"]) for r in rows])
    rmse = np.array([float(r["rmse_mean"]) for r in rows])
    heat = np.array([[float(r[d]) for d in ACTION_LABELS] for r in rows])

    order = np.argsort(mae)
    robots = [robots[i] for i in order]
    mae, rmse, heat = mae[order], rmse[order], heat[order]

    cmap = matplotlib.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(robots))]

    config = (metrics or {}).get("config", "")
    episode = (metrics or {}).get("episode", "?")
    step = (metrics or {}).get("step", "?")
    title = f"{config}  |  ep {episode}  |  step {step}  |  input=per-robot EEF, output=delta-EEF"

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    x = np.arange(len(robots))
    w = 0.36
    ax0.bar(x - w / 2, mae, width=w, color=colors, alpha=0.9, label="MAE")
    ax0.bar(x + w / 2, rmse, width=w, color=colors, alpha=0.45, edgecolor="black", linewidth=0.8, label="RMSE")
    ax0.set_xticks(x)
    ax0.set_xticklabels(robots, rotation=25, ha="right")
    ax0.set_ylabel("error (delta-EEF)")
    ax0.set_title("Open-loop error per robot")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(loc="upper left")
    for i, v in enumerate(mae):
        ax0.text(i - w / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax1 = fig.add_subplot(gs[1])
    im = ax1.imshow(heat, aspect="auto", cmap="YlOrRd")
    ax1.set_xticks(np.arange(len(ACTION_LABELS)))
    ax1.set_xticklabels(ACTION_LABELS)
    ax1.set_yticks(np.arange(len(robots)))
    ax1.set_yticklabels(robots)
    ax1.set_title("Per-dimension MAE")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax1.text(
                j,
                i,
                f"{heat[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if heat[i, j] > heat.max() * 0.55 else "black",
            )
    fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02).set_label("MAE")

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved metrics figure -> %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="openpi TrainConfig name.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint step dir (params/ + assets/).")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id (resolved under HF_LEROBOT_HOME).")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument(
        "--robots",
        default="all",
        help=f"Comma-separated subset, or 'all'. Options: {','.join(ALL_ROBOTS)}",
    )
    parser.add_argument("--prompt", default=None, help="Override the language instruction (default: dataset task).")
    parser.add_argument(
        "--chunk", type=int, default=None, help="Frames to advance per inference. Default = model action_horizon."
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Cap episode length (debugging).")
    parser.add_argument("--no-gt", action="store_true", help="Do not draw the ground-truth dashed lines.")
    parser.add_argument(
        "--layout",
        default="grid",
        choices=["grid", "overlay", "per-robot", "all"],
        help="grid: one row per robot (default); overlay: all robots on shared axes; "
        "per-robot: one file per robot; all: produce every layout.",
    )
    parser.add_argument(
        "--eval-base-dir",
        default=str(DEFAULT_EVAL_BASE),
        help="Root output dir. Results go to <eval-base-dir>/<experiment>/<step>/ep<episode>/.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override experiment folder name (default: parent dir of --checkpoint).",
    )
    parser.add_argument("--tolerance-s", type=float, default=0.2, help="LeRobot timestamp tolerance.")
    parser.add_argument(
        "--out",
        default=None,
        help="Override eval output directory (default: <eval-base-dir>/<experiment>/<step>/ep<episode>/).",
    )
    parser.add_argument(
        "--replot",
        default=None,
        help="Skip inference; re-plot from eval dir or data.npz (legacy flat .npz also accepted).",
    )
    args = parser.parse_args()

    robots = ALL_ROBOTS if args.robots == "all" else [r.strip() for r in args.robots.split(",") if r.strip()]
    unknown = [r for r in robots if r not in ALL_ROBOTS]
    if unknown:
        raise SystemExit(f"unknown robots {unknown}; choose from {ALL_ROBOTS}")

    # Fast path: re-plot from cached data.npz without loading the model or dataset.
    if args.replot:
        npz_path = resolve_replot_source(args.replot)
        preds, gts, saved_robots, chunk, meta = load_npz(npz_path)
        robots = [r for r in robots if r in saved_robots] if args.robots != "all" else saved_robots
        paths = eval_paths(pathlib.Path(args.out) if args.out else npz_path.parent)
        title = (
            f"{meta.get('config', args.config)}  |  ep {meta.get('episode', '?')}  |  step {meta.get('step', '?')}  |  "
            f"stride {chunk}  |  input=per-robot EEF, output=delta-EEF"
        )
        metrics = compute_metrics(preds, gts, robots, chunk, meta=meta)
        log_metrics(metrics)
        save_metrics(paths, metrics)
        _render(args.layout, preds, gts, robots, paths, title, chunk, draw_gt=not args.no_gt)
        return

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    horizon = int(train_config.model.action_horizon)
    chunk = args.chunk or horizon
    log.info("config=%s action_horizon=%d chunk(stride)=%d robots=%s", args.config, horizon, chunk, robots)

    log.info("Loading dataset %s episode %d ...", args.repo_id, args.episode)
    ds, start, end = load_episode(args.repo_id, args.episode, args.tolerance_s)
    if args.max_frames is not None:
        end = min(end, start + args.max_frames)
    T = end - start
    log.info("episode %d -> frames [%d, %d) (%d frames)", args.episode, start, end, T)

    # Per-frame language instruction (string column), used when --prompt is not given.
    try:
        nli = [str(np.asarray(v).reshape(-1)[0]) for v in ds.hf_dataset["natural_language_instruction"]]
    except (KeyError, ValueError):
        nli = None

    # Build each robot's state sequence in the model's [xyz, rpy, gripper] convention.
    src_state = _stack_col(ds.hf_dataset, "observation.state")
    src_eepose = _stack_col(ds.hf_dataset, "observation.ee_pose")
    convert = calibrate_quat_to_rpy(src_state, src_eepose)
    states = build_robot_states(ds.hf_dataset, convert)
    states = {r: s[start:end] for r, s in states.items()}

    log.info("Loading policy from %s ...", args.checkpoint)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)

    preds, gts = run_open_loop(
        policy, ds, start, end, states, robots, horizon, chunk, nli, args.prompt
    )

    experiment, step = parse_checkpoint_paths(args.checkpoint)
    eval_dir = pathlib.Path(args.out) if args.out else resolve_eval_dir(
        args.checkpoint, args.episode, eval_base=pathlib.Path(args.eval_base_dir), experiment=args.experiment
    )
    paths = eval_paths(eval_dir)
    log.info("eval output dir -> %s", paths.dir)

    meta = {
        "config": args.config,
        "experiment": args.experiment or experiment,
        "episode": args.episode,
        "step": step,
        "action_horizon": horizon,
        "repo_id": args.repo_id,
    }
    metrics = compute_metrics(preds, gts, robots, chunk, horizon=horizon, meta=meta)
    log_metrics(metrics)
    save_metrics(paths, metrics)

    save_npz(paths.data, preds, gts, robots, chunk, meta)
    log.info("Saved data -> %s", paths.data)

    title = (
        f"{args.config}  |  ep {args.episode}  |  step {step}  |  horizon {horizon}  |  stride {chunk}  |  "
        f"input=per-robot EEF, output=delta-EEF"
    )
    _render(args.layout, preds, gts, robots, paths, title, chunk, draw_gt=not args.no_gt)


def _render(layout, preds, gts, robots, paths: EvalPaths, title, chunk, draw_gt):
    """Dispatch to the requested plot layout(s)."""
    if layout in ("grid", "all"):
        plot_grid(preds, gts, robots, paths.grid, title, chunk, draw_gt)
        log.info("Saved grid figure -> %s", paths.grid)
    if layout in ("overlay", "all"):
        plot_overlay(preds, gts, robots, paths.overlay, title, chunk, draw_gt)
        log.info("Saved overlay figure -> %s", paths.overlay)
    if layout in ("per-robot", "all"):
        n = plot_per_robot_files(preds, gts, robots, paths.per_robot, title, chunk, draw_gt)
        log.info("Saved %d per-robot figures -> %s", len(n), paths.per_robot)


if __name__ == "__main__":
    main()
