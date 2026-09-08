"""Open-loop evaluation for the Franka pi0.5 `arrange_fruits` checkpoint.

This replays one recorded RoboChallenge episode through the *exact* deployment
I/O contract (the same state + images in, action chunk out as the RC robot
server uses) but offline, so we can overlay the model's predictions on the
recorded ground-truth actions.

RoboChallenge / model I/O for `pi05_franka_arrange_fruits`
(see `FrankaInputs` / `FrankaOutputs` in `openpi/training/config.py` and
`meta/modality.json` of the `local/franka_arrange_fruits` dataset):

    observation in:
        images.scene  -> base_0_rgb        (exterior camera)
        images.wrist  -> left_wrist_0_rgb  (wrist camera)
        state         -> 7D FR3 joint positions
    action out:
        8D = [gello_joint1..7, gripper]   (gripper binarized at 0.5)
        horizon = 16 (model.action_horizon)

Open-loop protocol ("chunk-length stride"):
    Starting at frame 0 we feed (state_t, images_t) to the policy, get a
    `horizon`-step action chunk, lay those predicted actions on the timeline at
    t..t+horizon, then jump forward by `--chunk` frames (default = horizon) and
    re-infer. No predicted action is ever fed back into the robot state -- the
    observations always come from the recorded episode -- hence "open loop".

The recorded `action` column is the ground truth. We plot, per action
dimension, the ground-truth trajectory against the stitched open-loop
prediction and save a single figure.

Run (must use the openpi venv so JAX + the model + lerobot are importable):

    cd /user/tanxiyuan/openpi
    OPENPI_DATA_HOME=/user/tanxiyuan/cache/openpi_cache \
    HF_LEROBOT_HOME=/user/tanxiyuan/cache/huggingface/lerobot HF_HUB_OFFLINE=1 \
    .venv/bin/python scripts/open_loop_eval_franka_arrange_fruits.py \
        --episode 0
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib

# Resolve local caches / offline mode before importing openpi or lerobot.
os.environ.setdefault("OPENPI_DATA_HOME", "/user/tanxiyuan/cache/openpi_cache")
os.environ.setdefault("HF_LEROBOT_HOME", "/user/tanxiyuan/cache/huggingface/lerobot")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("open_loop_franka")

DEFAULT_CONFIG = "pi05_franka_arrange_fruits"
DEFAULT_REPO_ID = "local/franka_arrange_fruits"
DEFAULT_CHECKPOINT = (
    "/user/tanxiyuan/openpi/checkpoints/pi05_franka_arrange_fruits/"
    "franka_arrange_fruits_8gpu_20ep_260623/2199"
)
DEFAULT_PROMPT = "pick up the fruits on the table one by one and place them into the basket"

# 8D action layout for this dataset.
ACTION_LABELS = [f"joint{i}" for i in range(1, 8)] + ["gripper"]


def build_repack():
    """Map the deployment-style obs keys to the keys `FrankaInputs` expects.

    `create_trained_policy` applies these BEFORE the data transforms, so we feed
    a clean nested observation, identical to `scripts/serve_franka_pick_up_cube.py`.
    """
    from openpi import transforms as _transforms

    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "scene_rgb": "images/scene",
                    "wrist_rgb": "images/wrist",
                    "state": "state",
                }
            )
        ]
    )


def _chw_float_to_hwc_uint8(t) -> np.ndarray:
    """LeRobot image tensor (C,H,W float in [0,1]) -> (H,W,C) uint8 RGB."""
    arr = np.asarray(t)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
    return np.ascontiguousarray(arr.astype(np.uint8))


def load_episode(repo_id: str, episode: int, tolerance_s: float):
    """Return (dataset, global_start, global_end, gt_actions[T,8]) for one episode."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:  # lerobot < 0.4
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id,
        episodes=[episode],
        tolerance_s=tolerance_s,
        video_backend="pyav",
    )

    # Episode frame ranges come from the non-video columns so we avoid decoding
    # video just to find boundaries. Rows are in global-index order.
    ep_idx = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int64)
    gt_actions_all = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    if ep_idx.shape[0] != len(ds):
        raise RuntimeError(f"hf_dataset rows ({ep_idx.shape[0]}) != len(ds) ({len(ds)}); cannot map episodes")

    matches = np.flatnonzero(ep_idx == episode)
    if matches.size == 0:
        raise ValueError(f"episode {episode} not found; available: {sorted(set(ep_idx.tolist()))}")
    start, end = int(matches[0]), int(matches[-1]) + 1
    return ds, start, end, gt_actions_all[start:end]


def run_open_loop(policy, ds, start: int, end: int, horizon: int, chunk: int, prompt: str):
    """Feed obs every `chunk` frames; return stitched prediction array (T, 8).

    Cells not covered by any chunk stay NaN. We also return the list of
    (t_local, pred_chunk) so each chunk can be drawn individually if desired.
    """
    T = end - start
    pred_traj = np.full((T, 8), np.nan, dtype=np.float32)
    chunks: list[tuple[int, np.ndarray]] = []

    for t in range(0, T, chunk):
        item = ds[start + t]
        obs = {
            "images": {
                "scene": _chw_float_to_hwc_uint8(item["observation.images.scene"]),
                "wrist": _chw_float_to_hwc_uint8(item["observation.images.wrist"]),
            },
            "state": np.asarray(item["observation.state"], dtype=np.float32),
            "prompt": prompt,
        }
        pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)  # (horizon, 8)
        n = min(horizon, T - t)
        pred_traj[t : t + n] = pred[:n]
        chunks.append((t, pred[:n]))
        log.info("frame %4d/%d: predicted chunk %s", t, T, tuple(pred.shape))

    return pred_traj, chunks


def plot_comparison(gt: np.ndarray, pred: np.ndarray, chunks, out_path: pathlib.Path, title: str, chunk: int):
    """Overlay ground-truth vs open-loop predicted actions, one subplot per dim."""
    T, D = gt.shape
    x = np.arange(T)
    fig, axes = plt.subplots(D, 1, figsize=(14, 2.1 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        ax.plot(x, gt[:, d], color="#1f77b4", lw=1.8, label="ground truth", zorder=1)
        # Stitched open-loop prediction (NaN gaps break the line automatically).
        ax.plot(x, pred[:, d], color="#d62728", lw=1.4, alpha=0.9, label="open-loop pred", zorder=2)

        # gripper is binarized in the model output; show the 0.5 threshold.
        if ACTION_LABELS[d] == "gripper":
            ax.axhline(0.5, color="green", lw=0.6, ls=":", alpha=0.6)
        ax.set_ylabel(ACTION_LABELS[d], fontsize=9)
        ax.grid(True, alpha=0.2)
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(f"frame index (inference every {chunk} frames)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="openpi TrainConfig name.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint step dir (params/ + assets/).")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id (the converted RC data).")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Task language instruction.")
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Frames to advance per inference (open-loop stride). Default = model action_horizon.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Cap episode length (debugging).")
    parser.add_argument("--tolerance-s", type=float, default=0.1, help="LeRobot timestamp tolerance.")
    parser.add_argument("--out", default=None, help="Output figure path (.png).")
    args = parser.parse_args()

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    horizon = int(train_config.model.action_horizon)
    chunk = args.chunk or horizon
    log.info("config=%s action_horizon=%d chunk(stride)=%d", args.config, horizon, chunk)

    log.info("Loading dataset %s episode %d ...", args.repo_id, args.episode)
    ds, start, end, gt = load_episode(args.repo_id, args.episode, args.tolerance_s)
    if args.max_frames is not None:
        end = min(end, start + args.max_frames)
        gt = gt[: end - start]
    log.info("episode %d -> frames [%d, %d) (%d frames)", args.episode, start, end, end - start)

    log.info("Loading policy from %s ...", args.checkpoint)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        repack_transforms=build_repack(),
        default_prompt=args.prompt,
    )

    pred, chunks = run_open_loop(policy, ds, start, end, horizon, chunk, args.prompt)

    # Open-loop error over the covered timesteps (ignore NaN gaps).
    valid = ~np.isnan(pred).any(axis=1)
    mae = np.nanmean(np.abs(pred[valid] - gt[valid]), axis=0)
    log.info("Open-loop MAE per dim:")
    for label, e in zip(ACTION_LABELS, mae):
        log.info("  %-8s %.4f", label, float(e))
    log.info("  %-8s %.4f", "mean", float(np.mean(mae)))

    step = pathlib.Path(args.checkpoint).name
    out_path = pathlib.Path(
        args.out
        or f"/user/tanxiyuan/openpi/eval_outputs/open_loop_{args.config}_ep{args.episode}_step{step}.png"
    )
    title = (
        f"{args.config}  |  ep {args.episode}  |  step {step}  |  "
        f"horizon {horizon}  |  stride {chunk}  |  mean MAE {np.mean(mae):.4f}"
    )
    plot_comparison(gt, pred, chunks, out_path, title, chunk)
    log.info("Saved figure -> %s", out_path)


if __name__ == "__main__":
    main()
