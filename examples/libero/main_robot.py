"""LIBERO evaluation with swappable robot embodiments (Panda / IIWA / UR5e).

Variant of ``main.py`` that lets the arm be swapped out. Non-Panda arms need initial states
regenerated for their own joint layout; those come from the Adapt3R release
(https://github.com/pairlab/Adapt3R), which ships them for the libero_90 suite.
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Optional

import imageio
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import extra_robots  # noqa: F401  -- registers MountedIIWA/OnTheGroundIIWA/MountedUR5e/OnTheGroundUR5e
from libero_state import gripper_state_2d

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_90"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # Number of rollouts per task
    num_tasks: Optional[int] = None  # Limit to the first N tasks of the suite (None = all)

    # Robot embodiment. "Panda" uses LIBERO's stock initial states; other arms require
    # `custom_init_states_dir` to point at states regenerated for their joint layout.
    robot: str = "Panda"
    gripper: Optional[str] = None
    custom_init_states_dir: Optional[str] = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"

    seed: int = 7  # Random Seed (for reproducibility)


def _load_init_states(task, args: Args) -> np.ndarray:
    """Loads initial states, preferring embodiment-specific ones when provided.

    LIBERO names its stock files ``<task>.pruned_init`` while the regenerated ones use
    ``<task>.init``, so the suffix is remapped here.
    """
    if args.custom_init_states_dir is None:
        path = (
            pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        )
    else:
        stem = task.init_states_file.replace(".pruned_init", "").replace(".init", "")
        path = pathlib.Path(args.custom_init_states_dir) / f"{stem}.init"

    if not path.exists():
        raise FileNotFoundError(f"Initial states not found for robot {args.robot}: {path}")
    return torch.load(path)


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = task_suite.n_tasks if args.num_tasks is None else min(args.num_tasks, task_suite.n_tasks)
    logging.info(f"Task suite: {args.task_suite_name} | robot: {args.robot} | tasks: {num_tasks}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name not in TASK_SUITE_MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = TASK_SUITE_MAX_STEPS[args.task_suite_name]

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    results = []
    for task_id in tqdm.tqdm(range(num_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = _load_init_states(task, args)
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed, args.robot, args.gripper
        )

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    gripper_state_2d(env, obs),
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Outcome is recorded in results.json rather than the filename, so re-running overwrites
            # instead of leaving stale videos from earlier runs behind.
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_id:03d}_ep{episode_idx:02d}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            results.append(
                {
                    "task_id": task_id,
                    "episode": episode_idx,
                    "task": task_description,
                    "success": bool(done),
                }
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        env.close()
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Robot: {args.robot}")
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    summary = {
        "robot": args.robot,
        "task_suite": args.task_suite_name,
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": float(total_successes) / float(total_episodes),
        "episodes_detail": results,
    }
    (pathlib.Path(args.video_out_path) / "results.json").write_text(json.dumps(summary, indent=2))


def _get_libero_env(task, resolution, seed, robot: str, gripper: Optional[str] = None):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "robots": [robot],
    }
    if gripper is not None:
        env_args["gripper_types"] = gripper
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
