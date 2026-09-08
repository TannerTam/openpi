"""Oracle upper bound for visual cross-embodiment transfer on LIBERO.

Measures how much of a non-Panda arm's success-rate drop is caused by the policy seeing an
unfamiliar arm, as opposed to the arm responding differently to the same end-effector command.

A second "mirror" environment holding the source arm (Panda) is kept synchronised with the
executing environment: objects are copied over verbatim each control step and the mirror arm tracks
the executing arm's end-effector pose. The policy is fed the mirror's images together with the
executing arm's proprioception, so it observes a Panda in the true scene while a different arm
actually moves. That is what a perfect feature translator would provide, so the resulting success
rate upper-bounds anything IGLT can achieve here.

If the oracle recovers the Panda baseline the gap is visual; if it stays near the native rate the
gap is kinematic and no visual translator will fix it.
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
from scipy.spatial.transform import Rotation

import extra_robots  # noqa: F401  -- registers the extra embodiments
from libero_state import StateLayout, gripper_state_2d

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    num_tasks: Optional[int] = None

    # Arm that executes actions, and the arm the policy is shown instead.
    robot: str = "IIWA"
    gripper: Optional[str] = None
    mirror_robot: str = "Panda"
    custom_init_states_dir: Optional[str] = None

    # OSC substeps used per control step to keep the mirror arm on the executing arm's pose.
    track_steps: int = 3

    video_out_path: str = "data/libero/oracle"
    seed: int = 7


def _load_init_states(task, states_dir: Optional[str]) -> np.ndarray:
    if states_dir is None:
        path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    else:
        stem = task.init_states_file.replace(".pruned_init", "").replace(".init", "")
        path = pathlib.Path(states_dir) / f"{stem}.init"
    if not path.exists():
        raise FileNotFoundError(f"Initial states not found: {path}")
    return torch.load(path)


def _eef_target(env, gripper_cmd: float) -> np.ndarray:
    """Absolute OSC action tracking the executing arm's end-effector pose.

    The pose is read from the controller rather than from ``robot0_eef_quat``: the observable is
    expressed in a frame rotated 90 degrees from the one absolute OSC targets, which leaves position
    tracking correct while the gripper (and with it the wrist camera) ends up sideways.
    """
    controller = env.env.robots[0].controller
    rotvec = Rotation.from_matrix(controller.ee_ori_mat).as_rotvec()
    return np.concatenate((controller.ee_pos, rotvec, [gripper_cmd]))


class MirrorView:
    """Renders the source arm inside the executing environment's scene."""

    def __init__(self, bddl: str, mirror_robot: str, resolution: int, track_steps: int):
        # The mirror is stepped `track_steps` times per control step, so it would otherwise hit
        # robosuite's episode horizon long before the executing env does and refuse to step further.
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            robots=[mirror_robot],
            camera_heights=resolution,
            camera_widths=resolution,
            ignore_done=True,
            horizon=10**9,
        )
        self.track_steps = track_steps
        self.layout = StateLayout(self.env)

    def reset(self, init_state: np.ndarray):
        self.env.reset()
        # Must follow reset(): robosuite rebuilds the controller there, restoring delta mode.
        self.env.env.robots[0].controller.use_delta = False
        self.env.set_init_state(init_state)

    def sync(self, real_parts: dict, target: np.ndarray):
        """Copies the live scene from the executing env, then tracks its end-effector pose.

        Only the object blocks are taken from the executing env; the mirror keeps its own arm and
        gripper state, including velocities. Clearing the arm's velocity here would restart the
        tracking controller from rest on every control step, leaving it unable to follow at all.
        """

        def pin_scene():
            own = self.layout.split(self.env.env.sim.get_state().flatten())
            return self.layout.build(
                time=real_parts["time"],
                arm_q=own["arm_q"],
                grip_q=own["grip_q"],
                obj_q=real_parts["obj_q"],
                arm_v=own["arm_v"],
                grip_v=own["grip_v"],
                obj_v=real_parts["obj_v"],
            )

        self.env.set_init_state(pin_scene())
        for _ in range(self.track_steps):
            self.env.step(target)

        # Re-pin before observing: the tracking steps advance the mirror's own physics, which would
        # otherwise leave the rendered objects several control steps ahead of the real scene.
        return self.env.set_init_state(pin_scene())

    def close(self):
        self.env.close()


def eval_oracle(args: Args) -> None:
    np.random.seed(args.seed)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks = task_suite.n_tasks if args.num_tasks is None else min(args.num_tasks, task_suite.n_tasks)
    max_steps = TASK_SUITE_MAX_STEPS[args.task_suite_name]
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    logging.info(
        f"Oracle: {args.robot} executes, policy sees {args.mirror_robot} | "
        f"suite={args.task_suite_name} tasks={num_tasks}"
    )

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    track_errors = []
    results = []

    for task_id in tqdm.tqdm(range(num_tasks)):
        task = task_suite.get_task(task_id)
        real_states = _load_init_states(task, args.custom_init_states_dir)
        mirror_states = _load_init_states(task, None)  # Panda uses LIBERO's stock states

        bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        real_env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            robots=[args.robot],
            **({"gripper_types": args.gripper} if args.gripper else {}),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        real_env.seed(args.seed)
        mirror = MirrorView(bddl, args.mirror_robot, LIBERO_ENV_RESOLUTION, args.track_steps)
        real_layout = StateLayout(real_env)

        task_episodes, task_successes = 0, 0
        for episode_idx in range(args.num_trials_per_task):
            real_env.reset()
            obs = real_env.set_init_state(real_states[episode_idx])
            mirror.reset(mirror_states[episode_idx])

            action_plan = collections.deque()
            replay_images = []
            gripper_cmd = -1.0
            done = False
            t = 0

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = real_env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    real_parts = real_layout.split(real_env.env.sim.get_state().flatten())
                    obs_mirror = mirror.sync(real_parts, _eef_target(real_env, gripper_cmd))
                    track_errors.append(
                        float(np.linalg.norm(obs_mirror["robot0_eef_pos"] - obs["robot0_eef_pos"]))
                    )

                    # Images come from the mirror; proprioception stays with the arm that moves.
                    img = np.ascontiguousarray(obs_mirror["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs_mirror["robot0_eye_in_hand_image"][::-1, ::-1])
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
                                    gripper_state_2d(real_env, obs),
                                )
                            ),
                            "prompt": str(task.language),
                        }
                        action_chunk = client.infer(element)["actions"]
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    gripper_cmd = float(action[6])
                    obs, _, done, _ = real_env.step(action.tolist())
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

            # These frames are the mirror's view, i.e. exactly what the policy was shown: the source
            # arm inside the executing arm's scene. Outcome lives in results.json so re-running
            # overwrites rather than accumulating stale videos.
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_id:03d}_ep{episode_idx:02d}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            results.append(
                {
                    "task_id": task_id,
                    "episode": episode_idx,
                    "task": task.language,
                    "success": bool(done),
                }
            )
            logging.info(f"Success: {done} | {total_successes}/{total_episodes}")

        real_env.close()
        mirror.close()

    logging.info(f"Oracle ({args.robot} executing, {args.mirror_robot} shown)")
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if track_errors:
        logging.info(
            f"Mirror tracking error: mean={np.mean(track_errors):.4f}m "
            f"p95={np.percentile(track_errors, 95):.4f}m max={np.max(track_errors):.4f}m"
        )

    summary = {
        "executing_robot": args.robot,
        "robot_shown_to_policy": args.mirror_robot,
        "task_suite": args.task_suite_name,
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": float(total_successes) / float(total_episodes),
        "mirror_tracking_error_m": {
            "mean": float(np.mean(track_errors)) if track_errors else None,
            "p95": float(np.percentile(track_errors, 95)) if track_errors else None,
            "max": float(np.max(track_errors)) if track_errors else None,
        },
        "episodes_detail": results,
    }
    (pathlib.Path(args.video_out_path) / "results.json").write_text(json.dumps(summary, indent=2))


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_oracle)
