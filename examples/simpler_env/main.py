"""Closed-loop evaluation of a Bridge/WidowX openpi policy in SimplerEnv.

Runs as a client against a policy served by ``scripts/serve_policy.py``. The policy server lives in
the openpi venv (Python 3.11 + JAX); this script must run in an environment with SAPIEN 2.x and
SimplerEnv installed (Python 3.10), hence the two-process split.

Frame conventions (verified against the BridgeData2 norm stats):
  * SimplerEnv reports ``tcp_pose``/``base_pose`` in the world frame, and the WidowX base carries a
    180 degree yaw relative to the world, so the EEF pose is transformed into the base frame.
  * Each robot's TCP frame is rotated relative to the EEF frame BridgeData2 uses, so every adapter
    carries its own ``TOOL_ROT_OFFSET``. Without it the state fed to the model is far outside the
    training distribution (rpy ~ [-3.06, 1.51, -3.08] instead of ~0).
  * Only the state needs that offset. Both ``ee_align`` controllers apply the commanded rotation by
    left multiplication, i.e. in base-frame axes, which is independent of the tool offset.

The Google Robot suite is supported as a cross-embodiment check. The model is trained on WidowX
only, so its state input is necessarily out of distribution there; ``--diagnose`` prints the
observed state range so that mismatch stays visible.

The Bridge tasks can also be driven by the Google Robot (see ``bridge_embodiment``), which keeps the
scene, camera and success criteria fixed and changes only the arm.

Example:
    python examples/simpler_env/main.py --suite bridge --robot widowx --episodes 0 1
    python examples/simpler_env/main.py --suite bridge --robot google_robot --diagnose
    python examples/simpler_env/main.py --suite google_robot --robot google_robot
"""

import argparse
import collections
import json
import pathlib
import time

import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d.euler import euler2axangle
from transforms3d.euler import euler2mat
from transforms3d.euler import mat2euler
from transforms3d.quaternions import quat2mat
import tqdm
import websockets.exceptions

import bridge_embodiment
from openpi_client import websocket_client_policy as _websocket_client_policy

WIDOWX_TASKS = (
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
)

GOOGLE_ROBOT_TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
    "google_robot_move_near",
)

# The Bridge and move_near environments lay objects out from `episode_id`; the coke can environments
# ignore it and draw the object pose from the episode seed instead.
SEED_DRIVEN_TASKS = tuple(t for t in GOOGLE_ROBOT_TASKS if "coke_can" in t)

# Rotation from a robot's TCP frame into the EEF frame BridgeData2 uses, where rpy = 0 means the
# gripper points straight down. Both robots close their fingers along TCP -y, but they differ in
# which axis approaches the object: the WidowX along TCP x, the Google Robot along TCP z. Each
# therefore needs its own offset, defined by eef z = -approach and eef y = -closing axis. Sharing
# the WidowX's offset leaves the Google Robot's roll and pitch ~1 rad outside the training range.


class WidowXAdapter:
    """The embodiment BridgeData2 was recorded on, so its conventions are the training conventions.

    The gripper controller takes an absolute command where +1 opens and -1 closes, and the finger
    joints open as their position grows.
    """

    FINGER_QPOS_CLOSED = 0.015
    FINGER_QPOS_OPEN = 0.037
    # Approaches along TCP x, so eef z = -tcp x and eef x = tcp z.
    TOOL_ROT_OFFSET = euler2mat(0.0, -np.pi / 2, 0.0)

    def reset(self) -> None:
        pass

    def gripper_command(self, opening: float) -> float:
        return 2.0 * float(opening > 0.5) - 1.0


class GoogleRobotAdapter:
    """The Google Robot, whose gripper conventions are inverted relative to the WidowX.

    Three differences matter here:
      * the finger joints close as their position grows, the opposite of the WidowX;
      * the gripper controller takes a *delta* command where +1 closes and -1 opens;
      * that delta controller needs several steps to finish a motion, so SimplerEnv's own policy
        adapters latch each gripper transition for 15 steps. This mirrors that logic, without which
        the gripper never fully closes before the arm moves on.
    """

    FINGER_QPOS_CLOSED = 1.07
    FINGER_QPOS_OPEN = 0.0
    STICKY_REPEATS = 15
    # Approaches along TCP z, so eef z = -tcp z and eef x = -tcp x.
    TOOL_ROT_OFFSET = euler2mat(0.0, np.pi, 0.0)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous_opening: float | None = None
        self._sticky_command = 0.0
        self._sticky_steps = 0
        self._sticky_on = False

    def gripper_command(self, opening: float) -> float:
        if self._previous_opening is None:
            command = 0.0
        else:
            command = self._previous_opening - opening
        self._previous_opening = opening

        if abs(command) > 0.5 and not self._sticky_on:
            self._sticky_on = True
            self._sticky_command = command
        if self._sticky_on:
            self._sticky_steps += 1
            command = self._sticky_command
        if self._sticky_steps == self.STICKY_REPEATS:
            self._sticky_on = False
            self._sticky_steps = 0
            self._sticky_command = 0.0
        return float(command)


class ReconnectingPolicyClient:
    """Policy client that reconnects when a request outlives the websocket keepalive timeout.

    The server JIT-compiles the model on its first request, which takes longer than the keepalive
    timeout and therefore drops the connection. The compiled program is cached, so reconnecting and
    resending the same request succeeds.
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    def get_server_metadata(self) -> dict:
        return self._client.get_server_metadata()

    def infer(self, obs: dict) -> dict:
        try:
            return self._client.infer(obs)
        except websockets.exceptions.ConnectionClosed:
            print("policy connection dropped (server still compiling?), reconnecting")
            self._client = _websocket_client_policy.WebsocketClientPolicy(self._host, self._port)
            return self._client.infer(obs)


def make_adapter(robot: str) -> WidowXAdapter | GoogleRobotAdapter:
    if robot == "widowx":
        return WidowXAdapter()
    if robot == "google_robot":
        return GoogleRobotAdapter()
    raise ValueError(f"unknown robot embodiment {robot!r}")


def finger_qpos_indices(env) -> list[int]:
    """Locates the gripper finger joints, which sit at different offsets on the two robots."""
    joints = [j.name for j in env.unwrapped.agent.robot.get_active_joints()]
    return [i for i, name in enumerate(joints) if "finger" in name]


def build_state(obs: dict, adapter, table_z: float, finger_indices: list[int]) -> np.ndarray:
    """Builds the 7-dim BridgeData2 state (xyz + rpy + gripper) from a SimplerEnv observation."""
    tcp_pose = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float64)
    base_pose = np.asarray(obs["agent"]["base_pose"], dtype=np.float64)

    # BridgeData2 measures the EEF against the WidowX base, which stands at table height, so its z
    # really means "height above the work surface". Anchoring z to the table carries that meaning
    # over to robots whose base is on the floor instead.
    anchor = np.array([base_pose[0], base_pose[1], table_z])
    base_rot = quat2mat(base_pose[3:])
    position = base_rot.T @ (tcp_pose[:3] - anchor)
    tcp_rot_in_base = base_rot.T @ quat2mat(tcp_pose[3:])
    rpy = mat2euler(tcp_rot_in_base @ adapter.TOOL_ROT_OFFSET)

    finger_qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float64)[finger_indices].mean()
    span = adapter.FINGER_QPOS_OPEN - adapter.FINGER_QPOS_CLOSED
    gripper = (finger_qpos - adapter.FINGER_QPOS_CLOSED) / span

    return np.concatenate([position, rpy, [gripper]]).astype(np.float32)


def to_env_action(action: np.ndarray, adapter) -> np.ndarray:
    """Converts a 7-dim delta-EEF model action into the 7-dim vector SimplerEnv expects."""
    axis, angle = euler2axangle(*action[3:6])
    gripper = adapter.gripper_command(float(action[6]))
    return np.concatenate([action[:3], axis * angle, [gripper]]).astype(np.float64)


def make_env(task: str, robot: str, episode_id: int):
    """Builds the environment and resets it to the requested episode.

    Bridge tasks go through ``bridge_embodiment`` so that the same scene can be driven by either
    arm; the Google Robot's own task suite is created by SimplerEnv directly.
    """
    options = {"obj_init_options": {"episode_id": episode_id}}
    if task in bridge_embodiment.BRIDGE_TASKS:
        env, robot_init_options = bridge_embodiment.make_bridge_task(task, robot)
        options["robot_init_options"] = robot_init_options
        obs, _ = env.reset(options=options)
    elif task in SEED_DRIVEN_TASKS:
        env = simpler_env.make(task)
        obs, _ = env.reset(seed=episode_id, options=options)
    else:
        env = simpler_env.make(task)
        obs, _ = env.reset(options=options)
    return env, obs


def observation_image(obs: dict, env, task: str) -> np.ndarray:
    """Picks the camera the policy should see.

    Bridge tasks always use the Bridge third-person view, which is the viewpoint the policy was
    trained on, even when a different arm is loaded; otherwise the robot's own default camera.
    """
    if task in bridge_embodiment.BRIDGE_TASKS:
        return bridge_embodiment.bridge_image(obs)
    return get_image_from_maniskill2_obs_dict(env.unwrapped, obs)


def run_episode(
    client: ReconnectingPolicyClient,
    task: str,
    robot: str,
    episode_id: int,
    *,
    replan_steps: int,
    max_steps: int,
    video_dir: pathlib.Path | None,
    diagnose: bool,
) -> dict:
    env, obs = make_env(task, robot, episode_id)
    instruction = env.unwrapped.get_language_instruction()

    adapter = make_adapter(robot)
    adapter.reset()
    table_z = env.unwrapped.scene_table_height
    finger_indices = finger_qpos_indices(env)

    frames = [observation_image(obs, env, task)]
    action_plan: collections.deque = collections.deque()
    states, raw_actions = [], []
    success, step = False, 0

    while step < max_steps:
        image = observation_image(obs, env, task)
        state = build_state(obs, adapter, table_z, finger_indices)
        states.append(state)

        if not action_plan:
            # The server resizes to 224x224 exactly as the training pipeline did, so send the raw frame.
            result = client.infer(
                {
                    "observation/image": image,
                    "observation/state": state,
                    "prompt": instruction,
                }
            )
            chunk = np.asarray(result["actions"])
            action_plan.extend(chunk[:replan_steps])

        action = np.asarray(action_plan.popleft())
        raw_actions.append(action)

        if diagnose and step < 3:
            print(f"    step {step}: state={np.round(state, 4)}")
            print(f"             action={np.round(action, 4)}")

        obs, _reward, done, truncated, info = env.step(to_env_action(action, adapter))
        frames.append(observation_image(obs, env, task))
        step += 1

        if done:
            success = True
            break
        if truncated:
            break

    if video_dir is not None:
        import mediapy

        video_dir.mkdir(parents=True, exist_ok=True)
        tag = "success" if success else "failure"
        mediapy.write_video(
            video_dir / f"ep{episode_id:03d}_{tag}.mp4", frames, fps=env.unwrapped.control_freq
        )

    env.close()
    return {
        "task": task,
        "robot": robot,
        "episode_id": episode_id,
        "instruction": instruction,
        "success": success,
        "steps": step,
        "state_min": np.min(states, axis=0).tolist(),
        "state_max": np.max(states, axis=0).tolist(),
        "action_min": np.min(raw_actions, axis=0).tolist(),
        "action_max": np.max(raw_actions, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--suite",
        default="bridge",
        choices=("bridge", "google_robot"),
        help="Which task set to run: the Bridge/WidowX tasks, or the Google Robot's own tasks.",
    )
    parser.add_argument(
        "--robot",
        default="widowx",
        choices=("widowx", "google_robot"),
        help="Which arm performs the tasks. The Bridge suite accepts either, so the same tasks can "
        "be compared across embodiments; the Google Robot suite only runs its own arm.",
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=("all", *WIDOWX_TASKS, *GOOGLE_ROBOT_TASKS),
        help="A single task, or 'all' for every task in the selected suite.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=list(range(24)),
        help="Episode ids to run per task (SimplerEnv's Bridge suite defines 24).",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=5,
        help="How many actions of each predicted chunk to execute before re-querying the policy.",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--video-dir", default="eval_outputs/simpler_env")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print the first few states/actions of every episode to check the frame conventions.",
    )
    args = parser.parse_args()

    if args.suite == "google_robot" and args.robot != "google_robot":
        parser.error("the Google Robot task suite can only be run with --robot google_robot")

    suite = WIDOWX_TASKS if args.suite == "bridge" else GOOGLE_ROBOT_TASKS
    tasks = list(suite) if args.task == "all" else [args.task]
    print(f"suite={args.suite} robot={args.robot} tasks={len(tasks)}")

    client = ReconnectingPolicyClient(args.host, args.port)
    print(f"connected to policy server: {client.get_server_metadata()}")

    out_root = pathlib.Path(args.video_dir)
    results, started = [], time.time()

    for task in tasks:
        successes = 0
        for episode_id in tqdm.tqdm(args.episodes, desc=task):
            result = run_episode(
                client,
                task,
                args.robot,
                episode_id,
                replan_steps=args.replan_steps,
                max_steps=args.max_steps,
                video_dir=None if args.no_video else out_root / task,
                diagnose=args.diagnose,
            )
            results.append(result)
            successes += int(result["success"])
        print(f"{task}: {successes}/{len(args.episodes)} = {successes / len(args.episodes):.1%}")

    total = sum(r["success"] for r in results)
    print(f"\noverall: {total}/{len(results)} = {total / len(results):.1%}  ({time.time() - started:.0f}s)")

    if args.diagnose:
        print("\nstate range over all episodes (BridgeData2 q01/q99 for reference):")
        print("  observed min:", np.round(np.min([r["state_min"] for r in results], axis=0), 3))
        print("  observed max:", np.round(np.max([r["state_max"] for r in results], axis=0), 3))
        print("  train  q01 : [ 0.168 -0.178 -0.061 -0.393 -0.574 -1.444  0.052]")
        print("  train  q99 : [ 0.458  0.242  0.204  0.405  0.295  1.931  1.012]")

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "results.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
