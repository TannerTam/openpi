"""Regenerate LIBERO initial states for a non-Panda arm.

LIBERO ships flattened MuJoCo sim states whose layout depends on the arm's joint count, so a
different embodiment needs its own. Each state is rebuilt by driving the new arm to the same
end-effector pose the Panda starts from, then splicing its joint angles into the original state so
every object stays exactly where it was.

Adapted from Adapt3R's ``make_robot_change_init.py``; the target pose is read from the observation
rather than the controller goal, since LIBERO's ``set_init_state`` does not refresh controller goals.

Usage:
    python make_robot_init_states.py --task-suite-name libero_10 --robot IIWA --out-dir <dir>
"""

import argparse
import pathlib
from typing import Optional

import numpy as np
import torch
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation

import extra_robots  # noqa: F401  -- registers the extra embodiments
from libero_state import StateLayout

# Steps of absolute-pose control given to the new arm to settle onto the Panda's starting pose.
SETTLE_STEPS = 15


def _target_pose(env) -> np.ndarray:
    """Absolute OSC action (position + axis-angle) matching the Panda's starting pose.

    Read from the controller rather than from ``robot0_eef_quat``: the observable is expressed in a
    frame rotated 90 degrees from the one absolute OSC targets, which would leave the new arm at the
    right position with a sideways gripper.
    """
    controller = env.env.robots[0].controller
    rotvec = Rotation.from_matrix(controller.ee_ori_mat).as_rotvec()
    return np.concatenate((controller.ee_pos, rotvec, [-1.0]))


def build_init_states(task, robot: str, gripper: Optional[str] = None, resolution: int = 128):
    bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    old_states = torch.load(
        pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    )

    env_panda = OffScreenRenderEnv(
        bddl_file_name=bddl, robots=["Panda"], camera_heights=resolution, camera_widths=resolution
    )
    new_kwargs = dict(
        bddl_file_name=bddl, robots=[robot], camera_heights=resolution, camera_widths=resolution
    )
    if gripper is not None:
        new_kwargs["gripper_types"] = gripper
    env_new = OffScreenRenderEnv(**new_kwargs)
    # Each env is reset exactly once. LIBERO defaults to hard resets, which rebuild the MuJoCo
    # model from XML and cost ~1.3 s; writing the sim state directly costs ~1.6 ms and fully
    # determines the episode, so per-episode resets are avoided.
    env_panda.reset()
    env_new.reset()

    # Interpret actions as absolute end-effector poses rather than deltas. This must follow
    # reset(): robosuite rebuilds the controller there, which would restore delta mode.
    env_new.env.robots[0].controller.use_delta = False

    home_state = env_new.env.sim.get_state().flatten()

    src_layout = StateLayout(env_panda)
    new_layout = StateLayout(env_new)
    home = new_layout.split(home_state)

    new_states, errors = [], []
    for old_state in old_states:
        obs_panda = env_panda.set_init_state(old_state)
        abs_action = _target_pose(env_panda)
        source = src_layout.split(old_state)

        # Settle from the home pose with the episode's real object layout in place, so any contact
        # the arm makes on the way matches the scene it will actually start in.
        env_new.set_init_state(new_layout.transplant(source, home["arm_q"], home["grip_q"]))
        for _ in range(SETTLE_STEPS):
            env_new.step(abs_action)
        settled = new_layout.split(env_new.env.sim.get_state().flatten())

        init_state = new_layout.transplant(source, settled["arm_q"], settled["grip_q"])
        obs_new = env_new.set_init_state(init_state)
        errors.append(np.linalg.norm(obs_new["robot0_eef_pos"] - obs_panda["robot0_eef_pos"]))
        new_states.append(init_state)

    env_panda.close()
    env_new.close()
    return np.array(new_states), float(np.mean(errors)), float(np.max(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--robot", required=True)
    parser.add_argument("--gripper", default=None, help="Override the arm's default gripper")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-tasks", type=int, default=None)
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n = suite.n_tasks if args.num_tasks is None else min(args.num_tasks, suite.n_tasks)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_id in range(n):
        task = suite.get_task(task_id)
        stem = task.init_states_file.replace(".pruned_init", "").replace(".init", "")
        out_path = out_dir / f"{stem}.init"
        if out_path.exists():
            print(f"[{task_id + 1}/{n}] skip (exists) {stem}", flush=True)
            continue

        states, mean_err, max_err = build_init_states(task, args.robot, args.gripper)
        torch.save(states, out_path)
        print(
            f"[{task_id + 1}/{n}] {stem} -> {states.shape} eef_err mean={mean_err:.4f} max={max_err:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
