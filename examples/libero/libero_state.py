"""Indexing helpers for LIBERO's flattened MuJoCo sim states.

A flattened state is ``[time][qpos][qvel]``. Both blocks are ordered arm, then gripper, then
objects, but ``nq != nv`` whenever the scene holds free-joint objects (7 position values against 6
velocity values each), so the two blocks cannot be split at the same offsets.

Splitting states this way is what lets a state be transplanted between embodiments: object entries
are carried over verbatim while the arm and gripper blocks, whose widths depend on the robot, are
taken from the target.
"""

import numpy as np


class StateLayout:
    """Index layout of a flattened sim state for one LIBERO environment."""

    def __init__(self, env):
        sim = env.env.sim
        robot = env.env.robots[0]
        self.nq = sim.model.nq
        self.nv = sim.model.nv
        self.n_arm = len(robot.robot_model.joints)
        self.n_grip = len(robot.gripper.joints)
        # Arm and gripper joints are all single-dof, so they occupy equal widths in qpos and qvel;
        # the objects absorb the difference.
        self.n_obj_q = self.nq - self.n_arm - self.n_grip
        self.n_obj_v = self.nv - self.n_arm - self.n_grip
        self.state_len = 1 + self.nq + self.nv

    def split(self, state: np.ndarray) -> dict:
        if len(state) != self.state_len:
            raise ValueError(f"State length {len(state)} does not match layout ({self.state_len})")
        q = state[1 : 1 + self.nq]
        v = state[1 + self.nq :]
        a, g = self.n_arm, self.n_grip
        return {
            "time": state[:1],
            "arm_q": q[:a],
            "grip_q": q[a : a + g],
            "obj_q": q[a + g :],
            "arm_v": v[:a],
            "grip_v": v[a : a + g],
            "obj_v": v[a + g :],
        }

    def build(self, time, arm_q, grip_q, obj_q, arm_v, grip_v, obj_v) -> np.ndarray:
        state = np.concatenate([time, arm_q, grip_q, obj_q, arm_v, grip_v, obj_v], axis=0)
        if len(state) != self.state_len:
            raise ValueError(f"Built state length {len(state)} does not match layout ({self.state_len})")
        return state

    def transplant(self, source_parts: dict, arm_q, grip_q, zero_robot_velocity: bool = True):
        """Places this robot's arm/gripper into another environment's scene.

        Object positions and velocities come from ``source_parts``; the robot blocks come from this
        layout's own environment, since their widths differ across embodiments.
        """
        return self.build(
            time=source_parts["time"],
            arm_q=arm_q,
            grip_q=grip_q,
            obj_q=source_parts["obj_q"],
            arm_v=np.zeros(self.n_arm) if zero_robot_velocity else source_parts.get("arm_v"),
            grip_v=np.zeros(self.n_grip) if zero_robot_velocity else source_parts.get("grip_v"),
            obj_v=source_parts["obj_v"],
        )


# Per-finger travel of the Panda gripper, i.e. the scale the policy's state input is expressed in.
PANDA_FINGER_MAX = 0.04


def gripper_state_2d(env, obs) -> np.ndarray:
    """Reduces gripper joint positions to the 2-value form the policy expects.

    π₀.₅ was trained with a two-finger Panda gripper, so its state carries one signed offset per
    finger, spanning 0 (closed) to 0.04 (open). Other grippers differ in joint count, travel and
    even sign — the Robotiq driving joint runs 0 to 0.7 and *increases* as it closes — so their
    opening is converted to a fraction of the joint's travel and re-expressed on the Panda scale.
    Feeding the raw values through instead puts the state far outside the range the policy has
    ever seen.
    """
    qpos = obs["robot0_gripper_qpos"]
    if len(qpos) == 2:
        return qpos

    sim = env.env.sim
    drive_joint = env.env.robots[0].gripper.joints[0]
    lo, hi = sim.model.jnt_range[sim.model.joint_name2id(drive_joint)]
    if hi <= lo:
        return np.array([qpos[0], -qpos[0]])

    fraction_closed = float(np.clip((qpos[0] - lo) / (hi - lo), 0.0, 1.0))
    half = (1.0 - fraction_closed) * PANDA_FINGER_MAX
    return np.array([half, -half])
