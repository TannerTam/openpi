"""Picks the Google Robot base placement that puts its start state inside the training distribution.

For each candidate base position, solves IK to the WidowX's resting gripper pose and reports the
resulting policy state plus whether the gripper is inside the Bridge camera frame.
"""

import sys

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat

sys.path.insert(0, "/home/test/test12/tanner/openpi/examples/simpler_env")
from bridge_embodiment import bridge_camera_world_pose, make_bridge_task  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
M_WIDOWX = euler2mat(0.0, -np.pi / 2, 0.0)
M_GOOGLE = euler2mat(0.0, np.pi, 0.0)
K = np.array([[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]])
W, H = 640, 480
cam = bridge_camera_world_pose("widowx")
cam_p, cam_R = np.asarray(cam.p), quat2mat(np.asarray(cam.q))


def in_frame(p_world):
    pc = cam_R.T @ (p_world - cam_p)
    cv = np.array([-pc[1], -pc[2], pc[0]])
    if cv[2] <= 1e-6:
        return False, None
    uv = (K @ (cv / cv[2]))[:2]
    return bool(0 <= uv[0] < W and 0 <= uv[1] < H), uv


def angle_between(R1, R2):
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


env_w, opts_w = make_bridge_task("widowx_spoon_on_towel", "widowx")
obs_w, _ = env_w.reset(options={"obj_init_options": {"episode_id": 0}, "robot_init_options": opts_w})
w = np.asarray(obs_w["extra"]["tcp_pose"], dtype=np.float64)
env_w.close()
target_R_world = quat2mat(w[3:]) @ M_WIDOWX @ np.linalg.inv(M_GOOGLE)
print("widowx target tcp world p:", np.round(w[:3], 4))
print("train xyz q01: [ 0.168 -0.178 -0.061]   q99: [0.458 0.242 0.204]\n")

CANDIDATES = [
    ("widowx base (state matches exactly)", [0.147, 0.028]),
    ("+0.10 back", [0.247, 0.028]),
    ("+0.15 back", [0.297, 0.028]),
    ("OXE-AugE +0.25 back", [0.397, 0.028]),
]

env, opts = make_bridge_task("widowx_spoon_on_towel", "google_robot")
u = env.unwrapped
ctrl = u.agent.controller.controllers["arm"]
robot = u.agent.robot
rng = np.random.default_rng(0)
limits = np.asarray([j.get_limits()[0] for j in robot.get_active_joints()], dtype=np.float64)

for label, xy in CANDIDATES:
    obs, _ = env.reset(
        options={"obj_init_options": {"episode_id": 0}, "robot_init_options": {**opts, "init_xy": xy}}
    )
    g_base = np.asarray(obs["agent"]["base_pose"], dtype=np.float64)
    base_rot = quat2mat(g_base[3:])
    target_pose = sapien.Pose(
        base_rot.T @ (w[:3] - g_base[:3]), mat2quat(base_rot.T @ target_R_world)
    )
    home = np.asarray(robot.get_qpos(), dtype=np.float64)

    best = None
    for trial in range(300):
        q_init = home.copy()
        if trial > 0:
            q_init[:7] = rng.uniform(limits[:7, 0], limits[:7, 1])
        robot.set_qpos(q_init)
        sol = ctrl.compute_ik(target_pose, max_iterations=200)
        if sol is None:
            continue
        q = home.copy()
        q[ctrl.joint_indices] = sol
        # SAPIEN's IK ignores joint limits, and an out-of-limit start pose gets pulled back by the
        # physics as soon as stepping begins, so only keep feasible solutions.
        arm = q[:7]
        if np.any(arm < limits[:7, 0] - 1e-6) or np.any(arm > limits[:7, 1] + 1e-6):
            continue
        robot.set_qpos(q)
        pose = u.tcp.get_pose()
        pe = np.linalg.norm(np.asarray(pose.p) - w[:3])
        ang = angle_between(quat2mat(np.asarray(pose.q)), target_R_world)
        if best is None or (pe + np.radians(ang)) < best[0]:
            best = (pe + np.radians(ang), pe, ang, q.copy())

    if best is None:
        print(f"{label:38} base={xy}  -> IK FAILED")
        continue
    _, pe, ang, q = best
    obs2, _ = env.reset(
        options={"obj_init_options": {"episode_id": 0}, "robot_init_options": {**opts, "init_xy": xy, "qpos": q}}
    )
    tcp = np.asarray(obs2["extra"]["tcp_pose"], dtype=np.float64)
    b = np.asarray(obs2["agent"]["base_pose"], dtype=np.float64)
    br = quat2mat(b[3:])
    st_xyz = br.T @ (tcp[:3] - np.array([b[0], b[1], u.scene_table_height]))
    st_rpy = mat2euler(br.T @ quat2mat(tcp[3:]) @ M_GOOGLE)
    vis, uv = in_frame(tcp[:3])
    print(
        f"{label:38} base={xy}  ik_pos_err={pe:.4f} ang={ang:5.1f}deg  "
        f"state_xyz={np.round(st_xyz, 4)} rpy={np.round(st_rpy, 3)}  "
        f"gripper_pixel={np.round(uv, 0) if uv is not None else None} {'IN' if vis else 'OUT'}"
    )
    print("      qpos:", [round(float(v), 6) for v in q])

env.close()
