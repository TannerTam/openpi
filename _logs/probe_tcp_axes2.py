"""Determines each robot's gripper frame from motion rather than link origins.

Closing the gripper moves the two fingertips toward each other along the closing axis, which is a
reliable read of the gripper frame. Combined with the home orientation it shows whether main.py's
single tool offset is valid for both robots.
"""

import sys

import numpy as np
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import quat2mat

sys.path.insert(0, "/home/test/test12/tanner/openpi/examples/simpler_env")
from bridge_embodiment import make_bridge_task  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
TOOL = euler2mat(0.0, -np.pi / 2, 0.0)

TIPS = {
    "widowx": ("left_finger_link", "right_finger_link"),
    "google_robot": ("link_finger_tip_left", "link_finger_tip_right"),
}
# Sign of the gripper command that closes the fingers on each robot.
CLOSE_CMD = {"widowx": -1.0, "google_robot": 1.0}


def tip_positions(u, names):
    links = {l.name: l for l in u.agent.robot.get_links()}
    return np.stack([np.asarray(links[n].get_pose().p, dtype=np.float64) for n in names])


for robot in ("widowx", "google_robot"):
    env, opts = make_bridge_task("widowx_spoon_on_towel", robot)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": 0}, "robot_init_options": opts})
    u = env.unwrapped

    tcp_R_home = quat2mat(np.asarray(u.tcp.get_pose().q, dtype=np.float64))
    tips_open = tip_positions(u, TIPS[robot])

    for _ in range(12):
        act = np.zeros(7)
        act[-1] = CLOSE_CMD[robot]
        obs, *_ = env.step(act)
    tips_closed = tip_positions(u, TIPS[robot])

    # The fingertips approach each other along the closing axis.
    close_world = (tips_closed[0] - tips_closed[1]) - (tips_open[0] - tips_open[1])
    moved = np.linalg.norm(close_world)
    print(f"\n=== {robot} ===")
    print(f"fingertip separation change: {moved:.4f} m")
    if moved > 1e-4:
        close_world /= moved
        print("closing axis in world:    ", close_world)
        print("closing axis in TCP frame:", tcp_R_home.T @ close_world)

    base = np.asarray(obs["agent"]["base_pose"], dtype=np.float64)
    base_rot = quat2mat(base[3:])
    tcp_in_base = base_rot.T @ tcp_R_home
    print("home TCP axes in base frame -> x:", tcp_in_base[:, 0], "y:", tcp_in_base[:, 1], "z:", tcp_in_base[:, 2])
    print("  which base-frame direction does each TCP axis point? (base -z is 'down')")
    print("current tool offset gives rpy:", np.asarray(mat2euler(tcp_in_base @ TOOL)))
    # What offset would make this home pose canonical (rpy = 0)?
    print("offset that would zero this home pose (R_tcp_in_base^T):\n", tcp_in_base.T)
    env.close()
