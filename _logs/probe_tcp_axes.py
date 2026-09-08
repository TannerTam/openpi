"""Checks which TCP axis is the gripper approach direction on each robot.

main.py assumes SimplerEnv uses the TCP z axis as the approach direction for both the WidowX and the
Google Robot. If that is wrong for one of them, the state's roll/pitch/yaw are rotated by 90 degrees
and the policy sees a pose that has nothing to do with what the arm is physically doing.
"""

import sys

import numpy as np
from transforms3d.quaternions import quat2mat

sys.path.insert(0, "/home/test/test12/tanner/openpi/examples/simpler_env")
from bridge_embodiment import make_bridge_task  # noqa: E402

np.set_printoptions(precision=3, suppress=True)


def axis_name(v):
    """Names the axis a unit vector is closest to."""
    labels = ["+x", "+y", "+z"]
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + labels[i][1]


for robot in ("widowx", "google_robot"):
    env, opts = make_bridge_task("widowx_spoon_on_towel", robot)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": 0}, "robot_init_options": opts})
    u = env.unwrapped

    tcp_pose = u.tcp.get_pose()
    tcp_p = np.asarray(tcp_pose.p, dtype=np.float64)
    tcp_R = quat2mat(np.asarray(tcp_pose.q, dtype=np.float64))

    links = {l.name: l for l in u.agent.robot.get_links()}
    finger_links = [n for n in links if "finger" in n.lower()]
    print(f"\n=== {robot} ({u.robot_uid}) ===")
    print("tcp link:", u.agent.config.ee_link_name, " world p:", tcp_p)
    print("finger links:", finger_links)

    # Approach direction: from the wrist/palm toward the fingertips, i.e. tcp -> mean fingertip.
    tips = np.stack([np.asarray(links[n].get_pose().p, dtype=np.float64) for n in finger_links])
    approach_world = tips.mean(0) - tcp_p
    n = np.linalg.norm(approach_world)
    print(f"tcp -> fingertip midpoint (world): {approach_world}  |d|={n:.4f}")
    if n > 1e-6:
        approach_world /= n
        in_tcp = tcp_R.T @ approach_world
        print(f"  that direction in the TCP frame: {in_tcp}  -> approach axis is {axis_name(in_tcp)}")

    # Closing direction: between the two fingers.
    if len(finger_links) == 2:
        close_world = tips[1] - tips[0]
        close_world /= np.linalg.norm(close_world)
        print(f"  finger closing axis in TCP frame: {tcp_R.T @ close_world} -> {axis_name(tcp_R.T @ close_world)}")

    print("  tcp axes in world:  x:", tcp_R[:, 0], " y:", tcp_R[:, 1], " z:", tcp_R[:, 2])
    env.close()
