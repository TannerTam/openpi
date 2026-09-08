"""Sweeps Google Robot base placements so its arm lands inside the Bridge camera view."""

import sys

import mediapy
import numpy as np

sys.path.insert(0, "/home/test/test12/tanner/openpi/examples/simpler_env")
from bridge_embodiment import bridge_image  # noqa: E402
from bridge_embodiment import make_bridge_task  # noqa: E402

np.set_printoptions(precision=3, suppress=True)

TASK = "widowx_spoon_on_towel"
# The WidowX starts with its TCP here; matching it should frame the arm the same way.
WIDOWX_TCP = np.array([-0.145, 0.034, 1.005])

candidates = [
    [0.42, 0.03],
    [0.29, -0.19],
    [0.35, -0.19],
    [0.29, -0.10],
    [0.35, -0.10],
    [0.45, -0.19],
]

for xy in candidates:
    env, opts = make_bridge_task(TASK, "google_robot")
    opts = {**opts, "init_xy": xy}
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": 0}, "robot_init_options": opts})
    tcp = np.asarray(obs["extra"]["tcp_pose"][:3], dtype=np.float64)
    spoon = [a for a in env.unwrapped._scene.get_all_actors() if "spoon" in a.name][0]
    spoon_p = np.asarray(spoon.pose.p, dtype=np.float64)
    qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float64)
    print(
        f"base={xy}  tcp={np.round(tcp, 3)}  d(tcp,widowx_tcp)={np.linalg.norm(tcp - WIDOWX_TCP):.3f}"
        f"  d(tcp,spoon)={np.linalg.norm(tcp - spoon_p):.3f}"
    )
    img = bridge_image(obs)
    mediapy.write_image(
        f"/home/test/test12/tanner/openpi/_logs/place_{xy[0]}_{xy[1]}.png", img
    )
    env.close()
