"""Checks the OXE-AugE placement in SimplerEnv: reach, camera framing, and collisions."""

import sys

import mediapy
import numpy as np
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import quat2mat

sys.path.insert(0, "/home/test/test12/tanner/openpi/examples/simpler_env")
from bridge_embodiment import bridge_image, google_base_pose, make_bridge_task  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
TOOL = euler2mat(0.0, -np.pi / 2, 0.0)


def policy_state(env, obs):
    tcp = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float64)
    base = np.asarray(obs["agent"]["base_pose"], dtype=np.float64)
    base_rot = quat2mat(base[3:])
    anchor = np.array([base[0], base[1], env.unwrapped.scene_table_height])
    pos = base_rot.T @ (tcp[:3] - anchor)
    rpy = mat2euler(base_rot.T @ quat2mat(tcp[3:]) @ TOOL)
    return np.concatenate([pos, rpy]), tcp[:3]


print("solved placement:")
for uid in ("widowx", "widowx_sink_camera_setup"):
    print(f"  {uid}: base_xy={google_base_pose(uid)[0]} height={google_base_pose(uid)[1]:.4f}")

for task in ("widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"):
    for robot in ("widowx", "google_robot"):
        env, opts = make_bridge_task(task, robot)
        obs, _ = env.reset(
            options={"obj_init_options": {"episode_id": 0}, "robot_init_options": opts}
        )
        st, tcp_world = policy_state(env, obs)
        print(f"\n{task} / {robot}")
        print(f"  base={np.round(np.asarray(obs['agent']['base_pose'][:3]), 3)}  tcp_world={np.round(tcp_world, 3)}")
        print(f"  policy state xyz={np.round(st[:3], 4)}  rpy={np.round(st[3:], 4)}")
        for a in env.unwrapped._scene.get_all_actors():
            if a.name == "arena":
                continue
            p = np.asarray(a.pose.p, dtype=np.float64)
            print(f"    {a.name!r:34} z={p[2]:.3f}  dist_from_tcp={np.linalg.norm(p - tcp_world):.3f}")
        out = f"/home/test/test12/tanner/openpi/_logs/auge_{task}_{robot}.png"
        mediapy.write_image(out, bridge_image(obs))
        print("  wrote", out)
        env.close()
