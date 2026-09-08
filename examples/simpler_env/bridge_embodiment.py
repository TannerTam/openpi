"""Runs SimplerEnv's Bridge (WidowX) task suite with a different robot arm.

The Bridge environments hardcode ``robot="widowx"`` in their prepackaged config, so this module
rebuilds them with ``prepackaged_config=False`` and supplies everything that config would have set.
Three things do not carry over when the arm is swapped and are handled here:

  * ``scene_offset`` falls back to a per-robot default, so the Bridge table would be placed at the
    Google Robot's own scene offset and every object would drop to the floor.
  * The Bridge third-person camera is declared on the WidowX *agent* and mounted to its base link,
    so it disappears with the arm. The WidowX base sits at a fixed world pose, so the camera is
    re-registered here as a world-anchored camera at exactly the pose it would have had. That also
    keeps the real-image overlay (SimplerEnv's "visual matching") valid, since the overlay is
    aligned to that specific viewpoint.
  * The control mode and control rate are robot-specific; the Google Robot has no ``align2`` end
    effector controller and runs at 3 Hz rather than 5 Hz.

The arm is the only thing that changes: scene, objects, episode layouts, camera and success
criteria all stay exactly as they are in the WidowX evaluation.
"""

import gymnasium as gym
from mani_skill2_real2sim.agents.configs.widowx.defaults import WidowXDefaultConfig
from mani_skill2_real2sim.agents.configs.widowx.defaults import WidowXSinkCameraSetupConfig
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutCarrotOnPlateInScene
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutEggplantInBasketScene
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutSpoonOnTableClothInScene
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import (
    StackGreenCubeOnYellowCubeBakedTexInScene,
)
from mani_skill2_real2sim.sensors.camera import CameraConfig
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import look_at  # noqa: F401  (used by the agent configs)
import numpy as np
import sapien.core as sapien

# The Bridge table glb's own offset. Without this the per-robot default is used instead.
BRIDGE_SCENE_OFFSET = [-2.0634, -2.8313, 0.0]

GOOGLE_CONTROL_MODE = (
    "arm_pd_ee_delta_pose_align_interpolate_by_planner"
    "_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
)

# Where each Bridge task puts the WidowX base. The third-person camera hangs off this pose, so it
# also fixes where the re-registered world-anchored camera goes.
_WIDOWX_BASE = {
    "widowx": {"xy": [0.147, 0.028], "height": 0.870, "config": WidowXDefaultConfig},
    "widowx_sink_camera_setup": {"xy": [0.127, 0.06], "height": 0.85, "config": WidowXSinkCameraSetupConfig},
}

# OXE-AugE places the Google Robot on Bridge by translating the source trajectory 0.25 m forward and
# 0.05 m up into the target arm's comfortable workspace; their augmented dataset reproduces the
# WidowX's world path with ~0 tracking error across 170k frames. That offset is deliberately *not*
# used here. It buys nothing for this evaluation, because the state the policy sees lives in the
# robot's own base frame and therefore does not depend on where the base stands, while it does push
# the arm out of the Bridge camera: with their placement the resting gripper projects to pixel x=706
# on a 640 wide image, and the arm never entered frame in 44 of 96 episodes.
#
# What the base placement does control is where the arm appears on screen, so it is instead solved to
# put the Google Robot's resting gripper exactly where the WidowX's rests. The offset below is the
# Google Robot's TCP relative to its base at the home pose SimplerEnv resets it to.
GOOGLE_BASE_HEIGHT = 0.079
GOOGLE_TCP_OFFSET_XY = np.array([-0.439, 0.218])

# Resting TCP of the WidowX in each Bridge setup, which the Google Robot's base is solved against.
_WIDOWX_HOME_TCP_XY = {
    "widowx": np.array([-0.145, 0.034]),
    "widowx_sink_camera_setup": np.array([-0.123, 0.128]),
}


def google_base_pose(widowx_uid: str) -> tuple[list[float], float]:
    """Places the Google Robot so its resting gripper coincides with the WidowX's."""
    origin_xy = _WIDOWX_HOME_TCP_XY[widowx_uid] - GOOGLE_TCP_OFFSET_XY
    return [float(origin_xy[0]), float(origin_xy[1])], GOOGLE_BASE_HEIGHT


def bridge_camera_world_pose(widowx_uid: str) -> sapien.Pose:
    """Resolves the Bridge third-person camera into the world frame.

    The camera is specified relative to the WidowX ``base_link``, and that link is placed at a fixed
    pose for evaluation, so composing the two gives a pose that no longer depends on the arm.
    """
    base = _WIDOWX_BASE[widowx_uid]
    camera_cfg = base["config"]().cameras[0]
    base_pose = sapien.Pose([base["xy"][0], base["xy"][1], base["height"]], [0, 0, 0, 1])
    return base_pose * sapien.Pose(camera_cfg.p, camera_cfg.q)


def _bridge_camera_config(widowx_uid: str) -> CameraConfig:
    """Builds a world-anchored copy of the Bridge camera, matching its resolution and intrinsics."""
    reference = _WIDOWX_BASE[widowx_uid]["config"]().cameras[0]
    pose = bridge_camera_world_pose(widowx_uid)
    return CameraConfig(
        uid="3rd_view_camera",
        p=pose.p,
        q=pose.q,
        width=reference.width,
        height=reference.height,
        actor_uid=None,  # world-anchored, so it survives swapping the arm
        intrinsic=reference.intrinsic,
    )


def _with_bridge_camera(base_cls, widowx_uid: str):
    """Subclasses a Bridge env so the WidowX viewpoint is available whatever arm is loaded."""

    class _BridgeCameraEnv(base_cls):
        def _register_cameras(self):
            cameras = super()._register_cameras()
            if not isinstance(cameras, list):
                cameras = [cameras]
            return [*cameras, _bridge_camera_config(widowx_uid)]

    _BridgeCameraEnv.__name__ = f"{base_cls.__name__}AnyEmbodiment"
    return _BridgeCameraEnv


# task name -> (env class, gym id, widowx uid it is normally evaluated with, scene, overlay image)
BRIDGE_TASKS = {
    "widowx_spoon_on_towel": (
        PutSpoonOnTableClothInScene,
        "PutSpoonOnTableClothInSceneAnyEmbodiment-v0",
        "widowx",
        "bridge_table_1_v1",
        "bridge_real_eval_1.png",
    ),
    "widowx_carrot_on_plate": (
        PutCarrotOnPlateInScene,
        "PutCarrotOnPlateInSceneAnyEmbodiment-v0",
        "widowx",
        "bridge_table_1_v1",
        "bridge_real_eval_1.png",
    ),
    "widowx_stack_cube": (
        StackGreenCubeOnYellowCubeBakedTexInScene,
        "StackGreenCubeOnYellowCubeBakedTexInSceneAnyEmbodiment-v0",
        "widowx",
        "bridge_table_1_v1",
        "bridge_real_eval_1.png",
    ),
    "widowx_put_eggplant_in_basket": (
        PutEggplantInBasketScene,
        "PutEggplantInBasketSceneAnyEmbodiment-v0",
        "widowx_sink_camera_setup",
        "bridge_table_1_v2",
        "bridge_sink.png",
    ),
}

_MAX_EPISODE_STEPS = {"widowx_put_eggplant_in_basket": 120}

for _task, (_cls, _env_id, _uid, _scene, _overlay) in BRIDGE_TASKS.items():
    register_env(_env_id, max_episode_steps=_MAX_EPISODE_STEPS.get(_task, 60))(
        _with_bridge_camera(_cls, _uid)
    )


def _overlay_path(filename: str) -> str:
    from mani_skill2_real2sim import ASSET_DIR

    return str(ASSET_DIR / "real_inpainting" / filename)


def make_bridge_task(task: str, robot: str, **kwargs):
    """Creates a Bridge task running either the stock WidowX or the Google Robot.

    Returns the env plus the reset options that place the arm, which differ per embodiment.
    """
    if task not in BRIDGE_TASKS:
        raise ValueError(f"{task!r} is not a Bridge task; expected one of {sorted(BRIDGE_TASKS)}")
    _cls, env_id, widowx_uid, scene_name, overlay = BRIDGE_TASKS[task]

    env_kwargs = dict(
        obs_mode="rgbd",
        prepackaged_config=False,
        scene_name=scene_name,
        scene_offset=BRIDGE_SCENE_OFFSET,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=_overlay_path(overlay),
        rgb_overlay_cameras=["3rd_view_camera"],
    )

    if robot == "widowx":
        env_kwargs.update(
            robot=widowx_uid,
            control_freq=5,
            sim_freq=500,
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        )
        base = _WIDOWX_BASE[widowx_uid]
        robot_init_options = {"init_xy": base["xy"], "init_height": base["height"], "init_rot_quat": [0, 0, 0, 1]}
    elif robot == "google_robot":
        env_kwargs.update(
            robot="google_robot_static",
            control_freq=3,
            sim_freq=513,
            control_mode=GOOGLE_CONTROL_MODE,
        )
        base_xy, base_height = google_base_pose(widowx_uid)
        robot_init_options = {
            "init_xy": base_xy,
            "init_height": base_height,
            "init_rot_quat": [0, 0, 0, 1],
        }
    else:
        raise ValueError(f"unsupported robot {robot!r}")

    env_kwargs.update(kwargs)
    env = gym.make(env_id, **env_kwargs)
    return env, robot_init_options


def bridge_image(obs: dict) -> np.ndarray:
    """Reads the Bridge third-person view, which is the camera the policy was trained on."""
    return obs["image"]["3rd_view_camera"]["rgb"]
