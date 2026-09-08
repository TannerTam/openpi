"""Extra LIBERO robot embodiments (IIWA, UR5e), ported from Adapt3R.

LIBERO prefixes robot names with ``Mounted``/``OnTheGround`` depending on the arena, so each
embodiment needs both variants. Importing this module registers them; robosuite auto-registers
``ManipulatorModel`` subclasses by class name via its metaclass.

Both embodiments override the stock gripper with ``PandaGripper`` so that ``robot0_gripper_qpos``
stays 2-dimensional, keeping the policy's 8-dim state layout unchanged.

Each class spells out its properties rather than sharing a base class: ``RobotModel.__init__``
assigns instance attributes such as ``self.mount``, which would silently shadow class-level
configuration of the same name.

Reference: https://github.com/pairlab/Adapt3R (adapt3r/envs/libero/robots)
"""

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm
from robosuite.utils.mjcf_utils import xml_path_completion


class MountedIIWA(ManipulatorModel):
    """KUKA LBR IIWA mounted on a table-height pedestal."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/iiwa/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class OnTheGroundIIWA(ManipulatorModel):
    """KUKA LBR IIWA standing directly on the floor."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/iiwa/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_iiwa"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.650, 0.000, -1.890, 0.000, 0.600, 0.000])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class MountedUR5e(ManipulatorModel):
    """Universal Robots UR5e mounted on a table-height pedestal."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_ur5e"

    @property
    def init_qpos(self):
        return np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -1.991])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class OnTheGroundUR5e(ManipulatorModel):
    """Universal Robots UR5e standing directly on the floor."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_ur5e"

    @property
    def init_qpos(self):
        return np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -1.991])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class MountedSawyer(ManipulatorModel):
    """Rethink Sawyer mounted on a table-height pedestal."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class OnTheGroundSawyer(ManipulatorModel):
    """Rethink Sawyer standing directly on the floor."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, -1.57])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class MountedKinova3(ManipulatorModel):
    """Kinova Gen3 mounted on a table-height pedestal."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/kinova3/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_kinova3"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.650, 0.000, 1.890, 0.000, 0.600, -np.pi / 2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class OnTheGroundKinova3(ManipulatorModel):
    """Kinova Gen3 standing directly on the floor."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/kinova3/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_kinova3"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.650, 0.000, 1.890, 0.000, 0.600, -np.pi / 2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


ROBOT_CLASS_MAPPING.update(
    {
        "MountedIIWA": SingleArm,
        "OnTheGroundIIWA": SingleArm,
        "MountedUR5e": SingleArm,
        "OnTheGroundUR5e": SingleArm,
        "MountedSawyer": SingleArm,
        "OnTheGroundSawyer": SingleArm,
        "MountedKinova3": SingleArm,
        "OnTheGroundKinova3": SingleArm,
    }
)
