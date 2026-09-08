import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bridge_example() -> dict:
    """Creates a random input example for the Bridge (WidowX) policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BridgeInputs(transforms.DataTransformFn):
    """Converts Bridge (WidowX) dataset inputs into the format expected by the model.

    The Bridge dataset uses a single third-person camera (image_0) and an 8-dim
    end-effector state (x, y, z, roll, pitch, yaw, pad, gripper). There are no wrist
    cameras, so the wrist image slots are zero-padded.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                # Bridge has no wrist cameras, pad with zeros.
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # Mask the padded wrist images for pi0 (flow) models; pi0-FAST does not mask.
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BridgeOutputs(transforms.DataTransformFn):
    """Converts model outputs back into the Bridge dataset format (inference only)."""

    def __call__(self, data: dict) -> dict:
        # Bridge actions are 7-dim (dx, dy, dz, droll, dpitch, dyaw, gripper); the rest is padding.
        return {"actions": np.asarray(data["actions"][..., :7])}


def make_bridge_aug_widowx_example() -> dict:
    """Creates a random input example for the OXE-AugE Bridge WidowX-source policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class BridgeAugWidowXInputs(transforms.DataTransformFn):
    """Inputs transform for the OXE-AugE augmented Bridge dataset, using only the WidowX source.

    The augmented dataset renders many robot embodiments, but here we exclusively use the source
    (WidowX) camera ``observation.images.image`` and the source 7-dim end-effector state
    ``observation.state`` (x, y, z, roll, pitch, yaw, gripper). The augmented robot views/states
    are ignored.

    The dataset has no explicit ``action`` field, so actions are supplied as a sequence of future
    absolute states (via ``action_sequence_keys=("observation.state",)`` + ``delta_timestamps``).
    A downstream ``DeltaActions`` transform then converts them to pi0-style deltas relative to the
    current state (gripper kept absolute). Because the state column doubles as the action source, it
    arrives stacked as ``(action_horizon, 7)`` during training; we take the first frame as the
    proprioceptive state.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # NOTE: copy (np.array, not np.asarray). The state column doubles as the action source, so
        # "observation/state" and "actions" alias the same buffer. The downstream DeltaActions mutates
        # actions in place (actions[0] -= state), which would otherwise zero the (aliased) state.
        state = np.array(data["observation/state"], dtype=np.float32)
        # During training the state column is stacked over the action horizon (H, 7) because it also
        # serves as the action source. Use the current (first) frame as the proprioceptive state.
        if state.ndim == 2:
            state = state[0]

        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                # No wrist cameras in the source view; pad with zeros.
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            # Copy so DeltaActions' in-place edit doesn't mutate the dataset's underlying tensor.
            inputs["actions"] = np.array(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BridgeAugWidowXOutputs(transforms.DataTransformFn):
    """Slices the model output down to the 7-dim WidowX-source action (inference only).

    No AbsoluteActions transform is applied upstream, so these 7 dims are delta-EEF (xyz + rpy deltas
    relative to the current state) with an absolute gripper.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}
