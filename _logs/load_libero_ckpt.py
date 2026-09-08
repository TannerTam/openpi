"""Loads the LeRobot-format pi05 LIBERO checkpoint and runs one forward pass on dummy inputs."""

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy

CKPT = "/home/test/test12/tanner/checkpoints/openpi/pi05_libero_finetuned_v044"
# The processor pipeline wants google/paligemma-3b-pt-224's tokenizer, which this box cannot reach.
# This directory pairs the cached tokenizer_config.json with the SentencePiece model openpi already
# downloaded to checkpoints/big_vision/paligemma_tokenizer.model.
TOKENIZER = "/home/test/test12/tanner/checkpoints/paligemma-3b-pt-224-tokenizer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI05Policy.from_pretrained(CKPT).to(device).eval()
print("loaded:", type(policy).__name__)
cfg = policy.config
print("  inputs :", {k: v.shape for k, v in cfg.input_features.items()})
print("  outputs:", {k: v.shape for k, v in cfg.output_features.items()})
print("  chunk_size:", cfg.chunk_size, " n_action_steps:", cfg.n_action_steps)
print("  normalization:", cfg.normalization_mapping)

preprocess, postprocess = make_pre_post_processors(
    cfg,
    CKPT,
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "tokenizer_processor": {"tokenizer_name": TOKENIZER},
    },
)
print("preprocessor/postprocessor built")

rng = np.random.default_rng(0)
frame = {
    "observation.images.image": torch.from_numpy(
        rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)
    ).float() / 255.0,
    "observation.images.image2": torch.from_numpy(
        rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)
    ).float() / 255.0,
    "observation.state": torch.from_numpy(rng.normal(size=8).astype(np.float32)),
    "task": "pick up the black bowl and place it on the plate",
}

batch = preprocess(frame)
with torch.inference_mode():
    action = policy.select_action(batch)
    action = postprocess(action)
print("action:", np.asarray(action.detach().float().cpu()).shape, np.round(np.asarray(action.detach().float().cpu()).ravel(), 4))
