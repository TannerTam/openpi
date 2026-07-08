"""Fast normalization-stats computation by reading state/action directly from parquet.

Unlike ``scripts/compute_norm_stats.py`` this skips video decoding entirely, so it is
dramatically faster (minutes instead of hours). It is the recommended way to (re)compute
norm stats for mixed-dataset configs (``data_config.repo_ids`` set), since the slow part of
the normal script is decoding videos that we don't even need for state/action statistics.

It reuses the config's ``repack_transforms`` + ``data_transforms.inputs`` so numeric effects
(e.g. gripper binarization) exactly match training. Image/video columns are not stored in
parquet, so they are filled with tiny dummy arrays (only ``state`` and ``actions`` matter).

Note on accuracy: training stacks ``action_horizon`` actions per frame via delta_timestamps,
while this reads one action per frame. State stats are therefore exact; action stats are a
near-identical approximation (the same action vectors, counted once instead of ~horizon
times across overlapping windows). This difference is negligible for normalization.

Usage:
    uv run python scripts/compute_norm_stats_fast.py --config-name <name> [--max-frames N]
"""

import pathlib

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as _transforms


def _resolve_repo_root(repo_id: str) -> pathlib.Path:
    try:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset  # lerobot >= 0.4
    except ImportError:
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # lerobot < 0.4
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    return pathlib.Path(meta.root)


def _parquet_files(root: pathlib.Path) -> list[pathlib.Path]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        # Fallback for non-standard layouts.
        files = sorted(p for p in root.rglob("*.parquet") if "meta" not in p.parts)
    return files


def _repack_image_sources(data_config: _config.DataConfig, available_columns: set[str]) -> list[str]:
    """Repack source keys that are not stored in parquet (i.e. videos/images)."""
    sources: set[str] = set()
    for t in data_config.repack_transforms.inputs:
        structure = getattr(t, "structure", None)
        if structure is None:
            continue
        flat = _transforms.flatten_dict(structure)
        sources.update(flat.values())
    return sorted(sources - available_columns)


def main(config_name: str, max_frames: int | None = None, seed: int = 0):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        raise ValueError("Fast norm stats only supports LeRobot (parquet) datasets, not RLDS.")

    config_repo_ids = getattr(data_config, "repo_ids", None)
    repo_ids = list(config_repo_ids) if config_repo_ids else [data_config.repo_id]
    if not repo_ids or repo_ids[0] is None:
        raise ValueError("Data config must have a repo_id (or repo_ids).")

    # Numeric transforms applied during training (repack + data transforms). We reuse them so
    # that effects like gripper binarization are reflected in the stats.
    transform = _transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])

    dummy_image = np.zeros((1, 1, 3), dtype=np.uint8)

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    rng = np.random.default_rng(seed)
    per_repo_cap = None if max_frames is None else max(2, max_frames // len(repo_ids))

    for repo_id in repo_ids:
        root = _resolve_repo_root(repo_id)
        files = _parquet_files(root)
        if not files:
            raise FileNotFoundError(f"No parquet files found under {root} for repo_id={repo_id!r}.")
        print(f"[{repo_id}] {len(files)} parquet files under {root}")

        import pandas as pd  # noqa: PLC0415

        seen = 0
        image_sources: list[str] | None = None
        for f in tqdm.tqdm(files, desc=f"Reading {repo_id}"):
            if per_repo_cap is not None and seen >= per_repo_cap:
                break
            df = pd.read_parquet(f)
            if image_sources is None:
                image_sources = _repack_image_sources(data_config, set(df.columns))

            records = df.to_dict("records")
            if per_repo_cap is not None:
                remaining = per_repo_cap - seen
                if len(records) > remaining:
                    idx = rng.choice(len(records), size=remaining, replace=False)
                    records = [records[i] for i in idx]

            batch_state, batch_actions = [], []
            for row in records:
                data = {k: np.asarray(v) for k, v in row.items()}
                for src in image_sources:
                    data[src] = dummy_image
                out = transform(data)
                batch_state.append(np.asarray(out["state"], dtype=np.float32))
                batch_actions.append(np.asarray(out["actions"], dtype=np.float32))

            if batch_state:
                stats["state"].update(np.stack(batch_state, axis=0))
                stats["actions"].update(np.stack(batch_actions, axis=0))
                seen += len(batch_state)

        print(f"[{repo_id}] processed {seen} frames")

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done.")


if __name__ == "__main__":
    tyro.cli(main)
