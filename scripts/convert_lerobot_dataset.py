from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import zarr

from xskill.common.replay_buffer import ReplayBuffer

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


HF_PREFIX = "https://huggingface.co/datasets/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-split", type=str, default="human")
    parser.add_argument(
        "--camera-key",
        type=str,
        default="observation.images.cam_azure_kinect_left.color",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--skip-episodes-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_hf_source(source: str):
    if source.startswith(HF_PREFIX):
        source = source[len(HF_PREFIX) :]
    return source.strip("/")


def resolve_dataset_root(source: str, cache_dir: Path | None):
    local_path = Path(source).expanduser()
    if local_path.exists():
        return local_path.resolve()

    repo_id = normalize_hf_source(source)
    if snapshot_download is None:
        raise ImportError("huggingface-hub is required to download remote datasets.")
    download_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=["meta/*", "videos/*"],
    )
    return Path(download_path)


def prepare_replay_buffer(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing zarr dataset {path}.")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(path)))


def load_info(dataset_root: Path):
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        info_path = dataset_root / "meta.json"
    with open(info_path, "r") as file:
        return json.load(file)


def get_episode_indices(info: dict):
    train_split = info.get("splits", {}).get("train")
    if train_split:
        start_index, end_index = train_split.split(":", maxsplit=1)
        return list(range(int(start_index), int(end_index)))
    return list(range(int(info["total_episodes"])))


def build_pattern_path(dataset_root: Path, pattern: str, episode_index: int, chunk_size: int, video_key: str):
    episode_chunk = episode_index // chunk_size
    relative_path = pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    return dataset_root / relative_path


def load_skip_mapping(path: Path | None):
    if path is None:
        return {}
    with open(path, "r") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("skip_episodes_file must contain a JSON object.")
    return {key: set(value) for key, value in data.items()}


def get_skip_set(skip_mapping: dict, dataset_name: str):
    skip_set = set(skip_mapping.get("*", set()))
    skip_set.update(skip_mapping.get(dataset_name, set()))
    return skip_set


def decode_video(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file {video_path}.")

    frames = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"Decoded zero frames from {video_path}.")
    return np.stack(frames).astype(np.uint8)


def main():
    args = parse_args()
    output_path = args.output_root / f"{args.output_split}.zarr"
    replay_buffer = prepare_replay_buffer(output_path, overwrite=args.overwrite)
    skip_mapping = load_skip_mapping(args.skip_episodes_file)

    source_episode_indices = []
    for dataset_source in args.dataset:
        dataset_root = resolve_dataset_root(dataset_source, cache_dir=args.cache_dir)
        dataset_name = normalize_hf_source(dataset_source)
        info = load_info(dataset_root)
        episode_indices = get_episode_indices(info)
        if args.limit_per_dataset is not None:
            episode_indices = episode_indices[: args.limit_per_dataset]
        skip_set = get_skip_set(skip_mapping, dataset_name)
        episode_indices = [index for index in episode_indices if index not in skip_set]

        for episode_index in tqdm(episode_indices, desc=f"Converting {dataset_name}"):
            video_path = build_pattern_path(
                dataset_root,
                info["video_path"],
                episode_index,
                chunk_size=int(info["chunks_size"]),
                video_key=args.camera_key,
            )
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing episode video file: {video_path}")

            replay_buffer.add_episode(
                {"camera_cam1": decode_video(video_path)},
                compressors="disk",
            )
            source_episode_indices.append(int(episode_index))

    replay_buffer.update_meta(
        {
            "camera_views": np.asarray(["cam1"], dtype="<U4"),
            "camera_file_stems": np.asarray(["cam1"], dtype="<U4"),
            "episode_indices": np.asarray(source_episode_indices, dtype=np.int64),
        }
    )
    print(f"Wrote zarr dataset to {output_path}")


if __name__ == "__main__":
    main()
