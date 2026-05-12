import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import zarr

from xskill.common.replay_buffer import ReplayBuffer

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


def write_json(path: Path, data):
    with open(path, "w") as file:
        json.dump(data, file)


HF_PREFIX = "https://huggingface.co/datasets/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot-format datasets into the folder-per-episode format used by xskill simulation training."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="One or more local dataset directories, Hugging Face dataset ids, or Hugging Face dataset URLs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/pick_red_mug"),
        help="Root directory that will contain the converted robot/ and human/ folders.",
    )
    parser.add_argument(
        "--output-split",
        default="human",
        help="Subdirectory name inside output-root.",
    )
    parser.add_argument(
        "--camera-key",
        default="observation.images.cam_azure_kinect_left.color",
        help="LeRobot video key to decode into PNG frames.",
    )
    parser.add_argument(
        "--state-key",
        default="observation.state",
        help="LeRobot parquet column to save as states.json.",
    )
    parser.add_argument(
        "--action-key",
        default="action",
        help="LeRobot parquet column to save as actions.json.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory used when dataset ids or URLs are provided.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional limit on the number of episodes converted from each dataset.",
    )
    parser.add_argument(
        "--skip-episodes-file",
        type=Path,
        default=None,
        help="JSON file mapping dataset names to episode indices to skip. "
             "Format: {\"dataset_id\": [1, 5, 10], \"other_dataset\": [3]}. "
             "Use \"*\" key for episodes to skip across all datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted episodes.",
    )
    parser.add_argument(
        "--write-episode-dirs",
        action="store_true",
        help=(
            "Also write legacy per-episode folders with PNGs, states.json, and actions.json. "
            "By default, --write-zarr writes only the zarr dataset."
        ),
    )
    parser.add_argument(
        "--write-zarr",
        action="store_true",
        help=(
            "Also write a sibling zarr replay buffer at <output-root>/<output-split>.zarr. "
            "Downstream datasets can read it directly without PNG folders."
        ),
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help="Optional explicit path for the output zarr dataset. Defaults to <output-root>/<output-split>.zarr.",
    )
    parser.add_argument(
        "--zarr-compressor",
        choices=["default", "disk"],
        default="disk",
        help="Compression profile used when writing the zarr dataset.",
    )
    return parser.parse_args()


def load_skip_mapping(file_path: Optional[Path]):
    """Load per-dataset episode skip lists from JSON file.
    
    Returns a dict mapping dataset names to skip sets, plus a global skip set under "*" key.
    Example file format:
    {
        "xiaochyVera/pick_red_mug_human_ss": [1, 5, 10],
        "xiaochyVera/pick_red_mug_human_1_ss": [3, 7],
        "*": [0, 19]
    }
    """
    skip_mapping = {}
    
    if file_path is not None:
        if not file_path.is_file():
            raise FileNotFoundError(f"Skip file not found: {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    skip_mapping[key] = set(value)
                else:
                    raise ValueError(f"Skip mapping values must be lists, got {type(value)} for key {key}")
        else:
            raise ValueError(f"Skip file must contain a JSON dict, got {type(data)}")
    
    return skip_mapping


def get_skip_set_for_dataset(skip_mapping: dict, dataset_name: str):
    """Get the union of dataset-specific and global skip sets."""
    skip_set = set()
    if "*" in skip_mapping:
        skip_set.update(skip_mapping["*"])
    if dataset_name in skip_mapping:
        skip_set.update(skip_mapping[dataset_name])
    return skip_set


def normalize_hf_source(source: str):
    if source.startswith(HF_PREFIX):
        source = source[len(HF_PREFIX) :]
    return source.strip("/")


def resolve_dataset_root(source: str, cache_dir: Optional[Path]):
    local_path = Path(source).expanduser()
    if local_path.exists():
        return local_path.resolve()

    repo_id = normalize_hf_source(source)
    if snapshot_download is None:
        raise ImportError(
            "huggingface-hub is required to download remote datasets. Install dependencies from requirement.txt first."
        )
    download_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=["meta/*", "data/*", "videos/*"],
    )
    return Path(download_path)


def prepare_output_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing directory {path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def decode_video_to_pngs(
    video_path: Path,
    output_dir: Optional[Path],
    frame_limit: int,
    collect_rgb_frames: bool = False,
):
    written_frames = 0
    rgb_frames = [] if collect_rgb_frames else None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file {video_path}.")

    try:
        while written_frames < frame_limit:
            success, frame = capture.read()
            if not success:
                break
            if output_dir is not None:
                out_path = output_dir / f"{written_frames}.png"
                if not cv2.imwrite(str(out_path), frame):
                    raise RuntimeError(f"Failed to write frame to {out_path}.")
            if collect_rgb_frames:
                rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            written_frames += 1
    finally:
        capture.release()

    if written_frames < frame_limit:
        raise RuntimeError(
            f"Decoded {written_frames} frames from {video_path}, but {frame_limit} were required."
        )

    if collect_rgb_frames:
        return np.stack(rgb_frames).astype(np.uint8)
    return None


def resolve_zarr_path(args):
    if args.zarr_path is not None:
        return args.zarr_path
    return args.output_root / f"{args.output_split}.zarr"


def prepare_zarr_buffer(zarr_path: Path, overwrite: bool):
    if zarr_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing zarr dataset {zarr_path}. Pass --overwrite to replace it."
            )
        if zarr_path.is_dir():
            shutil.rmtree(zarr_path)
        else:
            zarr_path.unlink()

    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    return ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(zarr_path)))


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


def build_pattern_path(
    dataset_root: Path,
    pattern: str,
    episode_index: int,
    chunk_size: int,
    video_key: Optional[str] = None,
):
    episode_chunk = episode_index // chunk_size
    relative_path = pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    return dataset_root / relative_path


def load_sequence_column(parquet_path: Path, column_name: str):
    if pq is None:
        raise ImportError(
            "pyarrow is required to read LeRobot parquet files. Install dependencies from requirement.txt first."
        )
    table = pq.read_table(parquet_path, columns=[column_name])
    return np.asarray(table[column_name].to_pylist(), dtype=np.float32)


def count_numeric_dirs(path: Path):
    if not path.is_dir():
        return 0
    return sum(1 for child in path.iterdir() if child.is_dir() and child.name.isdigit())


def count_output_episodes(split_path: Path):
    if split_path.is_dir():
        return count_numeric_dirs(split_path)

    zarr_path = Path(f"{split_path}.zarr")
    if zarr_path.exists():
        replay_buffer = ReplayBuffer.create_from_path(str(zarr_path))
        return replay_buffer.n_episodes

    return 0


def maybe_write_train_mask(output_root: Path):
    robot_count = count_output_episodes(output_root / "robot")
    human_count = count_output_episodes(output_root / "human")
    if robot_count == 0 or human_count == 0:
        return
    if robot_count != human_count:
        print(
            "Skipping train_mask.json because robot and human episode counts differ: "
            f"robot={robot_count}, human={human_count}."
        )
        return
    write_json(output_root / "train_mask.json", [True] * robot_count)


def main():
    args = parse_args()
    output_split_dir = args.output_root / args.output_split
    write_episode_dirs = args.write_episode_dirs or not args.write_zarr
    if write_episode_dirs:
        output_split_dir.mkdir(parents=True, exist_ok=True)

    zarr_buffer = None
    zarr_path = None
    if args.write_zarr:
        zarr_path = resolve_zarr_path(args)
        zarr_buffer = prepare_zarr_buffer(zarr_path, overwrite=args.overwrite)

    skip_mapping = load_skip_mapping(args.skip_episodes_file)

    global_episode_idx = 0
    source_index = []
    for dataset_source in args.dataset:
        dataset_root = resolve_dataset_root(dataset_source, cache_dir=args.cache_dir)
        dataset_name = normalize_hf_source(dataset_source)
        info = load_info(dataset_root)
        episode_indices = get_episode_indices(info)
        if args.limit_per_dataset is not None:
            episode_indices = episode_indices[: args.limit_per_dataset]
        
        skip_indices = get_skip_set_for_dataset(skip_mapping, dataset_name)
        if skip_indices:
            print(f"Skipping {len(skip_indices)} episodes from {dataset_name}: {sorted(skip_indices)}")
        
        episode_indices = [idx for idx in episode_indices if idx not in skip_indices]

        for episode_index in tqdm(
            episode_indices,
            desc=f"Converting {dataset_root.name}",
        ):
            parquet_path = build_pattern_path(
                dataset_root,
                info["data_path"],
                episode_index,
                chunk_size=int(info["chunks_size"]),
            )
            video_path = build_pattern_path(
                dataset_root,
                info["video_path"],
                episode_index,
                chunk_size=int(info["chunks_size"]),
                video_key=args.camera_key,
            )
            if not parquet_path.is_file():
                raise FileNotFoundError(f"Missing parquet episode file: {parquet_path}")
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing episode video file: {video_path}")

            states = load_sequence_column(parquet_path, args.state_key)
            actions = load_sequence_column(parquet_path, args.action_key)
            target_len = min(len(states), len(actions))
            if target_len <= 0:
                raise ValueError(
                    f"Episode {episode_index} from {dataset_root} does not contain any aligned state/action steps."
                )

            episode_zarr_data = None
            if zarr_buffer is not None:
                episode_zarr_data = {
                    "obs": states[:target_len].astype(np.float32),
                    "actions": actions[:target_len].astype(np.float32),
                }

            output_episode_dir = None
            if write_episode_dirs:
                output_episode_dir = output_split_dir / str(global_episode_idx)
                prepare_output_dir(output_episode_dir, overwrite=args.overwrite)

            rgb_frames = decode_video_to_pngs(
                video_path,
                output_episode_dir,
                frame_limit=target_len,
                collect_rgb_frames=zarr_buffer is not None,
            )
            if episode_zarr_data is not None:
                episode_zarr_data["camera_primary"] = rgb_frames[:target_len]

            if write_episode_dirs:
                write_json(output_episode_dir / "states.json", states[:target_len].tolist())
                write_json(output_episode_dir / "actions.json", actions[:target_len].tolist())
            if zarr_buffer is not None:
                zarr_buffer.add_episode(
                    episode_zarr_data,
                    compressors=args.zarr_compressor,
                )

            if write_episode_dirs:
                with open(output_episode_dir / "source_episode.json", "w") as file:
                    json.dump(
                        {
                            "source_dataset": normalize_hf_source(dataset_source),
                            "source_episode": int(episode_index),
                            "camera_key": args.camera_key,
                            "state_key": args.state_key,
                            "action_key": args.action_key,
                            "num_steps": int(target_len),
                        },
                        file,
                    )

            source_index.append(
                {
                    "output_episode": global_episode_idx,
                    "source_dataset": normalize_hf_source(dataset_source),
                    "source_episode": int(episode_index),
                    "num_steps": int(target_len),
                }
            )
            global_episode_idx += 1

    write_json(args.output_root / f"{args.output_split}_source_map.json", source_index)
    if zarr_buffer is not None:
        zarr_buffer.update_meta(
            {
                "camera_views": np.asarray(["."], dtype="<U1"),
                "camera_file_stems": np.asarray(["primary"], dtype="<U7"),
                "episode_indices": np.arange(len(source_index), dtype=np.int64),
            }
        )
        print(f"Wrote zarr dataset to {zarr_path}")
    maybe_write_train_mask(args.output_root)


if __name__ == "__main__":
    main()