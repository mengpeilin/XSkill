from __future__ import annotations

import argparse
import binascii
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_lerobot_dataset import (
    build_pattern_path,
    decode_video as decode_human_video,
    get_episode_indices,
    load_info,
    normalize_hf_source,
    prepare_replay_buffer as prepare_human_replay_buffer,
    resolve_dataset_root,
)
from convert_robot_video_dataset import (
    decode_video as decode_robot_video,
    prepare_replay_buffer as prepare_robot_replay_buffer,
)
from xskill.common.replay_buffer import ReplayBuffer

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_HUMAN_COUNTS = [40, 70, 100, 200]
DEFAULT_ROBOT_COUNTS = [5, 40]
MAX_HUMAN_EPISODES = 200
MAX_ROBOT_EPISODES = 40
EXPECTED_FPS = 15
DEFAULT_HUMAN_CAMERA_KEY = "observation.images.cam_azure_kinect_left.color"
DEFAULT_ROBOT_CAMERA_SPECS = (
    ("cam0", "observation.images.cam_azure_kinect_front.color"),
    ("cam1", "observation.images.cam_azure_kinect_left.color"),
    ("wrist_cam", "observation.images.cam_wrist"),
)
DEFAULT_ROBOT_OBS_KEY = "observation.right_eef_pose"
DEFAULT_ROBOT_ACTION_KEY = "action.right_eef_pose"
ROBOT_ALLOW_PATTERNS = ["meta/*", "videos/*", "data/*"]

TASK_SPECS = {
    "pick_place_red_mug": {
        "human_total": MAX_HUMAN_EPISODES,
        "robot_total": MAX_ROBOT_EPISODES,
        "human_sources": [
            {"dataset": "xiaochyVera/pick_red_mug_human_ss"},
            {"dataset": "xiaochyVera/pick_red_mug_human_1_ss"},
            {"dataset": "xiaochyVera/pick_red_mug_human_2_ss"},
            {"dataset": "xiaochyVera/pick_red_mug_human_3_ss"},
            {"dataset": "xiaochyVera/pick_red_mug_human_4_ss"},
        ],
        "robot_sources": [
            {"dataset": "xiaochyVera/pick_red_mug_realrobot_3_z45_reorder_binary_ss_427"},
            {"dataset": "xiaochyVera/pick_red_mug_realrobot_4_z45_reorder_binary_ss_427"},
        ],
    },
    "stack_bowls": {
        "human_total": MAX_HUMAN_EPISODES,
        "robot_total": MAX_ROBOT_EPISODES,
        "human_sources": [
            {"dataset": "xiaochyVera/stack_bowls_human_ss"},
        ],
        "robot_sources": [
            {"dataset": "xiaochyVera/stack_bowls_realrobot_z45_reorder_binary_ss"},
        ],
    },
}

PENDING_TASKS = {
    "pick_place_toys": "human and robot teleop dataset sources are still TBD",
    "insert_donut": "human teleop dataset sources are still TBD",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--human-counts", nargs="+", type=int, default=None)
    parser.add_argument("--robot-counts", nargs="+", type=int, default=None)
    parser.add_argument("--skip-zarr", action="store_true")
    parser.add_argument("--skip-masks", action="store_true")
    parser.add_argument("--overwrite-zarr", action="store_true")
    parser.add_argument("--skip-source-validation", action="store_true")
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[1]


def default_output_root():
    return repo_root() / "zarr" / "realworld_teleop_15fps"


def canonicalize_hf_source(source):
    normalized = normalize_hf_source(source)
    if "/tree/" in normalized:
        normalized = normalized.split("/tree/", maxsplit=1)[0]
    return normalized.strip("/")


def selected_tasks(task_args):
    if task_args == ["all"]:
        return list(TASK_SPECS.keys())

    unknown_tasks = [task for task in task_args if task not in TASK_SPECS]
    if unknown_tasks:
        pending = {task: PENDING_TASKS[task] for task in unknown_tasks if task in PENDING_TASKS}
        if pending:
            raise ValueError(f"Task specs not filled in yet: {pending}")
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    return task_args


def stable_seed(task_name, split_name, base_seed):
    salt = f"{task_name}:{split_name}".encode("utf-8")
    return int(base_seed + binascii.crc32(salt))


def unique_counts(counts):
    return sorted({int(count) for count in counts})


def task_relative_path(task_dir, path):
    return str(path.relative_to(task_dir))


def validate_dataset_fps(task_name, dataset_name, info, split_name):
    fps = info.get("fps")
    if fps is None:
        raise ValueError(f"[{task_name}] {split_name} source {dataset_name} is missing fps metadata.")
    if int(fps) != EXPECTED_FPS:
        raise ValueError(
            f"[{task_name}] {split_name} source {dataset_name} has fps={fps}, expected {EXPECTED_FPS}."
        )


def build_sources(task_name, source_specs, cache_dir, split_name, allow_patterns):
    prepared_sources = []
    for source_spec in source_specs:
        dataset_name = canonicalize_hf_source(source_spec["dataset"])
        dataset_root = resolve_dataset_root(
            dataset_name,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
        )
        info = load_info(dataset_root)
        validate_dataset_fps(task_name, dataset_name, info, split_name)
        episode_indices = get_episode_indices(info)
        skip_episodes = sorted(set(source_spec.get("skip_episodes", [])))
        skip_set = set(skip_episodes)
        keep_episodes = [episode for episode in episode_indices if episode not in skip_set]
        prepared_sources.append(
            {
                "dataset": dataset_name,
                "dataset_root": dataset_root,
                "info": info,
                "skip_episodes": skip_episodes,
                "episode_indices": keep_episodes,
            }
        )
        print(
            f"[{task_name}] {split_name} source {dataset_name}: kept {len(keep_episodes)} / {len(episode_indices)} episodes"
        )
    return prepared_sources


def load_existing_episode_count(zarr_path):
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
    return int(replay_buffer.n_episodes)


def convert_human_zarr(task_name, prepared_sources, output_path, overwrite):
    if output_path.exists() and not overwrite:
        existing_total = load_existing_episode_count(output_path)
        print(f"[{task_name}] keeping existing {output_path} with {existing_total} episodes")
        return existing_total

    replay_buffer = prepare_human_replay_buffer(output_path, overwrite=overwrite)
    manifest_entries = []

    for source in prepared_sources:
        if len(manifest_entries) >= MAX_HUMAN_EPISODES:
            break

        dataset_name = source["dataset"]
        info = source["info"]
        for episode_index in tqdm(source["episode_indices"], desc=f"[{task_name}] {dataset_name}"):
            if len(manifest_entries) >= MAX_HUMAN_EPISODES:
                break

            video_path = build_pattern_path(
                source["dataset_root"],
                info["video_path"],
                episode_index,
                chunk_size=int(info["chunks_size"]),
                video_key=DEFAULT_HUMAN_CAMERA_KEY,
            )
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing human demo video {video_path}")

            replay_buffer.add_episode(
                {"camera_cam1": decode_human_video(video_path)},
                compressors="disk",
            )
            manifest_entries.append({"dataset": dataset_name, "source_episode": int(episode_index)})

    replay_buffer.update_meta(
        {
            "camera_views": np.asarray(["cam1"], dtype="<U4"),
            "camera_file_stems": np.asarray(["cam1"], dtype="<U4"),
            "episode_indices": np.asarray(
                [entry["source_episode"] for entry in manifest_entries],
                dtype=np.int64,
            ),
        }
    )
    print(f"[{task_name}] wrote {output_path} with {len(manifest_entries)} episodes")
    return len(manifest_entries)


def load_parquet_matrix(parquet_path: Path, column_name: str):
    table = pq.read_table(parquet_path, columns=[column_name])
    values = table[column_name].to_pylist()
    if not values:
        raise ValueError(f"{parquet_path} column {column_name} is empty.")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"Expected {column_name} in {parquet_path} to decode to a 2D array, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{column_name} in {parquet_path} contains non-finite values.")
    return array


def normalize_vectors(vectors: np.ndarray, key: str):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)):
        raise ValueError(f"{key} contains non-finite rotation values.")
    if np.any(norms < 1e-8):
        raise ValueError(f"{key} contains near-zero rotation vectors that cannot form a valid 6D rotation.")
    return vectors / norms


def rot6d_to_rotvec(rot6d: np.ndarray, key: str):
    if rot6d.ndim != 2 or rot6d.shape[1] != 6:
        raise ValueError(f"Expected {key} to have shape (T, 6), got {rot6d.shape}.")

    basis_x = normalize_vectors(rot6d[:, 0:3], key)
    projected_y = rot6d[:, 3:6] - np.sum(basis_x * rot6d[:, 3:6], axis=1, keepdims=True) * basis_x
    basis_y = normalize_vectors(projected_y, key)
    basis_z = np.cross(basis_x, basis_y)
    rotation_matrices = np.stack([basis_x, basis_y, basis_z], axis=-1)
    return Rotation.from_matrix(rotation_matrices).as_rotvec().astype(np.float32)


def build_pose_width_array_from_rot6d(pose_array: np.ndarray, key: str):
    if pose_array.ndim != 2 or pose_array.shape[1] != 10:
        raise ValueError(f"Expected {key} to have shape (T, 10), got {pose_array.shape}.")
    position = pose_array[:, 6:9]
    rotvec = rot6d_to_rotvec(pose_array[:, 0:6], key)
    gripper = pose_array[:, 9:10]
    return np.concatenate([position, rotvec, gripper], axis=1).astype(np.float32)


def build_robot_episode_arrays(parquet_path: Path, obs_key: str, action_key: str):
    obs_pose = load_parquet_matrix(parquet_path, obs_key)
    action_pose = load_parquet_matrix(parquet_path, action_key)
    target_len = min(len(obs_pose), len(action_pose))
    if target_len <= 0:
        raise ValueError(f"{parquet_path} does not contain aligned robot observations and actions.")
    obs = build_pose_width_array_from_rot6d(obs_pose[:target_len], obs_key)
    actions = build_pose_width_array_from_rot6d(action_pose[:target_len], action_key)
    return obs, actions


def convert_robot_zarr(task_name, prepared_sources, output_path, overwrite):
    if output_path.exists() and not overwrite:
        existing_total = load_existing_episode_count(output_path)
        print(f"[{task_name}] keeping existing {output_path} with {existing_total} episodes")
        return existing_total

    replay_buffer = prepare_robot_replay_buffer(output_path, overwrite=overwrite)
    manifest_entries = []
    camera_views = [view_name for view_name, _ in DEFAULT_ROBOT_CAMERA_SPECS]
    max_view_len = max(len(view_name) for view_name in camera_views)

    for source in prepared_sources:
        if len(manifest_entries) >= MAX_ROBOT_EPISODES:
            break

        dataset_name = source["dataset"]
        info = source["info"]
        for episode_index in tqdm(source["episode_indices"], desc=f"[{task_name}] robot {dataset_name}"):
            if len(manifest_entries) >= MAX_ROBOT_EPISODES:
                break

            parquet_path = build_pattern_path(
                source["dataset_root"],
                info["data_path"],
                episode_index,
                chunk_size=int(info["chunks_size"]),
                video_key="unused",
            )
            if not parquet_path.is_file():
                raise FileNotFoundError(f"Missing robot episode parquet {parquet_path}")

            obs, actions = build_robot_episode_arrays(
                parquet_path,
                obs_key=DEFAULT_ROBOT_OBS_KEY,
                action_key=DEFAULT_ROBOT_ACTION_KEY,
            )

            episode_payload = {
                "obs": obs,
                "actions": actions,
            }
            for view_name, video_key in DEFAULT_ROBOT_CAMERA_SPECS:
                video_path = build_pattern_path(
                    source["dataset_root"],
                    info["video_path"],
                    episode_index,
                    chunk_size=int(info["chunks_size"]),
                    video_key=video_key,
                )
                if not video_path.is_file():
                    raise FileNotFoundError(f"Missing robot demo video {video_path}")
                episode_payload[f"camera_{view_name}"] = decode_robot_video(video_path, len(obs))

            replay_buffer.add_episode(episode_payload, compressors="disk")
            manifest_entries.append({"dataset": dataset_name, "source_episode": int(episode_index)})

    replay_buffer.update_meta(
        {
            "camera_views": np.asarray(camera_views, dtype=f"<U{max_view_len}"),
            "camera_file_stems": np.asarray(camera_views, dtype=f"<U{max_view_len}"),
            "episode_indices": np.asarray(
                [entry["source_episode"] for entry in manifest_entries],
                dtype=np.int64,
            ),
        }
    )
    print(f"[{task_name}] wrote {output_path} with {len(manifest_entries)} episodes")
    return len(manifest_entries)


def build_nested_masks(total_episodes, requested_counts, seed):
    requested_counts = unique_counts(requested_counts)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_episodes)
    mask_payloads = {}
    selected_positions = {}
    unavailable_counts = []

    for count in requested_counts:
        if count > total_episodes:
            unavailable_counts.append(int(count))
            continue
        mask = np.zeros(total_episodes, dtype=bool)
        chosen = np.sort(permutation[:count])
        mask[chosen] = True
        mask_payloads[int(count)] = mask.tolist()
        selected_positions[int(count)] = chosen.tolist()

    return mask_payloads, selected_positions, unavailable_counts


def write_masks(task_name, task_dir, split_name, total_episodes, requested_counts, seed):
    masks, selections, unavailable = build_nested_masks(total_episodes, requested_counts, seed)
    written_paths = {}
    for count, mask_payload in masks.items():
        mask_path = task_dir / f"{split_name}_mask_{count}.json"
        with open(mask_path, "w") as file:
            json.dump(mask_payload, file)
        written_paths[count] = mask_path
        print(f"[{task_name}] wrote {mask_path}")
    return written_paths, selections, unavailable


def write_manifest(task_name, task_dir, payload):
    manifest_path = task_dir / "eval_data_manifest.json"
    with open(manifest_path, "w") as file:
        json.dump(payload, file, indent=2)
    print(f"[{task_name}] wrote {manifest_path}")


def run_task(task_name, task_spec, args, output_root):
    task_dir = output_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    human_counts = unique_counts(args.human_counts or DEFAULT_HUMAN_COUNTS)
    robot_counts = unique_counts(args.robot_counts or DEFAULT_ROBOT_COUNTS)

    if args.skip_source_validation:
        if not args.skip_zarr:
            raise ValueError("--skip-source-validation can only be used together with --skip-zarr")
        prepared_human_sources = [
            {
                "dataset": canonicalize_hf_source(source["dataset"]),
                "skip_episodes": sorted(set(source.get("skip_episodes", []))),
                "episode_indices": [],
            }
            for source in task_spec["human_sources"]
        ]
        prepared_robot_sources = [
            {
                "dataset": canonicalize_hf_source(source["dataset"]),
                "skip_episodes": sorted(set(source.get("skip_episodes", []))),
                "episode_indices": [],
            }
            for source in task_spec["robot_sources"]
        ]
        human_total = int(task_spec["human_total"])
        robot_total = int(task_spec["robot_total"])
    else:
        prepared_human_sources = build_sources(
            task_name,
            task_spec["human_sources"],
            args.cache_dir,
            split_name="human",
            allow_patterns=["meta/*", "videos/*"],
        )
        prepared_robot_sources = build_sources(
            task_name,
            task_spec["robot_sources"],
            args.cache_dir,
            split_name="robot",
            allow_patterns=ROBOT_ALLOW_PATTERNS,
        )
        human_total = min(
            MAX_HUMAN_EPISODES,
            sum(len(source["episode_indices"]) for source in prepared_human_sources),
        )
        robot_total = min(
            MAX_ROBOT_EPISODES,
            sum(len(source["episode_indices"]) for source in prepared_robot_sources),
        )

    human_zarr_path = task_dir / "human.zarr"
    robot_zarr_path = task_dir / "robot.zarr"

    if not args.skip_zarr:
        human_total = convert_human_zarr(
            task_name,
            prepared_human_sources,
            human_zarr_path,
            overwrite=args.overwrite_zarr,
        )
        robot_total = convert_robot_zarr(
            task_name,
            prepared_robot_sources,
            robot_zarr_path,
            overwrite=args.overwrite_zarr,
        )

    human_mask_paths = {}
    human_selected = {}
    unavailable_human_counts = []
    robot_mask_paths = {}
    robot_selected = {}
    unavailable_robot_counts = []

    if not args.skip_masks:
        human_mask_paths, human_selected, unavailable_human_counts = write_masks(
            task_name,
            task_dir,
            "human",
            human_total,
            human_counts,
            stable_seed(task_name, "human", args.seed),
        )
        robot_mask_paths, robot_selected, unavailable_robot_counts = write_masks(
            task_name,
            task_dir,
            "robot",
            robot_total,
            robot_counts,
            stable_seed(task_name, "robot", args.seed),
        )

    available_human_counts = [count for count in human_counts if count in human_mask_paths]
    available_robot_counts = [count for count in robot_counts if count in robot_mask_paths]
    manifest_human_masks = {
        count: task_relative_path(task_dir, path)
        for count, path in human_mask_paths.items()
    }
    manifest_robot_masks = {
        count: task_relative_path(task_dir, path)
        for count, path in robot_mask_paths.items()
    }
    training_grid = [
        {
            "human_demos": human_count,
            "robot_demos": robot_count,
            "human_mask": manifest_human_masks[human_count],
            "robot_mask": manifest_robot_masks[robot_count],
        }
        for human_count in available_human_counts
        for robot_count in available_robot_counts
    ]

    manifest = {
        "task": task_name,
        "human_fps": EXPECTED_FPS,
        "robot_fps": EXPECTED_FPS,
        "human_zarr": task_relative_path(task_dir, human_zarr_path),
        "robot_zarr": task_relative_path(task_dir, robot_zarr_path),
        "human_total": human_total,
        "robot_total": robot_total,
        "requested_human_counts": human_counts,
        "requested_robot_counts": robot_counts,
        "unavailable_human_counts": unavailable_human_counts,
        "unavailable_robot_counts": unavailable_robot_counts,
        "human_masks": manifest_human_masks,
        "robot_masks": manifest_robot_masks,
        "human_selected_positions": human_selected,
        "robot_selected_positions": robot_selected,
        "training_grid": training_grid,
        "human_sources": [
            {
                "dataset": source["dataset"],
                "skip_episodes": source["skip_episodes"],
                "kept_episodes": len(source["episode_indices"]),
            }
            for source in prepared_human_sources
        ],
        "robot_sources": [
            {
                "dataset": source["dataset"],
                "skip_episodes": source["skip_episodes"],
                "kept_episodes": len(source["episode_indices"]),
            }
            for source in prepared_robot_sources
        ],
    }
    write_manifest(task_name, task_dir, manifest)


def main():
    args = parse_args()
    output_root = args.output_root or default_output_root()
    if PENDING_TASKS:
        pending_summary = ", ".join(f"{task}: {reason}" for task, reason in sorted(PENDING_TASKS.items()))
        print(f"Pending task specs not included in --tasks all: {pending_summary}")
    for task_name in selected_tasks(args.tasks):
        run_task(task_name, TASK_SPECS[task_name], args, output_root)


if __name__ == "__main__":
    main()