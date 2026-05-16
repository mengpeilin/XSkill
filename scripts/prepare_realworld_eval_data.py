import argparse
import binascii
import json
import sys
from pathlib import Path

import numpy as np

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
    build_episode_arrays,
    decode_video as decode_robot_video,
    list_episode_dirs,
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
DEFAULT_HUMAN_CAMERA_KEY = "observation.images.cam_azure_kinect_left.color"

TASK_SPECS = {
    "insert_donut": {
        "robot_input": "data/insert_donut",
        "human_total": 300,
        "robot_total": 200,
        "human_sources": [
            {"dataset": "xiaochyVera/insert_donut_human_2"},
            {"dataset": "xiaochyVera/insert_donut_human_3"},
        ],
    },
    "pick_place_red_mug": {
         "robot_input": "data/pick_place_red_mug",
         "human_total": 196,
         "robot_total": 100,
         "human_sources": [
             {"dataset": "Kovavavvavava/pick_place_red_mug_20260327_2"},
             {"dataset": "Kovavavvavava/pick_place_red_mug_20260327_1"},
             {"dataset": "xiaochyVera/pick_red_mug_human", "skip_episodes": [6, 7]},
             {"dataset": "xiaochyVera/pick_red_mug_human_1", "skip_episodes": [8]},
             {"dataset": "xiaochyVera/pick_red_mug_human_2", "skip_episodes": [4]},
             {"dataset": "xiaochyVera/pick_red_mug_human_3"},
             {"dataset": "xiaochyVera/pick_red_mug_human_4"},
         ],
     },
    "pick_place_toys": {
        "robot_input": "data/pick_place_toys",
        "human_total": 288,
        "robot_total": 200,
        "human_sources": [
            {"dataset": "xiaochyVera/pick_toys_human_5_1_1", "skip_episodes": [0, 12, 15, 36, 42, 43, 77, 85, 88]},
            {"dataset": "xiaochyVera/pick_toys_human_2", "skip_episodes": [12, 19]},
            {"dataset": "xiaochyVera/pick_toys_human_3", "skip_episodes": [0]},
        ],
    },
    "stack_bowls": {
        "robot_input": "data/stack_bowls",
        "human_total": 300,
        "robot_total": 200,
        "human_sources": [
            {"dataset": "xiaochyVera/stack_bowls_human_5_1"},
        ],
    },
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
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    return task_args


def stable_seed(task_name, split_name, base_seed):
    salt = f"{task_name}:{split_name}".encode("utf-8")
    return int(base_seed + binascii.crc32(salt))


def unique_counts(counts):
    return sorted({int(count) for count in counts})


def task_relative_path(task_dir, path):
    return str(path.relative_to(task_dir))


def build_human_sources(task_name, task_spec, cache_dir):
    prepared_sources = []
    for source_spec in task_spec["human_sources"]:
        dataset_name = canonicalize_hf_source(source_spec["dataset"])
        dataset_root = resolve_dataset_root(dataset_name, cache_dir=cache_dir)
        info = load_info(dataset_root)
        episode_indices = get_episode_indices(info)
        skip_episodes = sorted(set(source_spec.get("skip_episodes", [])))
        keep_episodes = [episode for episode in episode_indices if episode not in set(skip_episodes)]
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
            f"[{task_name}] human source {dataset_name}: kept {len(keep_episodes)} / {len(episode_indices)} episodes"
        )
    return prepared_sources


def build_robot_episodes(task_name, task_spec):
    input_dir = repo_root() / task_spec["robot_input"]
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing robot demo directory {input_dir}")
    episode_dirs = list_episode_dirs(input_dir)
    print(f"[{task_name}] robot source {input_dir}: found {len(episode_dirs)} episodes")
    return input_dir, episode_dirs


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
    MAX_HUMAN = 200

    for source in prepared_sources:
        if len(manifest_entries) >= MAX_HUMAN:
            break
        dataset_name = source["dataset"]
        info = source["info"]
        for episode_index in tqdm(source["episode_indices"], desc=f"[{task_name}] {dataset_name}"):
            if len(manifest_entries) >= MAX_HUMAN:
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


def convert_robot_zarr(task_name, episode_dirs, output_path, overwrite):
    if output_path.exists() and not overwrite:
        existing_total = load_existing_episode_count(output_path)
        print(f"[{task_name}] keeping existing {output_path} with {existing_total} episodes")
        return existing_total
    replay_buffer = prepare_robot_replay_buffer(output_path, overwrite=overwrite)
    source_episode_ids = []
    for episode_dir in tqdm(episode_dirs, desc=f"[{task_name}] robot"):
        trajectory_path = episode_dir / "trajectory.npz"
        cam0_path = episode_dir / "cam0.mp4"
        cam1_path = episode_dir / "cam1.mp4"
        wrist_path = episode_dir / "wrist_cam.mp4"

        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Missing trajectory file {trajectory_path}")
        if not cam0_path.is_file():
            raise FileNotFoundError(f"Missing camera file {cam0_path}")
        if not cam1_path.is_file():
            raise FileNotFoundError(f"Missing camera file {cam1_path}")
        if not wrist_path.is_file():
            raise FileNotFoundError(f"Missing camera file {wrist_path}")

        trajectory = np.load(trajectory_path, allow_pickle=False)
        obs, actions = build_episode_arrays(
            trajectory,
            state_key="states_ee",
            action_key="action_ee",
            gripper_width_key="gripper_width",
        )
        replay_buffer.add_episode(
            {
                "obs": obs,
                "actions": actions,
                "camera_cam0": decode_robot_video(cam0_path, len(obs)),
                "camera_cam1": decode_robot_video(cam1_path, len(obs)),
                "camera_wrist_cam": decode_robot_video(wrist_path, len(obs)),
            },
            compressors="disk",
        )
        source_episode_ids.append(int(episode_dir.name.split("_")[-1]))

    replay_buffer.update_meta(
        {
            "camera_views": np.asarray(["cam0", "cam1", "wrist_cam"], dtype="<U9"),
            "camera_file_stems": np.asarray(["cam0", "cam1", "wrist_cam"], dtype="<U9"),
            "episode_indices": np.asarray(source_episode_ids, dtype=np.int64),
        }
    )
    print(f"[{task_name}] wrote {output_path} with {len(source_episode_ids)} episodes")
    return len(source_episode_ids)


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
    robot_input_dir = repo_root() / task_spec["robot_input"]

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
        robot_episode_dirs = []
        human_total = int(task_spec["human_total"])
        robot_total = int(task_spec["robot_total"])
    else:
        prepared_human_sources = build_human_sources(task_name, task_spec, args.cache_dir)
        robot_input_dir, robot_episode_dirs = build_robot_episodes(task_name, task_spec)
        human_total = sum(len(source["episode_indices"]) for source in prepared_human_sources)
        robot_total = len(robot_episode_dirs)

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
            robot_episode_dirs,
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
        "robot_input_dir": str(robot_input_dir),
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
    }
    write_manifest(task_name, task_dir, manifest)


def main():
    args = parse_args()
    output_root = args.output_root or (repo_root() / "zarr")
    for task_name in selected_tasks(args.tasks):
        run_task(task_name, TASK_SPECS[task_name], args, output_root)


if __name__ == "__main__":
    main()
