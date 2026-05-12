import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import scipy.spatial.transform as st
import zarr

from xskill.common.replay_buffer import ReplayBuffer

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


def write_json(path: Path, data):
    with open(path, "w") as file:
        json.dump(data, file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert robot video episodes into the folder-per-episode format used by xskill simulation training."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/robot/pick_place_red_mug_100"),
        help="Directory that contains source episode folders such as episode_000000.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/pick_red_mug"),
        help="Root directory that will contain the converted robot/ and human/ folders.",
    )
    parser.add_argument(
        "--output-split",
        default="robot",
        help="Subdirectory name inside output-root.",
    )
    parser.add_argument(
        "--camera-file",
        nargs="+",
        default=["cam1.mp4", "wrist_cam.mp4"],
        help=(
            "One or more camera video filenames inside each source episode. The first camera is written to the episode root; "
            "additional cameras are written to subdirectories named after each video stem."
        ),
    )
    parser.add_argument(
        "--state-key",
        default="states_ee",
        help="Key from trajectory.npz to save as states.json.",
    )
    parser.add_argument(
        "--action-key",
        default="action_ee",

        help="Key from trajectory.npz to save as actions.json.",
    )
    parser.add_argument(
        "--gripper-width-key",
        default="gripper_width",
        help="Key from trajectory.npz used when composing 7D ee pose plus gripper states.",
    )
    parser.add_argument(
        "--state-representation",
        choices=["raw", "ee_pose_gripper_7d"],
        default="ee_pose_gripper_7d",
        help=(
            "How to build states.json. 'raw' writes the selected state key as-is. "
            "'ee_pose_gripper_7d' converts xyz + quaternion (wxyz) to xyz + rotvec and appends gripper width."
        ),
    )
    parser.add_argument(
        "--action-representation",
        choices=["raw", "ee_pose_gripper_7d", "delta_action_7d"],
        default="ee_pose_gripper_7d",
        help=(
            "How to build actions.json. 'raw' writes the selected action key as-is. "
            "'ee_pose_gripper_7d' converts xyz + quaternion (wxyz) to xyz + rotvec and appends the last action dim. "
            "'delta_action_7d' keeps the first 7 dimensions of the selected action key."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of source episodes to convert.",
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
            "KitchenBCDataset will auto-detect it and use it for faster training."
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


def numeric_episode_dirs(root: Path):
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode_"))


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


def quaternion_wxyz_to_rotvec(quat_wxyz: np.ndarray):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quaternion array with last dim 4, got {quat_wxyz.shape}.")

    quat_xyzw = np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)
    rotations = st.Rotation.from_quat(quat_xyzw.reshape(-1, 4))
    rotvec = rotations.as_rotvec().reshape(quat_xyzw.shape[:-1] + (3,))
    return rotvec.astype(np.float32)


def validate_pose_quaternion(pose: np.ndarray, pose_key: str):
    quat_wxyz = np.asarray(pose[:, 3:7], dtype=np.float32)
    quat_norm = np.linalg.norm(quat_wxyz, axis=1)
    if not np.all(np.isfinite(quat_norm)):
        raise ValueError(f"Pose key {pose_key} contains non-finite quaternion values.")

    max_deviation = float(np.max(np.abs(quat_norm - 1.0)))
    if max_deviation > 0.1:
        raise ValueError(
            f"Pose key {pose_key} does not look like xyz + quaternion (wxyz): "
            f"quaternion norms deviate from 1 by up to {max_deviation:.3f}. "
            "If this key stores joint actions, use --action-representation raw or delta_action_7d instead."
        )


def camera_view_name(camera_index: int, video_path: Path):
    if camera_index == 0:
        return "."
    return video_path.stem


def camera_array_key(camera_view: str):
    if camera_view in (None, "", ".", "root", "primary"):
        return "camera_primary"
    return f"camera_{camera_view}"


def camera_storage_key(camera_index: int, video_path: Path):
    return f"camera_{video_path.stem}"


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


def convert_ee_pose_gripper(trajectory, pose_key: str, gripper_width_key: str, include_action_gripper: bool):
    if pose_key not in trajectory.files:
        raise KeyError(f"Pose key {pose_key} not found in trajectory file.")
    if gripper_width_key not in trajectory.files and not include_action_gripper:
        raise KeyError(f"Gripper width key {gripper_width_key} not found in trajectory file.")

    pose = np.asarray(trajectory[pose_key], dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 7:
        raise ValueError(
            f"Expected pose key {pose_key} to have shape (T, >=7), got {pose.shape}."
        )

    validate_pose_quaternion(pose, pose_key)

    pos = pose[:, :3]
    quat_wxyz = pose[:, 3:7]
    rotvec = quaternion_wxyz_to_rotvec(quat_wxyz)

    if include_action_gripper:
        if pose.shape[1] < 8:
            raise ValueError(
                f"Expected action pose key {pose_key} to have an eighth gripper dimension, got shape {pose.shape}."
            )
        gripper = pose[:, 7:8]
    else:
        gripper = np.asarray(trajectory[gripper_width_key], dtype=np.float32)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        if gripper.ndim != 2 or gripper.shape[1] != 1:
            raise ValueError(
                f"Expected gripper width key {gripper_width_key} to have shape (T, 1), got {gripper.shape}."
            )

    return np.concatenate([pos, rotvec, gripper], axis=-1).astype(np.float32)


def build_state_array(trajectory, args):
    if args.state_representation == "raw":
        if args.state_key not in trajectory.files:
            raise KeyError(f"State key {args.state_key} not found in trajectory file.")
        return np.asarray(trajectory[args.state_key], dtype=np.float32)

    return convert_ee_pose_gripper(
        trajectory,
        pose_key=args.state_key,
        gripper_width_key=args.gripper_width_key,
        include_action_gripper=False,
    )


def build_action_array(trajectory, args):
    if args.action_representation == "raw":
        if args.action_key not in trajectory.files:
            raise KeyError(f"Action key {args.action_key} not found in trajectory file.")
        return np.asarray(trajectory[args.action_key], dtype=np.float32)

    if args.action_representation == "delta_action_7d":
        if args.action_key not in trajectory.files:
            raise KeyError(f"Action key {args.action_key} not found in trajectory file.")
        action = np.asarray(trajectory[args.action_key], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] < 7:
            raise ValueError(
                f"Expected action key {args.action_key} to have shape (T, >=7), got {action.shape}."
            )
        return action[:, :7]

    return convert_ee_pose_gripper(
        trajectory,
        pose_key=args.action_key,
        gripper_width_key=args.gripper_width_key,
        include_action_gripper=True,
    )


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
    camera_files = args.camera_file
    write_episode_dirs = args.write_episode_dirs or not args.write_zarr
    source_episodes = numeric_episode_dirs(args.input_dir)
    if args.limit is not None:
        source_episodes = source_episodes[: args.limit]
    output_split_dir = args.output_root / args.output_split
    if write_episode_dirs:
        output_split_dir.mkdir(parents=True, exist_ok=True)
    zarr_buffer = None
    zarr_path = None
    if args.write_zarr:
        zarr_path = resolve_zarr_path(args)
        zarr_buffer = prepare_zarr_buffer(zarr_path, overwrite=args.overwrite)

    source_index = []
    for output_episode_idx, episode_dir in enumerate(tqdm(source_episodes, desc="Converting robot episodes")):
        trajectory_path = episode_dir / "trajectory.npz"
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Missing trajectory file: {trajectory_path}")

        video_paths = [episode_dir / camera_file for camera_file in camera_files]
        for video_path in video_paths:
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing video file: {video_path}")

        trajectory = np.load(trajectory_path, allow_pickle=False)
        states = build_state_array(trajectory, args)
        actions = build_action_array(trajectory, args)
        target_len = min(len(states), len(actions))
        if target_len <= 0:
            raise ValueError(f"Episode {episode_dir} does not contain any aligned state/action steps.")

        episode_zarr_data = None
        if zarr_buffer is not None:
            episode_zarr_data = {
                "obs": states[:target_len].astype(np.float32),
                "actions": actions[:target_len].astype(np.float32),
            }

        output_episode_dir = None
        if write_episode_dirs:
            output_episode_dir = output_split_dir / str(output_episode_idx)
            prepare_output_dir(output_episode_dir, overwrite=args.overwrite)
        for camera_index, video_path in enumerate(video_paths):
            camera_output_dir = None
            if write_episode_dirs:
                if camera_index == 0:
                    camera_output_dir = output_episode_dir
                else:
                    camera_output_dir = output_episode_dir / video_path.stem
                    camera_output_dir.mkdir(parents=True, exist_ok=True)
            rgb_frames = decode_video_to_pngs(
                video_path,
                camera_output_dir,
                frame_limit=target_len,
                collect_rgb_frames=zarr_buffer is not None,
            )
            if episode_zarr_data is not None:
                episode_zarr_data[camera_storage_key(camera_index, video_path)] = rgb_frames[:target_len]

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
                        "source_episode": episode_dir.name,
                        "state_key": args.state_key,
                        "action_key": args.action_key,
                        "camera_files": camera_files,
                        "state_representation": args.state_representation,
                        "action_representation": args.action_representation,
                        "quaternion_order": "wxyz",
                        "gripper_width_key": args.gripper_width_key,
                        "num_steps": int(target_len),
                    },
                    file,
                )

        source_index.append(
            {
                "output_episode": output_episode_idx,
                "source_episode": episode_dir.name,
                "num_steps": int(target_len),
                "camera_files": camera_files,
            }
        )

    write_json(args.output_root / f"{args.output_split}_source_map.json", source_index)
    if zarr_buffer is not None:
        view_names = [camera_view_name(i, Path(path)) for i, path in enumerate(camera_files)]
        camera_stems = [Path(path).stem for path in camera_files]
        max_name_len = max(len(name) for name in view_names)
        max_stem_len = max(len(name) for name in camera_stems)
        zarr_buffer.update_meta(
            {
                "camera_views": np.asarray(view_names, dtype=f"<U{max_name_len}"),
                "camera_file_stems": np.asarray(camera_stems, dtype=f"<U{max_stem_len}"),
                "episode_indices": np.arange(len(source_index), dtype=np.int64),
            }
        )
        print(f"Wrote zarr dataset to {zarr_path}")
    maybe_write_train_mask(args.output_root)


if __name__ == "__main__":
    main()