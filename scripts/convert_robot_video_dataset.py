import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from xskill.common.replay_buffer import ReplayBuffer

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-split", type=str, default="robot")
    parser.add_argument("--state-key", type=str, default="states_ee")
    parser.add_argument("--action-key", type=str, default="action_ee")
    parser.add_argument("--gripper-width-key", type=str, default="gripper_width")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_episode_dirs(root: Path):
    return sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode_")
    )


def prepare_replay_buffer(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing zarr dataset {path}.")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(path)))


def decode_video(video_path: Path, frame_count: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file {video_path}.")

    frames = []
    try:
        while len(frames) < frame_count:
            success, frame = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if len(frames) != frame_count:
        raise RuntimeError(
            f"Decoded {len(frames)} frames from {video_path}, expected {frame_count}."
        )
    return np.stack(frames).astype(np.uint8)


def quaternion_wxyz_to_rotvec(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float32)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion array with last dimension 4, got {quaternion.shape}.")
    quaternion_xyzw = np.concatenate(
        [quaternion[..., 1:4], quaternion[..., 0:1]],
        axis=-1,
    )
    rotvec = Rotation.from_quat(quaternion_xyzw.reshape(-1, 4)).as_rotvec()
    return rotvec.reshape(quaternion_xyzw.shape[:-1] + (3,)).astype(np.float32)


def validate_pose_array(array: np.ndarray, key: str):
    if array.ndim != 2 or array.shape[1] < 7:
        raise ValueError(f"Expected {key} to have shape (T, >=7), got {array.shape}.")
    quaternion = np.asarray(array[:, 3:7], dtype=np.float32)
    norms = np.linalg.norm(quaternion, axis=1)
    if not np.all(np.isfinite(norms)):
        raise ValueError(f"{key} contains non-finite quaternion values.")
    max_deviation = float(np.max(np.abs(norms - 1.0)))
    if max_deviation > 0.1:
        raise ValueError(
            f"{key} does not look like xyz + quaternion (wxyz): maximum norm deviation is {max_deviation:.4f}."
        )


def build_pose_width_array(pose_array: np.ndarray, gripper_width: np.ndarray, key: str):
    validate_pose_array(pose_array, key)
    gripper_width = np.asarray(gripper_width, dtype=np.float32)
    if gripper_width.ndim == 1:
        gripper_width = gripper_width[:, None]
    if gripper_width.ndim != 2 or gripper_width.shape[1] != 1:
        raise ValueError(f"Expected gripper_width to have shape (T, 1), got {gripper_width.shape}.")

    position = pose_array[:, :3]
    rotvec = quaternion_wxyz_to_rotvec(pose_array[:, 3:7])
    return np.concatenate([position, rotvec, gripper_width], axis=-1).astype(np.float32)


def build_episode_arrays(trajectory, state_key: str, action_key: str, gripper_width_key: str):
    if state_key not in trajectory.files:
        raise KeyError(f"Missing state key {state_key} in trajectory.npz.")
    if action_key not in trajectory.files:
        raise KeyError(f"Missing action key {action_key} in trajectory.npz.")
    if gripper_width_key not in trajectory.files:
        raise KeyError(f"Missing gripper width key {gripper_width_key} in trajectory.npz.")

    states_ee = np.asarray(trajectory[state_key], dtype=np.float32)
    actions_ee = np.asarray(trajectory[action_key], dtype=np.float32)
    gripper_width = np.asarray(trajectory[gripper_width_key], dtype=np.float32)

    target_len = min(len(actions_ee), len(states_ee), len(gripper_width) - 1)
    if target_len <= 0:
        raise ValueError("Episode does not contain enough aligned state, action, and gripper samples.")

    obs = build_pose_width_array(states_ee[:target_len, :7], gripper_width[:target_len], state_key)
    actions_gripper_width = actions_ee[:target_len, -1]
    actions = build_pose_width_array(
        actions_ee[:target_len, :7],
        actions_gripper_width,
        action_key,
    )
    return obs, actions


def main():
    args = parse_args()
    output_path = args.output_root / f"{args.output_split}.zarr"
    replay_buffer = prepare_replay_buffer(output_path, overwrite=args.overwrite)

    episode_dirs = list_episode_dirs(args.input_dir)
    if args.limit is not None:
        episode_dirs = episode_dirs[: args.limit]

    source_episode_ids = []
    for episode_dir in tqdm(episode_dirs, desc="Converting robot episodes"):
        trajectory_path = episode_dir / "trajectory.npz"
        cam0_path = episode_dir / "cam0.mp4"
        cam1_path = episode_dir / "cam1.mp4"
        wrist_path = episode_dir / "wrist_cam.mp4"

        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Missing trajectory file: {trajectory_path}")
        if not cam0_path.is_file():
            raise FileNotFoundError(f"Missing camera file: {cam0_path}")
        if not cam1_path.is_file():
            raise FileNotFoundError(f"Missing camera file: {cam1_path}")
        if not wrist_path.is_file():
            raise FileNotFoundError(f"Missing camera file: {wrist_path}")

        trajectory = np.load(trajectory_path, allow_pickle=False)
        obs, actions = build_episode_arrays(
            trajectory,
            state_key=args.state_key,
            action_key=args.action_key,
            gripper_width_key=args.gripper_width_key,
        )

        replay_buffer.add_episode(
            {
                "obs": obs,
                "actions": actions,
                "camera_cam0": decode_video(cam0_path, len(obs)),
                "camera_cam1": decode_video(cam1_path, len(obs)),
                "camera_wrist_cam": decode_video(wrist_path, len(obs)),
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
    print(f"Wrote zarr dataset to {output_path}")


if __name__ == "__main__":
    main()
