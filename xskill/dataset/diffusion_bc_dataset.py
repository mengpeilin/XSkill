from collections import defaultdict
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import ListConfig
from tqdm import tqdm

from xskill.common.replay_buffer import ReplayBuffer
from xskill.utility.transform import get_transform_pipeline

normalize_threshold = 5e-2


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = []
    for episode_position in range(len(episode_ends)):
        start_idx = 0 if episode_position == 0 else int(episode_ends[episode_position - 1])
        end_idx = int(episode_ends[episode_position])
        episode_length = end_idx - start_idx
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    return np.asarray(indices)


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = {}
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if sample_start_idx > 0 or sample_end_idx < sequence_length:
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype,
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    return {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}


def normalize_data(data, stats):
    normalized = data.copy()
    for index in range(normalized.shape[1]):
        if stats["max"][index] - stats["min"][index] > normalize_threshold:
            normalized[:, index] = (data[:, index] - stats["min"][index]) / (
                stats["max"][index] - stats["min"][index]
            )
            normalized[:, index] = normalized[:, index] * 2 - 1
    return normalized


def unnormalize_data(normalized, stats):
    data = normalized.copy()
    for index in range(normalized.shape[1]):
        if stats["max"][index] - stats["min"][index] > normalize_threshold:
            normalized[:, index] = (normalized[:, index] + 1) / 2
            data[:, index] = (
                normalized[:, index] * (stats["max"][index] - stats["min"][index])
                + stats["min"][index]
            )
    return data


class KitchenBCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs,
        proto_dirs,
        pred_horizon,
        obs_horizon,
        action_horizon,
        resize_shape=None,
        proto_horizon=None,
        raw_representation=False,
        softmax_prototype=False,
        prototype=False,
        one_hot_prototype=False,
        prototype_snap=False,
        snap_frames=100,
        mask=None,
        camera_views=None,
        unnormal_list=None,
        pipeline=None,
        verbose=False,
        seed=0,
    ):
        self.data_dirs = [str(path) for path in data_dirs]
        self.proto_dirs = self._normalize_paths(proto_dirs, "proto_dirs")
        self.masks = self._normalize_masks(mask)
        self.resize_shape = tuple(resize_shape) if resize_shape is not None else None
        self.raw_representation = raw_representation
        self.softmax_prototype = softmax_prototype
        self.prototype = prototype
        self.one_hot_prototype = one_hot_prototype
        self.prototype_snap = prototype_snap
        self.snap_frames = snap_frames
        self.camera_views = list(camera_views) if camera_views is not None else ["cam1", "wrist_cam"]
        self.unnormal_list = list(unnormal_list or [])
        self.pipeline = self._normalize_pipeline(pipeline)
        self.verbose = verbose
        self.seed = seed
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.proto_horizon = obs_horizon if proto_horizon is None else proto_horizon

        if self.camera_views != ["cam1", "wrist_cam"]:
            raise ValueError(
                f"KitchenBCDataset requires camera_views=['cam1', 'wrist_cam'], got {self.camera_views}."
            )

        self.set_seed(seed)
        self.dataset_entries = self._load_dataset_entries()

        train_data = defaultdict(list)
        self.load_data(train_data)

        episode_lengths = [len(actions) for actions in train_data["actions"]]
        for key, values in train_data.items():
            train_data[key] = np.concatenate(values)

        self.episode_ends = np.cumsum(episode_lengths)
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        stats = {}
        for key, value in train_data.items():
            if key == "images" or key in self.unnormal_list:
                continue
            stats[key] = get_data_stats(value)
            train_data[key] = normalize_data(value, stats[key])

        self.stats = stats
        self.normalized_train_data = train_data

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _normalize_pipeline(self, pipeline):
        if pipeline is None:
            return None
        if isinstance(pipeline, ListConfig):
            pipeline = list(pipeline)
        if isinstance(pipeline, (list, tuple)):
            return get_transform_pipeline(list(pipeline))
        return pipeline

    def _load_mask(self, mask_value):
        if mask_value is None:
            return None
        if isinstance(mask_value, (list, tuple, np.ndarray)):
            return list(mask_value)
        with open(mask_value, "r") as file:
            return json.load(file)

    def _normalize_masks(self, mask):
        if mask is None:
            return [None] * len(self.data_dirs)
        if isinstance(mask, (str, Path)):
            return [self._load_mask(mask)] * len(self.data_dirs)
        if isinstance(mask, (list, tuple, np.ndarray)) and len(mask) > 0 and all(
            isinstance(value, (bool, np.bool_, int, np.integer)) for value in mask
        ):
            return [self._load_mask(mask)] * len(self.data_dirs)
        mask_list = list(mask)
        if len(mask_list) == 1 and len(self.data_dirs) > 1:
            mask_list = mask_list * len(self.data_dirs)
        if len(mask_list) != len(self.data_dirs):
            raise ValueError("mask must be a path or a list aligned with data_dirs")
        return [self._load_mask(value) for value in mask_list]

    def _normalize_paths(self, paths, name):
        if isinstance(paths, (str, Path)):
            return [str(paths)] * len(self.data_dirs)
        path_list = [str(path) for path in paths]
        if len(path_list) == 1 and len(self.data_dirs) > 1:
            path_list = path_list * len(self.data_dirs)
        if len(path_list) != len(self.data_dirs):
            raise ValueError(f"{name} must be a path or a list aligned with data_dirs")
        return path_list

    def _episode_indices(self, replay_buffer):
        if "episode_indices" in replay_buffer.meta:
            return np.asarray(replay_buffer.meta["episode_indices"][:], dtype=np.int64)
        return np.arange(replay_buffer.n_episodes, dtype=np.int64)

    def _episode_bounds(self, episode_ends, episode_position):
        episode_end = int(episode_ends[episode_position])
        episode_begin = 0 if episode_position == 0 else int(episode_ends[episode_position - 1])
        return episode_begin, episode_end

    def _selected_episode_positions(self, replay_buffer, mask):
        positions = np.arange(replay_buffer.n_episodes, dtype=np.int64)
        if mask is None:
            return positions
        if len(mask) != len(positions):
            raise ValueError(
                f"Mask length {len(mask)} does not match number of episodes {len(positions)}."
            )
        return positions[np.asarray(mask, dtype=bool)]

    def _proto_key(self):
        if self.raw_representation:
            return "traj_representation"
        if self.softmax_prototype or self.one_hot_prototype:
            return "softmax_encode_protos"
        if self.prototype:
            return "encode_protos"
        raise ValueError("One prototype mode must be enabled.")

    def _load_dataset_entries(self):
        entries = []
        for data_path, proto_path, mask in zip(self.data_dirs, self.proto_dirs, self.masks):
            replay_buffer = ReplayBuffer.create_from_path(data_path, mode="r")
            proto_buffer = ReplayBuffer.create_from_path(proto_path, mode="r")
            episode_positions = self._selected_episode_positions(replay_buffer, mask)
            data_episode_indices = self._episode_indices(replay_buffer)
            proto_episode_indices = self._episode_indices(proto_buffer)
            proto_lookup = {
                int(episode_index): position
                for position, episode_index in enumerate(proto_episode_indices.tolist())
            }
            proto_positions = []
            for episode_position in episode_positions.tolist():
                episode_index = int(data_episode_indices[episode_position])
                if episode_index not in proto_lookup:
                    raise KeyError(
                        f"Episode index {episode_index} from {data_path} is missing in {proto_path}."
                    )
                proto_positions.append(proto_lookup[episode_index])
            entries.append(
                {
                    "data_path": data_path,
                    "replay_buffer": replay_buffer,
                    "episode_positions": episode_positions,
                    "episode_indices": data_episode_indices,
                    "episode_ends": replay_buffer.episode_ends[:],
                    "proto_buffer": proto_buffer,
                    "proto_positions": np.asarray(proto_positions, dtype=np.int64),
                    "proto_episode_ends": proto_buffer.episode_ends[:],
                }
            )
        return entries

    def _load_images(self, replay_buffer, episode_begin, episode_end):
        views = []
        for camera_view in self.camera_views:
            key = f"camera_{camera_view}"
            if key not in replay_buffer:
                raise KeyError(
                    f"Replay buffer does not contain {key}. Available keys: {list(replay_buffer.keys())}"
                )
            images = np.asarray(replay_buffer[key][episode_begin:episode_end], dtype=np.uint8)
            if self.resize_shape is not None and images.shape[1:3] != (
                self.resize_shape[1],
                self.resize_shape[0],
            ):
                images = np.stack(
                    [cv2.resize(frame, self.resize_shape) for frame in images],
                    axis=0,
                )
            views.append(images)
        return np.stack(views, axis=1)

    def _load_proto_sequence(self, proto_buffer, episode_begin, episode_end):
        proto = np.asarray(
            proto_buffer[self._proto_key()][episode_begin:episode_end],
            dtype=np.float32,
        )
        if self.one_hot_prototype:
            one_hot = np.zeros_like(proto)
            max_index = np.argmax(proto, axis=1)
            one_hot[np.arange(len(proto)), max_index] = 1.0
            proto = one_hot
        return proto

    def _build_proto_snap(self, proto):
        sample_count = min(self.snap_frames, len(proto))
        sample_index = sorted(random.sample(list(range(len(proto))), k=sample_count))
        snap = np.asarray(proto[sample_index], dtype=np.float32)
        if sample_count < self.snap_frames:
            pad = np.repeat(snap[-1:], self.snap_frames - sample_count, axis=0)
            snap = np.concatenate([snap, pad], axis=0)
        flattened_snap = snap.reshape(1, -1)
        return np.repeat(flattened_snap, len(proto), axis=0)

    def load_data(self, train_data):
        iterator = self.dataset_entries
        for entry in iterator:
            episode_iterator = zip(entry["episode_positions"], entry["proto_positions"])
            if self.verbose:
                episode_iterator = tqdm(
                    list(episode_iterator),
                    desc=f"Loading {Path(entry['data_path']).name}",
                )
            for episode_position, proto_position in episode_iterator:
                data_begin, data_end = self._episode_bounds(entry["episode_ends"], int(episode_position))
                proto_begin, proto_end = self._episode_bounds(
                    entry["proto_episode_ends"],
                    int(proto_position),
                )
                obs = np.asarray(entry["replay_buffer"]["obs"][data_begin:data_end], dtype=np.float32)
                actions = np.asarray(entry["replay_buffer"]["actions"][data_begin:data_end], dtype=np.float32)
                images = self._load_images(entry["replay_buffer"], data_begin, data_end)
                proto = self._load_proto_sequence(entry["proto_buffer"], proto_begin, proto_end)

                target_len = min(len(obs), len(actions), len(images), len(proto))
                if target_len <= 0:
                    raise ValueError(f"Episode {int(episode_position)} in {entry['data_path']} is empty.")

                obs = obs[:target_len]
                actions = actions[:target_len]
                images = images[:target_len]
                proto = proto[:target_len]

                train_data["obs"].append(obs)
                train_data["actions"].append(actions)
                train_data["images"].append(images)
                train_data["protos"].append(proto)
                if self.prototype_snap:
                    train_data["proto_snap"].append(self._build_proto_snap(proto))

    def transform_images(self, images):
        images = torch.tensor(images, dtype=torch.float32) / 255.0
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape (T, V, H, W, C), got {tuple(images.shape)}")
        images = images.permute(0, 1, 4, 2, 3)
        if self.pipeline is not None:
            images = self.pipeline(images)
        return images

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[index]

        sample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        sample["obs"] = sample["obs"][: self.obs_horizon]
        sample["images"] = self.transform_images(sample["images"])[: self.obs_horizon]

        if self.prototype_snap:
            sample["protos"] = sample["protos"][: self.obs_horizon][-1:]
            sample["proto_snap"] = sample["proto_snap"][-1:]
        else:
            sample["protos"] = sample["protos"][: self.obs_horizon][-self.proto_horizon :]

        return sample
