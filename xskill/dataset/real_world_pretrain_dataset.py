import collections
import concurrent.futures
import json
import random
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch

from xskill.common.replay_buffer import ReplayBuffer

IndexBatch = namedtuple("IndexBatch", "im_q index info")


class RealWorldEpisodeTrajDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame_sampler,
        dataset_paths,
        mask=None,
        slide=None,
        seed=None,
        camera_view="cam1",
        max_get_threads=4,
        resize_shape=(320, 240),
    ):
        super().__init__()
        self._frame_sampler = frame_sampler
        self.slide = slide
        self.camera_view = camera_view
        self.max_get_threads = max_get_threads
        self.resize_shape = tuple(resize_shape) if resize_shape is not None else None
        self._seed = seed
        self.dataset_paths = [str(path) for path in dataset_paths]
        self.masks = self._normalize_masks(mask)
        self._index = {}
        self._buffers = collections.OrderedDict()
        self.seed_rng()
        self._build_index()

    def seed_rng(self):
        if self._seed is not None:
            random.seed(self._seed)

    def _camera_key(self):
        return f"camera_{self.camera_view}"

    def _load_mask(self, mask_value):
        if mask_value is None:
            return None
        if isinstance(mask_value, (list, tuple, np.ndarray)):
            return np.asarray(mask_value, dtype=bool)
        with open(mask_value, "r") as file:
            return np.asarray(json.load(file), dtype=bool)

    def _normalize_masks(self, mask):
        if mask is None:
            return [None] * len(self.dataset_paths)
        if isinstance(mask, (str, Path)):
            return [self._load_mask(mask)] * len(self.dataset_paths)
        if isinstance(mask, (list, tuple, np.ndarray)) and len(mask) > 0 and all(
            isinstance(value, (bool, np.bool_, int, np.integer)) for value in mask
        ):
            return [self._load_mask(mask)] * len(self.dataset_paths)
        mask_list = list(mask)
        if len(mask_list) == 1 and len(self.dataset_paths) > 1:
            mask_list = mask_list * len(self.dataset_paths)
        if len(mask_list) != len(self.dataset_paths):
            raise ValueError("mask must be a path or a list aligned with dataset_paths")
        return [self._load_mask(value) for value in mask_list]

    def _selected_episode_positions(self, replay_buffer, mask):
        episode_positions = np.arange(replay_buffer.n_episodes, dtype=np.int64)
        if mask is None:
            return episode_positions
        if len(mask) != len(episode_positions):
            raise ValueError(
                f"Mask length {len(mask)} does not match number of episodes {len(episode_positions)}."
            )
        return episode_positions[np.asarray(mask, dtype=bool)]

    def _build_index(self):
        global_index = 0
        for dataset_path, mask in zip(self.dataset_paths, self.masks):
            replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode="r")
            camera_key = self._camera_key()
            if camera_key not in replay_buffer:
                raise KeyError(
                    f"{dataset_path} does not contain {camera_key}. Available keys: {list(replay_buffer.keys())}"
                )
            episode_ends = replay_buffer.episode_ends[:]
            episode_indices = np.asarray(
                replay_buffer.meta.get("episode_indices", np.arange(replay_buffer.n_episodes)),
                dtype=np.int64,
            )
            self._buffers[dataset_path] = {
                "replay_buffer": replay_buffer,
                "episode_ends": episode_ends,
                "episode_indices": episode_indices,
                "camera_key": camera_key,
            }
            selected_positions = self._selected_episode_positions(replay_buffer, mask)
            for episode_position in selected_positions.tolist():
                self._index[global_index] = (dataset_path, int(episode_position))
                global_index += 1

    def _episode_bounds(self, episode_ends, episode_position):
        episode_end = int(episode_ends[episode_position])
        episode_begin = 0 if episode_position == 0 else int(episode_ends[episode_position - 1])
        return episode_begin, episode_end

    def _load_sequence_data(self, sample, image_zarr, episode_begin):
        indices = list(np.asarray(sample["ctx_idxs"]).reshape(-1) + episode_begin)
        frames = [None] * len(indices)

        def load_frame(frame_position, buffer_index):
            frame = image_zarr[buffer_index]
            if self.resize_shape is not None:
                frame = cv2.resize(frame, self.resize_shape)
            frames[frame_position] = frame
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_get_threads) as executor:
            futures = [executor.submit(load_frame, frame_position, buffer_index) for frame_position, buffer_index in enumerate(indices)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        return np.stack(frames)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        dataset_path, episode_position = self._index[index]
        entry = self._buffers[dataset_path]
        episode_begin, episode_end = self._episode_bounds(entry["episode_ends"], episode_position)
        episode_length = episode_end - episode_begin
        sample = self._frame_sampler.sample(np.arange(episode_length))
        sequence_data = self._load_sequence_data(
            sample,
            entry["replay_buffer"][entry["camera_key"]],
            episode_begin,
        )
        info = {
            "dataset_path": dataset_path,
            "episode_position": episode_position,
            "episode_index": int(entry["episode_indices"][episode_position]),
            "episode_begin": episode_begin,
            "episode_length": episode_length,
        }
        return IndexBatch(self.transform(sequence_data), index, info)

    def transform(self, sequence_data):
        sequence_data = np.transpose(sequence_data, (0, 3, 1, 2)).astype(np.float32)
        return sequence_data / 255.0
