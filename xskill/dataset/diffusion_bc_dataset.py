from collections import defaultdict, namedtuple
import numpy as np
import torch
from xskill.utility.file_utils import get_subdirs, get_files
import random
import collections
import os
import os.path as osp
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import cv2
from omegaconf import ListConfig

from xskill.common.replay_buffer import ReplayBuffer

normalize_threshold = 5e-2


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = data.copy()
    for i in range(ndata.shape[1]):
        if stats["max"][i] - stats["min"][i] > normalize_threshold:
            ndata[:, i] = (data[:, i] - stats["min"][i]) / (
                stats["max"][i] - stats["min"][i]
            )
            # normalize to [-1, 1]
            ndata[:, i] = ndata[:, i] * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    data = ndata.copy()
    for i in range(ndata.shape[1]):
        if stats["max"][i] - stats["min"][i] > normalize_threshold:
            ndata[:, i] = (ndata[:, i] + 1) / 2
            data[:, i] = (
                ndata[:, i] * (stats["max"][i] - stats["min"][i]) + stats["min"][i]
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
        obs_image_based=False,
        camera_views=None,
        unnormal_list=[],
        pipeline=None,
        verbose=False,
        seed=0,
    ):
        """
        Support 1) raw representation 2) softmax prototype 3) prototype 4) one-hot prototype
        """
        self.verbose = verbose
        self.resize_shape = resize_shape
        self.data_dirs = [str(path) for path in data_dirs]
        self.proto_dirs = self._normalize_proto_dirs(proto_dirs)
        self.masks = self._normalize_masks(mask)

        self.seed = seed
        self.set_seed(self.seed)
        self.raw_representation = raw_representation
        self.softmax_prototype = softmax_prototype
        self.prototype = prototype
        self.one_hot_prototype = one_hot_prototype
        self.obs_image_based = obs_image_based
        self.camera_views = list(camera_views) if camera_views is not None else ["."]
        self.prototype_snap = prototype_snap
        self.snap_frames = snap_frames
        self.pipeline = self._normalize_pipeline(pipeline)
        self.unnormal_list = unnormal_list
        self._episode_image_paths = []
        self._build_dir_tree()

        train_data = defaultdict(list)
        self.load_data(train_data)

        episode_ends = []
        for eps_action_data in train_data["actions"]:
            episode_ends.append(len(eps_action_data))

        for k, v in train_data.items():
            train_data[k] = np.concatenate(v)

        print(f"training data len {len(train_data['actions'])}")

        # Marks one-past the last index for each episode
        episode_ends = np.cumsum(episode_ends)
        self.episode_ends = episode_ends

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        # normalized_train_data = dict()
        for key, data in train_data.items():
            if key == "images" or key in self.unnormal_list:
                pass
            else:
                stats[key] = get_data_stats(data)

            if key == "images" or key in self.unnormal_list:
                pass
            else:
                train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        # self.normalized_train_data = normalized_train_data
        self.normalized_train_data = train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        if proto_horizon is None:
            self.proto_horizon = obs_horizon
        else:
            self.proto_horizon = proto_horizon

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
            from xskill.utility.transform import get_transform_pipeline

            return get_transform_pipeline(list(pipeline))
        return pipeline

    def _load_mask(self, mask_value):
        if mask_value is None:
            return None
        if isinstance(mask_value, (list, tuple, np.ndarray)):
            return list(mask_value)
        with open(mask_value, "r") as f:
            return json.load(f)

    def _normalize_proto_dirs(self, proto_dirs):
        if isinstance(proto_dirs, (str, Path)):
            return [str(proto_dirs)] * len(self.data_dirs)

        proto_dir_list = [str(path) for path in proto_dirs]
        if len(proto_dir_list) == 1 and len(self.data_dirs) > 1:
            proto_dir_list = proto_dir_list * len(self.data_dirs)
        if len(proto_dir_list) != len(self.data_dirs):
            raise ValueError(
                "proto_dirs must be a path or a list aligned with data_dirs"
            )
        return proto_dir_list

    def _normalize_masks(self, mask):
        if mask is None:
            return [None] * len(self.data_dirs)

        if isinstance(mask, (str, Path)):
            return [self._load_mask(mask)] * len(self.data_dirs)

        if isinstance(mask, (list, tuple, np.ndarray)) and mask and all(
            isinstance(value, (bool, np.bool_, int, np.integer)) for value in mask
        ):
            return [self._load_mask(mask)] * len(self.data_dirs)

        mask_list = list(mask)
        if len(mask_list) == 1 and len(self.data_dirs) > 1:
            mask_list = mask_list * len(self.data_dirs)
        if len(mask_list) != len(self.data_dirs):
            raise ValueError("mask must be a path or a list aligned with data_dirs")
        return [self._load_mask(value) for value in mask_list]

    def _resolve_zarr_path(self, data_dir):
        data_path = Path(data_dir)
        if data_path.suffix == ".zarr" and data_path.exists():
            return data_path

        zarr_path = Path(f"{data_dir}.zarr")
        if zarr_path.exists():
            return zarr_path
        return None

    def _resolve_episode_indices(self, replay_buffer):
        if "episode_indices" in replay_buffer.meta:
            return np.asarray(replay_buffer.meta["episode_indices"][:], dtype=np.int64)
        return np.arange(replay_buffer.n_episodes, dtype=np.int64)

    def _resolve_proto_zarr_path(self, proto_dir, data_dir):
        proto_path = Path(proto_dir)
        if proto_path.suffix == ".zarr" and proto_path.exists():
            return proto_path

        data_path = Path(data_dir)
        data_name = data_path.stem if data_path.suffix == ".zarr" else data_path.name
        candidate = proto_path / f"{data_name}.zarr"
        if candidate.exists():
            return candidate
        return None

    def _get_episode_slice(self, episode_ends, vid, episode_indices=None):
        episode_name = Path(vid).name
        if not episode_name.isdigit():
            raise ValueError(
                f"Expected numeric episode directory when using zarr cache, got {episode_name}."
            )
        episode_idx = int(episode_name)

        episode_position = episode_idx
        if episode_indices is not None:
            episode_indices = np.asarray(episode_indices, dtype=np.int64)
            matches = np.flatnonzero(episode_indices == episode_idx)
            if len(matches) == 0:
                raise IndexError(
                    f"Episode index {episode_idx} is not present in replay metadata {episode_indices.tolist()}."
                )
            episode_position = int(matches[0])

        if episode_position >= len(episode_ends):
            raise IndexError(
                f"Episode index {episode_idx} is out of bounds for zarr cache with {len(episode_ends)} episodes."
            )
        start_idx = 0 if episode_position == 0 else int(episode_ends[episode_position - 1])
        end_idx = int(episode_ends[episode_position])
        return start_idx, end_idx

    def _resolve_replay_array(self, replay_buffer, candidate_keys):
        for key in candidate_keys:
            if key in replay_buffer:
                return replay_buffer[key]
        raise KeyError(
            f"None of the replay buffer keys {candidate_keys} were found. Available keys: {list(replay_buffer.keys())}"
        )

    def _get_replay_meta_strings(self, replay_buffer, key):
        if key not in replay_buffer.meta:
            return []
        values = np.asarray(replay_buffer.meta[key])
        if values.ndim == 0:
            return [str(values.item())]
        return [str(value) for value in values.tolist()]

    def _camera_view_to_replay_keys(self, camera_view, replay_buffer=None):
        if camera_view in (None, "", ".", "root", "primary"):
            candidate_keys = ["camera_primary"]
            if replay_buffer is not None:
                camera_stems = self._get_replay_meta_strings(replay_buffer, "camera_file_stems")
                if camera_stems:
                    candidate_keys.append(f"camera_{camera_stems[0]}")
            candidate_keys.extend(["camera_0", "camera_2", "images"])
            return candidate_keys
        return [f"camera_{camera_view}", str(camera_view)]

    def _build_dir_tree(self):
        """Build a dict of indices for iterating over the dataset."""
        self._dir_tree = collections.OrderedDict()
        for data_dir, proto_dir, mask in zip(self.data_dirs, self.proto_dirs, self.masks):
            replay_buffer = None
            episode_ends = None
            episode_indices = None
            zarr_path = self._resolve_zarr_path(data_dir)

            if zarr_path is not None:
                replay_buffer = ReplayBuffer.create_from_path(str(zarr_path))
                episode_ends = replay_buffer.episode_ends[:]
                episode_indices = self._resolve_episode_indices(replay_buffer)
                synthetic_root = Path(data_dir).with_suffix("")
                vids = np.array([str(synthetic_root / str(index)) for index in episode_indices])
            else:
                vids = get_subdirs(
                    data_dir,
                    nonempty=False,
                    sort_numerical=True,
                )
            if len(vids) > 0:
                vids = np.array(vids)
                if mask is not None:
                    if len(mask) != len(vids):
                        raise ValueError(
                            f"Mask length {len(mask)} does not match number of episodes {len(vids)} in {data_dir}"
                        )
                    vids = vids[np.array(mask, dtype=bool)]
                entry = {
                    "vids": vids,
                    "proto_dir": proto_dir,
                }
                if replay_buffer is not None:
                    entry["replay_buffer"] = replay_buffer
                    entry["episode_ends"] = episode_ends
                    entry["episode_indices"] = episode_indices

                proto_zarr_path = self._resolve_proto_zarr_path(proto_dir, data_dir)
                if proto_zarr_path is not None:
                    proto_replay_buffer = ReplayBuffer.create_from_path(str(proto_zarr_path))
                    entry["proto_replay_buffer"] = proto_replay_buffer
                    entry["proto_episode_ends"] = proto_replay_buffer.episode_ends[:]
                    entry["proto_episode_indices"] = self._resolve_episode_indices(proto_replay_buffer)
                self._dir_tree[data_dir] = entry

    def load_action_and_to_tensor(self, vid, replay_buffer=None, episode_ends=None, episode_indices=None):
        if replay_buffer is not None:
            start_idx, end_idx = self._get_episode_slice(episode_ends, vid, episode_indices)
            action_arr = self._resolve_replay_array(replay_buffer, ["actions", "action"])
            return np.asarray(action_arr[start_idx:end_idx], dtype=np.float32)

        action_path = os.path.join(vid, "actions.json")
        with open(action_path, "r") as f:
            action_data = json.load(f)
        action_data = np.array(action_data)
        action_data = np.array(action_data, dtype=np.float32)
        return action_data

    def load_state_and_to_tensor(self, vid, replay_buffer=None, episode_ends=None, episode_indices=None):
        if replay_buffer is not None:
            start_idx, end_idx = self._get_episode_slice(episode_ends, vid, episode_indices)
            obs_arr = self._resolve_replay_array(replay_buffer, ["obs", "states", "state"])
            return np.asarray(obs_arr[start_idx:end_idx], dtype=np.float32)

        state_path = os.path.join(vid, "states.json")
        with open(state_path, "r") as f:
            state_data = json.load(f)
        state_data = np.array(state_data, dtype=np.float32)
        return state_data

    def load_proto_and_to_tensor(
        self,
        vid,
        proto_dir=None,
        data_dir=None,
        proto_replay_buffer=None,
        proto_episode_ends=None,
        proto_episode_indices=None,
    ):
        if proto_replay_buffer is not None:
            start_idx, end_idx = self._get_episode_slice(
                proto_episode_ends,
                vid,
                proto_episode_indices,
            )
            if self.raw_representation:
                proto_keys = ["traj_representation", "raw_rep", "raw_representation"]
            elif self.softmax_prototype or self.one_hot_prototype:
                proto_keys = ["softmax_encode_protos", "softmax_prototypes"]
            elif self.prototype:
                proto_keys = ["encode_protos", "prototypes"]
            else:
                raise ValueError("Unsupported prototype mode.")

            proto_data = np.asarray(
                self._resolve_replay_array(proto_replay_buffer, proto_keys)[start_idx:end_idx],
                dtype=np.float32,
            )
            if self.one_hot_prototype:
                one_hot_proto = np.zeros_like(proto_data)
                max_proto = np.argmax(proto_data, axis=1)
                one_hot_proto[np.arange(len(proto_data)), max_proto] = 1
                proto_data = one_hot_proto

            if self.prototype_snap:
                eps_len = len(proto_data)
                snap_idx = random.sample(list(range(eps_len)), k=self.snap_frames)
                snap_idx.sort()
                snap = proto_data[snap_idx]
                snap = snap.flatten()
                snap = np.tile(snap, (eps_len, 1))
                return proto_data, snap

            return proto_data

        if proto_dir is not None and data_dir is not None:
            relative_vid = osp.relpath(vid, data_dir)
            proto_path = osp.join(proto_dir, relative_vid)
        else:
            proto_path = osp.join(self.proto_dirs[0], os.path.basename(os.path.normpath(vid)))
        if self.raw_representation:
            proto_path = os.path.join(proto_path, "traj_representation.json")
        elif self.softmax_prototype or self.one_hot_prototype:
            proto_path = os.path.join(proto_path, "softmax_encode_protos.json")
        elif self.prototype:
            proto_path = os.path.join(proto_path, "encode_protos.json")

        with open(proto_path, "r") as f:
            proto_data = json.load(f)
        proto_data = np.array(proto_data, dtype=np.float32)  # (T,D)
        if self.one_hot_prototype:
            one_hot_proto = np.zeros_like(proto_data)
            max_proto = np.argmax(proto_data, axis=1)
            one_hot_proto[np.arange(len(proto_data)), max_proto] = 1
            proto_data = one_hot_proto

        if self.prototype_snap:
            eps_len = len(proto_data)
            snap_idx = random.sample(list(range(eps_len)), k=self.snap_frames)
            snap_idx.sort()
            snap = proto_data[snap_idx]
            snap = snap.flatten()
            snap = np.tile(snap, (eps_len, 1))  # (T,snap_frams*model_dim)
            return proto_data, snap

        return proto_data

    def _resolve_camera_dir(self, vid, camera_view):
        if camera_view in (None, "", ".", "root", "primary"):
            return Path(vid)
        return Path(vid) / str(camera_view)

    def _load_single_camera_images(self, camera_dir):
        images = []
        filenames = sorted(
            [f for f in os.listdir(camera_dir) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        if not filenames:
            raise RuntimeError(f"No PNG files found in camera directory {camera_dir}")

        for filename in filenames:
            img = Image.open(os.path.join(camera_dir, filename))
            img_arr = np.array(img)
            if self.resize_shape is not None:
                img_arr = cv2.resize(img_arr, self.resize_shape)
            images.append(img_arr)

        images_arr = np.array(images)
        assert images_arr.dtype == np.uint8
        return images_arr

    def load_images(self, vid, replay_buffer=None, episode_ends=None, episode_indices=None):
        if replay_buffer is not None:
            start_idx, end_idx = self._get_episode_slice(episode_ends, vid, episode_indices)
            multi_view_images = []

            for camera_view in self.camera_views:
                image_arr = self._resolve_replay_array(
                    replay_buffer,
                    self._camera_view_to_replay_keys(camera_view, replay_buffer=replay_buffer),
                )
                camera_images = np.asarray(image_arr[start_idx:end_idx], dtype=np.uint8)
                if self.resize_shape is not None and camera_images.shape[1:3] != (
                    self.resize_shape[1],
                    self.resize_shape[0],
                ):
                    camera_images = np.stack(
                        [cv2.resize(frame, self.resize_shape) for frame in camera_images],
                        axis=0,
                    )
                multi_view_images.append(camera_images)

            if len(multi_view_images) == 1:
                return multi_view_images[0]
            return np.stack(multi_view_images, axis=1)

        multi_view_images = []
        expected_len = None

        for camera_view in self.camera_views:
            camera_dir = self._resolve_camera_dir(vid, camera_view)
            camera_images = self._load_single_camera_images(camera_dir)
            if expected_len is None:
                expected_len = len(camera_images)
            elif len(camera_images) != expected_len:
                raise ValueError(
                    f"Camera view {camera_view} in {vid} has {len(camera_images)} frames, expected {expected_len}."
                )
            multi_view_images.append(camera_images)

        if len(multi_view_images) == 1:
            return multi_view_images[0]

        return np.stack(multi_view_images, axis=1)

    def transform_images(self, images_arr):
        images_tensor = torch.tensor(images_arr, dtype=torch.float32) / 255.0
        if images_tensor.ndim == 4:
            images_tensor = images_tensor.permute(0, 3, 1, 2)
        elif images_tensor.ndim == 5:
            images_tensor = images_tensor.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Unsupported image tensor shape: {tuple(images_tensor.shape)}")

        if self.pipeline is not None:
            images_tensor = self.pipeline(images_tensor)
        return images_tensor

    def load_data(self, train_data):
        print("loading data")
        for data_dir, entry in self._dir_tree.items():
            vids = entry["vids"]
            proto_dir = entry["proto_dir"]
            replay_buffer = entry.get("replay_buffer")
            episode_ends = entry.get("episode_ends")
            episode_indices = entry.get("episode_indices")
            proto_replay_buffer = entry.get("proto_replay_buffer")
            proto_episode_ends = entry.get("proto_episode_ends")
            proto_episode_indices = entry.get("proto_episode_indices")
            for _, vid in tqdm(
                enumerate(vids),
                desc=f"Loading data from {os.path.basename(data_dir)}",
                disable=not self.verbose,
            ):
                if self.obs_image_based:
                    images = self.load_images(
                        vid,
                        replay_buffer=replay_buffer,
                        episode_ends=episode_ends,
                        episode_indices=episode_indices,
                    )
                    train_data["images"].append(images)

                train_data["obs"].append(
                    self.load_state_and_to_tensor(
                        vid,
                        replay_buffer=replay_buffer,
                        episode_ends=episode_ends,
                        episode_indices=episode_indices,
                    )
                )
                if self.prototype_snap:
                    proto_data, proto_snap = self.load_proto_and_to_tensor(
                        vid,
                        proto_dir=proto_dir,
                        data_dir=data_dir,
                        proto_replay_buffer=proto_replay_buffer,
                        proto_episode_ends=proto_episode_ends,
                        proto_episode_indices=proto_episode_indices,
                    )
                    train_data["proto_snap"].append(proto_snap)
                else:
                    proto_data = self.load_proto_and_to_tensor(
                        vid,
                        proto_dir=proto_dir,
                        data_dir=data_dir,
                        proto_replay_buffer=proto_replay_buffer,
                        proto_episode_ends=proto_episode_ends,
                        proto_episode_indices=proto_episode_indices,
                    )

                train_data["protos"].append(proto_data)
                train_data["actions"].append(
                    self.load_action_and_to_tensor(
                        vid,
                        replay_buffer=replay_buffer,
                        episode_ends=episode_ends,
                        episode_indices=episode_indices,
                    )
                )

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        if self.prototype_snap:
            # set as prediction target
            nsample["protos"] = nsample["protos"][: self.obs_horizon, :]
            # most recent prototype
            nsample["protos"] = nsample["protos"][-1:, :]
            # duplicate. only take one
            nsample["proto_snap"] = nsample["proto_snap"][-1:, :]
        else:
            nsample["protos"] = nsample["protos"][: self.obs_horizon, :]
            nsample["protos"] = nsample["protos"][-self.proto_horizon :, :]

        if self.obs_image_based:
            nsample["images"] = self.transform_images(nsample["images"])
            nsample["images"] = nsample["images"][: self.obs_horizon, :]
            nsample["obs"] = nsample["obs"][: self.obs_horizon, :]

        return nsample