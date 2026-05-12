import json
import shutil
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import zarr
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from xskill.common.replay_buffer import ReplayBuffer
from xskill.utility.transform import get_transform_pipeline


def repeat_last_proto(proto_tensor, episode_len):
    if len(proto_tensor) >= episode_len:
        return proto_tensor
    repeated = proto_tensor[-1:].repeat(episode_len - len(proto_tensor), 1)
    return torch.cat([proto_tensor, repeated], dim=0)


def load_model(cfg):
    exp_cfg = OmegaConf.load(Path(cfg.exp_path) / ".hydra" / "config.yaml")
    model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)
    checkpoint = torch.load(
        Path(cfg.exp_path) / f"epoch={cfg.ckpt}.ckpt",
        map_location=cfg.device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def convert_images_to_tensors(images, pipeline):
    images = np.transpose(images, (0, 3, 1, 2))
    tensor = torch.tensor(images, dtype=torch.float32) / 255.0
    return pipeline(tensor)


def load_mask(mask_path):
    if mask_path is None:
        return None
    with open(mask_path, "r") as file:
        return json.load(file)


def prepare_proto_buffer(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(path)))


def episode_indices(replay_buffer):
    if "episode_indices" in replay_buffer.meta:
        return np.asarray(replay_buffer.meta["episode_indices"][:], dtype=np.int64)
    return np.arange(replay_buffer.n_episodes, dtype=np.int64)


def episode_bounds(episode_ends, episode_position):
    episode_end = int(episode_ends[episode_position])
    episode_begin = 0 if episode_position == 0 else int(episode_ends[episode_position - 1])
    return episode_begin, episode_end


def load_episode_images(replay_buffer, camera_key, episode_begin, episode_end, resize_shape):
    images = np.asarray(replay_buffer[camera_key][episode_begin:episode_end], dtype=np.uint8)
    if resize_shape is not None and images.shape[1:3] != (resize_shape[1], resize_shape[0]):
        images = np.stack([cv2.resize(frame, resize_shape) for frame in images], axis=0)
    return images


def encode_episode(images, model, pipeline, device):
    images_tensor = convert_images_to_tensors(images, pipeline).to(device)
    episode_len = images_tensor.shape[0]
    if episode_len <= model.slide:
        raise ValueError(
            f"Episode only has {episode_len} frames, but model.slide={model.slide} requires at least {model.slide + 1}."
        )

    im_q = torch.stack(
        [images_tensor[index : index + model.slide + 1] for index in range(episode_len - model.slide)]
    )

    with torch.inference_mode():
        encode_protos = model.encoder_q(im_q, None)
        softmax_encode_protos = torch.softmax(encode_protos / model.T, dim=1)
        state_representation = model.encoder_q.get_state_representation(im_q, None)
        traj_representation = model.encoder_q.get_traj_representation(state_representation)

    return {
        "encode_protos": repeat_last_proto(encode_protos, episode_len).cpu().numpy().astype(np.float32),
        "softmax_encode_protos": repeat_last_proto(softmax_encode_protos, episode_len)
        .cpu()
        .numpy()
        .astype(np.float32),
        "traj_representation": repeat_last_proto(traj_representation, episode_len)
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def label_dataset_zarr(dataset_cfg, model, pipeline, device):
    replay_buffer = ReplayBuffer.create_from_path(str(dataset_cfg.data_path), mode="r")
    proto_buffer = prepare_proto_buffer(Path(dataset_cfg.save_path))
    camera_key = f"camera_{dataset_cfg.camera_view}"
    if camera_key not in replay_buffer:
        raise KeyError(
            f"{dataset_cfg.data_path} does not contain {camera_key}. Available keys: {list(replay_buffer.keys())}"
        )

    raw_episode_indices = episode_indices(replay_buffer)
    selected_positions = np.arange(replay_buffer.n_episodes, dtype=np.int64)
    mask = load_mask(dataset_cfg.get("mask_path"))
    if mask is not None:
        if len(mask) != len(selected_positions):
            raise ValueError(
                f"Mask length {len(mask)} does not match number of episodes {len(selected_positions)} in {dataset_cfg.data_path}."
            )
        selected_positions = selected_positions[np.asarray(mask, dtype=bool)]

    saved_episode_indices = []
    episode_ends = replay_buffer.episode_ends[:]
    for episode_position in tqdm(selected_positions.tolist(), desc=f"Labelling {dataset_cfg.name}"):
        episode_begin, episode_end = episode_bounds(episode_ends, episode_position)
        images = load_episode_images(
            replay_buffer,
            camera_key,
            episode_begin,
            episode_end,
            tuple(dataset_cfg.resize_shape) if dataset_cfg.get("resize_shape") else None,
        )
        outputs = encode_episode(images, model, pipeline, device)
        proto_buffer.add_episode(outputs, compressors="disk")
        saved_episode_indices.append(int(raw_episode_indices[episode_position]))

    proto_buffer.update_meta({"episode_indices": np.asarray(saved_episode_indices, dtype=np.int64)})


@hydra.main(
    version_base=None,
    config_path="../config/realworld",
    config_name="label_skill",
)
def label_dataset(cfg: DictConfig):
    model = load_model(cfg)
    pipeline = get_transform_pipeline(cfg.augmentations)
    device = torch.device(cfg.device)

    for dataset_name, dataset_cfg in cfg.datasets.items():
        dataset_cfg = OmegaConf.merge(dataset_cfg, {"name": dataset_name, "resize_shape": cfg.resize_shape})
        label_dataset_zarr(dataset_cfg, model, pipeline, device)


if __name__ == "__main__":
    label_dataset()
