import json
import os
from pathlib import Path
import concurrent.futures
import shutil

import cv2
import hydra
import numpy as np
import omegaconf
import torch
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

from xskill.common.replay_buffer import ReplayBuffer
from xskill.utility.transform import get_transform_pipeline


def repeat_last_proto(encode_protos, eps_len):
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(
        eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])


def load_model(cfg):
    exp_cfg = omegaconf.OmegaConf.load(
        os.path.join(cfg.exp_path, '.hydra/config.yaml'))
    model = hydra.utils.instantiate(exp_cfg.Model).to(cfg.device)

    loadpath = os.path.join(cfg.exp_path, f'epoch={cfg.ckpt}.ckpt')
    checkpoint = torch.load(loadpath, map_location=cfg.device)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(cfg.device)
    model.eval()
    print("model loaded")
    return model


def convert_images_to_tensors(images_arr, pipeline):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    images_tensor = pipeline(images_tensor)

    return images_tensor


def load_episode_mask(mask_path):
    if mask_path is None:
        return None
    with open(mask_path, "r") as f:
        return json.load(f)


def get_episode_dirs(data_dir, mask_path=None):
    episode_dirs = sorted(
        [path for path in Path(data_dir).iterdir() if path.is_dir()],
        key=lambda path: int(path.name),
    )
    mask = load_episode_mask(mask_path)
    if mask is not None:
        if len(mask) != len(episode_dirs):
            raise ValueError(
                f"Mask length {len(mask)} does not match number of episodes {len(episode_dirs)} in {data_dir}"
            )
        episode_dirs = [path for path, keep in zip(episode_dirs, mask) if keep]
    return episode_dirs


def load_episode_images(episode_dir, resize_shape=None):
    frame_paths = sorted(episode_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in {episode_dir}")

    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_shape is not None:
            frame = cv2.resize(frame, resize_shape)
        frames.append(frame)

    return np.stack(frames)


def resolve_dataset_zarr_path(data_dir):
    data_dir = Path(data_dir)
    if data_dir.suffix == ".zarr" and data_dir.exists():
        return data_dir

    zarr_path = Path(f"{data_dir}.zarr")
    if zarr_path.exists():
        return zarr_path
    return None


def get_replay_meta_strings(replay_buffer, key):
    if key not in replay_buffer.meta:
        return []
    values = np.asarray(replay_buffer.meta[key])
    if values.ndim == 0:
        return [str(values.item())]
    return [str(value) for value in values.tolist()]


def get_primary_camera_key(replay_buffer):
    camera_stems = get_replay_meta_strings(replay_buffer, "camera_file_stems")
    if camera_stems:
        camera_key = f"camera_{camera_stems[0]}"
        if camera_key in replay_buffer:
            return camera_key

    for key in replay_buffer.keys():
        if key.startswith("camera_"):
            return key
    raise KeyError(f"No camera_* array found in replay buffer keys: {list(replay_buffer.keys())}")


def load_zarr_episode_images(replay_buffer, episode_idx, camera_key, resize_shape=None):
    episode_ends = replay_buffer.episode_ends[:]
    eps_begin = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    eps_end = int(episode_ends[episode_idx])
    sample = np.arange(eps_end - eps_begin)
    return get_sequence_data(sample, replay_buffer[camera_key], eps_begin, resize_shape)


def prepare_proto_replay_buffer(save_path):
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(save_path)))


def add_episode_outputs(proto_buffer, episode_outputs):
    proto_buffer.add_episode(
        {
            "encode_protos": episode_outputs["encode_protos"],
            "softmax_encode_protos": episode_outputs["softmax_encode_protos"],
            "traj_representation": episode_outputs["traj_representation"],
        }
    )


def encode_episode(images_arr, model, pipeline, device):
    images_tensor = convert_images_to_tensors(images_arr, pipeline).to(device)
    eps_len = images_tensor.shape[0]
    if eps_len <= model.slide:
        raise ValueError(
            f"Episode only has {eps_len} frames, but model.slide={model.slide} requires at least {model.slide + 1}"
        )

    im_q = torch.stack(
        [images_tensor[j:j + model.slide + 1] for j in range(eps_len - model.slide)]
    )

    with torch.no_grad():
        z = model.encoder_q(im_q, None)
        softmax_z = torch.softmax(z / model.T, dim=1)

        state_representation = model.encoder_q.get_state_representation(im_q, None)
        traj_representation = model.encoder_q.get_traj_representation(
            state_representation
        )

        encode_protos = repeat_last_proto(z, eps_len)
        softmax_encode_protos = repeat_last_proto(softmax_z, eps_len)
        traj_representation = repeat_last_proto(traj_representation, eps_len)

    return {
        "encode_protos": encode_protos.detach().cpu().numpy().astype(np.float32),
        "softmax_encode_protos": softmax_encode_protos.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "traj_representation": traj_representation.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def save_episode_outputs(save_folder, episode_outputs):
    save_folder.mkdir(parents=True, exist_ok=True)

    with open(save_folder / "encode_protos.json", "w") as f:
        json.dump(episode_outputs["encode_protos"].tolist(), f)

    with open(save_folder / "softmax_encode_protos.json", "w") as f:
        json.dump(episode_outputs["softmax_encode_protos"].tolist(), f)

    with open(save_folder / "traj_representation.json", "w") as f:
        json.dump(episode_outputs["traj_representation"].tolist(), f)


def get_sequence_data(sample, image_zarr, eps_begin, resize_shape=None):
    sample_index = list(np.array(sample) + eps_begin)
    frames = [None for _ in range(len(sample))]

    def get_image(image_index, sample_index, frames, image_zarr, resize_shape):
        try:
            if resize_shape is not None:
                frame = image_zarr[sample_index]
                resized_frames = cv2.resize(frame, resize_shape)
                frames[image_index] = resized_frames
            else:
                frames[image_index] = image_zarr[sample_index]
            return True
        except Exception as e:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = set()
        for i, idx in enumerate(sample_index):
            futures.add(
                executor.submit(get_image, i, idx, frames, image_zarr,
                                resize_shape))

        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to get image!')

    sequence_data = np.stack(frames)  # Shape: (S * X, H, W, C)
    return sequence_data


def label_episode_dir_dataset(cfg, model, pipeline):
    device = torch.device(cfg.device)
    mask_paths = cfg.get("mask_paths", {})

    for demo_type in ["human", "robot"]:
        data_dir = Path(cfg.data_path) / demo_type
        save_root = Path(cfg.save_path)
        if demo_type == "human":
            save_root = save_root / "human_encode_protos" / f"ckpt_{cfg.ckpt}"
        else:
            save_root = save_root / "encode_protos" / f"ckpt_{cfg.ckpt}"

        proto_buffer = prepare_proto_replay_buffer(save_root / f"{demo_type}.zarr")
        saved_episode_indices = []
        zarr_path = resolve_dataset_zarr_path(data_dir)

        if zarr_path is not None:
            replay_buffer = ReplayBuffer.create_from_path(str(zarr_path))
            episode_indices = np.arange(replay_buffer.n_episodes, dtype=np.int64)
            mask = load_episode_mask(mask_paths.get(demo_type))
            if mask is not None:
                if len(mask) != len(episode_indices):
                    raise ValueError(
                        f"Mask length {len(mask)} does not match number of episodes {len(episode_indices)} in {zarr_path}"
                    )
                episode_indices = episode_indices[np.asarray(mask, dtype=bool)]

            camera_key = get_primary_camera_key(replay_buffer)
            for episode_idx in tqdm(episode_indices, desc=f"labelling {demo_type} episodes"):
                episode_outputs = encode_episode(
                    load_zarr_episode_images(replay_buffer, int(episode_idx), camera_key, cfg.resize_shape),
                    model,
                    pipeline,
                    device,
                )
                add_episode_outputs(proto_buffer, episode_outputs)
                saved_episode_indices.append(int(episode_idx))
        else:
            episode_dirs = get_episode_dirs(data_dir, mask_paths.get(demo_type))
            for episode_dir in tqdm(episode_dirs, desc=f"labelling {demo_type} episodes"):
                episode_outputs = encode_episode(
                    load_episode_images(episode_dir, cfg.resize_shape),
                    model,
                    pipeline,
                    device,
                )
                add_episode_outputs(proto_buffer, episode_outputs)
                saved_episode_indices.append(int(episode_dir.name))

        proto_buffer.update_meta(
            {
                "episode_indices": np.asarray(saved_episode_indices, dtype=np.int64),
            }
        )


def label_replay_buffer_dataset(cfg, model, pipeline, robot_dataset, human_dataset):
    device = torch.device(cfg.device)
    save_path = os.path.join(cfg.save_path, f'ckpt_{cfg.ckpt}', 'prototype.zarr')
    prototype_store = zarr.DirectoryStore(save_path)
    prototype_zarr = zarr.group(prototype_store)

    for embodiment in ['human', 'robot']:
        dataset_to_label = robot_dataset if embodiment == 'robot' else human_dataset
        for key, zarr_data in tqdm(dataset_to_label.in_replay_buffer.items(), desc="labelling task"):
            eps_end = zarr_data['/meta/episode_ends'][:]
            image_zarr = zarr_data['/data/camera_2']
            z_store = []
            softmax_z_store = []
            raw_rep_store = []
            for i in tqdm(range(len(eps_end)), desc="labelling episode"):
                if i == 0:
                    eps_start_index = 0
                    eps_end_index = eps_end[i]
                else:
                    eps_start_index = eps_end[i - 1]
                    eps_end_index = eps_end[i]

                sample = np.arange(eps_end_index - eps_start_index)[::cfg.frequency]
                images_arr = get_sequence_data(sample, image_zarr, eps_start_index, cfg.resize_shape)
                episode_outputs = encode_episode(images_arr, model, pipeline, device)

                z_store.append(episode_outputs['encode_protos'])
                softmax_z_store.append(episode_outputs['softmax_encode_protos'])
                raw_rep_store.append(episode_outputs['traj_representation'])

            eps_end = np.cumsum([len(zs) for zs in z_store]).astype(np.int64)
            z_store = np.concatenate(z_store).astype(np.float32)
            softmax_z_store = np.concatenate(softmax_z_store).astype(np.float32)
            raw_rep_store = np.concatenate(raw_rep_store).astype(np.float32)
            save_key = Path(key).name

            prototype_zarr.require_dataset(
                f'{embodiment}/{save_key}/prototypes',
                shape=z_store.shape,
                dtype=np.float32,
            )
            prototype_zarr.require_dataset(
                f'{embodiment}/{save_key}/softmax_prototypes',
                shape=softmax_z_store.shape,
                dtype=np.float32,
            )
            prototype_zarr.require_dataset(
                f'{embodiment}/{save_key}/raw_rep',
                shape=raw_rep_store.shape,
                dtype=np.float32,
            )
            prototype_zarr.require_dataset(
                f'{embodiment}/{save_key}/eps_end',
                shape=eps_end.shape,
                dtype=np.int64,
            )

            prototype_zarr[f'{embodiment}/{save_key}/prototypes'] = z_store
            prototype_zarr[f'{embodiment}/{save_key}/softmax_prototypes'] = softmax_z_store
            prototype_zarr[f'{embodiment}/{save_key}/raw_rep'] = raw_rep_store
            prototype_zarr[f'{embodiment}/{save_key}/eps_end'] = eps_end


@hydra.main(version_base=None,
            config_path="../../config/realworld",
            config_name="label_real_kitchen_dataset")
def label_dataset(cfg: DictConfig):
    model = load_model(cfg)
    pretrain_pipeline = get_transform_pipeline(cfg.augmentations)

    if cfg.get("data_path") is not None:
        label_episode_dir_dataset(cfg, model, pretrain_pipeline)
        return

    robot_dataset = hydra.utils.instantiate(cfg.robot_dataset)
    human_dataset = hydra.utils.instantiate(cfg.human_dataset)
    label_replay_buffer_dataset(
        cfg,
        model,
        pretrain_pipeline,
        robot_dataset,
        human_dataset,
    )


if __name__ == "__main__":
    label_dataset()
