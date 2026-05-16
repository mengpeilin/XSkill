from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import random
import sys
from collections import deque

import cv2
import hydra
import numpy as np
import torch
import torch.nn as nn
import zmq
from omegaconf import OmegaConf


def _find_xskill_root(script_path: pathlib.Path) -> pathlib.Path | None:
	for candidate in [script_path.parent, *script_path.parents]:
		if (candidate / "xskill").exists():
			return candidate
	return None


WORKSPACE_ROOT = _find_xskill_root(pathlib.Path(__file__).resolve())
if WORKSPACE_ROOT is not None:
	xskill_root_str = str(WORKSPACE_ROOT)
	if xskill_root_str not in sys.path:
		sys.path.insert(0, xskill_root_str)

from xskill.common.replay_buffer import ReplayBuffer
from xskill.dataset.diffusion_bc_dataset import normalize_data, unnormalize_data
from xskill.model.diffusion_model import get_resnet, replace_bn_with_gn
from xskill.model.encoder import ResnetConv
from xskill.utility.transform import get_transform_pipeline


CRITICAL_CONFIG_FIELDS = (
	"obs_dim",
	"action_dim",
	"proto_dim",
	"vision_feature_dim",
	"obs_horizon",
	"pred_horizon",
	"action_horizon",
	"proto_horizon",
	"bc_resize",
	"inference_augmentations",
	"raw_representation",
	"prototype",
	"softmax_prototype",
	"one_hot_prototype",
	"upsample_proto",
	"dataset.camera_views",
	"dataset.snap_frames",
	"num_diffusion_iters",
)

REQUIRED_BC_CAMERA_VIEWS = ["cam0", "cam1", "wrist_cam"]


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def _normalize_config_value(field_name: str, value):
	if OmegaConf.is_config(value):
		value = OmegaConf.to_container(value, resolve=True)
	if isinstance(value, str) and field_name.endswith("_path"):
		return os.path.abspath(os.path.expanduser(value))
	if isinstance(value, list):
		return [_normalize_config_value(field_name, item) for item in value]
	return value


def _collect_config_mismatches(requested_cfg, checkpoint_cfg) -> list[str]:
	mismatches = []
	for field_name in CRITICAL_CONFIG_FIELDS:
		requested_value = _normalize_config_value(
			field_name,
			OmegaConf.select(requested_cfg, field_name, default=None),
		)
		checkpoint_value = _normalize_config_value(
			field_name,
			OmegaConf.select(checkpoint_cfg, field_name, default=None),
		)
		if requested_value != checkpoint_value:
			mismatches.append(
				f"{field_name}: requested={requested_value!r}, checkpoint={checkpoint_value!r}"
			)
	return mismatches


def resolve_runtime_config(args):
	requested_cfg = OmegaConf.load(args.config_path) if args.config_path is not None else None
	checkpoint_cfg_path = pathlib.Path(args.ckpt_path).resolve().parent / "hydra_config.yaml"
	if checkpoint_cfg_path.exists():
		checkpoint_cfg = OmegaConf.load(checkpoint_cfg_path)
		if requested_cfg is not None:
			requested_cfg_path = pathlib.Path(args.config_path).resolve()
			if requested_cfg_path != checkpoint_cfg_path.resolve():
				mismatches = _collect_config_mismatches(requested_cfg, checkpoint_cfg)
				if mismatches:
					print(
						"Warning: --config_path does not match the checkpoint training config. "
						f"Using {checkpoint_cfg_path} instead."
					)
					for mismatch in mismatches:
						print(f"  - {mismatch}")
		return checkpoint_cfg, str(checkpoint_cfg_path)
	if requested_cfg is None:
		raise ValueError(
			"Could not find hydra_config.yaml next to the checkpoint and no --config_path was provided."
		)
	return requested_cfg, os.path.abspath(args.config_path)


def sample_proto_snap(proto_sequence: np.ndarray, snap_frames: int, device: torch.device):
	if len(proto_sequence) == 0:
		raise ValueError("Prototype sequence is empty.")
	sample_count = min(snap_frames, len(proto_sequence))
	sample_indices = sorted(random.sample(list(range(len(proto_sequence))), k=sample_count))
	snap = np.asarray(proto_sequence[sample_indices], dtype=np.float32)
	if sample_count < snap_frames:
		pad = np.repeat(snap[-1:], snap_frames - sample_count, axis=0)
		snap = np.concatenate([snap, pad], axis=0)
	return torch.from_numpy(snap).to(device=device, dtype=torch.float32).unsqueeze(0)


def proto_sequence_key(cfg):
	if cfg.get("raw_representation"):
		return "traj_representation"
	if cfg.get("softmax_prototype") or cfg.get("one_hot_prototype"):
		return "softmax_encode_protos"
	if cfg.get("prototype"):
		return "encode_protos"
	raise ValueError("One prototype mode must be enabled.")


def load_proto_snap(cfg, proto_path: str, demo_episode: int, device: torch.device):
	replay_buffer = ReplayBuffer.create_from_path(proto_path, mode="r")
	episode_indices = np.arange(replay_buffer.n_episodes, dtype=np.int64)
	if "episode_indices" in replay_buffer.meta:
		episode_indices = np.asarray(replay_buffer.meta["episode_indices"][:], dtype=np.int64)
	matches = np.flatnonzero(episode_indices == demo_episode)
	if len(matches) == 0:
		raise KeyError(f"Episode index {demo_episode} is not present in {proto_path}.")
	episode_position = int(matches[0])
	episode_ends = replay_buffer.episode_ends[:]
	episode_begin = 0 if episode_position == 0 else int(episode_ends[episode_position - 1])
	episode_end = int(episode_ends[episode_position])
	proto_sequence = np.asarray(
		replay_buffer[proto_sequence_key(cfg)][episode_begin:episode_end],
		dtype=np.float32,
	)
	if cfg.get("one_hot_prototype"):
		one_hot = np.zeros_like(proto_sequence)
		max_proto = np.argmax(proto_sequence, axis=1)
		one_hot[np.arange(len(proto_sequence)), max_proto] = 1.0
		proto_sequence = one_hot
	snap_frames = int(cfg.dataset.snap_frames)
	return sample_proto_snap(proto_sequence, snap_frames, device)


def get_runtime_pipeline(cfg):
	augmentations = cfg.get("inference_augmentations")
	if not augmentations:
		return None
	return get_transform_pipeline(augmentations)


def validate_config(cfg):
	if int(cfg.obs_dim) != 7:
		raise ValueError(f"Expected obs_dim=7, got {cfg.obs_dim}.")
	if int(cfg.action_dim) != 7:
		raise ValueError(f"Expected action_dim=7, got {cfg.action_dim}.")
	camera_views = list(cfg.dataset.camera_views)
	if camera_views != REQUIRED_BC_CAMERA_VIEWS:
		raise ValueError(
			"Inference requires dataset.camera_views="
			f"{REQUIRED_BC_CAMERA_VIEWS}, got {camera_views}."
		)


def build_policy(cfg, ckpt_path: str, device: torch.device):
	vision_feature_dim = int(cfg.vision_feature_dim)
	obs_horizon = int(cfg.obs_horizon)
	obs_dim = int(cfg.obs_dim)
	proto_horizon = int(cfg.proto_horizon)
	proto_dim = int(cfg.proto_dim)
	visual_feature_dim_per_step = vision_feature_dim * len(cfg.dataset.camera_views)

	if vision_feature_dim == 512:
		vision_encoder = get_resnet("resnet18")
	else:
		vision_encoder = ResnetConv(embedding_size=vision_feature_dim)
	vision_encoder = replace_bn_with_gn(vision_encoder)

	if cfg.get("upsample_proto"):
		upsample_proto_out = int(cfg.upsample_proto_net.out_size)
		global_cond_dim = (
			visual_feature_dim_per_step * obs_horizon
			+ obs_dim * obs_horizon
			+ proto_horizon * upsample_proto_out
		)
	else:
		global_cond_dim = (
			visual_feature_dim_per_step * obs_horizon
			+ obs_dim * obs_horizon
			+ proto_horizon * proto_dim
		)

	noise_pred_net = hydra.utils.instantiate(
		cfg.noise_pred_net,
		global_cond_dim=global_cond_dim,
	)
	proto_pred_net = hydra.utils.instantiate(
		cfg.proto_pred_net,
		input_dim=visual_feature_dim_per_step * obs_horizon + obs_dim * obs_horizon,
	)

	nets = nn.ModuleDict(
		{
			"vision_encoder": vision_encoder,
			"proto_pred_net": proto_pred_net,
			"noise_pred_net": noise_pred_net,
		}
	)
	if cfg.get("upsample_proto"):
		nets["upsample_proto_net"] = hydra.utils.instantiate(cfg.upsample_proto_net)

	state_dict = torch.load(ckpt_path, map_location=device)
	nets.load_state_dict(state_dict)
	nets.to(device)
	nets.eval()
	noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
	return nets, noise_scheduler


def load_stats(stats_path: str):
	with open(stats_path, "rb") as file:
		return pickle.load(file)


def image_to_tensor(image: np.ndarray, pipeline):
	if image.ndim != 4:
		raise ValueError(f"Expected images with shape (V, H, W, 3), got {image.shape}.")
	image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
	if pipeline is not None:
		image_tensor = pipeline(image_tensor)
	return image_tensor


def infer_action_chunk(cfg, nets, noise_scheduler, stats, proto_snap, image_buf, state_buf, device):
	obs_horizon = int(cfg.obs_horizon)
	pred_horizon = int(cfg.pred_horizon)
	action_horizon = int(cfg.action_horizon)
	action_dim = int(cfg.action_dim)

	visual_seq = torch.stack(list(image_buf), dim=0).to(device)
	obs_seq = np.stack(list(state_buf), axis=0).astype(np.float32)

	with torch.inference_mode():
		num_steps, num_views = visual_seq.shape[:2]
		visual_feature = nets["vision_encoder"](visual_seq.flatten(start_dim=0, end_dim=1))
		visual_feature = visual_feature.reshape(num_steps, num_views, -1).flatten(start_dim=1)

		normalized_obs = normalize_data(obs_seq.copy(), stats=stats["obs"])
		normalized_obs = torch.from_numpy(normalized_obs).to(device=device, dtype=torch.float32)
		obs_feature = torch.cat([visual_feature, normalized_obs], dim=-1)

		predicted_proto = nets["proto_pred_net"](
			obs_feature.unsqueeze(0).flatten(start_dim=1),
			proto_snap,
		)

		if cfg.get("upsample_proto"):
			upsample_proto = nets["upsample_proto_net"](predicted_proto)
			obs_cond = torch.cat(
				[
					obs_feature.unsqueeze(0).flatten(start_dim=1),
					upsample_proto,
				],
				dim=1,
			)
		else:
			obs_cond = torch.cat(
				[obs_feature.unsqueeze(0).flatten(start_dim=1), predicted_proto],
				dim=1,
			)

		action_tensor = torch.randn((1, pred_horizon, action_dim), device=device)
		noise_scheduler.set_timesteps(int(cfg.num_diffusion_iters))
		for timestep in noise_scheduler.timesteps:
			noise_pred = nets["noise_pred_net"](
				sample=action_tensor,
				timestep=timestep,
				global_cond=obs_cond,
			)
			action_tensor = noise_scheduler.step(
				model_output=noise_pred,
				timestep=timestep,
				sample=action_tensor,
			).prev_sample

	action_pred = action_tensor[0].detach().cpu().numpy()
	action_pred = unnormalize_data(action_pred, stats=stats["actions"])
	start = obs_horizon - 1
	end = start + action_horizon
	return action_pred[start:end].astype(np.float32)


def resize_images(images: np.ndarray, resize_shape):
	if resize_shape is None:
		return images
	return np.stack([cv2.resize(image, resize_shape) for image in images], axis=0)


def select_request_images(request: dict, resize_shape, expected_num_views: int):
	if "images" not in request:
		raise KeyError("Request must contain 'images'.")
	images = np.asarray(request["images"], dtype=np.uint8)
	if images.ndim != 4:
		raise ValueError(f"Expected 'images' with shape (V, H, W, 3), got {images.shape}.")
	if images.shape[0] != expected_num_views:
		raise ValueError(
			f"Expected {expected_num_views} camera views in request, got {images.shape[0]}."
		)
	return resize_images(images, resize_shape)


def select_request_state(request: dict):
	for key in ("state_ee", "state"):
		if key not in request:
			continue
		state = np.asarray(request[key], dtype=np.float32).reshape(-1)
		if state.shape[0] == 7:
			return state
	raise ValueError("Request must contain a 7D 'state_ee' or 'state'.")


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt_path", type=str, required=True)
	parser.add_argument("--config_path", type=str, default=None)
	parser.add_argument("--proto_path", type=str, required=True)
	parser.add_argument("--demo_episode", type=int, required=True)
	parser.add_argument("--stats_path", type=str, default=None)
	parser.add_argument("--port", type=int, default=5558)
	parser.add_argument("--device", type=str, default=None)
	return parser.parse_args()


def main():
	args = parse_args()
	device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(device_name)
	cfg, config_source = resolve_runtime_config(args)
	validate_config(cfg)
	set_seed(int(cfg.get("seed", 0)))

	stats_path = args.stats_path or os.path.join(os.path.dirname(args.ckpt_path), "stats.pickle")
	print(f"Loading xskill config from {config_source}...")
	print(f"Loading xskill checkpoint from {args.ckpt_path} on {device}...")
	nets, noise_scheduler = build_policy(cfg, args.ckpt_path, device)
	stats = load_stats(stats_path)
	proto_snap = load_proto_snap(cfg, args.proto_path, args.demo_episode, device)
	print(f"Loaded prototype demo episode {args.demo_episode} from {args.proto_path}")

	obs_horizon = int(cfg.obs_horizon)
	expected_num_views = len(cfg.dataset.camera_views)
	resize_shape = tuple(cfg.bc_resize) if cfg.get("bc_resize") else None
	runtime_pipeline = get_runtime_pipeline(cfg)
	image_buf = deque(maxlen=obs_horizon)
	state_buf = deque(maxlen=obs_horizon)

	context = zmq.Context()
	socket = context.socket(zmq.REP)
	socket.bind(f"tcp://*:{args.port}")
	print(f"xskill server listening on port {args.port}")

	while True:
		raw = socket.recv()
		try:
			request = pickle.loads(raw)
			if request.get("describe"):
				socket.send(
					pickle.dumps(
						{
							"status": "ok",
							"camera_views": list(cfg.dataset.camera_views),
						}
					)
				)
				continue
			if request.get("reset"):
				image_buf.clear()
				state_buf.clear()
				socket.send(
					pickle.dumps(
						{
							"status": "ok",
							"camera_views": list(cfg.dataset.camera_views),
						}
					)
				)
				continue

			image = select_request_images(request, resize_shape, expected_num_views)
			state = select_request_state(request)
			image_buf.append(image_to_tensor(image, runtime_pipeline))
			state_buf.append(state)
			while len(image_buf) < obs_horizon:
				image_buf.appendleft(image_buf[0].clone())
				state_buf.appendleft(state_buf[0].copy())

			action_chunk = infer_action_chunk(
				cfg,
				nets,
				noise_scheduler,
				stats,
				proto_snap,
				image_buf,
				state_buf,
				device,
			)
			socket.send(
				pickle.dumps(
					{
						"action_chunk": action_chunk,
						"action_representation": "ee_pose_gripper_7d",
					}
				)
			)
		except Exception as exc:
			print(f"[xskill_server] {exc}")
			socket.send(pickle.dumps({"status": "error", "message": str(exc)}))


if __name__ == "__main__":
	main()

