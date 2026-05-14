"""
Policy inference server for xskill diffusion checkpoints.

The xskill policy uses the live robot observation plus a fixed demonstration
video. The demonstration is read from a folder of numbered PNG frames and is
encoded once at startup into the prototype context used for action generation.

Usage:
	python src/polaris/server/xskill_server.py \
		--ckpt_path /path/to/ckpt_199.pt \
		--config_path /path/to/skill_transfer_composing_pick_mug.yaml \
		--video_dir /path/to/demo_episode_folder \
		--port 5558
"""

from __future__ import annotations

import argparse
import json
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
	"pretrain_path",
	"pretrain_ckpt",
	"dataset.camera_views",
	"dataset.snap_frames",
	"num_diffusion_iters",
)


def set_seed(seed: int) -> None:
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


def resolve_runtime_config(args) -> tuple[object, str]:
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


def repeat_last_proto(proto_tensor: torch.Tensor, episode_len: int) -> torch.Tensor:
	if len(proto_tensor) >= episode_len:
		return proto_tensor
	repeat_count = episode_len - len(proto_tensor)
	repeated = proto_tensor[-1].unsqueeze(0).repeat(repeat_count, 1)
	return torch.cat([proto_tensor, repeated], dim=0)


def load_pretrain_model(pretrain_path: str, pretrain_ckpt: int, device: torch.device):
	pretrain_cfg = OmegaConf.load(os.path.join(pretrain_path, ".hydra", "config.yaml"))
	model = hydra.utils.instantiate(pretrain_cfg.Model).to(device)

	ckpt_path = os.path.join(pretrain_path, f"epoch={pretrain_ckpt}.ckpt")
	checkpoint = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(checkpoint["state_dict"])
	model.to(device)
	model.eval()

	pipeline = get_transform_pipeline(pretrain_cfg.augmentations)
	resize_shape = tuple(pretrain_cfg.resize_shape) if pretrain_cfg.get("resize_shape") else None
	return model, pipeline, resize_shape


def load_episode_images(video_dir: str, resize_shape: tuple[int, int] | None) -> np.ndarray:
	video_path = pathlib.Path(video_dir)
	frame_paths = sorted(video_path.glob("*.png"), key=lambda path: int(path.stem))
	if not frame_paths:
		raise RuntimeError(f"No PNG frames found in {video_dir}")

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


def convert_images_to_tensors(images_arr: np.ndarray, pipeline) -> torch.Tensor:
	images_tensor = np.transpose(images_arr, (0, 3, 1, 2))
	images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255.0
	if pipeline is not None:
		images_tensor = pipeline(images_tensor)
	return images_tensor


def encode_episode(images_arr: np.ndarray, model, pipeline, device: torch.device) -> dict[str, np.ndarray]:
	images_tensor = convert_images_to_tensors(images_arr, pipeline).to(device)
	episode_len = images_tensor.shape[0]
	if episode_len <= model.slide:
		raise ValueError(
			f"Episode only has {episode_len} frames, but model.slide={model.slide} requires at least {model.slide + 1}"
		)

	im_q = torch.stack(
		[images_tensor[index:index + model.slide + 1] for index in range(episode_len - model.slide)]
	)

	with torch.inference_mode():
		z = model.encoder_q(im_q, None)
		softmax_z = torch.softmax(z / model.T, dim=1)
		state_representation = model.encoder_q.get_state_representation(im_q, None)
		traj_representation = model.encoder_q.get_traj_representation(state_representation)

		encode_protos = repeat_last_proto(z, episode_len)
		softmax_encode_protos = repeat_last_proto(softmax_z, episode_len)
		traj_representation = repeat_last_proto(traj_representation, episode_len)

	return {
		"encode_protos": encode_protos.detach().cpu().numpy().astype(np.float32),
		"softmax_encode_protos": softmax_encode_protos.detach().cpu().numpy().astype(np.float32),
		"traj_representation": traj_representation.detach().cpu().numpy().astype(np.float32),
	}


def sample_proto_snap(proto_sequence: np.ndarray, snap_frames: int, device: torch.device) -> torch.Tensor:
	episode_len = len(proto_sequence)
	if episode_len == 0:
		raise ValueError("Prototype sequence is empty")

	sample_count = min(snap_frames, episode_len)
	snap_indices = sorted(random.sample(list(range(episode_len)), k=sample_count))
	snap = np.asarray(proto_sequence[snap_indices], dtype=np.float32)

	if sample_count < snap_frames:
		pad = np.repeat(snap[-1:], snap_frames - sample_count, axis=0)
		snap = np.concatenate([snap, pad], axis=0)

	return torch.from_numpy(snap).to(device=device, dtype=torch.float32).unsqueeze(0)


def load_proto_snap(cfg, video_dir: str, pretrain_path: str | None, pretrain_ckpt: int | None, device: torch.device) -> torch.Tensor:
	resolved_pretrain_path = pretrain_path or cfg.get("pretrain_path")
	resolved_pretrain_ckpt = pretrain_ckpt if pretrain_ckpt is not None else cfg.get("pretrain_ckpt")
	if resolved_pretrain_path is None or resolved_pretrain_ckpt is None:
		raise ValueError("xskill demo encoding requires pretrain_path and pretrain_ckpt")

	pretrain_model, pretrain_pipeline, resize_shape = load_pretrain_model(
		resolved_pretrain_path,
		int(resolved_pretrain_ckpt),
		device,
	)
	episode_images = load_episode_images(video_dir, resize_shape)
	episode_outputs = encode_episode(episode_images, pretrain_model, pretrain_pipeline, device)

	if cfg.get("raw_representation"):
		proto_sequence = episode_outputs["traj_representation"]
	elif cfg.get("softmax_prototype") or cfg.get("one_hot_prototype"):
		proto_sequence = episode_outputs["softmax_encode_protos"]
		if cfg.get("one_hot_prototype"):
			one_hot = np.zeros_like(proto_sequence)
			max_proto = np.argmax(proto_sequence, axis=1)
			one_hot[np.arange(len(proto_sequence)), max_proto] = 1.0
			proto_sequence = one_hot
	elif cfg.get("prototype"):
		proto_sequence = episode_outputs["encode_protos"]
	else:
		raise ValueError("Enable one of raw_representation, softmax_prototype, one_hot_prototype, or prototype")

	snap_frames = int(cfg.dataset.snap_frames) if cfg.get("dataset") and cfg.dataset.get("snap_frames") else 100
	return sample_proto_snap(proto_sequence, snap_frames, device)


def get_num_camera_views(cfg) -> int:
	if cfg.get("dataset") and cfg.dataset.get("camera_views"):
		return len(cfg.dataset.camera_views)
	return 1


def get_runtime_pipeline(cfg):
	augmentations = cfg.get("inference_augmentations")
	if augmentations is None and cfg.get("dataset"):
		augmentations = cfg.dataset.get("pipeline")
	if not augmentations:
		return None
	return get_transform_pipeline(augmentations)


def resize_images(images: np.ndarray, resize_shape: tuple[int, int] | None) -> np.ndarray:
	if resize_shape is None:
		return images
	if images.ndim == 3:
		if images.shape[-1] != 3:
			raise ValueError(f"Expected image with shape (H, W, 3), got {images.shape}")
		if (images.shape[1], images.shape[0]) != resize_shape:
			images = cv2.resize(images, resize_shape)
		return images
	if images.ndim == 4:
		if images.shape[-1] != 3:
			raise ValueError(f"Expected images with shape (V, H, W, 3), got {images.shape}")
		return np.stack([resize_images(image, resize_shape) for image in images], axis=0)
	raise ValueError(f"Expected image array with 3 or 4 dims, got {images.shape}")


def ensure_uint8_images(images) -> np.ndarray:
	images_arr = np.asarray(images)
	if np.issubdtype(images_arr.dtype, np.floating):
		max_value = float(np.nanmax(images_arr)) if images_arr.size else 0.0
		if max_value <= 1.0:
			images_arr = images_arr * 255.0
		images_arr = np.clip(images_arr, 0.0, 255.0).astype(np.uint8)
	elif images_arr.dtype != np.uint8:
		images_arr = np.clip(images_arr, 0, 255).astype(np.uint8)
	return images_arr


def select_request_images(request: dict, expected_views: int, resize_shape: tuple[int, int] | None) -> np.ndarray:
	if "images" in request:
		images = ensure_uint8_images(request["images"])
	elif "image" in request:
		images = ensure_uint8_images(request["image"])
	else:
		raise KeyError("Request must contain either 'images' or 'image'")

	if images.ndim == 3:
		if expected_views != 1:
			raise ValueError(f"Expected {expected_views} camera views, got 1")
		return resize_images(images, resize_shape)

	if images.ndim != 4:
		raise ValueError(f"Expected image batch with shape (V, H, W, 3), got {images.shape}")
	if images.shape[0] < expected_views:
		raise ValueError(f"Expected at least {expected_views} camera views, got {images.shape[0]}")
	images = images[:expected_views]
	if expected_views == 1:
		return resize_images(images[0], resize_shape)
	return resize_images(images, resize_shape)


def select_request_state(request: dict, obs_dim: int) -> np.ndarray:
	preferred_keys = []
	if obs_dim == 7:
		preferred_keys.append("state_ee")
	elif obs_dim == 8:
		preferred_keys.append("state_joint")
	preferred_keys.append("state")
	preferred_keys.extend(["state_ee", "state_joint"])

	seen_keys = set()
	for key in preferred_keys:
		if key in seen_keys or key not in request:
			continue
		seen_keys.add(key)
		state = np.asarray(request[key], dtype=np.float32).reshape(-1)
		if state.shape[0] == obs_dim:
			return state

	available_shapes = {
		key: tuple(np.asarray(request[key]).shape)
		for key in ("state", "state_ee", "state_joint")
		if key in request
	}
	raise ValueError(
		f"Could not find a state matching obs_dim={obs_dim}. Available request state shapes: {available_shapes}"
	)


def build_policy(cfg, ckpt_path: str, device: torch.device) -> tuple[nn.ModuleDict, object]:
	vision_feature_dim = int(cfg.vision_feature_dim)
	num_camera_views = get_num_camera_views(cfg)
	visual_feature_dim_per_step = vision_feature_dim * num_camera_views
	obs_horizon = int(cfg.obs_horizon)
	obs_dim = int(cfg.obs_dim)
	proto_horizon = int(cfg.proto_horizon)
	proto_dim = int(cfg.proto_dim)

	if vision_feature_dim == 512:
		vision_encoder = get_resnet("resnet18")
	else:
		vision_encoder = ResnetConv(embedding_size=vision_feature_dim)
	vision_encoder = replace_bn_with_gn(vision_encoder)

	if cfg.get("upsample_proto"):
		upsample_proto_out = int(cfg.upsample_proto_net.out_size)
		global_cond_dim = visual_feature_dim_per_step * obs_horizon + obs_dim * obs_horizon + proto_horizon * upsample_proto_out
	else:
		global_cond_dim = visual_feature_dim_per_step * obs_horizon + obs_dim * obs_horizon + proto_horizon * proto_dim

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


def load_stats(stats_path: str) -> dict:
	with open(stats_path, "rb") as file:
		return pickle.load(file)


def image_to_tensor(image: np.ndarray, pipeline) -> torch.Tensor:
	image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
	if image.ndim == 3:
		image_tensor = image_tensor.permute(2, 0, 1)
	elif image.ndim == 4:
		image_tensor = image_tensor.permute(0, 3, 1, 2)
	else:
		raise ValueError(f"Expected image array with 3 or 4 dims, got {image.shape}")
	if pipeline is not None:
		image_tensor = pipeline(image_tensor)
	return image_tensor


def infer_action_chunk(
	cfg,
	nets: nn.ModuleDict,
	noise_scheduler,
	stats: dict,
	proto_snap: torch.Tensor,
	image_buf: deque,
	state_buf: deque,
	device: torch.device,
) -> np.ndarray:
	obs_horizon = int(cfg.obs_horizon)
	pred_horizon = int(cfg.pred_horizon)
	action_horizon = int(cfg.action_horizon)
	action_dim = int(cfg.action_dim)

	visual_seq = torch.stack(list(image_buf), dim=0).to(device)
	obs_seq = np.stack(list(state_buf), axis=0).astype(np.float32)

	with torch.inference_mode():
		if visual_seq.ndim == 4:
			visual_feature = nets["vision_encoder"](visual_seq)
		elif visual_seq.ndim == 5:
			num_steps, num_views = visual_seq.shape[:2]
			visual_feature = nets["vision_encoder"](visual_seq.flatten(start_dim=0, end_dim=1))
			visual_feature = visual_feature.reshape(num_steps, num_views, -1).flatten(start_dim=1)
		else:
			raise ValueError(f"Expected visual sequence with 4 or 5 dims, got {tuple(visual_seq.shape)}")

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

		noisy_action = torch.randn((1, pred_horizon, action_dim), device=device)
		action_tensor = noisy_action
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


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt_path", type=str, required=True)
	parser.add_argument(
		"--config_path",
		type=str,
		default=None,
		help="Optional fallback config. If hydra_config.yaml exists next to the checkpoint, the server uses it instead.",
	)
	parser.add_argument("--video_dir", type=str, required=True)
	parser.add_argument("--stats_path", type=str, default=None)
	parser.add_argument("--pretrain_path", type=str, default=None)
	parser.add_argument("--pretrain_ckpt", type=int, default=None)
	parser.add_argument("--port", type=int, default=5558)
	parser.add_argument("--device", type=str, default=None)
	return parser.parse_args()


def main():
	args = parse_args()

	device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(device_name)
	cfg, config_source = resolve_runtime_config(args)
	seed = int(cfg.get("seed", 0))
	set_seed(seed)

	stats_path = args.stats_path or os.path.join(os.path.dirname(args.ckpt_path), "stats.pickle")

	print(f"Loading xskill config from {config_source}...")
	print(f"Loading xskill checkpoint from {args.ckpt_path} on {device}...")
	nets, noise_scheduler = build_policy(cfg, args.ckpt_path, device)
	stats = load_stats(stats_path)
	proto_snap = load_proto_snap(
		cfg,
		args.video_dir,
		args.pretrain_path,
		args.pretrain_ckpt,
		device,
	)
	print(f"Loaded demo video from {args.video_dir}")

	obs_horizon = int(cfg.obs_horizon)
	resize_shape = tuple(cfg.bc_resize) if cfg.get("bc_resize") else None
	obs_dim = int(cfg.obs_dim)
	num_camera_views = get_num_camera_views(cfg)
	runtime_pipeline = get_runtime_pipeline(cfg)
	action_dim = int(cfg.action_dim)
	if action_dim == 7:
		action_representation = "ee_pose_gripper_7d"
	elif action_dim == 8:
		action_representation = "joint_pos_gripper_8d"
	else:
		action_representation = f"action_dim_{action_dim}"

	image_buf: deque[torch.Tensor] = deque(maxlen=obs_horizon)
	state_buf: deque[np.ndarray] = deque(maxlen=obs_horizon)

	context = zmq.Context()
	socket = context.socket(zmq.REP)
	socket.bind(f"tcp://*:{args.port}")
	print(f"xskill server listening on port {args.port}")

	while True:
		raw = socket.recv()
		try:
			request = pickle.loads(raw)
			if request.get("reset"):
				image_buf.clear()
				state_buf.clear()
				socket.send(pickle.dumps({"status": "ok"}))
				continue

			image = select_request_images(request, num_camera_views, resize_shape)
			state = select_request_state(request, obs_dim)

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
			socket.send(pickle.dumps({"action_chunk": action_chunk, "action_representation": action_representation}))
		except Exception as exc:
			print(f"[xskill_server] {exc}")
			socket.send(pickle.dumps({"status": "error", "message": str(exc)}))


if __name__ == "__main__":
	main()
