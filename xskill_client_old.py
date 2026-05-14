import pickle

import cv2
import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation

from polaris.client.abstract_client import InferenceClient
from polaris.config import PolicyArgs


VIZ_H = 240
VIZ_W = 426


@InferenceClient.register(client_name="XSkill")
class XSkillClient(InferenceClient):
	"""
	Client for the xskill diffusion policy served by xskill_server.py.

	The xskill checkpoint can run either in joint space or in 7D absolute EE pose
	space. The client forwards both camera views plus both state representations,
	and converts EE-pose actions back to env joint actions when needed.
	"""

	def __init__(self, args: PolicyArgs) -> None:
		self.args = args
		self.device = args.device if args.device is not None else "cuda:1"
		self.camera_keys = self._resolve_camera_keys(args)
		host = args.host if args.host is not None else "localhost"
		port = args.port if args.port is not None else 5558
		self.open_loop_horizon = args.open_loop_horizon or 1

		context = zmq.Context()
		self.socket = context.socket(zmq.REQ)
		self.socket.connect(f"tcp://{host}:{port}")
		print(f"Connected to xskill server at {host}:{port}")

		self.action_chunk: np.ndarray | None = None
		self.action_representation: str | None = None
		self.actions_from_chunk_completed = 0
		self.motion_gen = None

	@property
	def rerender(self) -> bool:
		return self._needs_replan()

	def reset(self):
		self.action_chunk = None
		self.actions_from_chunk_completed = 0
		self.socket.send(pickle.dumps({"reset": True}))
		self.socket.recv()

	def infer(
		self, obs: dict, instruction: str, return_viz: bool = False
	) -> tuple[np.ndarray, np.ndarray | None]:
		del instruction
		images, joint_state, ee_state = self._extract_observation(obs)

		if self._needs_replan():
			request = {
				"images": images,
				"state_joint": joint_state,
				"state_ee": ee_state,
			}
			self.socket.send(pickle.dumps(request))
			response = pickle.loads(self.socket.recv())
			if response.get("status") == "error":
				raise RuntimeError(response.get("message", "xskill server error"))

			action_chunk = np.asarray(response["action_chunk"], dtype=np.float32)
			if action_chunk.ndim == 1:
				action_chunk = action_chunk[None, :]
			if len(action_chunk) == 0:
				raise RuntimeError("xskill server returned an empty action chunk")

			self.action_chunk = action_chunk
			self.action_representation = response.get("action_representation") or self._infer_action_representation(action_chunk)
			self.actions_from_chunk_completed = 0

		assert self.action_chunk is not None
		raw_action = self.action_chunk[self.actions_from_chunk_completed].copy()
		self.actions_from_chunk_completed += 1

		action = self._policy_action_to_env_action(raw_action, obs)

		viz = None
		if return_viz:
			viz = self._build_viz(images)

		return action.astype(np.float32), viz

	def _needs_replan(self) -> bool:
		if self.action_chunk is None:
			return True

		chunk_horizon = len(self.action_chunk)
		if self.open_loop_horizon is not None:
			chunk_horizon = min(chunk_horizon, self.open_loop_horizon)
		return self.actions_from_chunk_completed >= chunk_horizon

	@staticmethod
	def _resolve_camera_keys(args: PolicyArgs) -> tuple[str, ...]:
		camera_keys = getattr(args, "camera_keys", None)
		if camera_keys:
			return tuple(camera_keys)

		cam_key = getattr(args, "cam_key", None)
		if isinstance(cam_key, str) and cam_key.strip():
			parsed_keys = tuple(key.strip() for key in cam_key.split(",") if key.strip())
			if len(parsed_keys) > 1:
				return parsed_keys
			if parsed_keys and parsed_keys[0] != "cam1":
				return parsed_keys

		return ("cam1", "wrist_cam")

	def _extract_observation(self, obs_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		splat_obs = obs_dict["splat"]
		images = []
		for key in self.camera_keys:
			print(f"Looking for camera key '{key}' in obs['splat']")
			if key in splat_obs:
				images.append(np.asarray(splat_obs[key], dtype=np.uint8))
		if not images:
			raise KeyError(f"None of the configured camera keys were found in obs['splat']: {self.camera_keys}")
		image = images[0] if len(images) == 1 else np.stack(images, axis=0)

		robot_state = obs_dict["policy"]
		joint_pos = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
		gripper_pos = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]
		joint_state = np.concatenate([joint_pos, gripper_pos], axis=0).astype(np.float32)

		ee_pose = robot_state["ee_pose"][0].detach().cpu().numpy().astype(np.float32)
		ee_state = self._ee_pose_to_rotvec_state(ee_pose)

		return image, joint_state, ee_state

	@staticmethod
	def _ee_pose_to_rotvec_state(ee_pose: np.ndarray) -> np.ndarray:
		pos = ee_pose[:3]
		quat_wxyz = ee_pose[3:7]
		quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
		rotvec = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
		gripper = ee_pose[7:8]
		return np.concatenate([pos, rotvec, gripper], axis=0).astype(np.float32)

	@staticmethod
	def _infer_action_representation(action_chunk: np.ndarray) -> str:
		action_dim = int(action_chunk.shape[-1])
		if action_dim == 7:
			return "ee_pose_gripper_7d"
		if action_dim == 8:
			return "joint_pos_gripper_8d"
		return f"action_dim_{action_dim}"

	def _policy_action_to_env_action(self, action: np.ndarray, obs: dict) -> np.ndarray:
		representation = self.action_representation or self._infer_action_representation(action[None, :])
		if representation == "joint_pos_gripper_8d" or action.shape[0] == 8:
			env_action = action.astype(np.float32).copy()
			env_action[-1] = 1.0 if env_action[-1] > 0.5 else 0.0
			return env_action
		if representation == "ee_pose_gripper_7d" or action.shape[0] == 7:
			return self._ee_action_to_env_action(action.astype(np.float32), obs)
		raise ValueError(f"Unsupported xskill action representation: {representation}")

	def _ee_action_to_env_action(self, action: np.ndarray, obs: dict) -> np.ndarray:
		self._ensure_motion_gen()

		from curobo.types.math import Pose

		pos = action[:3]
		rotvec = action[3:6]
		gripper = action[6]

		quat_xyzw = Rotation.from_rotvec(rotvec).as_quat().astype(np.float32)
		quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

		ee_pos = torch.from_numpy(pos).float().unsqueeze(0).to(self.device)
		ee_quat = torch.from_numpy(quat_wxyz).float().unsqueeze(0).to(self.device)
		goal_pose = Pose(position=ee_pos, quaternion=ee_quat)
		ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)

		if ik_result.success.item():
			joint_pos = ik_result.solution.squeeze(0)[0, :7].cpu().numpy().astype(np.float32)
		else:
			print("[XSkillClient] IK failed, holding current joints")
			joint_pos = obs["policy"]["arm_joint_pos"][0].detach().cpu().numpy().astype(np.float32)

		gripper_bin = np.array([1.0 if gripper > 0.5 else 0.0], dtype=np.float32)
		return np.concatenate([joint_pos, gripper_bin], axis=0)

	def _ensure_motion_gen(self) -> None:
		if self.motion_gen is not None:
			return
		from polaris.utils_.planner_utils import setup_curobo

		self.motion_gen = setup_curobo()

	def _build_viz(self, images: np.ndarray) -> np.ndarray:
		if images.ndim == 3:
			return cv2.resize(images, (VIZ_W, VIZ_H))

		num_views = images.shape[0]
		view_width = max(1, VIZ_W // num_views)
		resized_views = [cv2.resize(view, (view_width, VIZ_H)) for view in images]
		return np.concatenate(resized_views, axis=1)