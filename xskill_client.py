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
GRIPPER_WIDTH_THRESHOLD = 0.025


@InferenceClient.register(client_name="XSkill")
class XSkillClient(InferenceClient):
	def __init__(self, args: PolicyArgs) -> None:
		self.args = args
		self.device = args.device if args.device is not None else "cuda:1"
		host = args.host if args.host is not None else "localhost"
		port = args.port if args.port is not None else 5558
		self.open_loop_horizon = args.open_loop_horizon or 1

		context = zmq.Context()
		self.socket = context.socket(zmq.REQ)
		self.socket.connect(f"tcp://{host}:{port}")
		print(f"Connected to xskill server at {host}:{port}")

		self.action_chunk: np.ndarray | None = None
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
		images, ee_state = self._extract_observation(obs)

		if self._needs_replan():
			request = {"images": images, "state_ee": ee_state}
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
			self.actions_from_chunk_completed = 0

		assert self.action_chunk is not None
		raw_action = self.action_chunk[self.actions_from_chunk_completed].copy()
		self.actions_from_chunk_completed += 1
		action = self._ee_action_to_env_action(raw_action.astype(np.float32), obs)

		viz = self._build_viz(images) if return_viz else None
		return action.astype(np.float32), viz

	def _needs_replan(self) -> bool:
		if self.action_chunk is None:
			return True
		chunk_horizon = min(len(self.action_chunk), self.open_loop_horizon)
		return self.actions_from_chunk_completed >= chunk_horizon

	def _extract_observation(self, obs_dict: dict) -> tuple[np.ndarray, np.ndarray]:
		splat_obs = obs_dict["splat"]
		missing_keys = [key for key in ("cam1", "wrist_cam") if key not in splat_obs]
		if missing_keys:
			raise KeyError(f"Missing camera keys in obs['splat']: {missing_keys}")
		images = np.stack(
			[
				np.asarray(splat_obs["cam1"], dtype=np.uint8),
				np.asarray(splat_obs["wrist_cam"], dtype=np.uint8),
			],
			axis=0,
		)

		robot_state = obs_dict["policy"]
		ee_pose = robot_state["ee_pose"][0].detach().cpu().numpy().astype(np.float32)
		gripper_width = self._extract_gripper_width(robot_state, ee_pose)
		ee_state = self._ee_pose_to_rotvec_state(ee_pose, gripper_width)
		return images, ee_state

	def _extract_gripper_width(self, robot_state: dict, ee_pose: np.ndarray) -> np.ndarray:
		for key in ("gripper_width", "gripper_pos"):
			if key not in robot_state:
				continue
			value = robot_state[key]
			if isinstance(value, torch.Tensor):
				value = value.detach().cpu().numpy()
			else:
				value = np.asarray(value)
			return value.reshape(-1)[:1].astype(np.float32)
		return ee_pose[7:8].astype(np.float32)

	@staticmethod
	def _ee_pose_to_rotvec_state(ee_pose: np.ndarray, gripper_width: np.ndarray) -> np.ndarray:
		position = ee_pose[:3]
		quat_wxyz = ee_pose[3:7]
		quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
		rotvec = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
		return np.concatenate([position, rotvec, gripper_width], axis=0).astype(np.float32)

	def _ee_action_to_env_action(self, action: np.ndarray, obs: dict) -> np.ndarray:
		self._ensure_motion_gen()
		from curobo.types.math import Pose

		position = action[:3]
		rotvec = action[3:6]
		gripper_width = float(action[6])
		quat_xyzw = Rotation.from_rotvec(rotvec).as_quat().astype(np.float32)
		quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

		ee_pos = torch.from_numpy(position).float().unsqueeze(0).to(self.device)
		ee_quat = torch.from_numpy(quat_wxyz).float().unsqueeze(0).to(self.device)
		goal_pose = Pose(position=ee_pos, quaternion=ee_quat)
		ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)

		if ik_result.success.item():
			joint_pos = ik_result.solution.squeeze(0)[0, :7].cpu().numpy().astype(np.float32)
		else:
			print("[XSkillClient] IK failed, holding current joints")
			joint_pos = obs["policy"]["arm_joint_pos"][0].detach().cpu().numpy().astype(np.float32)

		gripper_command = np.array(
			[1.0 if gripper_width < GRIPPER_WIDTH_THRESHOLD else 0.0],
			dtype=np.float32,
		)
		return np.concatenate([joint_pos, gripper_command], axis=0)

	def _ensure_motion_gen(self) -> None:
		if self.motion_gen is not None:
			return
		from polaris.utils_.planner_utils import setup_curobo

		self.motion_gen = setup_curobo()

	def _build_viz(self, images: np.ndarray) -> np.ndarray:
		view_width = max(1, VIZ_W // images.shape[0])
		resized_views = [cv2.resize(view, (view_width, VIZ_H)) for view in images]
		return np.concatenate(resized_views, axis=1)
