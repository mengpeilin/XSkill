from __future__ import annotations

from contextlib import nullcontext
import os
import pickle
import random
import uuid
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import DictConfig, OmegaConf

from xskill.model.diffusion_model import get_resnet, replace_bn_with_gn
from xskill.model.encoder import ResnetConv


def configure_runtime_speedups(cfg):
    matmul_precision = cfg.get("matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))

    if not torch.cuda.is_available():
        return

    allow_tf32 = bool(cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True


def create_optimizer(parameters, cfg, device):
    optimizer_kwargs = {
        "params": parameters,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
    }
    if device.type == "cuda" and bool(cfg.get("fused_optimizer", False)):
        try:
            return torch.optim.AdamW(**optimizer_kwargs, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(**optimizer_kwargs)


def maybe_compile_module(module, cfg, device):
    if not bool(cfg.get("compile_model", False)):
        return module
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return module
    return torch.compile(module, mode=str(cfg.get("compile_mode", "reduce-overhead")))


def autocast_context(cfg, device):
    mixed_precision = str(cfg.get("mixed_precision", "")).lower()
    use_bf16 = device.type == "cuda" and mixed_precision in {"bf16", "bf16-mixed", "bfloat16"}
    if not use_bf16:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def checkpoint_epoch_from_path(path: str | os.PathLike[str]) -> int | None:
    stem = Path(path).stem
    for prefix in ["resume_", "ckpt_"]:
        if not stem.startswith(prefix):
            continue
        value = stem[len(prefix) :]
        if value.isdigit():
            return int(value)
    return None


def save_resume_state(save_dir, epoch_idx, nets, ema, optimizer, lr_scheduler):
    torch.save(
        {
            "epoch": int(epoch_idx),
            "nets": nets.state_dict(),
            "ema_averaged_model": ema.averaged_model.state_dict(),
            "ema_optimization_step": int(ema.optimization_step),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        os.path.join(save_dir, f"resume_{epoch_idx}.pt"),
    )


def load_resume_state(resume_path, nets, ema, optimizer, lr_scheduler, device):
    checkpoint = torch.load(resume_path, map_location=device)

    if isinstance(checkpoint, dict) and "nets" in checkpoint:
        nets.load_state_dict(checkpoint["nets"])
        ema.averaged_model.load_state_dict(checkpoint["ema_averaged_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        ema.optimization_step = int(checkpoint.get("ema_optimization_step", 0))
        return int(checkpoint["epoch"]) + 1, "full_state"

    warm_start_epoch = checkpoint_epoch_from_path(resume_path)
    if warm_start_epoch is None:
        raise ValueError(f"Could not infer epoch from resume checkpoint path: {resume_path}")

    nets.load_state_dict(checkpoint)
    ema.averaged_model.load_state_dict(checkpoint)
    return warm_start_epoch + 1, "weights_only"


@hydra.main(
    version_base=None,
    config_path="../config/realworld",
    config_name="skill_transfer_composing_pick_mug",
)
def train_diffusion_bc(cfg: DictConfig):
    configure_runtime_speedups(cfg)

    resume_path = cfg.get("resume_path")
    resume_run_dir = cfg.get("resume_run_dir")
    if resume_path is not None:
        resume_path = os.path.abspath(os.path.expanduser(str(resume_path)))
    if resume_run_dir is None and resume_path is not None:
        resume_run_dir = str(Path(resume_path).resolve().parent)

    if resume_run_dir is not None:
        save_dir = os.path.abspath(os.path.expanduser(str(resume_run_dir)))
    else:
        unique_id = str(uuid.uuid4())
        save_dir = os.path.join(cfg.save_dir, unique_id)
    cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "hydra_config.yaml"))
    print(f"output_dir: {save_dir}")

    wandb.init(project=cfg.project_name)
    wandb.config.update(OmegaConf.to_container(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    proto_horizon = cfg.proto_horizon

    dataset = hydra.utils.instantiate(cfg.dataset)
    stats = dataset.stats
    with open(os.path.join(save_dir, "stats.pickle"), "wb") as f:
        pickle.dump(stats, f)

    dataloader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "shuffle": True,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers if cfg.num_workers > 0 else False,
    }
    if cfg.num_workers > 0 and cfg.get("prefetch_factor") is not None:
        dataloader_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    print("dataset len", len(dataset))
    num_camera_views = len(cfg.dataset.camera_views)
    if list(cfg.dataset.camera_views) != ["cam1", "wrist_cam"]:
        raise ValueError(
            f"Expected dataset.camera_views=['cam1', 'wrist_cam'], got {list(cfg.dataset.camera_views)}"
        )
    print("num_camera_views", num_camera_views)

    if cfg.vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        vision_encoder = ResnetConv(embedding_size=cfg.vision_feature_dim)

    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_feature_dim = cfg.vision_feature_dim

    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim
    proto_dim = cfg.proto_dim
    visual_feature_dim_per_step = vision_feature_dim * num_camera_views

    if cfg.upsample_proto:
        noise_pred_net = hydra.utils.instantiate(
            cfg.noise_pred_net,
            global_cond_dim=visual_feature_dim_per_step * obs_horizon
            + obs_dim * obs_horizon
            + proto_horizon * cfg.upsample_proto_net.out_size,
        )
    else:
        noise_pred_net = hydra.utils.instantiate(
            cfg.noise_pred_net,
            global_cond_dim=visual_feature_dim_per_step * obs_horizon
            + obs_dim * obs_horizon
            + proto_horizon * proto_dim,
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

    if cfg.upsample_proto:
        upsample_proto_net = hydra.utils.instantiate(cfg.upsample_proto_net)
        nets["upsample_proto_net"] = upsample_proto_net

    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    device = torch.device(cfg.device)
    _ = nets.to(device)

    ema = EMAModel(model=nets, power=0.75)

    optimizer = create_optimizer(nets.parameters(), cfg, device)

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * cfg.num_epochs,
    )

    start_epoch = 0
    if resume_path is not None:
        start_epoch, resume_mode = load_resume_state(
            resume_path,
            nets,
            ema,
            optimizer,
            lr_scheduler,
            device,
        )
        print(f"Resuming BC from {resume_path} ({resume_mode}) at epoch {start_epoch}")

    compiled_vision_encoder = maybe_compile_module(nets["vision_encoder"], cfg, device)
    compiled_proto_pred_net = maybe_compile_module(nets["proto_pred_net"], cfg, device)
    compiled_noise_pred_net = maybe_compile_module(nets["noise_pred_net"], cfg, device)
    compiled_upsample_proto_net = None
    if cfg.upsample_proto:
        compiled_upsample_proto_net = maybe_compile_module(nets["upsample_proto_net"], cfg, device)

    grad_accumulation_steps = max(int(cfg.get("gradient_accumulation_steps", 1)), 1)
    non_blocking = bool(cfg.pin_memory) and device.type == "cuda"

    for epoch_idx in range(start_epoch, cfg.num_epochs):
        epoch_loss = []
        epoch_action_loss = []
        epoch_proto_prediction_loss = []
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, nbatch in enumerate(dataloader):
            nobs = nbatch["obs"].to(device, non_blocking=non_blocking)
            batch_size = nobs.shape[0]
            nimage = nbatch["images"].to(device, non_blocking=non_blocking)
            nproto = nbatch["protos"].to(device, non_blocking=non_blocking)
            proto_snap = nbatch["proto_snap"].to(device, non_blocking=non_blocking)
            proto_snap = proto_snap.reshape(batch_size, dataset.snap_frames, -1)
            naction = nbatch["actions"].to(device, non_blocking=non_blocking)

            if nimage.ndim != 6:
                raise ValueError(
                    f"Expected images with shape (B, T, 2, C, H, W), got {tuple(nimage.shape)}"
                )

            with autocast_context(cfg, device):
                image_features = compiled_vision_encoder(
                    nimage.reshape(
                        batch_size * nimage.shape[1] * nimage.shape[2],
                        nimage.shape[3],
                        nimage.shape[4],
                        nimage.shape[5],
                    )
                )
                image_features = image_features.reshape(
                    batch_size,
                    nimage.shape[1],
                    nimage.shape[2],
                    -1,
                )
                image_features = image_features.flatten(start_dim=2)

                obs_feature = torch.cat([image_features, nobs], dim=-1)
                predict_proto = compiled_proto_pred_net(obs_feature.flatten(start_dim=1), proto_snap)

                nobs = nobs[:, :obs_horizon, :]

                if cfg.upsample_proto:
                    upsample_proto = compiled_upsample_proto_net(nproto.flatten(start_dim=1))
                    upsample_proto = upsample_proto.reshape(batch_size, cfg.proto_horizon, -1)
                    obs_cond = torch.cat(
                        [
                            obs_feature.flatten(start_dim=1),
                            upsample_proto.flatten(start_dim=1),
                        ],
                        dim=1,
                    )
                else:
                    obs_cond = torch.cat(
                        [obs_feature.flatten(start_dim=1), nproto.flatten(start_dim=1)],
                        dim=1,
                    )

                noise = torch.randn(naction.shape, device=device)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=device,
                ).long()

                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                noise_pred = compiled_noise_pred_net(
                    noisy_actions,
                    timesteps,
                    global_cond=obs_cond,
                )

                action_loss = nn.functional.mse_loss(noise_pred, noise)
                proto_prediction_loss = nn.functional.mse_loss(
                    predict_proto, nproto.squeeze(1)
                )
                loss = action_loss + proto_prediction_loss

            (loss / grad_accumulation_steps).backward()

            should_step = (batch_idx + 1) % grad_accumulation_steps == 0 or batch_idx == len(dataloader) - 1
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                ema.step(nets)

            epoch_loss.append(float(loss.detach().cpu()))
            epoch_action_loss.append(float(action_loss.detach().cpu()))
            epoch_proto_prediction_loss.append(float(proto_prediction_loss.detach().cpu()))

        wandb.log(
            {
                "epoch": epoch_idx,
                "epoch loss": np.mean(epoch_loss),
                "epoch action loss": np.mean(epoch_action_loss),
                "epoch proto prediction loss": np.mean(epoch_proto_prediction_loss),
            }
        )

        if epoch_idx % cfg.ckpt_frequency == 0 or epoch_idx == cfg.num_epochs - 1:
            torch.save(
                ema.averaged_model.state_dict(),
                os.path.join(save_dir, f"ckpt_{epoch_idx}.pt"),
            )
            save_resume_state(save_dir, epoch_idx, nets, ema, optimizer, lr_scheduler)


if __name__ == "__main__":
    train_diffusion_bc()