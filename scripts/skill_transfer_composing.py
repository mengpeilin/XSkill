import os
import pickle
import random

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


@hydra.main(
    version_base=None,
    config_path="../config/realworld",
    config_name="skill_transfer_composing_pick_mug_a6000",
)
def train_diffusion_bc(cfg: DictConfig):
    save_dir = cfg.save_dir
    cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "hydra_config.yaml"))
    print(f"output_dir: {save_dir}")

    wandb.init(project=cfg.wandb.project, name=cfg.wandb.run, config=OmegaConf.to_container(cfg))

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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
    )

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

    optimizer = torch.optim.AdamW(
        params=nets.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * cfg.num_epochs,
    )

    for epoch_idx in range(cfg.num_epochs):
        epoch_loss = []
        epoch_action_loss = []
        epoch_proto_prediction_loss = []

        for nbatch in dataloader:
            nobs = nbatch["obs"].to(device)
            batch_size = nobs.shape[0]
            nimage = nbatch["images"].to(device)
            nproto = nbatch["protos"].to(device)
            proto_snap = nbatch["proto_snap"].to(device)
            proto_snap = proto_snap.reshape(batch_size, dataset.snap_frames, -1)
            naction = nbatch["actions"].to(device)

            if nimage.ndim != 6:
                raise ValueError(
                    f"Expected images with shape (B, T, 2, C, H, W), got {tuple(nimage.shape)}"
                )

            image_features = nets["vision_encoder"](
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
            predict_proto = proto_pred_net(obs_feature.flatten(start_dim=1), proto_snap)

            nobs = nobs[:, :obs_horizon, :]

            if cfg.upsample_proto:
                upsample_proto = nets["upsample_proto_net"](nproto.flatten(start_dim=1))
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
            noise_pred = nets["noise_pred_net"](
                noisy_actions,
                timesteps,
                global_cond=obs_cond,
            )

            action_loss = nn.functional.mse_loss(noise_pred, noise)
            proto_prediction_loss = nn.functional.mse_loss(
                predict_proto, nproto.squeeze(1)
            )
            loss = action_loss + proto_prediction_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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


if __name__ == "__main__":
    train_diffusion_bc()