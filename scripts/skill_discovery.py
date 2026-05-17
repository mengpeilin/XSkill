from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from xskill.dataset.dataset import ConcatDataset
from xskill.utility.transform import get_transform_pipeline


def configure_runtime_speedups(cfg: DictConfig):
    matmul_precision = cfg.get("matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))

    if not torch.cuda.is_available():
        return

    allow_tf32 = bool(cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True


def sample_shape(dataset):
    sample = dataset[0]
    if isinstance(sample, tuple) and len(sample) == 2 and hasattr(sample[0], "im_q"):
        return sample[0].im_q.shape
    if hasattr(sample, "im_q"):
        return sample.im_q.shape
    raise TypeError("Unsupported dataset sample format.")


def build_wandb_init_kwargs(cfg: DictConfig, output_dir: str) -> dict:
    wandb_cfg = cfg.wandb
    robot_mask = cfg.robot_dataset.get("mask")
    human_mask = cfg.human_dataset.get("mask")
    robot_mask_token = Path(str(robot_mask)).stem if robot_mask else "robot_mask_all"
    human_mask_token = Path(str(human_mask)).stem if human_mask else "human_mask_all"
    group_name = str(wandb_cfg.group)
    run_name = wandb_cfg.get("run_name")
    if not run_name:
        run_name = "__".join(
            [
                group_name,
                robot_mask_token,
                human_mask_token,
                f"seed_{int(cfg.get('seed', 0))}",
                Path(output_dir).name,
            ]
        )
    tags = [str(tag) for tag in OmegaConf.to_container(wandb_cfg.tags, resolve=True)]
    return {
        "project": str(wandb_cfg.project),
        "group": group_name,
        "job_type": str(wandb_cfg.job_type),
        "name": str(run_name),
        "tags": tags,
        "dir": output_dir,
        "config": OmegaConf.to_container(cfg, resolve=False),
    }


@hydra.main(version_base=None,
            config_path="../config/realworld",
            config_name="skill_discovery_a6000")
def pretrain(cfg: DictConfig):
    configure_runtime_speedups(cfg)
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"output_dir: {output_dir}")
    pretrain_pipeline = get_transform_pipeline(cfg.augmentations)

    robot_dataset = hydra.utils.instantiate(cfg.robot_dataset)
    human_dataset = hydra.utils.instantiate(cfg.human_dataset)
    robot_len = len(robot_dataset)
    human_len = len(human_dataset)

    if robot_len == 0 and human_len == 0:
        raise ValueError("robot_dataset and human_dataset are both empty.")

    if robot_len == 0:
        train_dataset = human_dataset
        dataset_mode = "single_human"
    elif human_len == 0:
        train_dataset = robot_dataset
        dataset_mode = "single_robot"
    else:
        combine_mode = "max" if cfg.get("repeat_shorter_dataset", False) else "min"
        train_dataset = ConcatDataset(robot_dataset, human_dataset, mode=combine_mode)
        dataset_mode = f"concat_{combine_mode}"

    print("robot dataset len:", robot_len)
    print("human dataset len:", human_len)
    print("dataset mode:", dataset_mode)

    dataloader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "shuffle": True,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers if cfg.num_workers > 0 else False,
        "drop_last": cfg.drop_last,
    }
    if cfg.num_workers > 0 and cfg.get("prefetch_factor") is not None:
        dataloader_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)

    steps_per_epoch = len(dataloader)

    model = hydra.utils.instantiate(
        cfg.Model,
        steps_per_epoch=steps_per_epoch,
        pretrain_pipeline=pretrain_pipeline,
    )

    print("dataset len: ", len(train_dataset))
    print(sample_shape(train_dataset))

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=cfg.callback.every_n_epoch,
        save_top_k=-1,
        dirpath=output_dir,
        filename="{epoch:02d}",
    )

    wandb_init_kwargs = build_wandb_init_kwargs(cfg, output_dir)
    print(f"wandb project: {wandb_init_kwargs['project']}")
    print(f"wandb group: {wandb_init_kwargs['group']}")
    print(f"wandb run: {wandb_init_kwargs['name']}")
    wandb.init(**wandb_init_kwargs)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        default_root_dir=output_dir,
        **cfg.Trainer,
    )

    trainer.fit(model=model, train_dataloaders=dataloader, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    pretrain()
