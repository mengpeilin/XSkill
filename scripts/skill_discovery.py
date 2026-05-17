import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from xskill.dataset.dataset import ConcatDataset
from xskill.utility.transform import get_transform_pipeline


def sample_shape(dataset):
    sample = dataset[0]
    if isinstance(sample, tuple) and len(sample) == 2 and hasattr(sample[0], "im_q"):
        return sample[0].im_q.shape
    if hasattr(sample, "im_q"):
        return sample.im_q.shape
    raise TypeError("Unsupported dataset sample format.")


@hydra.main(version_base=None,
            config_path="../config/realworld",
            config_name="skill_discovery")
def pretrain(cfg: DictConfig):
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

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        drop_last=cfg.drop_last)

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

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run,
        save_dir=output_dir,
    )
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg))
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        default_root_dir=output_dir,
        **cfg.Trainer,
    )

    trainer.fit(model=model, train_dataloaders=dataloader, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    pretrain()
