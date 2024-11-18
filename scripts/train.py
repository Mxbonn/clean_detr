import logging
import tempfile
from collections import defaultdict
from datetime import datetime

import hydra
import torch
import torch.distributed as dist
from hydra.utils import get_method, instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm

import wandb
from detr.lr_scheduler import get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    if dist.is_torchelastic_launched():
        dist.init_process_group()

    if dist.is_initialized():
        device = torch.device(f"cuda:{dist.get_rank()}")
    else:
        device = torch.device(cfg.device)

    model = instantiate(cfg.model)
    model = model.to(device)
    uncompiled_model = model  # needed for saving the model
    if cfg.compile_model:
        model = torch.compile(model)

    if torch.distributed.is_initialized():
        model = DistributedDataParallel(model)

    loss = instantiate(cfg.loss)

    assert isinstance(model, torch.nn.Module)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)

    logger.info("Loading datasets")
    train_dataset: Dataset = instantiate(cfg.data.datasets.train)
    validation_dataset: Dataset = instantiate(cfg.data.datasets.validation)

    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
        validation_sampler = DistributedSampler(validation_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)  # pyright: ignore
        validation_sampler = SequentialSampler(validation_dataset)  # pyright: ignore

    batch_train_sampler = BatchSampler(train_sampler, batch_size=cfg.batch_size, drop_last=True)

    train_collate = get_method(cfg.data.train_collate)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=train_collate,
        pin_memory=True,
    )
    validation_collate = get_method(cfg.data.validation_collate)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        sampler=validation_sampler,
        num_workers=cfg.num_workers,
        collate_fn=validation_collate,
        pin_memory=True,
    )

    num_training_steps = len(train_loader) * cfg.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.lr_scheduler.warmup_steps, num_training_steps)
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.LRScheduler)

    # wandb stuff
    if not dist.is_initialized() or dist.get_rank() == 0:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        run_name = f"{cfg.semantic_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=cfg_dict,
        )

    for epoch in tqdm(range(cfg.epochs), desc="Epochs", disable=(dist.is_initialized() and dist.get_rank() != 0)):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        train_epoch(model, loss, train_loader, optimizer, lr_scheduler, device, cfg)
        validate(model, loss, validation_loader, device)
        if dist.is_initialized():
            dist.barrier()
        if not dist.is_initialized() or dist.get_rank() == 0:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fp:
                path = fp.name
                torch.save(uncompiled_model.state_dict(), path)
                checkpoint_name = f"model-{wandb.run.id}"  # pyright: ignore
                artifact = wandb.Artifact(checkpoint_name, type="model")
                artifact.add_file(path)
                wandb.log_artifact(artifact)
            wandb.log({"epoch": epoch})


def train_epoch(model, loss_module, dataloader, optimizer, scheduler, device, cfg):
    model.train()
    gradient_accumulation_steps = max(1, cfg.gradient_accumulation_steps)
    for i, (inputs, targets) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        leave=False,
        disable=(dist.is_initialized() and dist.get_rank() != 0),
    ):
        inputs = inputs.to(device)
        if isinstance(targets, (list, tuple)):
            targets = [target.to(device) for target in targets]
        else:
            targets = targets.to(device)

        outputs = model(inputs)

        loss, loss_dict = loss_module(outputs, targets)

        # wandb
        if not dist.is_initialized() or dist.get_rank() == 0:
            if i % cfg.wandb.log_interval == 0:
                lrs = scheduler.get_last_lr()
                for idx, lr in enumerate(lrs):
                    wandb.log({f"lr/{idx}": lr}, commit=False)
                for key, value in loss_dict.items():
                    wandb.log({f"train/loss/{key}": value}, commit=False)
                wandb.log({"train/loss/weighted": loss.item()})

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            if cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
    # zero_grad for the last batch in case it was not a multiple of gradient_accumulation_steps
    optimizer.zero_grad()


@torch.no_grad()
def validate(model, loss_module, dataloader, device):
    model.eval()
    mean_loss_metrics = defaultdict(MeanMetric)
    for inputs, targets in tqdm(dataloader, leave=False, disable=(dist.is_initialized() and dist.get_rank() != 0)):
        inputs = inputs.to(device)
        if isinstance(targets, (list, tuple)):
            targets = [target.to(device) for target in targets]
        else:
            targets = targets.to(device)

        outputs = model(inputs)

        loss, loss_dict = loss_module(outputs, targets)
        for key, value in loss_dict.items():
            mean_loss_metrics[key].update(value)
        mean_loss_metrics["weighted"].update(loss.item())
    for key, metric in mean_loss_metrics.items():
        mean_loss = metric.compute()
        if not dist.is_initialized() or dist.get_rank() == 0:
            wandb.log({f"validation/loss/{key}": mean_loss}, commit=False)


if __name__ == "__main__":
    main()
