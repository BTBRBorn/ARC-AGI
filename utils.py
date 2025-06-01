import torch
import model as pt
import model_transformer as tt
from pathlib import Path
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP

def configure_optimizer(model):
    config = model.module.config

    optim_groups = [
        {
            "params": [p for p in model.parameters() if p.dim() >= 2],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for p in model.parameters() if p.dim() < 2],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )

    return optimizer

def save_checkpoint(
    checkpoint_path, model, optimizer, scheduler, tokenizer, config, results
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.parent.exists():
        checkpoint_path.parent.mkdir(parents=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "tokenizer": tokenizer,
            "config": config,
            "results": results,
        },
        Path(checkpoint_path),
    )


def load_checkpoint(checkpoint_path, device, compile_model, with_model=True, weight_only=False):

    checkpoint = torch.load(Path(checkpoint_path), weights_only=weight_only)

    config = checkpoint["config"]

    if with_model:
        if config.model_type == "PT":
            base_model = pt.GPT(config=config, device=device).to(device)
            assert config.token_len == 1, "Pixel based model has to have token_len equal to 1"
        elif config.model_type == "TT":
            base_model = tt.Transformer(config=config).to(device)

        base_model.load_state_dict(checkpoint["model_state_dict"])

        if compile_model:
            compiled_model = torch.compile(base_model)
            model = DDP(compiled_model, device_ids=[device])
        else:
            model = DDP(base_model, device_ids=[device])

        optimizer = configure_optimizer(model)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.scheduler_iter
        )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        model, optimizer, scheduler = None, None, None

    tokenizer = checkpoint["tokenizer"]

    results = checkpoint["results"]

    return_dict = {
        "base_model": base_model,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "tokenizer": tokenizer,
        "config": config,
        "results": results,
    }

    return return_dict


def plot_losses(results):
    train_loss, val_loss = results["train_losses"], results["val_losses"]
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="training", color="green")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="validation", color="red")
    plt.title("Train and Validation Losses vs Num Epochs")
    plt.xlabel("Num Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
