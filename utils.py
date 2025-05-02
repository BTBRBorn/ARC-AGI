import torch
import model as pt
import model_transformer as tt
from pathlib import Path
import matplotlib.pyplot as plt


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


def load_checkpoint(checkpoint_path, weight_only=False):
    checkpoint = torch.load(Path(checkpoint_path), weights_only=weight_only)

    config = checkpoint["config"]

    if config.model_type == "PT":
        model = pt.GPT(config=config).to(config.device)
        assert config.token_len == 1, "Pixel based model has to have token_len equal to 1"
    elif config.model_type == "TT":
        model = tt.Transformer(config=config).to(config.device)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = model.configure_optimizer()
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_iter
    )
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    tokenizer = checkpoint["tokenizer"]

    results = checkpoint["results"]

    return_dict = {
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
