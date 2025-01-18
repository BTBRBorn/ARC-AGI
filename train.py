from pathlib import Path
import argparse
import torch
import model
import get_dataloaders
import engine
from get_tokenizer import Tokenizer
import utils
from configurations import Config

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--vocab_size", type=int, default=16)
parser.add_argument("--block_size", type=int, default=2048)
parser.add_argument("--n_layer", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--head_size", type=int, default=32)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--data_path_train", type=str, default="data/combined")
parser.add_argument("--data_path_val", type=str, default="data/training")
parser.add_argument("--dataloader_num_workers", type=int, default=2)
parser.add_argument("--compile_model", type=int, choices={0, 1}, default=0)
parser.add_argument("--attention_mode", type=str, default="flash_attention")
parser.add_argument("--use_mixed_precision", type=int, choices={0, 1}, default=1)
parser.add_argument("--checkpoint_save_path", type=str, default="")
parser.add_argument("--checkpoint_load_path", type=str, default="")
parser.add_argument("--scheduler_iter", type=int, default=1000)
parser.add_argument("--weight_decay", type=float, default=1.0)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("-" * 50)
print(f"Device: {device}")
print("-" * 50)

torch.set_float32_matmul_precision("high")


if args.checkpoint_load_path:
    checkpoint_dict = utils.load_checkpoint(Path(args.checkpoint_load_path))

    config = checkpoint_dict["config"]

    gpt = checkpoint_dict["model"]

    optimizer = checkpoint_dict["optimizer"]

    scheduler = checkpoint_dict["scheduler"]

    tokenizer = checkpoint_dict["tokenizer"]

    results = checkpoint_dict["results"]

else:
    config = Config(args, device)

    gpt = model.GPT(config=config).to(config.device)

    optim_groups = [
        {
            "params": [param for param in gpt.parameters() if param.dim() >= 2],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [param for param in gpt.parameters() if param.dim() < 2],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.learning_rate, fused=True, betas=(0.9, 0.95)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_iter
    )

    tokenizer = Tokenizer(config.vocab_size)

    results = {"train_losses": [], "val_losses": []}

print("Model:")
print(gpt)
print("-" * 50)
print(optimizer)
print("-" * 50)
print(f"Total number of parameters: {sum(p.numel() for p in gpt.parameters())}")

if config.compile_model:
    gpt = torch.compile(gpt)

# Create the training data
utils.create_data(
    data_path=config.data_path_train,
    vocab_size=config.vocab_size,
    tokenizer=tokenizer,
    save_folder="pretraining/",
    is_train=True,
    rolled=True,
    augmented=True,
)
# Create the validation data
utils.create_data(
    data_path=config.data_path_val,
    vocab_size=config.vocab_size,
    tokenizer=tokenizer,
    save_folder="pretraining/",
    is_train=False,
    rolled=False,
    augmented=False,
)
train_dataloader, val_dataloader = get_dataloaders.create_dataloaders(config)

results = engine.train(
    model=gpt,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    tokenizer=tokenizer,
    config=config,
    args=args,
    results=results,
)

if args.checkpoint_save_path:
    utils.save_checkpoint(
        checkpoint_path=Path(args.checkpoint_save_path),
        model=gpt,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        config=config,
        results=results,
    )

utils.plot_losses(results)
