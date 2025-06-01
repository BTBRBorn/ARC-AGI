from pathlib import Path
import argparse
import pickle
import itertools
import torch

from get_tokenizer import Tokenizer
from get_dataloaders import create_dataloaders
import model as pt
import model_transformer as tt
import engine
import utils
from configurations import Config

# Modules needed for multi-gpu training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# torchrun will setup rank, local_rank and world_size for us
dist.init_process_group(backend="nccl")
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
master_process = ddp_rank == 0


parser = argparse.ArgumentParser()


parser.add_argument("--model_type", type=str, choices={"PT", "TT"}, default="PT")
parser.add_argument("--max_iter", type=int, default=100)
parser.add_argument("--tokens_per_iter", type=int, default=1e6)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--vocab_size", type=int, default=16)
parser.add_argument("--block_size", type=int, default=2048)
parser.add_argument("--token_len", type=int, default=1)
parser.add_argument("--n_layer", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--batch_accum_num", type=int, default=1)
parser.add_argument("--head_size", type=int, default=32)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--data_path", type=str, default="data/pretraining")
parser.add_argument("--dataloader_num_workers", type=int, default=2)
parser.add_argument("--compile_model", type=int, choices={0, 1}, default=0)
parser.add_argument("--attention_mode", type=str, default="flash_attention")
parser.add_argument("--use_mixed_precision", type=int, choices={0, 1}, default=1)
parser.add_argument("--checkpoint_save_path", type=str, default="")
parser.add_argument("--checkpoint_load_path", type=str, default="")
parser.add_argument("--scheduler_iter", type=int, default=1200)
parser.add_argument("--weight_decay", type=float, default=1.0)
parser.add_argument("--tokenizer_path", type=str, default="")

args = parser.parse_args()


torch.set_float32_matmul_precision("high")


if args.checkpoint_load_path:
    checkpoint_dict = utils.load_checkpoint(
        Path(args.checkpoint_load_path),
        compile_model=args.compile_model,
        device=device,
    )

    config = checkpoint_dict["config"]

    base_model, model = checkpoint_dict["base_model"], checkpoint_dict["model"]
    models = (base_model, model)

    optimizer = checkpoint_dict["optimizer"]

    scheduler = checkpoint_dict["scheduler"]

    tokenizer = checkpoint_dict["tokenizer"]

    results = checkpoint_dict["results"]

else:
    config = Config(args)

    if config.model_type == "PT":
        base_model = pt.GPT(config=config, device=device).to(device)

        assert config.token_len == 1, (
            "Pixel based model has to have token_len equal to 1"
        )
    elif config.model_type == "TT":
        model = tt.Transformer(config=config).to(device)
        base_model = model

    if args.compile_model:
        model_compiled = torch.compile(base_model)
        model = DDP(model_compiled, device_ids=[device])
    else:
        model = DDP(base_model, device_ids=[device])

    models = (base_model, model)

    optimizer = utils.configure_optimizer(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_iter,
    )

    if args.tokenizer_path:
        with open(Path(args.tokenizer_path), "rb") as fhandle:
            tokenizer = pickle.load(fhandle)
    else:
        tokenizer = Tokenizer(config.vocab_size)

    results = {"train_losses": [], "val_losses": []}

if master_process:
    print("-" * 50)
    print(f"Device: {device}")
    print("-" * 50)
    print("Model:")
    print(model)
    print("-" * 50)
    print(optimizer)
    print("-" * 50)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    print("-" * 50)
    print(
        f"Total number of tokens in every training step: {config.batch_size * args.batch_accum_num * config.block_size}"
    )


train_dataloader, val_dataloader = create_dataloaders(config, args.data_path)
train_dataloader_cycle = itertools.cycle(train_dataloader)

results = engine.train(
    models=models,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config,
    train_dataloader=train_dataloader_cycle,
    val_dataloader=val_dataloader,
    max_iter=args.max_iter,
    results=results,
    tokenizer=tokenizer,
    checkpoint_save_path=Path(args.checkpoint_save_path),
    batch_accum_num=args.batch_accum_num,
    tokens_per_iter=args.tokens_per_iter,
    is_master=master_process,
    world_size=world_size,
    device=device,
)


if args.checkpoint_save_path and len(results["val_losses"]) < 300 and master_process:
    utils.save_checkpoint(
        checkpoint_path=Path(args.checkpoint_save_path),
        model=base_model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        config=config,
        results=results,
    )

dist.destroy_process_group()
