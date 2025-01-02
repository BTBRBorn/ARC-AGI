from dataclasses import dataclass
from pathlib import Path
import argparse
import torch
import model
import get_dataloaders
import engine
from get_tokenizer import Tokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--vocab_size", type=int, default=16)
parser.add_argument("--block_size", type=int, default=7500)
parser.add_argument("--test_block_size", type=int, default=2056)
parser.add_argument("--n_layer", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--head_size", type=int, default=2)
parser.add_argument("--n_head", type=int, default=2)
parser.add_argument("--data_path", type=str, default="data/training")
parser.add_argument("--dl_num_workers", type=int, default=2)
parser.add_argument("--compile_model", type=int, choices={0, 1}, default=1)
parser.add_argument("--attention_mode", type=str, default="flash_attention")
parser.add_argument("--use_mixed_precision", type=int, choices={0, 1}, default=1)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("-" * 50)
print(f"Device: {device}")
print("-" * 50)

torch.set_float32_matmul_precision("high")


@dataclass
class Config:
    num_epochs: int = args.num_epochs
    learning_rate: float = args.learning_rate
    vocab_size: int = args.vocab_size
    block_size: int = args.block_size
    test_block_size: int = args.test_block_size
    n_layer: int = args.n_layer
    batch_size: int = args.batch_size
    head_size: int = args.head_size
    n_head: int = args.n_head
    emb_dim: int = head_size * n_head
    data_path: Path = Path(args.data_path)
    device: str = device
    dl_num_workers: int = args.dl_num_workers
    compile_model: int = args.compile_model
    attention_mode: str = args.attention_mode
    use_mixed_precision: int = args.use_mixed_precision


config = Config()

gpt = model.GPT(config=config).to(config.device)
if config.compile_model:
    gpt = torch.compile(gpt)

print("Model:")
print(gpt)
print("-" * 50)
print(f"Total number of parameters: {sum(p.numel() for p in gpt.parameters())}")

no_decay_params = [p for p in gpt.parameters() if p.dim() <= 1]
decay_params = [p for p in gpt.parameters() if p.dim() > 1]

optim_groups = [
    {"params": decay_params, "weight_decay": 0.0},
    {"params": no_decay_params, "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, fused=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

tokenizer = Tokenizer(config.vocab_size)
train_dataloader, val_dataloader = get_dataloaders.create_dataloaders(config, tokenizer)

engine.train(gpt, train_dataloader, val_dataloader, optimizer, scheduler, config)
