from dataclasses import dataclass
from pathlib import Path
import argparse
import torch
import model
import get_dataloaders
import engine

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--vocab_size", type=int, default=16)
parser.add_argument("--block_size", type=int, default=4096)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--head_size", type=int, default=16)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--data_path", type=str, default="data/pretraining")
parser.add_argument("--dl_num_workers", type=int, default=4)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print('-'*50)
print(f"Device: {device}")
print('-'*50)

@dataclass
class Config:
    num_epochs: int = args.num_epochs
    learning_rate: float = args.learning_rate
    vocab_size: int = args.vocab_size
    block_size: int = args.block_size
    n_layer: int = args.n_layer
    batch_size: int = args.batch_size
    head_size: int = args.head_size
    n_head: int = args.n_head
    emb_dim: int = head_size * n_head
    data_path: int = Path(args.data_path)
    device: str = device
    dl_num_workers: int = args.dl_num_workers

config = Config()

gpt = model.GPT(config=config).to(config.device)
print('Model:')
print(gpt)
print('-'*50)
print(f'Total number of parameters: {sum(p.numel() for p in gpt.parameters())}')
optimizer = torch.optim.Adam(gpt.parameters(), lr=config.learning_rate)

train_dataloader, val_dataloader = get_dataloaders.create_dataloaders(config.data_path / 'training.npy',
                                                                      config.data_path / 'validation.npy',
                                                                      config.batch_size,
                                                                      config.block_size,
                                                                      config.dl_num_workers)

engine.train(gpt,
             train_dataloader,
             val_dataloader,
             optimizer,
             config)