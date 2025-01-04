from pathlib import Path


class Config:
    def __init__(self, args, device):
        self.num_epochs: int = args.num_epochs
        self.learning_rate: float = args.learning_rate
        self.vocab_size: int = args.vocab_size
        self.block_size: int = args.block_size
        self.test_block_size: int = args.test_block_size
        self.n_layer: int = args.n_layer
        self.batch_size: int = args.batch_size
        self.head_size: int = args.head_size
        self.n_head: int = args.n_head
        self.emb_dim: int = self.head_size * self.n_head
        self.data_path: Path = Path(args.data_path)
        self.device: str = device
        self.dataloader_num_workers: int = args.dataloader_num_workers
        self.compile_model: int = args.compile_model
        self.attention_mode: str = args.attention_mode
        self.use_mixed_precision: int = args.use_mixed_precision

    def __str__(self):
        arg_list = [
            f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
            for k, v in vars(self).items()
        ]
        return f'Config({', '.join(arg_list)})'
