import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs["config"]
        self.e_proj = nn.Linear(config.emb_dim, 4 * config.emb_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.emb_dim * 4, config.emb_dim)

    def forward(self, x: torch.Tensor):
        return self.c_proj(self.gelu(self.e_proj(x)))


class MaskedSelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs["config"]
        self.config = config
        self.QKV = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=False)
        self.proj = nn.Linear(config.emb_dim, config.emb_dim)

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(config.block_size, config.block_size, device=config.device)
            ),
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        Q, K, V = self.QKV(x).split(
            self.config.emb_dim, dim=2
        )  # (B, T, 3C) -> ((B, T, C), (B, T, C), (B, T, C))

        Q = Q.view(B, T, self.config.n_head, self.config.head_size).transpose(
            1, 2
        )  # (B, T, C) -> (B, n_head, T, head_size)
        K = K.view(B, T, self.config.n_head, self.config.head_size).transpose(
            1, 2
        )  # (B, T, C) -> (B, n_head, T, head_size)
        V = V.view(B, T, self.config.n_head, self.config.head_size).transpose(
            1, 2
        )  # (B, T, C) -> (B, n_head, T, head_size)

        att = (Q @ K.transpose(-2, -1)) * (
            self.config.head_size**-0.5
        )  # (B, n_head, T, head_size) -> (B, n_head, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0.0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = (
            att @ V
        )  # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)

        y = y.transpose(2, 1).reshape(B, T, C)

        return self.proj(y)


class Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs["config"]
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.msa = MaskedSelfAttention(config=config)
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP(config=config)

    def forward(self, x: torch.Tensor):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs["config"]
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pte = nn.Embedding(config.block_size, config.emb_dim)
        self.blocks = nn.ModuleList(
            [Block(config=config) for _ in range(config.n_layer)]
        )
        self.ln = nn.LayerNorm(config.emb_dim)
        self.f_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self.register_buffer(
            "pos_inx", torch.arange(config.block_size, device=config.device)
        )

        self.f_head.weight = self.wte.weight
        assert id(self.f_head.weight) == id(self.wte.weight)

    def forward(self, x: torch.Tensor):
        B, T = x.size()
        x = self.wte(x) + self.pte(self.pos_inx[:T])  # (B, T, C) + (T, C) -> (B, T, C)
        for block in self.blocks:
            x = block(x)  # (B, T, C) -> (B, T, C)
        return self.f_head(self.ln(x))  # (B, T, vocab_size)
