import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import time


def train_step(model, dataloader, optimizer, config):
    total_loss = 0.0
    model.train()
    for x, y in dataloader:
        x, y = x.to(config.device), y.to(config.device)
        B, T = x.size()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(B * T, config.vocab_size), y.view(B * T))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def val_step(model, dataloader, config):
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(config.device), y.to(config.device)
            B, T = x.size()
            logits = model(x)
            loss = F.cross_entropy(logits.view(B * T, config.vocab_size), y.view(B * T))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train(model, train_dataloader, val_dataloader, optimizer, config):
    total_tokens = (
        len(train_dataloader) * config.batch_size * config.block_size
        + len(val_dataloader) * config.batch_size * config.block_size
    )
    for epoch in tqdm(range(config.num_epochs)):
        start = time.time()
        train_loss = train_step(model, train_dataloader, optimizer, config)
        val_loss = val_step(model, val_dataloader, config)
        end = time.time()
        token_per_sec = total_tokens / (end - start)
        print(
            f"Epoch: {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, \
              tokens/sec:{token_per_sec:.2f}"
        )
