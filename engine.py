import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import time
from utils import create_data


def train_step(model, dataloader, optimizer, config):
    total_loss = 0.0
    model.train()
    for x, y in dataloader:
        x, y = x.to(config.device), y.to(config.device)
        B, T = x.size()
        optimizer.zero_grad()
        if config.use_mixed_precision:
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                logits = model(x, config.attention_mode)
                loss = F.cross_entropy(
                    logits.view(B * T, config.vocab_size), y.view(B * T)
                )
        else:
            logits = model(x, config.attention_mode)
            loss = F.cross_entropy(logits.view(B * T, config.vocab_size), y.view(B * T))
        total_loss += loss.item()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return total_loss / len(dataloader), norm


def val_step(model, dataloader, config):
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(config.device), y.to(config.device)
            B, T = x.size()
            if config.use_mixed_precision:
                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                    logits = model(x, config.attention_mode)
                    loss = F.cross_entropy(
                        logits.view(B * T, config.vocab_size), y.view(B * T)
                    )
            else:
                logits = model(x, config.attention_mode)
                loss = F.cross_entropy(
                    logits.view(B * T, config.vocab_size), y.view(B * T)
                )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    tokenizer,
    config,
    args,
    results,
):
    total_tokens = (
        len(train_dataloader) * config.batch_size * config.block_size
        + len(val_dataloader) * config.batch_size * config.block_size
    )

    for epoch in tqdm(range(args.num_epochs)):
        # Change the training data
        create_data(
            config=config,
            tokenizer=tokenizer,
            save_folder="pretraining/",
            is_train=True,
            rolled=True,
            augmented=True,
        )
        start = time.time()
        train_loss, norm = train_step(model, train_dataloader, optimizer, config)
        val_loss = val_step(model, val_dataloader, config)
        scheduler.step(train_loss)
        lr = scheduler.get_last_lr()
        end = time.time()
        token_per_sec = total_tokens / (end - start)
        results["train_losses"].append(train_loss)
        results["val_losses"].append(val_loss)
        print(
            f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
            + f"tokens/sec: {token_per_sec:.2f}, norm: {norm:.4f}, learning_rate: {lr}"
        )
    return results
