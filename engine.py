import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import time
from get_dataloaders import create_dataloaders
import utils
from pathlib import Path


def train_step(model, dataloader, optimizer, config, batch_accum_num):
    total_loss = 0.0
    total_norm = 0.0
    optimizer_steps = 0
    model.train()
    optimizer.zero_grad()
    for batch_num, (x, y) in enumerate(dataloader, start=1):
        x, y = x.to(config.device), y.to(config.device)
        B, T = x.size()
        if config.use_mixed_precision:
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                logits = model(x, config.attention_mode)
                loss = F.cross_entropy(
                    logits.view(B * T, config.vocab_size), y.view(B * T)
                )
                loss = loss / batch_accum_num
        else:
            logits = model(x, config.attention_mode)
            loss = F.cross_entropy(logits.view(B * T, config.vocab_size), y.view(B * T))
            loss = loss / batch_accum_num

        loss.backward()
        total_loss += loss.item()

        if batch_num % batch_accum_num == 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_norm += norm.item()
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1

    # If there is any leftover loss accumulation
    # Its effect will be less but it is still okay
    # We are clipping the gradient anyways
    if len(dataloader) % batch_accum_num != 0:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_norm += norm.item()
        optimizer.step()
        optimizer.zero_grad()
        optimizer_steps += 1

    return total_loss / optimizer_steps, total_norm / optimizer_steps


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
    optimizer,
    scheduler,
    config,
    num_epochs,
    results,
    tokenizer,
    checkpoint_save_path,
    batch_accum_num,
):
    num_shard = len(results["train_losses"]) + 1
    train_dataloader, val_dataloader = create_dataloaders(config, num_shard=num_shard)

    total_tokens = (
        len(train_dataloader) * config.batch_size * config.block_size
        + len(val_dataloader) * config.batch_size * config.block_size
    )

    train_loss = val_step(model, train_dataloader, config)
    val_loss = val_step(model, val_dataloader, config)
    print(f"Continuing from epoch: {len(results['val_losses']) + 1}")
    print(f"Starting training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}")
    print("-" * 100)

    if len(results["val_losses"]) == 0:
        min_val_loss = 1000
    else:
        min_val_loss = min(results["val_losses"])

    checkpoint_flag = False
    for epoch in tqdm(range(num_epochs)):
        num_shard = len(results["train_losses"]) + 1
        train_dataloader, val_dataloader = create_dataloaders(
            config, num_shard=num_shard
        )

        start = time.perf_counter()
        train_loss, norm = train_step(
            model, train_dataloader, optimizer, config, batch_accum_num
        )
        val_loss = val_step(model, val_dataloader, config)
        scheduler.step()
        lr = scheduler.get_last_lr()
        end = time.perf_counter()
        token_per_sec = total_tokens / (end - start)
        results["train_losses"].append(train_loss)
        results["val_losses"].append(val_loss)
        print(
            f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
            + f"tokens/sec: {token_per_sec:.2f}, norm: {norm:.4f}, learning_rate: {lr[0]:.6e}"
        )

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            checkpoint_flag = True

        if (
            checkpoint_save_path
            and len(results["val_losses"]) >= 300
            and checkpoint_flag
        ):
            checkpoint_flag = False
            utils.save_checkpoint(
                checkpoint_path=Path(checkpoint_save_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                config=config,
                results=results,
            )

    return results
