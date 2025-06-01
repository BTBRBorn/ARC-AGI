import torch
import torch.nn.functional as F
import time


def train_step(
    model,
    dataloader,
    optimizer,
    config,
    grad_accum_num,
    tokens_per_iter,
    world_size,
    device,
):
    total_loss = 0.0
    total_norm = 0.0
    optimizer_steps = 0
    tokens_processed = 0
    model.train()
    optimizer.zero_grad()
    grad_accum_num //= world_size
    tokens_per_iter //= world_size
    for batch_num, (x, y) in enumerate(dataloader, start=1):
        x, y = x.to(device), y.to(device)
        B, T = x.size()
        num_tokens = B * T
        if config.use_mixed_precision:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(x, config.attention_mode)
                loss = F.cross_entropy(
                    logits.view(B * T, config.vocab_size), y.view(B * T)
                )
                loss = loss / grad_accum_num
        else:
            logits = model(x, config.attention_mode)
            loss = F.cross_entropy(logits.view(B * T, config.vocab_size), y.view(B * T))
            loss = loss / grad_accum_num

        loss.backward()
        total_loss += loss.item()

        tokens_processed += num_tokens * world_size
        if batch_num % grad_accum_num == 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_norm += norm.item()
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1
            if tokens_processed >= tokens_per_iter:
                break

    return (
        total_loss / optimizer_steps,
        total_norm / optimizer_steps,
        tokens_processed,
    )


def val_step(model, dataloader, config, device):
    total_loss = 0.0
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            B, T = x.size()
            if config.use_mixed_precision:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
    train_dataloader,
    config,
    max_iter,
    results,
    grad_accum_num,
    tokens_per_iter,
    is_master,
    world_size,
    device,
):
    assert (grad_accum_num % world_size) == 0
    if is_master:
        print(f"Continuing from iteration: {len(results['train_losses']) + 1}")

    for i in range(max_iter):
        start = time.perf_counter()
        train_loss, norm, train_tokens = train_step(
            model,
            train_dataloader,
            optimizer,
            config,
            grad_accum_num,
            tokens_per_iter,
            world_size,
            device,
        )

        end = time.perf_counter()
        scheduler.step()

        if is_master:
            lr = scheduler.get_last_lr()
            token_per_sec = train_tokens / (end - start)
            results["train_losses"].append(train_loss)

            print(
                f"Iter: {i + 1}/{max_iter}, Train Loss: {train_loss:.4f}, "
                + f"tokens/sec: {token_per_sec:.2f}, norm: {norm:.4f}, learning_rate: {lr[0]:.6e}"
            )

    return results
