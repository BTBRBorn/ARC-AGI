import torch
import torch.nn.functional as F
import time
import torch.distributed as dist

#Calculate the loss for one batch
def calculate_loss(model, config, x, y, grad_accum_num=1):
    B, T = x.size()
    if config.use_mixed_precision:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, config.attention_mode)
            loss = F.cross_entropy(
                logits.view(B * T, config.vocab_size), y.view(B * T)
            )
            loss = loss / grad_accum_num
    else:
        logits = model(x, config.attention_mode)
        loss = F.cross_entropy(logits.view(B * T, config.vocab_size), y.view(B * T))
        loss = loss / grad_accum_num

    return loss

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
    tokens_per_iter //= world_size
    for batch_num, (x, y) in enumerate(dataloader, start=1):
        x, y = x.to(device), y.to(device)
        B, T = x.size()
        tokens_processed += B * T
        if batch_num % grad_accum_num != 0:
            with model.no_sync():
                loss = calculate_loss(model, config, x, y, grad_accum_num)
                total_loss += loss.item()
                loss.backward()
        else: 
            loss = calculate_loss(model, config, x, y, grad_accum_num)
            total_loss += loss.item()
            #Sync all the gradients
            loss.backward()
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
    avg_loss = 0.0
    model.eval()
    iter_steps = 0
    with torch.inference_mode():
        for x, y in dataloader:
            iter_steps += 1
            x, y = x.to(device), y.to(device)
            loss = calculate_loss(model, config, x, y)
            avg_loss += loss.detach()

        avg_loss /= iter_steps
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG) 

    return avg_loss 


def train(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    config,
    num_iter,
    results,
    grad_accum_num,
    tokens_per_iter,
    is_master,
    world_size,
    device,
    train_sampler,
):
    assert (grad_accum_num % world_size) == 0

    current_iter = len(results["train_losses"]) + 1
    end_iter = current_iter + num_iter
    if is_master:
        print(f"Continuing from iteration: {current_iter}")

    for i in range(current_iter, end_iter):
        start = time.perf_counter()
        train_sampler.set_epoch(i)
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
        dt = end - start
        scheduler.step()

        if is_master:
            lr = scheduler.get_last_lr()
            token_per_sec = train_tokens / dt
            results["train_losses"].append(train_loss)

            print(
                f"Iter: {i}/{end_iter - 1}, Train Loss: {train_loss:.4f}, dt: {dt:.4f} s, "
                + f"tokens/sec: {token_per_sec:.2f}, norm: {norm:.4f}, learning_rate: {lr[0]:.6e}"
            )

    return results
