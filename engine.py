import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def train_step(model,
               dataloader,
               optimizer,
               config):
    
    model.train()
    for (x, y) in dataloader:
        x, y = x.to(config.device), y.to(config.device)
        B, T = x.size()
        optimizer.zero_grad()
        logits = model(x) 
        loss = F.cross_entropy(logits.view(B*T, config.vocab_size), y.view(B*T))
        loss.backward()
        optimizer.step()

    return loss.item()

def val_step(model,
             dataloader,
             config):

    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for (x, y) in dataloader:
            x, y = x.to(config.device), y.to(config.device)
            B, T = x.size()
            logits = model(x)
            loss = F.cross_entropy(logits.view(B*T, config.vocab_size), y.view(B*T))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train(model,
          train_dataloader,
          val_dataloader,
          optimizer,
          config):

    for epoch in tqdm(range(config.num_epochs)):
        train_loss = train_step(model, train_dataloader, optimizer, config) 
        val_loss = val_step(model, val_dataloader, config)
        print(f'Epoch: {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f} ')
