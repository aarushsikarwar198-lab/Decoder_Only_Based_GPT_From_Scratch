import torch 
import torch.nn as nn
import torch.nn.functional as F
from decoder import GPT
from data import get_batch , train , val , vos
import math

# hyperparameters
vocab_size = vos          # vocab size comes from the data module
block_size = 256          # context length (tokens the model sees at once)
d_model = 128             # embedding / hidden dimension
heads = 4                 # attention heads per layer
batch_size = 64
max_lr = 5e-4             # peak learning rate after warmup
min_lr = max_lr/10        # lr floor at the end of cosine decay
steps = 5000              # total training steps
dropout = 0.2
n_layers = 6              # number of transformer blocks
warmup_steps = 500        # steps of linear lr ramp-up
device = "cuda" if torch.cuda.is_available() else "cpu"


# lr schedule: linear warmup -> cosine decay -> flat at min_lr
def get_lr(step):
    # warmup: ramp linearly from ~0 up to max_lr
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # past the schedule, just hold the minimum
    if step >= steps:
        return min_lr
    # cosine decay: progress goes 0 -> 1, coeff goes 1 -> 0
    progress = (step - warmup_steps) / (steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)

# forward pass + cross entropy over every token position
def compute_loss(model , x , y):
    logits = model(x)
    B , T , V = logits.shape
    # flatten (B, T) so each token is one classification over the vocab
    return F.cross_entropy(logits.view(B*T , V) , y.view(B*T))

# average loss over several random batches for train and val (no gradients, eval mode)
@torch.no_grad()
def estimate_loss(model , n_batches = 100):
    model.eval()   # turns off dropout so the numbers are stable
    out = {}
    for name, data in [("train", train), ("val", val)]:
        losses = []
        for _ in range(n_batches):
            x, y = get_batch(data, batch_size, block_size)
            x, y = x.to(device), y.to(device)
            losses.append(compute_loss(model, x, y).item()) # the tensor lives on the GPU. .item() copies the value to the CPU
        out[name] = sum(losses) / len(losses)
    model.train()  # back to train mode (dropout on)
    return out 

def Train_model(model , optimizer , steps , device):
    model.train()
    best_val = float("inf")   # track best val loss to checkpoint on
    for step in range(steps):
       # set this step's lr on every param group
       lr = get_lr(step)
       for groups in optimizer.param_groups:
           groups["lr"]  = lr
       # grab a random batch and move it to the device
       x , y  = get_batch(train, batch_size, block_size)
       x , y = x.to(device) , y.to(device)
       B , T = x.shape
       # standard update: forward, backward, clip, step
       loss = compute_loss(model , x , y)
       optimizer.zero_grad()
       loss.backward()
       torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip grads to avoid blowups
       optimizer.step()

       # every 500 steps (and at the end): evaluate and log
       if (step%500 == 0 or step == steps-1):
           l = estimate_loss(model)
           print(f"step {step}: train {l['train']:.4f} | val {l['val']:.4f} , lr {lr :.2e}")
           # save a checkpoint only when val loss improves
           if(l["val"] < best_val):
               best_val = l["val"]
               torch.save(model.state_dict(), "model.pt")
               print("saved best!")
    # also keep the final weights regardless of val loss
    torch.save(model.state_dict(), "last.pt")

       
if __name__ == "__main__":
    torch.manual_seed(0)   # reproducibility
    model = GPT(vocab_size , dropout , block_size , d_model , n_layers , heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr , weight_decay = 0.1)
    # sanity checks before training
    x, y = get_batch(train, 4, block_size)
    x, y = x.to(device), y.to(device)
    logits = model(x)
    print(logits.shape)      # expect (4, block_size, vocab_size)
    print(compute_loss(model, x, y).item())  # untrained loss should be near ln(vocab_size)

    Train_model(model, optimizer, steps, device)