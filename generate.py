import torch 
import torch.nn as nn
import torch.nn.functional as F
from decoder import GPT
from data import get_batch , train , val , vos , encode, decode
import math

# ---- Hyperparameters (must match the ones used during training) ----
vocab_size = vos          # vocabulary size, imported from data.py
block_size = 256          # max context length (tokens)
d_model = 128             # embedding / hidden dimension
heads = 4                 # attention heads per layer
batch_size = 64           # (unused here, training leftover)
max_lr = 5e-4             # peak learning rate (training leftover)
min_lr = max_lr/10        # final learning rate (training leftover)
steps = 5000              # training steps (training leftover)
dropout = 0.2             # dropout probability
n_layers = 6              # number of transformer blocks
warmup_steps = 500        # LR warmup steps (training leftover)
device = "cuda" if torch.cuda.is_available() else "cpu"   # use GPU if available


# Build the model with the same architecture as training and move it to the device
model = GPT(vocab_size, dropout, block_size, d_model, n_layers, heads).to(device)

# Load the trained weights from disk (map_location handles CPU/GPU mismatch)
model.load_state_dict(torch.load("model.pt" , map_location = device))


@torch.no_grad()  # no gradients needed for inference
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: (B, T) tensor of token ids used as the starting context
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens if it grew too long
            idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
            # forward pass -> logits of shape (B, T, vocab_size)
            logits = model(idx_cond)
            # keep only the last position's logits and apply temperature scaling
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # find the k-th largest logit and mask everything below it
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample the next token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append the sampled token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



# Read the starting text from the user
prompt = str(input("prompt:"))
model.eval()   # switch to eval mode (disables dropout)
# Encode the prompt into token ids and wrap in a batch dimension -> (1, T)
context = torch.tensor(
     [encode(prompt)],
     dtype  = torch.long , 
     device = device
)

# Generate 32 new tokens, temperature 0.8, top-k sampling with k=10
output = generate(model , context , 32 , 0.8 , 10)
# Decode the token ids back to text and print
print(decode(output[0].tolist()))