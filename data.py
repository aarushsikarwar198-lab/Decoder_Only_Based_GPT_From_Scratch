import torch

# ---- Load text and build character-level vocabulary ----
with open("input.txt" , 'r') as file:
    content = file.read()
    chars = sorted(set(content))       # sorted unique characters -> stable ordering
    v_id = {}                          # char -> integer id
    id = 0
    for i in chars:
        v_id[i] = id
        id+= 1
    rv_id = {}                         # integer id -> char (reverse lookup for decoding)
    for j in v_id.items():
        rv_id[j[1]] = j[0]
    ls = []
    for j in content:
        ls.append(v_id[j])             # encode the whole text as a list of ids
    data = torch.tensor(ls , dtype = torch.long)   # long dtype required for embedding / cross_entropy

# ---- 90/10 train/val split (contiguous, no shuffling) ----
n = int(0.9*len(content))
train = data[:n]
val = data[n:]
vos = len(v_id)                        # vocab size, imported by the training script


# Sample a random batch of (input, target) windows from a 1D token tensor
def get_batch(split_data , batch_size , block_size):
    # random start positions; upper bound leaves room for the +1 shifted target window
    ix = torch.randint(0, len(split_data) - block_size - 1, (batch_size,))
    x = torch.stack([split_data[i : i + block_size] for i in ix])          # inputs: tokens i .. i+block_size-1
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])  # targets: same window shifted by 1
    return x, y
# for example x[0] = h e l l and y[0] = e l l o. At each position, y is the next char



v_s = len(chars)                       # vocab size (duplicate of vos)

# string -> list of ids
def encode(s):
    return [v_id[c] for c in s]

# list of ids -> string
def decode(l):
    return "".join(rv_id[i] for i in l)

# ---- Sanity checks, only run when this file is executed directly ----
if __name__ == "__main__":
    print(len(content), len(chars), data.shape)        # 1115394 65 torch.Size([1115394])
    print(len(train), len(val))                        # 1003854 111540

    x, y = get_batch(train, 4, 8)
    print(x.shape, y.shape, x.dtype)                   # [4, 8] [4, 8] torch.int64

    print("".join(rv_id[i] for i in x[0].tolist()))    # decode first input window
    print("".join(rv_id[i] for i in y[0].tolist()))    # decode its target (shifted by 1)

    print(torch.equal(x[:, 1:], y[:, :-1]))            # x shifted left == y shifted right -> confirms alignment
    print(vos)          # True