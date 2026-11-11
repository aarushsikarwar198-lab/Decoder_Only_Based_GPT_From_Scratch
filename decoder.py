import torch
import torch.nn as nn 
from attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self , d_model , heads , dropout ):
        super(DecoderBlock , self).__init__()
        self.d_model = d_model 
        self.heads =heads
        self.dropout = dropout 
        self.drop = nn.Dropout(dropout)                   # applied on each sublayer output before the residual add
        self.MMHA = MultiHeadAttention(d_model , heads)   # causal self-attention, each token sees only the past
        self.FFN_1 = nn.Linear(d_model , 4*d_model)       # blow up to 4x, standard GPT ratio
        self.FFN_2 = nn.Linear(4*d_model , d_model)       # and squeeze back down
        self.activation = nn.GELU()                       # GPT uses GELU rather than ReLU
        self.ln1 = nn.LayerNorm(d_model)                  # norm before attention
        self.ln2 = nn.LayerNorm(d_model)                  # norm before FFN

    def forward(self ,x):
        # x: (B, T, d_model)
        # norm first, then attention, then add back the residual
        x = x + self.drop(self.MMHA(self.ln1(x))) #GPT's pre norm 
        # same deal for the feed-forward part: expand -> GELU -> project back
        # the FFN works on each token independently, no mixing across positions
        x = x + self.drop(self.FFN_2(self.activation(self.FFN_1(self.ln2(x)))))
        return x
        

class GPT(nn.Module):
    def __init__(self, vocab_size , dropout , block_size , d_model , n_layers , heads):
        super(GPT , self).__init__()
        self.vocab_size = vocab_size                      # number of distinct tokens
        self.dropout = dropout 
        self.block_size = block_size                      # max context length the model can handle
        self.d_model = d_model                            # width of the residual stream
        self.n_layers = n_layers                          # how many decoder blocks to stack
        self.heads = heads                                # attention heads per block
        self.drop = nn.Dropout(self.dropout)              # dropout right after the embeddings
        self.token_embedding = nn.Embedding(vocab_size , d_model)        # token id -> learned vector
        self.positional_embedding = nn.Embedding(block_size , d_model)   # learned, not sinusoidal
        self.nx = nn.ModuleList([DecoderBlock(d_model , heads , dropout) for _ in range(n_layers)])   # stack of decoder blocks
        self.Final_out = nn.Linear(d_model , vocab_size)  # maps back to vocab for logits
        self.final_norm = nn.LayerNorm(d_model)           # last norm before the head

    def forward(self, x):
        # x: (B, T) integer token ids
        B, T = x.shape
        t_e = self.token_embedding(x)                     # (B, T, d_model) what each token is
        p_e = self.positional_embedding(torch.arange(T, device = x.device) )   # (T, d_model) positions 0..T-1, broadcasts over batch
        e_e = self.drop(t_e + p_e)                        # token + position info together
        for k in self.nx:                                 # run through the decoder blocks one by one
            e_e = k(e_e)
        return self.Final_out(self.final_norm(e_e))       # (B, T, vocab_size) raw logits, no softmax here