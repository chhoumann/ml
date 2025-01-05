import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # Prevent division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), GELU(), nn.Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        drop_rate: float,
        num_heads: int,
        qkv_bias=False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduces projection dim to match desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # To combine head outputs
        self.dropout = nn.Dropout(drop_rate)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # Tensor shape (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  # implicitly split the matrix by adding num_heads dimension, then unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(
            2, 3
        )  # compute dot product for each head
        mask_bool = self.mask.bool()[
            :num_tokens, :num_tokens
        ]  # masks truncated to the number of tokens

        attn_scores.masked_fill_(mask_bool, -torch.inf)  # uses mask to fill attn scores

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(
            1, 2
        )  # tensor shape: (b, num_tokens, n_heads, head_dim)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional linear projection
        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length, num_heads, drop_rate, qkv_bias):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm(emb_dim),
                    MultiHeadAttention(
                        d_in=emb_dim,
                        d_out=emb_dim,
                        context_length=context_length,
                        drop_rate=drop_rate,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                    ),
                    nn.Dropout(drop_rate),
                ),
                nn.Sequential(
                    LayerNorm(emb_dim), FeedForward(emb_dim), nn.Dropout(drop_rate)
                ),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        emb_dim,
        n_heads,
        n_layers,
        drop_rate,
        qkv_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(drop_rate)
        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    context_length=context_length,
                    emb_dim=emb_dim,
                    num_heads=n_heads,
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias,
                    **kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
