import torch
import torch.nn as nn
from torch.nn import functional as F
num_embeddings = 8192
embedding_dim = 256
d_model = 256
max_len = 1024
num_heads = 4
dropout = 0.1
d_ff = 1024
n_layer = 4
n_heads = 4

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self,d_model,max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self,x):
        L = x.shape[1]
        assert L <= self.max_len
        return x + self.pe[:L]

class  CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads,dropout,position_type="sinusoidal",max_len=1024,attention_type="sdpa"):
        super().__init__()
        self.d_model = d_model
        self.attention_type = attention_type
        if attention_type not in ("sdpa", "linear"):
            raise ValueError(f"未知注意力类型：{attention_type}")

        self.n_heads = n_heads
        self.position_type = position_type
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        if position_type == "rope":
            self.rope = RotaryPositionEncoding(self.head_dim, max_len)


    def forward(self,x):
        B, L, D = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        if self.attention_type == "linear":
            rope = self.rope if self.position_type == "rope" else None
            output = causal_linear_attention(Q, K, V, rope=rope)
        else:
            if self.position_type == "rope":
                Q = self.rope(Q)
                K = self.rope(K)

            output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        output = output.transpose(1, 2).reshape(B, L, D)
        output = self.out(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout,position_type="sinusoidal",max_len=1024,attention_type="sdpa"):
        super().__init__()
        self.attention_type = attention_type
        self.attention = CausalSelfAttention(
            d_model, n_heads, dropout,
            position_type=position_type,
            max_len=max_len,
            attention_type=attention_type,
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.layer_norm1(x + self.dropout(self.attention(x)))
        x = self.layer_norm2(x + self.feed_forward(x))
        return x


class languageModel(nn.Module):
    def __init__(self, d_model, n_heads, d_ff,n_layer, dropout, max_len, num_embeddings,position_type="sinusoidal",attention_type="sdpa"):
        super().__init__()
        self.position_type = position_type
        self.token_embedding = nn.Embedding(num_embeddings, d_model)
        self.position_encoding = SinusoidalPositionEncoding(d_model, max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, num_embeddings,bias = False)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads, d_ff, dropout,position_type=position_type,max_len=max_len,attention_type=attention_type) for _ in range(n_layer)])
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self,x):
        x = self.token_embedding(x)
        if self.position_type == "sinusoidal":
            x = self.position_encoding(x)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.lm_head(x)
        return x

class RotaryPositionEncoding(nn.Module):
    def __init__(self, head_dim, max_len):
        super().__init__()
        self.max_len = max_len
        self.head_dim = head_dim
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_len, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def forward(self, x):
        L = x.shape[-2]
        assert L <= self.max_len
        cos = self.cos_cached[:L][None, None, :, :].to(dtype=x.dtype)
        sin = self.sin_cached[:L][None, None, :, :].to(dtype=x.dtype)
        # 每两个相邻维度组成一组
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 二维旋转
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # 交替排列旋转后的维度
        x_rotated = torch.stack(
            [rotated_even, rotated_odd], dim=-1
        ).flatten(-2)

        return x_rotated

def causal_linear_attention(q, k, v, eps=1e-6, rope=None):
    q_features = F.elu(q) + 1
    k_features = F.elu(k) + 1

    # 分子使用旋转后的特征
    q_rotated = rope(q_features) if rope is not None else q_features
    k_rotated = rope(k_features) if rope is not None else k_features

    kv = k_rotated.unsqueeze(-1) * v.unsqueeze(-2)
    kv_cumsum = kv.cumsum(dim=2)

    numerator = torch.einsum(
        "bhld,bhldv->bhlv", q_rotated, kv_cumsum
    )

    # 分母使用未旋转的正值特征
    k_cumsum = k_features.cumsum(dim=2)
    denominator = torch.einsum(
        "bhld,bhld->bhl", q_features, k_cumsum
    ).unsqueeze(-1)

    return numerator / (denominator + eps)


if __name__ == "__main__":
    model = languageModel(d_model, n_heads, d_ff, 1, dropout, max_len, num_embeddings)



