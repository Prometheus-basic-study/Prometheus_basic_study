import numpy as np
import matplotlib.pyplot as plt

import torch as th
import torch.nn as nn
import torch.nn.functional as F


# class PositionalEncoding(nn.Module):
#     def __init__(self, batch_first=True):
#         """
#         if not batch_first: [seq_len, batch_size, d_model]
#         if batch_first: [batch_size, seq_len, d_model]
#         """
#         super().__init__()
#         self.batch_first = batch_first
#
#     def positional_vector(self, d_model, seq_len):
#         angles = 10000 ** (2 * (th.arange(0, d_model) // 2) / d_model)
#         angles = th.arange(0, seq_len).unsqueeze(1) / angles.unsqueeze(0)   # pos : [0 ~ seq_len)
#         # even indices in the array are replaced with sin and odd indices are replaced with cos
#         angles = th.where(th.arange(0, d_model) % 2 == 0, th.sin(angles), th.cos(angles))
#         return angles.unsqueeze(0 if self.batch_first else 1)
#
#     def forward(self, x):
#         seq_len = x.shape[1 if self.batch_first else 0]
#         d_model = x.shape[-1]
#         x = x + self.positional_vector(d_model, seq_len)
#         return x

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 batch_first: bool = True):

        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        density = th.exp(-th.arange(0, emb_dim, 2) * np.log(10000) / emb_dim)
        pos = th.arange(0, max_len).unsqueeze(1)
        pos_embedding = th.zeros((max_len, emb_dim))
        pos_embedding[:, 0::2] = th.sin(pos * density)
        pos_embedding[:, 1::2] = th.cos(pos * density)
        pos_embedding = pos_embedding.unsqueeze(-2)  # [max_len, 1, emb_dim]

        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        # x = [seq_len, batch_size, emb_dim] or [batch_size, seq_len, emb_dim]
        if self.batch_first:
            return self.dropout(x + self.pos_embedding[:x.size(1), :].permute(1, 0, 2))
        else:
            return self.dropout(x + self.pos_embedding[:x.size(0), :])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, batch_first=True):
        """
        Multi-Head Attention
        :param d_model: hidden size of the model, which is also the embedding size
        :param n_heads: number of heads
        """
        super().__init__()
        self.batch_first = batch_first
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def split_head(self, x):
        batch_size = x.shape[0 if self.batch_first else 1]
        x = x.view(batch_size, -1, self.n_heads, self.d_head)   # [bs, seq_len, n_heads, d_head]
        if self.batch_first:
            return x.permute(0, 2, 1, 3)    # [bs, n_heads, seq_len, d_head]

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0 if self.batch_first else 1]
        query = self.Wq(query)  # [bs, seq_len, d_model]
        key = self.Wk(key)
        value = self.Wv(value)

        query = self.split_head(query)  # [bs, n_heads, seq_len, d_head]
        key = self.split_head(key)
        value = self.split_head(value)

        scores = th.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_head)  # [bs, n_heads, seq_len, seq_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        output = scores.matmul(value).view(batch_size, -1, self.d_model)    # [bs, seq_len, d_model]
        output = self.Wo(output)    # [bs, seq_len, d_model]
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers    # number of encoder layers (encoder can be stacked)

        self.dropout = nn.Dropout(dropout)
        self.multi_attention = MultiHeadAttention(d_model, n_heads, batch_first)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )

    def forward(self, x, mask=None):
        for _ in range(self.n_layers):
            attn = self.dropout(self.multi_attention(x, x, x, mask))    # [bs, seq_len, d_model]
            out = self.norm1(x + attn)          # [bs, seq_len, d_model] + [bs, seq_len, d_model]
            out = self.dropout(self.ffn(out))   # [bs, seq_len, d_model]
            x = self.norm2(attn + out)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(batch_first)
        self.encoder = TransformerEncoderLayer(d_model, n_heads, n_layers, dropout, batch_first)

    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)
        self.multi_attention1 = MultiHeadAttention(d_model, n_heads, batch_first)
        self.multi_attention2 = MultiHeadAttention(d_model, n_heads, batch_first)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for _ in range(self.n_layers):
            attn1 = self.dropout(self.multi_attention1(x, x, x, tgt_mask))
            out1 = self.norm1(x + attn1)
            attn2 = self.dropout(self.multi_attention2(out1, encoder_output, encoder_output, src_mask))
            out2 = self.norm2(out1 + attn2)
            out3 = self.dropout(self.ffn(out2))
            x = self.norm3(out2 + out3)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(batch_first)
        self.decoder = TransformerDecoderLayer(d_model, n_heads, n_layers, dropout, batch_first)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, dropout, batch_first)
        self.decoder = TransformerDecoder(d_model, n_heads, n_layers, dropout, batch_first)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output

