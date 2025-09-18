import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=128, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, embed_dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RecipeEncoder(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=128, max_len=20, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)                   # [B, T, D]
        x = self.pos_encoding(x)                              # [B, T, D]
        x = self.dropout(x)
        x = self.transformer(x)                               # [B, T, D]
        x = x.transpose(1, 2)                                 # [B, D, T]
        x = self.pool(x).squeeze(-1)                          # [B, D]
        x = self.norm(x)
        return x
    

# import torch
# import torch.nn as nn

# class RecipeEncoder(nn.Module):
#     def __init__(self, vocab_size=20, embed_dim=128, max_len=20, num_layers=2, num_heads=4):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size, embed_dim)
#         self.pos_embedding = nn.Embedding(max_len, embed_dim)

#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.pool = nn.AdaptiveAvgPool1d(1)

#     def forward(self, token_ids):
#         B, T = token_ids.shape
#         pos = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
#         x = self.token_embedding(token_ids) + self.pos_embedding(pos)
#         x = self.transformer(x)
#         x = x.transpose(1, 2)
#         x = self.pool(x).squeeze(-1)
#         return x