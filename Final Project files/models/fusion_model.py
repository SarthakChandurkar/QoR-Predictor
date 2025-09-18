import torch
import torch.nn as nn
from models.graph_encoder import GraphEncoder, EnhancedGraphEncoder
from models.recipe_encoder import RecipeEncoder
from bidirectional_cross_attention import BidirectionalCrossAttention

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=128, heads=4, dim_head=32):
        super().__init__()
        self.bidirectional_attn = BidirectionalCrossAttention(
            dim=embed_dim,
            heads=heads,
            dim_head=dim_head
        )
        self.linear = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, g_embed, r_embed):
        # Ensure inputs are 3D tensors: (batch_size, sequence_length, embed_dim)
        g_embed = g_embed.unsqueeze(1)  # (B, 1, D)
        r_embed = r_embed.unsqueeze(1)  # (B, 1, D)

        # Apply bidirectional cross-attention
        g_attn, r_attn = self.bidirectional_attn(g_embed, r_embed)

        # Fuse the attended embeddings (e.g., by averaging)
        fused = torch.cat([g_attn, r_attn], dim=-1)  # (B, 1, 2D)
        fused = self.linear(fused.squeeze(1))       # Linear: 2D → D
        return fused
       


class DelayPredictor(nn.Module):
    def __init__(self, graph_input_dim=8, embed_dim=128):
        super().__init__()
        self.graph_encoder = EnhancedGraphEncoder(in_channels=graph_input_dim, hidden_dim=embed_dim)
        self.recipe_encoder = RecipeEncoder(embed_dim=embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, graph_data, recipe_tokens):
        g_embed = self.graph_encoder(graph_data.x, graph_data.edge_index, graph_data.batch)
        r_embed = self.recipe_encoder(recipe_tokens)
        fused = self.fusion(g_embed, r_embed)
        fused = self.norm(fused)
        return self.predictor(fused).squeeze(-1)
    
