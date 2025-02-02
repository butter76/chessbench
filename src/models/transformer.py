from typing import Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.bfloat16)
torch.set_printoptions(profile="full")
import dataclasses


from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import tokenizer
from searchless_chess.src import utils

from flash_attn import flash_attn_func
from flash_attn.modules.mha import FlashSelfAttention

@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
    """Hyperparameters used in the Transformer architectures."""
    
    # The dimension of the first embedding.
    embedding_dim: int = 64
    # The number of multi-head attention layers.
    num_layers: int = 4
    # The number of heads per layer.
    num_heads: int = 8
    # How much larger the hidden layer of the feedforward network should be
    # compared to the `embedding_dim`.
    widening_factor: int = 4
    # The dropout rate.
    dropout: float = 0.1

    ## TODO: Implement
    # Whether to apply QK normalization trick in attention layer.
    apply_qk_layernorm: bool = False
    # Whether to apply post LN after attention + MLP blocks
    apply_post_ln: bool = True

class FlashAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # Pass causal=True directly in the constructor
        self.flash_attention = FlashSelfAttention(causal=True, attention_dropout=dropout)
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=True):
        batch_size, seq_len, d_model = x.shape
        
        # Fuse QKV projections into one operation
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Use flash_attn_func directly instead of FlashSelfAttention module
        output = cast(Any, flash_attn_func(q, k, v, causal=causal, dropout_p=self.dropout))
        
        output = output.contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(output)

class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.flash_attention = FlashAttentionLayer(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        x = x + self.flash_attention(self.norm1(x))
        x = x + self.dropout(self.linear2(self.activation(self.linear1(self.norm2(x)))))
        return x

class ChessTransformer(nn.Module):
    """PyTorch implementation of the transformer model."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.vocab_size = len(tokenizer._CHARACTERS)
        self.seq_len = tokenizer.SEQUENCE_LENGTH
        
        self.embedding = nn.Embedding(self.vocab_size, config.embedding_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, config.embedding_dim)
        )

        # Replace standard transformer layers with Flash Attention layers
        self.transformer = nn.ModuleList([
            FlashTransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.embedding_dim * config.widening_factor,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])

        self.self_head = nn.Linear(config.embedding_dim, self.vocab_size)
        
        # Complex Projection for action matrix
        self.final_num_heads = config.num_heads
        self.final_head_dim = config.embedding_dim // config.num_heads
        self.final_scaling = self.final_head_dim ** -0.5

        self.final_q_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.final_k_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.final_out_proj = nn.Linear(self.final_head_dim, 2)
        
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.embedding(x)
        x = x + self.pos_embedding
        for layer in self.transformer:
            x = layer(x)

        # Flatten the output
        batch_size = x.shape[0]
        flat_x = x.reshape(batch_size, -1)  # [batch, seq_len * embedding_dim]

        # Compute Q, K, V for the final attention op.
        q = self.final_q_proj(x)  # [batch, seq_len, embed_dim]
        k = self.final_k_proj(x)  # [batch, seq_len, embed_dim]

        # Reshape for multi-head attention.
        # We'll reshape and transpose such that the shape becomes
        # [batch, num_heads, seq_len, head_dim]
        def reshape_for_heads(t):
            return t.reshape(batch_size, self.seq_len, self.final_num_heads, self.final_head_dim).transpose(1, 2)
        
        q = reshape_for_heads(q)
        k = reshape_for_heads(k)

        # Compute scaled dot-product attention.
        # q: [batch, num_heads, seq_len, head_dim]
        # k: [batch, num_heads, seq_len, head_dim]
        # => scores: [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.final_scaling
        # Move num_heads to the end: [batch, seq_len, seq_len, num_heads]
        attn_scores = attn_scores.permute(0, 2, 3, 1)
        # Apply final projection: [batch, seq_len, seq_len, 2]
        attn_scores = self.final_out_proj(attn_scores)
        # Shorten seq_len to 64: [batch, 64, 64, 2]
        attn_scores = attn_scores[:, :64, :64, :]
        # Apply sigmoid to constrain values between 0 and 1
        attn_scores = torch.sigmoid(attn_scores)

        return {
            'self': self.self_head(x),
            'value': attn_scores[:, :, :, 0],
            'legal': attn_scores[:, :, :, 1],
        }
    
    def losses(self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        return {
            'self': F.cross_entropy(output['self'].view(-1, output['self'].size(-1)), target['self'].view(-1)),
            'legal': F.binary_cross_entropy(output['legal'], target['legal']),
            'value': (F.mse_loss(output['value'], target['value'], reduction='none') * target['legal']).sum() / target['legal'].sum(),
        }