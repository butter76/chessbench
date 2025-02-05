from typing import Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
torch.set_printoptions(profile="full")
import dataclasses


from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import tokenizer
from searchless_chess.src import utils
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attn import flash_attn_qkvpacked_func
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
    # repeater for tokens
    repeater: int = 1

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
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=True):
        batch_size, seq_len, d_model = x.shape
        
        # Single QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        
        # Use dropout only during training
        dropout_p = self.dropout if self.training else 0.0
        output = cast(Any, flash_attn_qkvpacked_func(qkv, causal=causal, dropout_p=dropout_p))
        
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
        self.seq_len = tokenizer.SEQUENCE_LENGTH * config.repeater
        
        self.embedding = nn.Embedding(self.vocab_size, config.embedding_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, config.embedding_dim)
        )
        self.sq_embedding = nn.Parameter(
            torch.randn(1, self.seq_len // config.repeater, config.embedding_dim)
        )

        self.transformer = nn.ModuleList([
            FlashTransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.embedding_dim * config.widening_factor,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])

        self.self_head = nn.Linear(config.embedding_dim, self.vocab_size)

        self.value_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1 for win probability
        )

        # Complex Projection for action matrix
        self.final_ln = nn.LayerNorm(config.embedding_dim)
        self.final_num_heads = config.num_heads
        self.final_head_dim = config.embedding_dim // self.final_num_heads
        self.final_scaling = self.final_head_dim ** -0.5


        self.final_qk_proj = nn.Linear(config.embedding_dim, 2 * config.embedding_dim)
        self.final_out_proj = nn.Sequential(
            nn.Linear(self.final_num_heads * self.config.repeater * self.config.repeater, 
                    self.final_num_heads * self.config.repeater),
            nn.GELU(),
            nn.Linear(self.final_num_heads * self.config.repeater, 2),
            nn.Sigmoid(),
        )
        
        
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x + self.sq_embedding
        x = x.repeat_interleave(self.config.repeater, dim=1)
        x = x + self.pos_embedding
        for layer in self.transformer:
            x = layer(x)


        # qk = self.final_qk_proj(self.final_ln(x))
        # qk = qk.reshape(batch_size, self.seq_len, 2, self.final_num_heads, self.final_head_dim).transpose(1, 3)

        # q, k = qk.unbind(2, dim=2)

        # attn_scores = torch.baddbmm(
        #     torch.empty(batch_size, self.final_num_heads, self.seq_len, self.seq_len, dtype=q.dtype, device=q.device),
        #     q,
        #     k.transpose(-2, -1),
        #     beta=0.0,
        #     alpha=self.final_scaling,
        # )
        # attn_scores = torch.tanh(attn_scores)

        # s = self.seq_len // self.config.repeater

        # attn_scores = attn_scores.reshape(batch_size, self.final_num_heads, self.config.repeater, s, self.config.repeater, s).permute(0, 3, 5, 1, 2, 4)
        # attn_scores = attn_scores.reshape(batch_size, s, s, self.final_num_heads * self.config.repeater * self.config.repeater)

        # attn_scores = self.final_out_proj(attn_scores)

        return {
            'self': self.self_head(x),
            'value': self.value_head(x[:, -1, :]),
        }
    
    def losses(self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        return {
            'self': F.cross_entropy(output['self'].view(-1, output['self'].size(-1)), target['self'].repeat_interleave(self.config.repeater, dim=1).view(-1)),
            'value': F.mse_loss(output['value'], target['value']),
        }