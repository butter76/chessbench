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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * config.widening_factor,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )

        self.self_head = nn.Linear(config.embedding_dim, self.vocab_size)

        self.value_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embedding_dim // 2, 1),
            nn.Sigmoid()  # Output between -1 and 1 for win probability
        )
        
        
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = self.transformer(x)




        return {
            'self': self.self_head(x),
            'value': self.value_head(x[:, -1, :]),
        }
    
    def losses(self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        return {
            'self': F.cross_entropy(output['self'].view(-1, output['self'].size(-1)), target['self'].view(-1)),
            'value': F.mse_loss(output['value'], target['value']),
        }