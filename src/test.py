from typing import cast
import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
)
import torch.nn as nn
import torch.nn.functional as F

from searchless_chess.src import tokenizer
torch.set_default_dtype(torch.float16)
import dataclasses

vocab_size = 1024
seq_length = 1024

def score_mod(score, batch, head, token_q, token_kv):
    return score
compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
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
    widening_factor: float = 4
    # The dropout rate.
    dropout: float = 0.1
    # repeater for tokens
    repeater: int = 1

    ## TODO: Implement
    # Whether to apply QK normalization trick in attention layer.
    apply_qk_layernorm: bool = False
    # Whether to apply post LN after attention + MLP blocks
    apply_post_ln: bool = True
class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """
    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
          self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
          self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
          self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
          self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None, is_causal=False) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_q, E_qk)
            key (torch.Tensor): key of shape (N, L_kv, E_qk)
            value (torch.Tensor): value of shape (N, L_kv, E_v)
            attn_mask (torch.Tensor, optional): attention mask of shape (N, L_q, L_kv) to pass to sdpa. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = F.linear(query, q_weight, q_bias), F.linear(key, k_weight, k_bias), F.linear(value, v_weight, v_bias)

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        # with torch.nn.attention.sdpa_kernel(
        #     SDPBackend.CUDNN_ATTENTION
        # ):
        attn_output = flex_attention(
            query, key, value, score_mod=None)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output

class MyTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation = F.relu,
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(
            d_model,
            d_model,
            d_model,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        

    def _sa_block(self, x, attn_mask, is_causal):
        x = self.self_attn(x, x, x, is_causal=is_causal)
        return x

    def _ff_block(self, x):
        x = self.linear2(self.activation(self.linear1(x)))
        return x

    def forward(self, src, src_mask=None, is_causal=False):
        '''
        Arguments:
            src: (batch_size, seq_len, d_model)
            src_mask: (batch_size, seq_len, seq_len)
            is_causal: bool
        '''
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

class ChessTransformer(nn.Module):
    """PyTorch implementation of the transformer model."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.vocab_size = vocab_size
        self.seq_len = seq_length
        
        self.embedding = nn.Embedding(self.vocab_size, config.embedding_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, config.embedding_dim)
        )

        self.transformer = nn.ModuleList([
            MyTransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embedding_dim * config.widening_factor),
                dropout=config.dropout,
                activation=F.gelu,
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
        # self.final_num_heads = config.num_heads
        # self.final_head_dim = config.embedding_dim // self.final_num_heads
        # self.final_scaling = self.final_head_dim ** -0.5


        # self.final_qk_proj = nn.Linear(config.embedding_dim, 2 * config.embedding_dim)
        # self.final_out_proj = nn.Sequential(
        #     nn.Linear(self.final_num_heads, 
        #             self.final_num_heads),
        #     nn.GELU(),
        #     nn.Linear(self.final_num_heads, 2),
        # )
        
        
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x + self.pos_embedding
        for layer in self.transformer:
            x = layer(x)
        x = self.final_ln(x)


        # qk = self.final_qk_proj(x)
        # qk = qk.reshape(batch_size, self.seq_len, 2, self.final_num_heads, self.final_head_dim).transpose(1, 3)

        # q, k = qk.unbind(dim=2)

        # attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.final_scaling
        # attn_scores = torch.tanh(attn_scores)

        # attn_scores = attn_scores.permute(0, 2, 3, 1)


        # attn_scores = self.final_out_proj(attn_scores)

        return {
            'self': self.self_head(x),
            'value': self.value_head(x[:, -1, :]),
            # 'legal': attn_scores[:, :64, :64, 1],
            # 'avs': torch.sigmoid(attn_scores[:, :64, :64, 0]),
        }
    
    def losses(self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        return {
            'self': F.cross_entropy(output['self'].view(-1, output['self'].size(-1)), target['self'].view(-1)),
            'value': F.mse_loss(output['value'], target['value']),
            # 'legal': F.binary_cross_entropy_with_logits(output['legal'], target['legal']),
            # 'avs': F.mse_loss(output['avs'][legal_moves], target['avs'][legal_moves]),
        }


# Create config
config = TransformerConfig(
    embedding_dim=1024,
    num_layers=8,
    num_heads=32,
    widening_factor=3,
    dropout=0
)

# Initialize model
model = ChessTransformer(config)
model = torch.compile(model).cuda()
model.train()

# Create dummy input data
batch_size = 64
x = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()  # Random integers between 0-63


# Create dummy target data
target = {
    'self': x,
    'value': torch.rand((batch_size, 1)).cuda(),  # Random values between 0-1
}

# Forward pass
output = model(x)
print(output)
losses = model.losses(output, target)
print(losses)

# Compute total loss and do backprop
total_loss = sum(loss for loss in losses.values())
total_loss.backward()
print("DONE", total_loss)