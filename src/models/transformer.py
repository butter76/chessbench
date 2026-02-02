from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
torch.set_printoptions(profile="full")
import dataclasses
import math

from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader, tokenizer

S = tokenizer.SEQUENCE_LENGTH

# Policy head dimensions (matching CatGPT)
POLICY_TO_DIM = 73  # 64 normal + 9 underpromotions
POLICY_SHAPE = (64, POLICY_TO_DIM)

@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
    """Hyperparameters used in the Transformer architectures."""

    # The dimension of the first embedding.
    embedding_dim: int = 512
    # The number of multi-head attention layers.
    num_layers: int = 16
    # The number of heads per layer.
    num_heads: int = 32
    # How much larger the hidden layer of the feedforward network should be
    # compared to the `embedding_dim`.
    widening_factor: float = 3  # 1536 / 512 = 3
    # The dropout rate.
    dropout: float = 0.0
    # repeater for tokens
    repeater: int = 1

    # Position Embedding Type
    # Whether to use smolgen (now with proper LC0-style implementation)
    use_smolgen: bool = True
    # Smolgen configuration (matching jax_base.yaml)
    smolgen_hidden_channels: int = 32
    smolgen_hidden_size: int = 256
    smolgen_gen_size: int = 256
    # Whether to use simple attention bias
    use_attention_bias: bool = False

    # Output heads configuration (matching jax_base.yaml)
    self_weight: float = 0.1      # Token reconstruction weight
    value_weight: float = 0.7     # Value head weight
    policy_weight: float = 1.5    # Policy weight
    hard_policy_temperature: float = 0.25  # p^4 sharpening
    hard_policy_weight: float = 0.1  # Small weight (harder targets are noisier)
    # Policy head Q/K dimension for attention-based policy
    policy_qk_dim: int = 32

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
        use_smolgen (bool, optional): Whether to use smolgen. Default: False
        smolgen_hidden_channels (int): Compression dimension for smolgen
        smolgen_hidden_size (int): Hidden layer size for smolgen
        smolgen_gen_size (int): Per-head generation dimension for smolgen
        use_attention_bias (bool, optional): Whether to use simple attention bias. Default: False
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
        use_smolgen: bool = False,
        smolgen_hidden_channels: int = 32,
        smolgen_hidden_size: int = 256,
        smolgen_gen_size: int = 256,
        use_attention_bias: bool = False,
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
        self.use_smolgen = use_smolgen
        self.use_attention_bias = use_attention_bias

        if use_smolgen:
            # LC0-style Smolgen: dynamic attention bias generation
            # Compress input: (batch, seq, E_q) -> (batch, seq, hidden_channels)
            self.smolgen_compress = nn.Linear(E_q, smolgen_hidden_channels, **factory_kwargs)
            # Global transform: (batch, seq * hidden_channels) -> (batch, hidden_size)
            self.smolgen_dense1 = nn.Linear(S * smolgen_hidden_channels, smolgen_hidden_size, **factory_kwargs)
            self.smolgen_ln = nn.LayerNorm(smolgen_hidden_size, eps=1e-5, **factory_kwargs)
            # Generate per-head codes: (batch, hidden_size) -> (batch, nheads * gen_size)
            self.smolgen_dense2 = nn.Linear(smolgen_hidden_size, nheads * smolgen_gen_size, **factory_kwargs)
            # Per-head attention bias generation (shared weights across heads)
            self.smolgen_out_ln = nn.LayerNorm(smolgen_gen_size, eps=1e-5, **factory_kwargs)
            self.smolgen_out = nn.Linear(smolgen_gen_size, S * S, **factory_kwargs)
            self.smolgen_gen_size = smolgen_gen_size
        elif use_attention_bias:
            self.attn_bias = nn.Parameter(torch.empty(1, nheads, S, S, **factory_kwargs).uniform_(-0.02, 0.02))


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
        if self.use_smolgen:
            batch_size = query.size(0)
            # Compress input: (batch, seq, E_q) -> (batch, seq, hidden_channels)
            compressed = self.smolgen_compress(query)
            # Flatten: (batch, seq * hidden_channels)
            flat = compressed.view(batch_size, -1)
            # Global transform with LayerNorm
            hidden = F.gelu(self.smolgen_ln(self.smolgen_dense1(flat)))
            # Generate per-head codes: (batch, nheads, gen_size)
            codes = self.smolgen_dense2(hidden).view(batch_size, self.nheads, self.smolgen_gen_size)
            # Apply LayerNorm and generate attention bias per head
            codes = self.smolgen_out_ln(codes)
            attn_bias = self.smolgen_out(codes).view(batch_size, self.nheads, S, S)
        elif self.use_attention_bias:
            attn_bias = self.attn_bias
        else:
            attn_bias = None

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
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias, is_causal=is_causal)
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
        norm_first=False,
        bias=True,
        use_smolgen=False,
        smolgen_hidden_channels=32,
        smolgen_hidden_size=256,
        smolgen_gen_size=256,
        use_attention_bias=False,
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
            use_smolgen=use_smolgen,
            smolgen_hidden_channels=smolgen_hidden_channels,
            smolgen_hidden_size=smolgen_hidden_size,
            smolgen_gen_size=smolgen_gen_size,
            use_attention_bias=use_attention_bias,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
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

        self.vocab_size = len(tokenizer._CHARACTERS)
        self.seq_len = tokenizer.SEQUENCE_LENGTH

        self.embedding = nn.Embedding(self.vocab_size, config.embedding_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, config.embedding_dim)
        )

        self.activation = F.gelu

        # Mixed-LN Attention
        post_ln_layers = min(5, config.num_layers // 4)
        pre_ln_layers = config.num_layers - post_ln_layers
        self.transformer = nn.ModuleList([
            *[MyTransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embedding_dim * config.widening_factor),
                dropout=config.dropout,
                activation=self.activation,
                norm_first=False,
                use_smolgen=config.use_smolgen,
                smolgen_hidden_channels=config.smolgen_hidden_channels,
                smolgen_hidden_size=config.smolgen_hidden_size,
                smolgen_gen_size=config.smolgen_gen_size,
                use_attention_bias=config.use_attention_bias,
            ) for _ in range(post_ln_layers)],
            *[MyTransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embedding_dim * config.widening_factor),
                dropout=config.dropout,
                activation=self.activation,
                norm_first=True,
                use_smolgen=config.use_smolgen,
                smolgen_hidden_channels=config.smolgen_hidden_channels,
                smolgen_hidden_size=config.smolgen_hidden_size,
                smolgen_gen_size=config.smolgen_gen_size,
                use_attention_bias=config.use_attention_bias,
            ) for _ in range(pre_ln_layers)]
        ])

        # Self reconstruction head (auxiliary task for stability)
        self.self_head = nn.Linear(config.embedding_dim, self.vocab_size)

        # Value head with HL-Gauss distribution
        self.value_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, data_loader.NUM_BINS),
        )

        # Final layer norm
        self.final_ln = nn.LayerNorm(config.embedding_dim)

        # LC0-style attention policy head: uses Q·K^T for 64x73 move logits
        # Q projects from first 64 positions (from squares)
        # K projects to 73 dimensions (to squares + underpromotions)
        policy_qk_dim = config.policy_qk_dim
        self.policy_q_proj = nn.Linear(config.embedding_dim, policy_qk_dim)
        self.policy_k_proj = nn.Linear(config.embedding_dim, policy_qk_dim)
        # Additional projection for the 9 underpromotion targets
        # These are special "virtual" positions that don't exist in the sequence
        self.policy_promo_embed = nn.Parameter(
            torch.randn(9, policy_qk_dim) * 0.02  # 9 underpromotion targets
        )
        self.policy_scaling = policy_qk_dim ** -0.5




    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device

        x = self.embedding(x)
        x = x + self.pos_embedding
        for layer in self.transformer:
            x = layer(x)

        x = self.final_ln(x)

        # Value head: use last position for value prediction
        bin_width = 1.0 / data_loader.NUM_BINS
        bin_centers = torch.arange(bin_width / 2, 1.0, bin_width, device=device)

        hl = self.value_head(x[:, -1, :])
        value = torch.sum(F.softmax(hl, dim=-1) * bin_centers, dim=-1, keepdim=True)

        # Self reconstruction head
        self_logits = self.self_head(x)

        # LC0-style attention policy head
        # Q from first 64 positions (from squares): (batch, 64, qk_dim)
        policy_q = self.policy_q_proj(x[:, :64, :])  # (batch, 64, qk_dim)

        # K for normal destination squares (first 64): (batch, 64, qk_dim)
        policy_k_normal = self.policy_k_proj(x[:, :64, :])  # (batch, 64, qk_dim)

        # K for underpromotion targets: (batch, 9, qk_dim)
        promo_embed = self.policy_promo_embed.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate normal and underpromotion K: (batch, 73, qk_dim)
        policy_k = torch.cat([policy_k_normal, promo_embed], dim=1)  # (batch, 73, qk_dim)

        # Compute attention scores: Q @ K^T -> (batch, 64, 73)
        policy_logits = torch.matmul(policy_q, policy_k.transpose(-2, -1)) * self.policy_scaling

        return {
            'self': self_logits,
            'value': value,
            'hl': hl,
            'policy': policy_logits,  # Shape: (batch, 64, 73)
        }

    def losses(self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute losses matching jax_base.yaml config.

        Args:
            output: Model outputs with keys 'self', 'value', 'hl', 'policy'
            target: Targets with keys 'self', 'policy', 'hard_policy', 'hl', 'value'

        Loss weights (from jax_base.yaml):
            - self_weight: 0.1 (token reconstruction)
            - value_weight: 0.7 (win probability)
            - policy_weight: 1.5 (main policy)
            - hard_policy_weight: 0.1 (sharpened policy)
        """
        batch_size = output['policy'].shape[0]
        config = self.config

        # Self reconstruction loss (cross entropy)
        self_loss = F.cross_entropy(
            output['self'].view(-1, output['self'].size(-1)),
            target['self'].view(-1)
        ) * config.self_weight

        # HL-Gauss distribution cross entropy (actual training loss)
        hl_loss = -torch.sum(
            target['hl'] * F.log_softmax(output['hl'], dim=-1),
            dim=-1
        ).mean() * config.value_weight

        # MSE loss for value (metrics only, excluded from training loss)
        value_mse = F.mse_loss(output['value'], target['value'])

        # Policy loss: cross entropy on (64, 73) policy
        # Mask out zero-probability moves (illegal) by setting logits to -inf
        legal_mask = target['policy'] > 0  # (batch, 64, 73)
        masked_policy = output['policy'].clone()
        masked_policy[~legal_mask] = -1e9

        # Flatten for cross entropy computation
        masked_policy_flat = masked_policy.view(batch_size, -1)  # (batch, 64*73)
        target_policy_flat = target['policy'].view(batch_size, -1)  # (batch, 64*73)

        policy_loss = -torch.sum(
            target_policy_flat * F.log_softmax(masked_policy_flat, dim=-1),
            dim=-1
        ).mean() * config.policy_weight

        # Hard policy loss (sharpened target)
        target_hard_policy_flat = target['hard_policy'].view(batch_size, -1)
        hard_policy_loss = -torch.sum(
            target_hard_policy_flat * F.log_softmax(masked_policy_flat, dim=-1),
            dim=-1
        ).mean() * config.hard_policy_weight

        return {
            'self': self_loss,
            'hl': hl_loss,
            'value': value_mse,  # Metrics only (excluded from loss via k not in ['value', 'draw'])
            'policy': policy_loss,
            'hard_policy': hard_policy_loss,
        }



