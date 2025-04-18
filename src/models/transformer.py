from typing import Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
torch.set_printoptions(profile="full")
import dataclasses
import math


from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import tokenizer
from searchless_chess.src import utils
from torch.nn.attention import SDPBackend, sdpa_kernel

from searchless_chess.src.models.dense_attention.danet_layers import DANetLayer
from searchless_chess.src.models.dense_attention.model_config import ModelConfig

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

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value

class Expert(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.activation = activation
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

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
        attn_output = F.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal)
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
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 1,
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
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        
        self.activation = activation
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k

        if use_moe:
            self.experts = nn.ModuleList([Expert(d_model, dim_feedforward, activation, bias, **factory_kwargs) for _ in range(num_experts)])
            # Shared expert always runs
            self.shared_expert = Expert(d_model, dim_feedforward, activation, bias, **factory_kwargs)
            # Router determines weights for the experts (not the shared one)
            self.router = nn.Linear(d_model, num_experts, bias=False, **factory_kwargs)
        else:
            # Standard FFN layers if not using MoE
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)


    def _sa_block(self, x, attn_mask, is_causal):
        x = self.self_attn(x, x, x, is_causal=is_causal)
        return x

    def _ff_block(self, x):
        x = self.linear2(self.activation(self.linear1(x)))
        return x

    def _moe_block(self, ffn_input: torch.Tensor) -> torch.Tensor:
        """Applies the Mixture of Experts block with sequence-level routing."""
        batch_size, seq_len, d_model = ffn_input.shape
        # Use first token representation for sequence-level routing
        sequence_repr = ffn_input[:, 0, :] # Shape: [B, D]
        router_logits = self.router(sequence_repr) # Shape: [B, num_experts]

        # Select Top-K experts per sequence
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1) # Shape: [B, top_k]
        gates = F.softmax(top_k_logits, dim=-1) # Shape: [B, top_k]

        # Initialize output tensor for the selected experts' contribution
        moe_output = torch.zeros_like(ffn_input)

        # Iterate over the K selected experts for each sequence
        for k in range(self.top_k):
            # Get the index of the k-th expert for each sequence in the batch
            expert_indices_k = top_k_indices[:, k] # Shape: [B]
            # Get the gate value for the k-th expert for each sequence
            gate_values_k = gates[:, k].unsqueeze(-1).unsqueeze(-1) # Shape: [B, 1, 1]

            # Process each expert that was selected by at least one sequence as their k-th choice
            for expert_idx in range(self.num_experts):
                # Find which sequences in the batch selected this 'expert_idx' as their k-th choice
                batch_indices = torch.where(expert_indices_k == expert_idx)[0]

                if batch_indices.numel() > 0:
                    # Select the input sequences for this expert at this k-th position
                    expert_input = ffn_input[batch_indices] # Shape: [num_selected, S, D]

                    # Compute the output of the selected expert
                    expert_output = self.experts[expert_idx](expert_input) # Shape: [num_selected, S, D]

                    # Apply the corresponding gates for the k-th expert choice
                    # We need gate_values_k specific to the selected batch_indices
                    weighted_output = expert_output * gate_values_k[batch_indices] # Shape: [num_selected, S, D]

                    # Add (accumulate) the weighted output to the final MoE output tensor
                    # index_add_ handles adding contributions if an expert is chosen multiple times (though not possible with top_k=1)
                    # and correctly places results for the selected batch items.
                    moe_output.index_add_(0, batch_indices, weighted_output)

        # Compute output of the shared expert (applied to all sequences)
        shared_output = self.shared_expert(ffn_input)

        # Combine MoE output (sum of top-k weighted experts) and shared expert output
        ff_output = moe_output + shared_output
        return ff_output

    def forward(self, src, src_mask=None, is_causal=False):
        '''
        Arguments:
            src: (batch_size, seq_len, d_model)
            src_mask: (batch_size, seq_len, seq_len)
            is_causal: bool
        '''
        x = src
        if self.norm_first:
            # --- Self-Attention (Pre-LN) ---
            attn_output = self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + attn_output # Residual 1

            # --- FFN / MoE (Pre-LN) ---
            ffn_input = self.norm2(x) # Norm before FFN/MoE
            if self.use_moe:
                 ff_output = self._moe_block(ffn_input)
            else:
                 # Original FFN block if MoE is not used
                 ff_output = self._ff_block(ffn_input)

            x = x + ff_output # Residual 2

        else: # Not norm_first (Post-LN)
            # --- Self-Attention (Post-LN) ---
            attn_output = self._sa_block(x, src_mask, is_causal)
            x = self.norm1(x + attn_output) # Residual 1 + Norm 1

            # --- FFN / MoE (Post-LN) ---
            ffn_input = x # Input to FFN/MoE is the output of the first block
            if self.use_moe:
                ff_output = self._moe_block(ffn_input)
            else:
                 # Original FFN block if MoE is not used
                 ff_output = self._ff_block(ffn_input)

            x = self.norm2(x + ff_output) # Residual 2 + Norm 2

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

        # VANILLA ATTENTION
        self.transformer = nn.ModuleList([
            *[MyTransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embedding_dim * config.widening_factor),
                dropout=config.dropout,
                activation=self.activation,
                norm_first=False,
            ) for _ in range(config.num_layers // 4)],
            *[MyTransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embedding_dim * config.widening_factor),
                dropout=config.dropout,
                activation=self.activation,
                norm_first=True,
            ) for _ in range(3 * config.num_layers // 4 - 2)],
            *[MyTransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embedding_dim * config.widening_factor * 2),
                dropout=config.dropout,
                activation=self.activation,
                norm_first=True,
            ) for _ in range(2)],
        ])

        # # DENSE ATTENTION
        # self.transformer = nn.ModuleList([
        #     DANetLayer(
        #         ModelConfig(
        #             self.vocab_size,
        #             hidden_size= config.embedding_dim,
        #             num_hidden_layers=config.num_layers,
        #             num_attention_heads=config.num_heads,
        #             intermediate_size=int(config.widening_factor),
        #             hidden_act="gelu",
        #             max_position_embeddings=self.seq_len,

                    
        #         ), layer_number
        #     ) for layer_number in range(config.num_layers)
        # ])

        self.self_head = nn.Linear(config.embedding_dim, self.vocab_size)

        self.value_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, data_loader.NUM_BINS),
        )

        # Complex Projection for action matrix
        self.final_ln = nn.LayerNorm(config.embedding_dim)
        self.final_num_heads = config.num_heads
        self.final_head_dim = config.embedding_dim // self.final_num_heads
        self.final_scaling = self.final_head_dim ** -0.5


        self.final_qk_proj = nn.Linear(config.embedding_dim, 2 * config.embedding_dim)
        self.final_out_proj = nn.Sequential(
            nn.Linear(self.final_num_heads, 
                    self.final_num_heads),
            nn.GELU(),
            nn.Linear(self.final_num_heads, 5),
        )
        
        
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x + self.pos_embedding
        for layer in self.transformer:
            x = layer(x)

        x = self.final_ln(x)

        qk = self.final_qk_proj(x)
        qk = qk.reshape(batch_size, self.seq_len, 2, self.final_num_heads, self.final_head_dim).transpose(1, 3)

        q, k = qk.unbind(dim=2)

        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.final_scaling
        attn_scores = torch.tanh(attn_scores)

        attn_scores = attn_scores.permute(0, 2, 3, 1)


        attn_scores = self.final_out_proj(attn_scores)

        bin_width = 1.0 / (data_loader.NUM_BINS)
        bin_centers = torch.arange(bin_width / 2, 1.0, bin_width).to('cuda')


        hl = self.value_head(x[:, -1, :])
        value = torch.sum(F.softmax(hl, dim=-1) * bin_centers, dim=-1, keepdim=True)

        # valuel = self.value_head(x[:, -1, :])

        avsl = attn_scores[:, :, :, 0]
        avs2l = attn_scores[:, :, :, 3]

        return {
            'self': self.self_head(x),
            # 'hl': hl,
            'value': value,
            'hl': hl,
            # 'valuel': valuel,
            'legal': attn_scores[:, :, :, 1],
            'avs': torch.sigmoid(avsl),
            'avsl': avsl,
            'avs2': torch.sigmoid(avs2l),
            'avs2l': avs2l,
            'policy': attn_scores[:, :, :, 2],
            'opt_policy_split': attn_scores[:, :, :, 4],
        }
    
    def losses(self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        legal_moves = target['legal'] == 1

        # Policy loss with masking
        batch_size = output['policy'].shape[0]
        # Clone policy logits to avoid modifying the original
        masked_policy = output['policy'].clone()
        opt_masked_policy_split = output['opt_policy_split'].clone()

        # Apply masking - set illegal moves to large negative value
        masked_policy[~legal_moves] = -1e9  
        opt_masked_policy_split[~legal_moves] = -1e9

        # Reshape for softmax over all possible moves
        masked_policy_flat = masked_policy.view(batch_size, -1)  # [batch_size, S*S]
        target_policy_flat = target['policy'].view(batch_size, -1)  # [batch_size, S*S]

        # Compute cross entropy loss
        policy_loss = -torch.sum(target_policy_flat * F.log_softmax(masked_policy_flat, dim=-1), dim=-1).mean()

        return {
            'self': F.cross_entropy(output['self'].view(-1, output['self'].size(-1)), target['self'].view(-1)),
            'value': F.mse_loss(output['value'], target['value']),
            # 'valuel': F.binary_cross_entropy_with_logits(output['valuel'], target['value']),
            'hl': -0.1 * torch.sum(target['hl'] * F.log_softmax(output['hl'], dim=-1), dim=-1).mean(),
            'legal': F.binary_cross_entropy_with_logits(output['legal'], target['legal']),
            'avs': ((F.mse_loss(output['avs'], target['avs'], reduction='none') * target['weights']).view(batch_size, -1).sum(dim=-1) / target['weights'].view(batch_size, -1).sum(dim=-1)).mean(),
            'avsl': 0.1 * ((F.binary_cross_entropy_with_logits(output['avsl'], target['avs'], reduction='none') * target['weights']).view(batch_size, -1).sum(dim=-1) / target['weights'].view(batch_size, -1).sum(dim=-1)).mean(),
            'avs2': ((F.mse_loss(output['avs2'], target['avs'], reduction='none') * target['legal']).view(batch_size, -1).sum(dim=-1) / target['legal'].view(batch_size, -1).sum(dim=-1)).mean(),
            'avs2l': ((F.binary_cross_entropy_with_logits(output['avs2l'], target['avs'], reduction='none') * target['legal']).view(batch_size, -1).sum(dim=-1) / target['legal'].view(batch_size, -1).sum(dim=-1)).mean(),
            'policy': policy_loss * 0.1,
            'opt_policy_split': 0.1 * ((F.binary_cross_entropy_with_logits(opt_masked_policy_split, target['policy'], reduction='none') * target['weights']).view(batch_size, -1).sum(dim=-1) / target['weights'].view(batch_size, -1).sum(dim=-1)).mean(),
        }