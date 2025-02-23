import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
torch.set_default_dtype(torch.bfloat16)
def checkerboard(score, batch, head, token_q, token_kv):
    return score

# Create input tensors
x = torch.randn(8, 2048, 256, device="cuda")

# Create an attention layer
proj = nn.Linear(256, 256 * 3, device="cuda")
result = proj(x)
query, key, value = torch.chunk(result, 3, dim=-1)
query = query.unflatten(-1, [16, 16]).transpose(1, 2)
key = key.unflatten(-1, [16, 16]).transpose(1, 2)
value = value.unflatten(-1, [16, 16]).transpose(1, 2)

# Compile and run
compiled_flex_attention = torch.compile(flex_attention)
out_compiled = compiled_flex_attention(query, key, value, score_mod=checkerboard)

# Run Backward Pass
out_compiled.sum().backward()
print(out_compiled)