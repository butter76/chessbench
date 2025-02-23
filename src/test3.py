import torch
from torch.nn.attention.flex_attention import flex_attention
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
def noop(score, batch, head, token_q, token_kv):
    return score
compiled_flex_attention = torch.compile(flex_attention)

# Create input tensors
x = torch.randn(128, 2048, 256, device="cuda", dtype=torch.float32)


class MyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(256, 256 * 3, device="cuda")
    
    def forward(self, x):
        result = self.proj(x)
        query, key, value = torch.chunk(result, 3, dim=-1)
        query = query.unflatten(-1, [16, 16]).transpose(1, 2)
        key = key.unflatten(-1, [16, 16]).transpose(1, 2)
        value = value.unflatten(-1, [16, 16]).transpose(1, 2)
        out_compiled = compiled_flex_attention(query, key, value, score_mod=noop)
        return out_compiled
    

model = MyAttention().to("cuda")
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=1e-4,
#     weight_decay=0.01,
# )
# scaler = GradScaler('cuda')


with autocast('cuda', dtype=torch.bfloat16):
    out_compiled = model(x)

    loss = out_compiled.sum()
loss.backward()
# optimizer.zero_grad()
# scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()