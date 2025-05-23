import torch

# right at the top of your main entrypoint, *before* any other torch imports:
torch.set_default_dtype(torch.float32)