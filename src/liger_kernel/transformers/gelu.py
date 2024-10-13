import torch.nn as nn

from liger_kernel.ops.gelu import LigerGELUFunction


class LigerGELUMLP(nn.Module):
    def __init__(self, config, exact=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.exact = exact

    def forward(self, x):
        return self.down_proj(LigerGELUFunction.apply(self.gate_proj(x), self.exact))
