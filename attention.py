from torch import nn
import torch

import math
from torch.nn import Parameter
class TAA(nn.Module):
    def __init__(self, in_channels=2, kernel_size=3, expansion=4, act='hsigmoid', dilation=1):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1

        hidden_channels = in_channels * expansion

        self.pw1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False)

        self.dw = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                            padding=(kernel_size-1)//2 * dilation, groups=hidden_channels,
                            dilation=dilation, bias=False)
        self.pw2 = nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=False)

        # 激活函数
        if act == 'hsigmoid':
            self.act_fn = nn.Hardsigmoid()
        elif act == 'tanh':
            self.act_fn = lambda x: 0.5 * (torch.tanh(x) + 1)
        else:
            self.act_fn = torch.sigmoid

    def forward(self, x):
        # x: [B,F,T]
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool, _ = x.max(dim=1, keepdim=True)
        cat = torch.cat([avg_pool, max_pool], dim=1)  # [B,2,T]

        out = self.pw1(cat)
        out = self.dw(out)
        out = self.pw2(out)
        w_time = self.act_fn(out)  # [B,1,T]
        return x * w_time


