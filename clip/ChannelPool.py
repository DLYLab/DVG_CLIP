import sys

import torch
import torch.nn as nn


class GroupedChannelPool(nn.Module):
    def __init__(self, num_layers=24, group_num=4, feature_dim=768):
        super().__init__()
        self.num_layers = num_layers
        self.group_num = group_num
        self.group_size = num_layers // group_num  # 每组数量（24/4 = 6）
        self.feature_dim = feature_dim

        # AvgPool1d：压缩 768*6 → 768
        self.pool = nn.AvgPool1d(kernel_size=self.group_size, stride=self.group_size)

    def forward(self, layers):
        """
        layers: list of 24 tensors, each of shape [B, 1370, 768]
        """
        assert len(layers) == self.num_layers, f"需要 {self.num_layers} 层，但收到 {len(layers)} 层"

        outputs = []

        # 分组
        for g in range(self.group_num):
            group = []
            for i in range(self.num_layers):
                if i % self.group_num == g:
                    group.append(layers[i])

            # 拼接通道：[B, 1370, 768*6]
            # x = torch.cat(group, dim=-1)

            stacked = torch.stack(group, dim=0)  # [6, B, 1370, 768]
            x = stacked.mean(dim=0)
            # print(x.shape)

            # x = self.pool(x)
            outputs.append(x)

        return outputs
