import torch
import torch.nn as nn


class CLSTokenProcessor(nn.Module):
    """
    将 cls_tokens 从 [batch, 4, 1024] 处理为 [batch, 786]

    提供多种处理方式:
    1. flatten + linear: 展平后通过线性层
    2. pool + linear: 池化后通过线性层
    3. attention + linear: 注意力聚合后通过线性层
    """

    def __init__(self, method='flatten'):
        """
        Args:
            method: 处理方式，可选 'flatten', 'pool', 'attention'
        """
        super().__init__()
        self.method = method

        if method == 'flatten':
            # 4 * 1024 = 4096 -> 786
            self.fc = nn.Linear(4 * 1024, 768)

        elif method == 'pool':
            # 池化: [batch, 4, 1024] -> [batch, 1024] -> [batch, 786]
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(1024, 786)

        elif method == 'attention':
            # 注意力机制聚合
            self.attention = nn.MultiheadAttention(
                embed_dim=1024,
                num_heads=8,
                batch_first=True
            )
            self.query = nn.Parameter(torch.randn(1, 1, 1024))
            self.fc = nn.Linear(1024, 786)

        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, cls_tokens):
        """
        Args:
            cls_tokens: [batch, 4, 1024]
        Returns:
            output: [batch, 786]
        """
        batch_size = cls_tokens.size(0)

        if self.method == 'flatten':
            # 展平: [batch, 4, 1024] -> [batch, 4096]
            x = cls_tokens.view(batch_size, -1)
            output = self.fc(x)

        elif self.method == 'pool':
            # 池化: [batch, 4, 1024] -> [batch, 1024, 4] -> [batch, 1024, 1] -> [batch, 1024]
            x = cls_tokens.permute(0, 2, 1)  # [batch, 1024, 4]
            x = self.pool(x).squeeze(-1)  # [batch, 1024]
            output = self.fc(x)

        elif self.method == 'attention':
            # 注意力聚合: [batch, 4, 1024] -> [batch, 1, 1024] -> [batch, 1024]
            query = self.query.expand(batch_size, -1, -1)
            x, _ = self.attention(query, cls_tokens, cls_tokens)
            x = x.squeeze(1)  # [batch, 1024]
            output = self.fc(x)

        return output