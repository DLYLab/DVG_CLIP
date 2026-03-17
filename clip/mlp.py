import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,):
        super().__init__()
        self.fc1 = nn.Linear(2 * 768, 768 * 4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(768 * 4, 768)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, cls_tokens):
        batch_size = cls_tokens.size(0)
        x = cls_tokens.view(batch_size, -1)
        output = self.fc2(self.gelu(self.fc1(x)))

        output = self.layer_norm(output)
        return output
