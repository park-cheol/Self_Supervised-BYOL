import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, proj_channels):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                 nn.BatchNorm1d(hidden_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, proj_channels))

    def forward(self, inputs: torch.Tensor):
        out = self.mlp(inputs)
        return out
