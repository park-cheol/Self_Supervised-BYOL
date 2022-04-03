import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        batch, _, _, _ = x.size()
        return self.model(x.view(batch, -1))
