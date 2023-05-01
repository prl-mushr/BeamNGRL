import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicMLP(nn.Module):
    def __init__(
            self,
            input_size=1,
            hidden_depth=2,
            hidden_dim=512,
            output_dim=1,
            batch_norm=False,
            **kwargs,
    ):
        super().__init__()

        self.input_size = input_size

        fc_layers = [
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(hidden_depth):
            fc_layers += [nn.Linear(hidden_dim, hidden_dim)]
            if batch_norm:
                fc_layers += [nn.BatchNorm1d(hidden_dim)]
            fc_layers += [nn.ReLU()]
        fc_layers += [nn.Linear(hidden_dim, output_dim)]
        fc_layers += [nn.ReLU()]

        self.main = nn.Sequential(*fc_layers)

    def forward(self, x):
        out = self.main(x)
        return out