"""
Part B - PyTorch MLP for Automatic Modulation Classification.

Problem 6:
12 input features -> 128 -> 128 -> 5 classes
"""

import torch
import torch.nn as nn


class AMCMLP(nn.Module):
    """
    Feedforward neural network for AMC classification.
    """

    def __init__(
        self,
        input_dim=12,
        hidden_dim=128,
        num_classes=5,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":

    # Simple shape test
    model = AMCMLP()

    x = torch.randn(8, 12)

    logits = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", logits.shape)
    print(model)