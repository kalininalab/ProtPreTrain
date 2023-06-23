from torch import nn
from torch.nn.utils import weight_norm


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        )

    def forward(self, x):
        return self.main(x)
