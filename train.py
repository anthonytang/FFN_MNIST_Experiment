import torch, random, matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import numpy as np

class FFN_GeGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.02)
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))
        self.w_gate = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.02)
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.w_out = nn.Parameter(torch.randn(hidden_dim, out_dim) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(out_dim))
        self.act = nn.GELU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = torch.einsum("bi,ih->bh", x, self.w_in) + self.b_in
        g = torch.einsum("bi,ih->bh", x, self.w_gate) + self.b_gate
        h = z * self.act(g)
        return torch.einsum("bh,ho->bo", h, self.w_out) + self.b_out

class FFN_ReLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.02)
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))
        self.w_out = nn.Parameter(torch.randn(hidden_dim, out_dim) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(out_dim))
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.act(torch.einsum("bi,ih->bh", x, self.w_in) + self.b_in)
        return torch.einsum("bh,ho->bo", h, self.w_out) + self.b_out

class LitModel(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        return self.loss_fn(self(batch[0]), batch[1])

    def validation_step(self, batch, _):
        acc = (self(batch[0]).argmax(1) == batch[1]).float().mean()
        self.log("val_acc", acc, prog_bar=False)
        return acc

    def test_step(self, batch, _):
        acc = (self(batch[0]).argmax(1) == batch[1]).float().mean()
        self.log("test_acc", acc, prog_bar=False)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
