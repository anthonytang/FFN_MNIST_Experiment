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

def bootstrap(data, k=1000):
    return np.std([np.mean(np.random.choice(data, len(data))) for _ in range(k)])

def loader(s, b, shuffle=False):
    return DataLoader(s, batch_size=b, shuffle=shuffle, num_workers=4)

def run(hidden_dims=[2, 4, 8, 16], ks=[2, 4, 8]):
    data = datasets.MNIST('', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('', train=False, download=True, transform=transforms.ToTensor())
    all_results = {"FFN_ReLU": {}, "FFN_GeGLU": {}}

    for Model in [FFN_ReLU, FFN_GeGLU]:
        name = Model.__name__
        for k in ks:
            accs = []
            for d in hidden_dims:
                best_acc, best_model = 0, None
                for _ in range(k):
                    bs = random.choice([8, 64])
                    lr = random.choice([1e-1, 1e-2, 1e-3, 1e-4])
                    train_set, val_set = random_split(data, [50000, 10000])
                    m = Model(28 * 28, d, 10)
                    lit = LitModel(m, lr)
                    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
                    trainer.fit(lit, loader(train_set, bs, shuffle=True), loader(val_set, bs))
                    val_result = trainer.validate(lit, loader(val_set, bs), verbose=False)
                    val_acc = val_result[0].get('val_acc', 0.0) if val_result else 0.0
                    if val_acc > best_acc:
                        best_acc, best_model = val_acc, lit
                test_accs = [x['test_acc'] for x in pl.Trainer(logger=False).test(best_model, DataLoader(test_data, batch_size=64))]
                accs.append((test_accs[0], bootstrap(test_accs)))
            all_results[name][k] = accs

    for k in ks:
        plt.figure()
        for model in all_results:
            means = [all_results[model][k][i][0] for i in range(len(hidden_dims))]
            errs = [all_results[model][k][i][1] for i in range(len(hidden_dims))]
            plt.errorbar(hidden_dims, means, yerr=errs, label=model, marker='o', capsize=5)
        plt.title(f"MNIST Test Accuracy vs Hidden Dim (k={k})")
        plt.xlabel("Hidden Dim")
        plt.ylabel("Test Accuracy")
        plt.ylim(0.5, 1.0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"k{k}.png")
        plt.close()

        relu_best = max([acc for acc, _ in all_results["FFN_ReLU"][k]])
        geglu_best = max([acc for acc, _ in all_results["FFN_GeGLU"][k]])
        if geglu_best > relu_best:
            print(f"[k={k}] GeGLU outperformed ReLU: {geglu_best:.4f} vs {relu_best:.4f}")
        else:
            print(f"[k={k}] ReLU outperformed or matched GeGLU: {relu_best:.4f} vs {geglu_best:.4f}")

if __name__ == "__main__":
    run()
