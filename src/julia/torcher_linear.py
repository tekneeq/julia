import torch
import torch.nn as nn
import torch.optim as optim


# X: (n, 3) with columns: [revenue, net_income, rev_growth]
# y: (n,) market cap (same currency/units; e.g., USD billions)
def standardize(train_X, X):
    mu = train_X.mean(dim=0, keepdim=True)
    sigma = train_X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (X - mu) / sigma, mu, sigma


class LinearRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)  # predicts log_cap


class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # predicts log_cap


def fit_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    lr=1e-2,
    wd=1e-3,
    epochs=200,
):
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    for ep in range(1, epochs + 1):
        model.train()
        pred_log = model(X_train)
        loss = loss_fn(pred_log, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if X_val is not None and ep % 25 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), y_val).item()
            print(
                f"Epoch {ep:4d} | train MSE(log): {loss.item():.4f} | val MSE(log): {val_loss:.4f}"
            )
    return model


# ---------- Example (replace with your real data) ----------
# Toy data (numbers are made up / illustrative)
X_raw = torch.tensor(
    [
        [6.59, 1.17, 0.0334],  # $6.59B rev, $1.17B NI, 3.34% growth
        [3.35, 0.1392, 0.1131],
        [59.93, 18.93, 0.2801],
        [1.06, -0.31172, 0.2302],
        [1.78, -0.43391, -0.1374],
        [1.43, -0.08793, 0.3032],
    ],
    dtype=torch.float32,
)

y_cap = torch.tensor(
    [23.28, 30.82, 1670, 7.86, 15.76, 22.52], dtype=torch.float32
)

# Train/val split (here trivial)
X_train, y_train = X_raw, y_cap
# Standardize inputs; transform target -> log1p
X_train_std, mu, sigma = standardize(X_train, X_train)
y_train_log = torch.log1p(y_train)

# ----- Baseline: Linear on log(cap) -----
lin = LinearRegressor(in_dim=3)
fit_model(lin, X_train_std, y_train_log, epochs=300, lr=5e-3, wd=1e-2)
lin.eval()

# Predict for a new company
ntnx = torch.tensor([[2.537, 0.1884, 0.1814]], dtype=torch.float32)
ntnx_std = (ntnx - mu) / sigma
with torch.no_grad():
    pred_log = lin(ntnx_std)
    pred_cap = torch.expm1(pred_log)  # back to original scale
print("Linear prediction (cap):", pred_cap.item())

# ----- Small MLP (if you need mild nonlinearity) -----
mlp = SmallMLP(in_dim=3, hidden=32)
fit_model(mlp, X_train_std, y_train_log, epochs=500, lr=3e-3, wd=5e-3)
mlp.eval()
with torch.no_grad():
    pred_log_mlp = mlp(ntnx_std)
    pred_cap_mlp = torch.expm1(pred_log_mlp)
print("MLP prediction (cap):", pred_cap_mlp.item())
