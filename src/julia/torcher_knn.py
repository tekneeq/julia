import torch


class KNNRegressor:
    def __init__(self, k=5, weighted=True, standardize=True, eps=1e-8):
        self.k = k
        self.weighted = weighted  # inverse-distance weights if True
        self.standardize = standardize  # z-score features using train stats
        self.eps = eps  # avoid div-by-zero
        self.train_X = None
        self.train_y = None
        self.mu = None
        self.sigma = None

    def _standardize(self, X):
        if not self.standardize:
            return X
        return (X - self.mu) / (self.sigma + self.eps)

    def fit(self, X, y):
        """
        X: (n_samples, n_features)  -> features: [revenue, net_income, rev_growth]
        y: (n_samples,)             -> target: market_cap
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        if self.standardize:
            self.mu = X.mean(dim=0, keepdim=True)
            self.sigma = X.std(dim=0, unbiased=False, keepdim=True)
        self.train_X = self._standardize(X)
        self.train_y = y

    def predict(self, X):
        """
        X: (n_test, n_features)
        returns: (n_test,) predicted market caps
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        Xs = self._standardize(X)  # (n_test, d)

        # pairwise L2 distances: (n_test, n_train)
        dists = torch.cdist(Xs, self.train_X)  # Euclidean

        # k nearest neighbor indices
        knn = dists.topk(self.k, largest=False)
        idxs = knn.indices  # (n_test, k)
        dsel = knn.values  # (n_test, k)

        neigh_vals = self.train_y[idxs]  # (n_test, k)

        if self.weighted:
            # inverse-distance weights (safe for zero distance)
            w = 1.0 / (dsel + self.eps)
            w = w / (w.sum(dim=1, keepdim=True) + self.eps)
            preds = (w * neigh_vals).sum(dim=1)
        else:
            preds = neigh_vals.mean(dim=1)

        return preds


# ------------------------
# Example usage
# ------------------------
# Suppose your rows are: [revenue, net_income, revenue_growth]
# Units matter! Use consistent units (e.g., USD billions, % as decimal).

train_X = torch.tensor(
    [
        [6.59, 1.17, 0.0334],  # $6.59B rev, $1.17B NI, 3.34% growth
        [3.35, 0.1392, 0.1131],
        [59.93, 18.93, 0.2801],
        [1.06, -0.31172, 0.2302],
        [1.78, -0.43391, -0.1374],
        [1.43, -0.08793, 0.3032],
    ]
)  # shape (6, 3)

# Market caps (e.g., in $ billions)
train_y = torch.tensor([23.28, 30.82, 1670, 7.86, 15.76, 22.52])

# Predict for two hypothetical companies
ntnx_list = [2.537, 0.1884, 0.1814]
# ntnx_list = [2.537, 0.1884, 0.1]
test_X = torch.tensor(
    [
        ntnx_list,
    ]
)

knn = KNNRegressor(k=3, weighted=True, standardize=True)
knn.fit(train_X, train_y)
pred_caps = knn.predict(test_X)
print(pred_caps)  # e.g., tensor([...])
