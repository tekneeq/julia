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

data = {
    "PRCH": {
        "revenue": 435.60,
        "net_income": 55.83,
        "revenue_yoy": -0.0741,
        "revenue_fwd": 0.0634,
        "market_cap": 1570,
        "rank": 1,
    },
    "EGAN": {
        "revenue": 88.43,
        "net_income": 32.25,
        "revenue_yoy": -0.0471,
        "revenue_fwd": 0,
        "market_cap": 389.62,
        "rank": 2,
    },
    "HUT": {
        "revenue": 138.54,
        "net_income": 156.3,
        "revenue_yoy": -0.0564,
        "revenue_fwd": 0,
        "market_cap": 5350,
        "rank": 3,
    },
    "WDAY": {
        "revenue": 8960,
        "net_income": 583,
        "revenue_yoy": 0.1394,
        "revenue_fwd": 0.1387,
        "market_cap": 64060,
        "rank": 50,
    },
    "TEAM": {
        "revenue": 5460,
        "net_income": -184.79,
        "revenue_yoy": 0.1951,
        "revenue_fwd": 0.1961,
        "market_cap": 44880,
        "rank": 74,
    },
    "DOCU": {
        "revenue": 3100,
        "net_income": 280.97,
        "revenue_yoy": 0.0829,
        "revenue_fwd": 0.0734,
        "market_cap": 14710,
        "rank": 75,
    },
    "GWRE": {
        "revenue": 1200,
        "net_income": 69.8,
        "revenue_yoy": 0.2264,
        "revenue_fwd": 0.1797,
        "market_cap": 19860,
        "rank": 84,
    },
    "PD": {
        "revenue": 483.61,
        "net_income": -13.88,
        "revenue_yoy": 0.082,
        "revenue_fwd": 0.069,
        "market_cap": 1500,
        "rank": 87,
    },
    "HUBS": {
        "revenue": 2850,
        "net_income": -11.92,
        "revenue_yoy": 0.1895,
        "revenue_fwd": 0.1823,
        "market_cap": 25920,
        "rank": 93,
    },
    "CLBT": {
        "revenue": 436.73,
        "net_income": -150.95,
        "revenue_yoy": 0.2048,
        "revenue_fwd": 0.1905,
        "market_cap": 4170,
        "rank": 100,
    },
    "BOX": {
        "revenue": 1130,
        "net_income": 228.54,
        "revenue_yoy": 0.0629,
        "revenue_fwd": 0.0679,
        "market_cap": 4650,
        "rank": 101,
    },
    "PTC": {
        "revenue": 2470,
        "net_income": 512.73,
        "revenue_yoy": 0.1142,
        "revenue_fwd": 0.1018,
        "market_cap": 23780,
        "rank": 111,
    },
    "ASAN": {
        "revenue": 756.42,
        "net_income": -208,
        "revenue_yoy": 0.0974,
        "revenue_fwd": 0.0936,
        "market_cap": 3320,
        "rank": 120,
    },
    "APPF": {
        "revenue": 906.29,
        "net_income": 203.74,
        "revenue_yoy": 0.1888,
        "revenue_fwd": 0.2192,
        "market_cap": 9120,
        "rank": 121,
    },
    "OPRA": {
        "revenue": 583.45,
        "net_income": 81.27,
        "revenue_yoy": 0.3029,
        "revenue_fwd": 0.2037,
        "market_cap": 1320,
        "rank": 123,
    },
    "INFA": {
        "revenue": 1660,
        "net_income": -7.54,
        "revenue_yoy": 0.0116,
        "revenue_fwd": 0.037,
        "market_cap": 7580,
        "rank": 159,
    },
    "SNPS": {
        "revenue": 6440,
        "net_income": 2000,
        "revenue_yoy": 0.08,
        "revenue_fwd": 0.1829,
        "market_cap": 84300,
        "rank": 173,
    },
}

train_X = torch.tensor(
    [
        [v["revenue"], v["net_income"], v["revenue_yoy"], v["revenue_fwd"]]
        for v in data.values()
    ]
)  # shape (6, 3)

# Market caps (e.g., in $ billions)
train_y = torch.tensor([v["market_cap"] for v in data.values()])

# Predict for two hypothetical companies
ntnx_list = [2540, 188.37, 0.1814, 0.1589]
# ntnx_list = [2540, 188.37, 0.1, 0.1]
# ntnx_list = [5000, 500, 0.1814, 0.1589]
test_X = torch.tensor(
    [
        ntnx_list,
    ]
)

knn = KNNRegressor(k=3, weighted=True, standardize=True)
knn.fit(train_X, train_y)
pred_caps = knn.predict(test_X)
print(pred_caps)  # e.g., tensor([...])
