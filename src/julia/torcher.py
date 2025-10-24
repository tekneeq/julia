import torch


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.train_X = None
        self.train_y = None

    def fit(self, X, y):
        """
        X: (num_samples, num_features)
        y: (num_samples,)
        """
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        """
        X: (num_test_samples, num_features)
        Returns predicted labels.
        """
        # Compute pairwise L2 distances
        # (num_test, num_train)
        distances = torch.cdist(X, self.train_X)

        # Get indices of k nearest neighbors for each example
        knn_idxs = distances.topk(self.k, largest=False).indices

        # Lookup labels of these neighbors
        knn_labels = self.train_y[knn_idxs]

        # Majority vote
        preds = knn_labels.mode(dim=1).values
        return preds


# ------------------------
# Example usage
# ------------------------

# Fake dataset: 6 samples, 2 features
train_X = torch.tensor(
    [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 3.0],
        [6.0, 5.0],
        [7.0, 8.0],
        [8.0, 7.0],
    ]
)

train_y = torch.tensor([0, 0, 0, 1, 1, 1])  # Two classes: 0 and 1

test_X = torch.tensor(
    [
        [2.5, 2.5],
        [7.0, 6.5],
    ]
)

knn = KNNClassifier(k=3)
knn.fit(train_X, train_y)
preds = knn.predict(test_X)

print(preds)  # tensor([0, 1])
