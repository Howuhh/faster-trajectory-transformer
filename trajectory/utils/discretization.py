import torch
import numpy as np


class KBinsDiscretizer:
    # simplified and modified version of KBinsDiscretizer from sklearn, see:
    # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/preprocessing/_discretization.py#L21
    def __init__(self, dataset, num_bins=100, strategy="uniform"):
        self.strategy = strategy
        self.n_bins = num_bins
        self.feature_dim = dataset.shape[-1]

        # compute edges for binning
        self.bin_edges = self.__find_bin_edges(dataset)  # [feature_dim, num_bins]
        self.bin_centers = (self.bin_edges[:, 1:] + self.bin_edges[:, :-1]) * 0.5

        # for beam search, to be in the same device (for speed)
        self.bin_centers_torch = torch.from_numpy(self.bin_centers)

    def __find_bin_edges(self, X):
        if self.strategy == "uniform":
            mins, maxs = X.min(axis=0), X.max(axis=0)
            bin_edges = np.linspace(mins, maxs, self.n_bins + 1).T
        elif self.strategy == "quantile":
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.percentile(X, quantiles, axis=0).T
        else:
            raise RuntimeError("Unknown strategy, should be uniform or quatile.")

        return bin_edges

    def encode(self, X, subslice=None):
        if X.ndim == 1:
            X = X[None]

        if subslice is None:
            bin_edges = self.bin_edges
        else:
            start, end = subslice
            bin_edges = self.bin_edges[start:end]

        # See documentation of numpy.isclose for an explanation of ``rtol`` and ``atol``.
        rtol = 1.0e-5
        atol = 1.0e-8

        Xt = np.zeros_like(X, dtype=np.long)
        for jj in range(X.shape[1]):
            # Values which are close to a bin edge are susceptible to numeric
            # instability. Add eps to X so these values are binned correctly
            # with respect to their decimal truncation.
            eps = atol + rtol * np.abs(X[:, jj])
            Xt[:, jj] = np.digitize(X[:, jj] + eps, bin_edges[jj][1:])

        np.clip(Xt, 0, self.n_bins - 1, out=Xt)

        return Xt

    def decode(self, Xt, subslice=None):
        if Xt.ndim == 1:
            Xt = Xt[None]

        if subslice is None:
            bin_centers = self.bin_centers
        else:
            start, end = subslice
            bin_centers = self.bin_centers[start:end]

        X = np.zeros_like(Xt, dtype=np.float64)
        for jj in range(Xt.shape[1]):
            X[:, jj] = bin_centers[jj, np.int_(Xt[:, jj])]

        return X

    def expectation(self, probs, subslice=None):
        if probs.ndim == 1:
            probs = probs[None]

        # probs: [batch_size, num_dims, num_bins]
        # bins: [1, num_dims, num_bins]
        if torch.is_tensor(probs):
            bin_centers = self.bin_centers_torch.unsqueeze(0)
        else:
            bin_centers = self.bin_centers.unsqueeze(0)

        if subslice is not None:
            start, end = subslice
            bin_centers = bin_centers[:, start:end]

        assert probs.shape[1:] == bin_centers.shape[1:]

        # expectation: [batch_size, num_dims]
        exp = (probs * bin_centers).sum(axis=-1)

        return exp

    def to(self, device):
        self.bin_centers_torch = self.bin_centers_torch.to(device)

    def eval(self):
        return self



