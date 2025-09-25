#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN classifier (from scratch) on CIFAR-10 for VS Code
- Uses torchvision to import CIFAR-10 (no manual download needed)
- Supports L1 / L2 distances
- Batched prediction to fit in memory
- Subsampling options for quick runs during practice
"""
'''
기본 실행

python knn_cifar10.py

L1 거리 + k=7
python knn_cifar10.py --metric l1 --k 7

더 빨리 시험: train 2000, test 500
python knn_cifar10.py --train_size 2000 --test_size 500

메모리/속도 조절: 배치 크기 변경
python knn_cifar10.py --batch_size 128
'''
import argparse
import time
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def load_cifar10(root: str = "./data",
                train_size: int = 5000,
                test_size: int = 1000,
                seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 and return flattened numpy arrays (0~1 float32)."""
    tfm = transforms.ToTensor()  # [0,1], shape (C,H,W)
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_set  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    rng = np.random.default_rng(seed)
    if train_size is not None and train_size < len(train_set):
        train_idx = rng.choice(len(train_set), size=train_size, replace=False)
        train_set = Subset(train_set, train_idx)
    if test_size is not None and test_size < len(test_set):
        test_idx = rng.choice(len(test_set), size=test_size, replace=False)
        test_set = Subset(test_set, test_idx)

    def dataset_to_numpy(ds):
        loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
        xs, ys = [], []
        for x, y in loader:
            # x: (B,3,32,32) -> (B, 3072)
            xs.append(x.view(x.size(0), -1).numpy().astype(np.float32))
            ys.append(y.numpy())
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return X, y

    X_train, y_train = dataset_to_numpy(train_set)
    X_test,  y_test  = dataset_to_numpy(test_set)

    return X_train, y_train, X_test, y_test

class KNNClassifier:
    def __init__(self, k: int = 3, metric: str = "l2", batch_size: int = 256):
        assert metric in {"l1", "l2"}
        assert k >= 1
        self.k = k
        self.metric = metric
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Memorize training data."""
        self.X_train = X.astype(np.float32)
        self.y_train = y.astype(np.int64)

    def _pairwise_l2(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Efficient squared L2 distances between A (n,d) and B (m,d):
        ||A-B||^2 = ||A||^2 + ||B||^2 - 2 A·B^T
        Returns (n,m) distances.
        """
        # Cast to float32 to control memory
        A2 = np.sum(A*A, axis=1, keepdims=True)        # (n,1)
        B2 = np.sum(B*B, axis=1, keepdims=True).T      # (1,m)
        # Use matrix multiply for -2AB^T
        ABt = A @ B.T                                   # (n,m)
        D2 = A2 + B2 - 2.0 * ABt
        # Numerical floor to zero
        np.maximum(D2, 0.0, out=D2)
        return D2

    def _pairwise_l1(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        L1 distances between A (n,d) and B (m,d).
        Computes in chunks to reduce peak memory.
        """
        n, d = A.shape
        m = B.shape[0]
        # Choose a chunk size for B to bound memory
        # Aim for ~200MB peak:  n*chunk floats ~ 50M -> with n<=1000 works well.
        # We'll instead chunk over A (test) in predict(), so here do full B.
        # This function expects relatively small n.
        D = np.empty((n, m), dtype=np.float32)
        # Chunk over feature dimension to reduce memory if needed
        # But vectorized abs difference over all dims is typically fine for given n.
        for i in range(n):
            # broadcast |A[i] - B| over axis 0
            D[i, :] = np.sum(np.abs(B - A[i, :]), axis=1, dtype=np.float32)
        return D

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for X using batched distance computation."""
        assert self.X_train is not None, "Call fit() first."
        X = X.astype(np.float32)
        N = X.shape[0]
        preds = np.empty((N,), dtype=np.int64)

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            Xb = X[start:end]  # (b,d)
            if self.metric == "l2":
                D = self._pairwise_l2(Xb, self.X_train)  # (b, n_train), squared L2 is fine for NN
            else:
                D = self._pairwise_l1(Xb, self.X_train)  # (b, n_train)

            # Get indices of k smallest distances
            # argpartition is O(n) and memory-efficient
            nn_idx = np.argpartition(D, kth=self.k-1, axis=1)[:, :self.k]  # (b, k)
            # Gather neighbor labels
            nn_labels = self.y_train[nn_idx]  # (b, k)
            # Majority vote
            # For speed, use bincount per row
            for i in range(nn_labels.shape[0]):
                counts = np.bincount(nn_labels[i], minlength=10)  # CIFAR-10 has 10 classes
                preds[start + i] = np.argmax(counts)

            # Optional: free memory
            del D, nn_idx, nn_labels

        return preds

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())

def main():
    parser = argparse.ArgumentParser(description="KNN (from scratch) on CIFAR-10")
    parser.add_argument("--root", type=str, default="./data", help="dataset root (cache)")
    parser.add_argument("--train_size", type=int, default=5000, help="number of training samples (None for full)")
    parser.add_argument("--test_size", type=int, default=1000, help="number of test samples (None for full)")
    parser.add_argument("--k", type=int, default=5, help="number of neighbors")
    parser.add_argument("--metric", type=str, default="l2", choices=["l1", "l2"], help="distance metric")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for batched prediction")
    parser.add_argument("--seed", type=int, default=42, help="random seed for subsampling")
    args = parser.parse_args()

    print("==> Loading CIFAR-10 (import via torchvision)...")
    X_train, y_train, X_test, y_test = load_cifar10(
        root=args.root, train_size=args.train_size, test_size=args.test_size, seed=args.seed
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    clf = KNNClassifier(k=args.k, metric=args.metric, batch_size=args.batch_size)
    print("==> Fitting (memorizing training data)...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()
    print(f"Fit done in {t1 - t0:.2f}s")

    print(f"==> Predicting with k={args.k}, metric={args.metric} ...")
    t2 = time.time()
    y_pred = clf.predict(X_test)
    t3 = time.time()
    acc = accuracy(y_test, y_pred)
    print(f"Predict done in {t3 - t2:.2f}s")
    print(f"[RESULT] Accuracy: {acc*100:.2f}% (k={args.k}, metric={args.metric}, "
          f"train_size={args.train_size}, test_size={args.test_size})")

    # Show a tiny confusion matrix summary (counts of correct/wrong)
    correct = int((y_test == y_pred).sum())
    total = len(y_test)
    print(f"Correct: {correct}/{total} | Wrong: {total - correct}")

if __name__ == "__main__":
    main()
