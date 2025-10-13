#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-Nearest Neighbors (KNN) on CIFAR-10
--------------------------------------
"""

import argparse
import numpy as np
import torch
from torchvision import datasets, transforms

def load_cifar10(root="./data", train=True):
    """CIFAR-10을 torchvision에서 불러온다."""
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)
    return ds

def dataset_to_numpy(ds, max_samples=None):
    """torch Dataset → numpy 배열로 변환"""
    if max_samples is None:
        max_samples = len(ds)
    idxs = np.arange(len(ds))[:max_samples]
    Xs, ys = [], []
    for i in idxs:
        img, label = ds[i]
        Xs.append(img.numpy().reshape(-1))
        ys.append(label)
    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="KNN on CIFAR-10 (base structure)")
    args = parser.parse_args()
    print("Base structure ready.")

if __name__ == "__main__":
    main()
