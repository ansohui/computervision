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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def run_kfold_cv(X, y, k_list, n_splits=5, random_state=42, use_scaler=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {k: [] for k in k_list}

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"[Fold {fold}/{n_splits}]")
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        X_tr_f, meta = build_features(X_tr, use_scaler=use_scaler)
        X_te_f = meta["scaler"].transform(X_te) if "scaler" in meta else X_te

        for k in k_list:
            clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            clf.fit(X_tr_f, y_tr)
            pred = clf.predict(X_te_f)
            results[k].append(evaluate(y_te, pred)["accuracy"])

    for k in k_list:
        accs = np.array(results[k])
        print(f"k={k}: mean={accs.mean():.4f}, std={accs.std(ddof=1):.4f}")

def run_split_with_val(X, y, k_list, val_size, test_size, random_state=42, use_scaler=True):
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_ratio = val_size / len(X_tmp)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=random_state)

    X_tr_f, meta = build_features(X_tr, use_scaler=use_scaler)
    X_val_f = meta["scaler"].transform(X_val) if "scaler" in meta else X_val
    X_te_f = meta["scaler"].transform(X_te) if "scaler" in meta else X_te

    best_k, best_f1 = None, -1
    for k in k_list:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(X_tr_f, y_tr)
        val_pred = clf.predict(X_val_f)
        metrics = evaluate(y_val, val_pred)
        if metrics["f1"] > best_f1:
            best_k, best_f1 = k, metrics["f1"]

    clf = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    clf.fit(X_tr_f, y_tr)
    test_pred = clf.predict(X_te_f)
    print(f"[Best k={best_k}] Test → {evaluate(y_te, test_pred)}")

def run_simple_split(X, y, k_list, test_size, random_state=42, use_scaler=True):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_tr_f, meta = build_features(X_tr, use_scaler=use_scaler)
    X_te_f = meta["scaler"].transform(X_te) if "scaler" in meta else X_te

    print("[Simple Split] Train:", X_tr_f.shape, "Test:", X_te_f.shape)
    for k in k_list:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(X_tr_f, y_tr)
        y_pred = clf.predict(X_te_f)
        print(f"k={k} → {evaluate(y_te, y_pred)}")

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def build_features(X, use_scaler=True):
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, {"scaler": scaler}
    return X, {}

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
