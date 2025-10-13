#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-Nearest Neighbors (KNN) on CIFAR-10 — Assignment Version
----------------------------------------------------------
이 스크립트는 CIFAR-10 데이터셋에 대해 KNN 분류기를 적용하고,
다음의 세 가지 실험 모드를 지원한다.

[체크리스트 ✅]
1) CIFAR-10 데이터셋 (torchvision)
2) KNN 분류 (scikit-learn)
3) 실험 모드
   - split      : train/test 분할
   - split_val  : train/validation/test (val로 best-k 선택)
   - cv         : 5-fold 교차검증
4) 평가지표: accuracy, precision, recall, F1(macro)
5) k-스윕 결과 그래프 저장 (필요 시 save_plot_k_sweep 사용)

[실행 예시]
# 1) 단순 train/test (k=5만 검사)
python knn_cifar10_assignment.py --mode split --k_list 5 --train_size 10000 --test_size 5000

# 2) train/val/test (val로 best-k 선택)
python knn_cifar10_assignment.py --mode split_val --k_list 1 3 5 7 9 \
  --train_size 10000 --val_size 5000 --test_size 5000

# 3) 5-fold 교차검증 (k-스윕)
python knn_cifar10_assignment.py --mode cv --k_list 1 3 5 7 9 --folds 5
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
import matplotlib.pyplot as plt

def save_plot_k_sweep(rows, title, out):
    ks = [r["k"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    plt.figure(figsize=(6,4))
    plt.plot(ks, accs, "-o")
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[Saved plot] {out}")

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
    # 1) 먼저 test를 분리 → 남은 데이터에서 train/val 분리
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / len(X_tmp)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=random_state
    )

    # 2) 전처리 fit은 train으로만 → val/test에는 transform만 (데이터 누수 방지)
    X_tr_f, meta = build_features(X_tr, use_scaler=use_scaler)
    X_val_f = meta["scaler"].transform(X_val) if "scaler" in meta else X_val
    X_te_f  = meta["scaler"].transform(X_te)  if "scaler" in meta else X_te

    # 3) 검증셋으로 각 k 성능 측정 (→ 플롯에 쓸 rows 만들어두기)
    val_scores = []
    best_k, best_f1 = None, -1.0
    for k in k_list:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(X_tr_f, y_tr)
        val_pred = clf.predict(X_val_f)
        metrics = evaluate(y_val, val_pred)
        val_scores.append({"k": k, **metrics})
        if metrics["f1"] > best_f1:
            best_k, best_f1 = k, metrics["f1"]

    # 4) best-k로 재학습 후 test 평가
    clf = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    clf.fit(X_tr_f, y_tr)
    test_pred = clf.predict(X_te_f)
    print(f"[Best k={best_k}] Test → {evaluate(y_te, test_pred)}")

    # 5) 검증 성능 그래프 저장
    save_plot_k_sweep(val_scores, title="Validation Performance vs k", out="plot_val_k.png")

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
    """CIFAR-10을 torchvision에서 불러온다.
    - transform: ToTensor() → Tensor(C,H,W) in [0,1] 범위
    - train=True  : 50,000장 (클래스별 5,000장)
    - train=False : 10,000장 (여기서는 직접 분할을 하므로 주로 train=True 사용)
    """
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)
    return ds

def dataset_to_numpy(ds, max_samples=None):
    """torch Dataset → numpy 배열
    - 고전적인 KNN(거리기반) 실험을 위해 픽셀을 평탄화(flatten)하여 사용 (3072차원)
    - max_samples: 빠른 실험을 위한 상한
    반환:
      X: (N, 3072) float32, y: (N,) int64
    """
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
    parser = argparse.ArgumentParser(description="KNN on CIFAR-10 (Assignment)")
    parser.add_argument("--mode", choices=["split","split_val","cv"], required=True)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--k_list", type=int, nargs="+", default=[1,3,5,7,9])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    ds = load_cifar10(args.data_root, train=True)
    X, y = dataset_to_numpy(ds, max_samples=args.train_size + args.val_size + args.test_size)
    if args.mode == "split":
        run_simple_split(X, y, args.k_list, args.test_size)
    elif args.mode == "split_val":
        run_split_with_val(X, y, args.k_list, args.val_size, args.test_size)
    elif args.mode == "cv":
        run_kfold_cv(X, y, args.k_list, args.folds)


if __name__ == "__main__":
    main()
