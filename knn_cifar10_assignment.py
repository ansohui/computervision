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
    """k-스윕 결과를 선 그래프로 저장
    - rows: [{"k": int, "accuracy": float, ...}, ...] 형태 가정
    - title: 그림 제목
    - out: 저장 파일 경로 (예: 'plot_split_k.png')
    [사용 예시]
      rows = [{"k":1,"accuracy":0.45}, {"k":3,"accuracy":0.50}, ...]
      save_plot_k_sweep(rows, "Simple Split Accuracy vs k", "plot_split_k.png")
    """
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
    """
    StratifiedKFold 교차검증
    - 각 폴드에서 train/test를 반복하며 k별 성능 측정
    - fold별 평균/표준편차 계산 후 그래프로 저장 (plot_cv_k.png)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {k: [] for k in k_list}

    # fold 반복
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[CV] Fold {fold}/{n_splits}")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 각 fold에서 scaler는 새로 학습해야 함 (데이터 누수 방지)
        X_tr_f, meta = build_features(X_tr, use_scaler=use_scaler)
        X_te_f = meta["scaler"].transform(X_te) if "scaler" in meta else X_te

        # k 스윕
        for k in k_list:
            clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            clf.fit(X_tr_f, y_tr)
            y_pred = clf.predict(X_te_f)
            metrics = evaluate(y_te, y_pred)
            results[k].append(metrics["accuracy"])
            print(f"k={k} → acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

    # 평균/표준편차 계산
    summary_rows = []
    print("\n[CV Summary] mean ± std")
    for k in k_list:
        accs = np.array(results[k])
        mean, std = accs.mean(), accs.std(ddof=1)
        summary_rows.append({"k": k, "accuracy_mean": mean, "accuracy_std": std})
        print(f"k={k}: {mean:.4f} ± {std:.4f}")

    # 그래프 저장 (error bar 포함)
    plt.figure(figsize=(7, 5))
    ks = [r["k"] for r in summary_rows]
    acc_mean = [r["accuracy_mean"] for r in summary_rows]
    acc_std = [r["accuracy_std"] for r in summary_rows]

    plt.errorbar(ks, acc_mean, yerr=acc_std, fmt="-o", capsize=5)
    plt.title(f"{n_splits}-Fold CV: Accuracy vs k (±std)")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_cv_k.png")
    print("[Saved plot] plot_cv_k.png")

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

    # 3) 검증셋으로 각 k 성능 측정 (→ 플롯에 쓸 rows 만들어두기) (여기선 macro-F1 기준)
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
    # 1) Stratified split: 클래스 비율 보존
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 2) 전처리(학습셋 기준으로 fit) → 테스트셋에는 transform만 적용 (데이터 누수 방지)
    X_tr_f, meta = build_features(X_tr, use_scaler=use_scaler)
    X_te_f = meta["scaler"].transform(X_te) if "scaler" in meta else X_te

    # 3) 여러 k 스윕: 학습→예측→지표 출력
    print("[Simple Split] Train:", X_tr_f.shape, "Test:", X_te_f.shape)
    results = []  # 👈 그래프용 데이터 저장 리스트
    for k in k_list:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(X_tr_f, y_tr)
        y_pred = clf.predict(X_te_f)
        metrics = evaluate(y_te, y_pred)
        results.append({"k": k, **metrics})
        print(f"k={k} → {metrics}")

    # 4) 결과 요약 + 그래프 저장
    print("\n[Simple Split Summary]")
    for row in results:
        print(row)

    save_plot_k_sweep(results, title="Simple Split: Test Accuracy vs k", out="plot_split_k.png")


def evaluate(y_true, y_pred):
    """평가지표 계산
    - accuracy: 전체 정확도
    - precision/recall/F1: macro-average (클래스 불균형 영향 완화)
    - zero_division=0: 특정 클래스 미예측 시 division by zero 방지
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def build_features(X, use_scaler=True):
    """특징 전처리 (KNN 최적화)
    - KNN은 '거리' 기반 → 각 차원의 스케일(분산)이 다르면 왜곡 발생
    - StandardScaler: 평균 0, 표준편차 1로 정규화하여 거리 계산을 안정화
    """
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
