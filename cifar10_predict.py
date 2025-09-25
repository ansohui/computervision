#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 (import via torchvision) + HOG + IPCA + SGDClassifier
- Replaces local 7z extraction & CSV scanning with torchvision.datasets.CIFAR10()
- Keeps original feature pipeline structure (HOG -> StandardScaler -> IncrementalPCA -> SGDClassifier)
- Adds CLI options for quick experimentation
"""
'''
라이브러리
pip install torch torchvision numpy opencv-python-headless pillow scikit-learn tqdm

실행
# 1) IPCA=300, 배치 1000 (DataLoader도 1000, IPCA도 1000)
python cifar10_predict.py --train_size 10000 --test_size 2000 --batch 1000 --ipca 300

# 2) 더 높은 차원 시도 (IPCA=500)
python cifar10_predict.py --train_size 10000 --test_size 2000 --batch 1000 --ipca 500

# 3) 분류기 로지스틱 + 규제 약화 + 반복수 ↑
python cifar10_predict.py --train_size 10000 --test_size 2000 \
--batch 1000 --ipca 500 --loss log_loss --alpha 1e-5 --max_iter 2000

2
python cifar10_predict.py \
  --train_size 10000 \
  --test_size 2000 \
  --ipca 500 \
  --batch 1000
  
python cifar10_predict.py --train_size 10000 --test_size 2000 --batch 1000 --ipca 500

빠른 테스트
python cifar10_predict.py --train_size 2000 --test_size 500

'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 (import via torchvision) + Grayscale + HOG + IPCA + SGD
- --batch 가 DataLoader(batch_size)와 IPCA(batch_size)에 모두 반영됨
- IPCA n_components가 batch_size/특징차원보다 크면 자동 조정
- 모든 학습용 미니배치를 동일 크기로 맞추기 위해 drop_last=True (IPCA 제약 때문)
"""
import argparse, math
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import SGDClassifier

# -------------------- 기본 설정 --------------------
HOG_SIZE_DEFAULT = (64, 64)
NBINS_DEFAULT    = 9
BATCH_DEFAULT    = 512
N_COMP_DEFAULT   = 150

# -------------------- HOG 준비 --------------------
def make_hog(win=(64,64), block=(16,16), stride=(8,8), cell=(8,8), nbins=9):
    return cv2.HOGDescriptor(_winSize=win, _blockSize=block, _blockStride=stride, _cellSize=cell, _nbins=nbins)

def to_gray01(img_chw: np.ndarray, out_size) -> np.ndarray:
    """
    img_chw: (C,H,W) float32 in [0,1]  ->  (H,W) float32 in [0,1] resized to out_size
    """
    c, h, w = img_chw.shape
    if c == 3:
        rgb = np.transpose(img_chw, (1,2,0))
        pil = Image.fromarray(np.clip(rgb*255,0,255).astype(np.uint8))
        pil = pil.convert("L").resize(out_size, Image.BILINEAR)
        return np.asarray(pil, dtype=np.float32)/255.0
    else:
        g = img_chw[0]
        g = cv2.resize(g, out_size, interpolation=cv2.INTER_LINEAR)
        return g.astype(np.float32)

def extract_hog(hog: cv2.HOGDescriptor, img01: np.ndarray, out_size) -> np.ndarray:
    u8 = np.ascontiguousarray(np.clip(img01*255,0,255).astype(np.uint8))
    if u8.shape != out_size:
        u8 = cv2.resize(u8, out_size, interpolation=cv2.INTER_LINEAR)
        u8 = np.ascontiguousarray(u8)
    return hog.compute(u8).ravel()

def fe_batch_tensor(hog, xb: torch.Tensor, out_size) -> np.ndarray:
    """
    xb: (B,C,H,W) torch float tensor in [0,1]
    returns: (B, D_hog) float32
    """
    xb_np = xb.numpy()
    feats = [extract_hog(hog, to_gray01(xb_np[i], out_size), out_size) for i in range(xb_np.shape[0])]
    return np.asarray(feats, dtype=np.float32)

# -------------------- 데이터 로드 (torchvision import) --------------------
def load_cifar10(root="./data", train_size=None, test_size=None, seed=42, bs=512):
    tfm = transforms.ToTensor()  # PIL -> [0,1] tensor (C,H,W)
    train_set = datasets.CIFAR10(root=root, train=True,  download=True, transform=tfm)
    test_set  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    rng = np.random.default_rng(seed)
    if train_size is not None and train_size < len(train_set):
        idx = rng.choice(len(train_set), size=train_size, replace=False)
        train_set = Subset(train_set, idx)
    if test_size is not None and test_size < len(test_set):
        idx = rng.choice(len(test_set), size=test_size, replace=False)
        test_set = Subset(test_set, idx)

    # IPCA 제약을 만족시키기 위해 train은 drop_last=True (모든 배치 크기 동일)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False, num_workers=0, drop_last=False)
    return train_loader, test_loader

# -------------------- 파이프라인 --------------------
def run_pipeline(root="./data",
                train_size=5000, test_size=1000, seed=42,
                ipca_components=N_COMP_DEFAULT, batch=BATCH_DEFAULT,
                hog_win=HOG_SIZE_DEFAULT, hog_block=(16,16), hog_stride=(8,8), hog_cell=(8,8),
                nbins=NBINS_DEFAULT,
                sgd_loss="hinge", sgd_alpha=1e-4, sgd_max_iter=1000):
    # 데이터
    train_loader, test_loader = load_cifar10(root=root, train_size=train_size, test_size=test_size, seed=seed, bs=batch)
    n_train = len(train_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(f"Train samples: {n_train} | Test samples: {n_test} | train batch_size: {batch}")

    # HOG 구성
    hog = make_hog(win=hog_win, block=hog_block, stride=hog_stride, cell=hog_cell, nbins=nbins)
    hog_out_size = hog_win

    # 첫 배치에서 특징 차원 파악 + IPCA 차원 자동 조정
    xb0, yb0 = next(iter(train_loader))
    feats0 = fe_batch_tensor(hog, xb0, hog_out_size)
    feat_dim = feats0.shape[1]
    # IPCA는 "각 partial_fit 배치 샘플 수 >= n_components" 여야 함
    # train_loader는 drop_last=True라서 모든 배치 크기 = batch
    ncomp_eff = min(ipca_components, batch, feat_dim)
    if ncomp_eff != ipca_components:
        print(f"[Info] Adjust IPCA n_components: requested={ipca_components} -> used={ncomp_eff} "
            f"(limited by batch={batch}, feat_dim={feat_dim})")

    # 1) Fit scaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    # 첫 배치 반영
    scaler.partial_fit(feats0)
    # 나머지 배치
    for xb, _ in tqdm(list(train_loader)[1:], desc="Fit scaler"):
        feats = fe_batch_tensor(hog, xb, hog_out_size)
        if feats.size: scaler.partial_fit(feats)

    # 2) Fit IPCA
    ipca = IncrementalPCA(n_components=ncomp_eff, batch_size=batch)
    # 다시 한 번 전체 배치 순회 (첫 배치 포함)
    for xb, _ in tqdm(train_loader, desc="Fit IPCA"):
        feats = fe_batch_tensor(hog, xb, hog_out_size)
        feats = scaler.transform(feats)
        ipca.partial_fit(feats)

    # 3) Train SGDClassifier incrementally
    clf = SGDClassifier(loss=sgd_loss, alpha=sgd_alpha, random_state=42, max_iter=sgd_max_iter)
    n_classes = 10
    first = True
    for xb, yb in tqdm(train_loader, desc="Train SGD"):
        feats = fe_batch_tensor(hog, xb, hog_out_size)
        feats = ipca.transform(scaler.transform(feats))
        y_np = yb.numpy()
        if first:
            clf.partial_fit(feats, y_np, classes=np.arange(n_classes))
            first = False
        else:
            clf.partial_fit(feats, y_np)

    # 4) Evaluate
    correct, total = 0, 0
    for xb, yb in tqdm(test_loader, desc="Eval"):
        feats = fe_batch_tensor(hog, xb, hog_out_size)
        feats = ipca.transform(scaler.transform(feats))
        pred = clf.predict(feats)
        y_np = yb.numpy()
        correct += int((pred == y_np).sum())
        total   += y_np.shape[0]
    acc = correct / total if total else 0.0
    print(f"[RESULT] Accuracy: {acc*100:.2f}%  (IPCA={ncomp_eff}, HOG={hog_win}, nbins={nbins}, loss={sgd_loss}, alpha={sgd_alpha})")
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./data")
    ap.add_argument("--train_size", type=int, default=5000)
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT, help="DataLoader & IPCA batch size (train drop_last=True)")
    ap.add_argument("--ipca", type=int, default=N_COMP_DEFAULT, help="Requested IPCA components (auto-bounded by batch & feat_dim)")

    # HOG 옵션
    ap.add_argument("--hog_w", type=int, default=HOG_SIZE_DEFAULT[0], help="HOG width (and height)")
    ap.add_argument("--hog_h", type=int, default=HOG_SIZE_DEFAULT[1], help="HOG height (and width)")
    ap.add_argument("--nbins", type=int, default=NBINS_DEFAULT)
    ap.add_argument("--cell", type=int, default=8, help="cell size (square)")
    ap.add_argument("--block", type=int, default=16, help="block size (square)")
    ap.add_argument("--stride", type=int, default=8, help="block stride (square)")

    # SGD 옵션
    ap.add_argument("--loss", type=str, default="hinge", choices=["hinge","log_loss"])
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--max_iter", type=int, default=1000)

    args = ap.parse_args()

    run_pipeline(root=args.root,
                train_size=args.train_size, test_size=args.test_size, seed=args.seed,
                ipca_components=args.ipca, batch=args.batch,
                hog_win=(args.hog_w, args.hog_h),
                hog_block=(args.block, args.block),
                hog_stride=(args.stride, args.stride),
                hog_cell=(args.cell, args.cell),
                nbins=args.nbins,
                sgd_loss=args.loss, sgd_alpha=args.alpha, sgd_max_iter=args.max_iter)

if __name__ == "__main__":
    main()
