#  **Final Term — POC Dataset Classification using GoogLeNet**

## **Development Process**

---

### **1. Environment Setup**

* Configured a PyTorch-based deep learning environment
* Implemented data loading using `torchvision.datasets.ImageFolder`
* Added augmentation and normalization pipelines
* Organized the project structure (`googlenet.py`, `train.py`, `result/`)

---

### **2. Implementation of GoogLeNet Architecture**

* Reconstructed the Inception modules following the original paper
* Implemented Auxiliary Classifiers to stabilize gradient flow
* Built the complete GoogLeNet architecture from scratch
* Ensured auxiliary outputs are used only during training, not inference

---

### **3. Dataset Preparation**

The POC dataset contains two folders:

```
POC_Dataset/
  ├── Training/
  └── Testing/
```

* **Training** → split into **Train : Validation = 90 : 10**
* **Train set** → augmentation applied
* **Validation / Test sets** → only resize + normalize
* **Testing** folder was strictly used only at the very end
  → prevents data leakage and ensures proper generalization evaluation

---

## **4. Baseline Training**

The initial baseline training used a minimal pipeline:

* No augmentation
* No LR scheduler
* Direct train/test split

**Baseline performance:** ~50–56% accuracy
<img width="319" height="32" alt="스크린샷 2025-12-09 오후 5 48 30" src="https://github.com/user-attachments/assets/b86ae148-4553-45a1-bc01-b957629c33f2" />

**Issues detected:**

* Unstable training
* High confusion in certain classes
* Sensitivity to class imbalance

---

## **5. Data Augmentation & Normalization**

To improve generalization, the following augmentations were added:

* `RandomHorizontalFlip`
* `RandomRotation`
* `ColorJitter`
* Input normalization (`mean=0.5`, `std=0.5`)

**Result:**
Validation accuracy increased significantly — reaching **~71%**, with more stable loss curves.
This showed augmentation was essential for this medical dataset.
<img width="339" height="34" alt="스크린샷 2025-12-09 오후 5 48 49" src="https://github.com/user-attachments/assets/4142197a-04e4-4f3f-be6b-03c0a8f0d042" />

---

## **6. Training Stabilization**

To build a more reliable training procedure:

* Added auxiliary classifier loss (GoogLeNet aux branches)
* Introduced **StepLR** scheduler
* Implemented **Early Stopping (patience = 5)**
* Created a full **train/val/test split**
* Added automatic logging (CSV)
* Enabled intermediate Confusion Matrix visualization

**Effect:**

* Training stabilized
* Overfitting became easier to detect
* Best-performing model was saved automatically

---

## **7. Evaluation & Error Analysis**

Confusion matrices were generated for both validation and final test sets.

### **Key observations**

* **Chorionic_villi ↔ Trophoblastic_tissue** showed noticeable misclassification
* **Hemorrhage** was classified relatively accurately
* Visualization clearly revealed class imbalance and inter-class similarity issues

These insights guided tuning decisions throughout development.

---

## **8. Final Results**

* **Best Validation Accuracy:** 87.23%
* **Final Test Accuracy:** 81.34%
 <img width="303" height="88" alt="image" src="https://github.com/user-attachments/assets/3232ff3f-d678-418f-aa15-12487abbfc69" />

* Exported results include:

  * Per-epoch validation confusion matrices
  * Final test confusion matrix
  * Training log CSV
  * Best model checkpoint

Despite limited dataset size, the model shows strong improvement compared to the initial baseline.

---

## **9. Training Pipeline**

* **Loss:** CrossEntropyLoss + weighted auxiliary losses
* **Optimizer:** Adam (lr = 1e-3)
* **Scheduler:** StepLR(step_size=7, gamma=0.1)
* Added tqdm progress bars
* Enabled Early Stopping (patience = 5)
* Saved best model to:

  ```
  result/googlenet_poc_best.pt
  ```

---

## **10. Metrics & Visualization**

* Logged **training loss** and **validation accuracy** per epoch
* Saved validation confusion matrices:

  ```
  result/cm_val_epoch_XX.png
  ```
* Saved final test confusion matrix:

  ```
  result/cm_test_final.png
  ```
* Exported training log:

  ```
  result/training_log.csv
  ```

---

## **11. Final Test (Hold-out Evaluation)**

After training, the reserved Testing dataset was used for unbiased evaluation:

* Final Test Accuracy was computed
* Final Confusion Matrix generated
* Confirms generalization performance on unseen data

---

## **Final Results Summary**

### **Metrics**

* **Best Validation Accuracy:** 87.23%
* **Final Test Accuracy:** 81.34%

### **Confusion Matrix Insights**

* Some confusion between **Chorionic_villi** and **Trophoblastic_tissue**
* **Hemorrhage** was reliably classified
* Auxiliary classifiers helped stabilize training on a small dataset

---

## **Project Structure**

```
ComputerVision/
 ├── googlenet.py
 ├── train.py
 └── result/
      ├── cm_val_epoch_01.png
      ├── cm_val_epoch_02.png
      ├── cm_test_final.png
      ├── training_log.csv
      └── googlenet_poc_best.pt
```

---

## **Summary**

This project implements a complete training pipeline for classifying medical images using a reconstructed GoogLeNet architecture.

It includes:

* A clean train/validation/test workflow
* Auxiliary classifier integration
* Training stabilization techniques (scheduler, early stopping)
* Automated logging and visual analysis
* Rigorous evaluation on a dedicated hold-out test set

This work demonstrates practical deep learning engineering skills suitable for academic submissions or portfolio use.

## [MIDTERM]K-Nearest Neighbors (KNN) on CIFAR-10 — Assignment Version

이 프로젝트는 CIFAR-10 이미지 데이터셋을 이용하여
K-Nearest Neighbors (KNN) 분류기를 구현하고,
세 가지 실험 모드에 따라 모델 성능을 평가하는 과제용 코드이다.

## 주요 기능
| 기능             | 설명                                                               |
| -------------- | ---------------------------------------------------------------- |
| **KNN 분류기 구현** | `scikit-learn`의 `KNeighborsClassifier` 사용                        |
| **데이터셋 로드**    | `torchvision.datasets.CIFAR10`로 자동 다운로드 및 변환                     |
| **데이터 전처리**    | `StandardScaler`로 픽셀 단위 정규화 (거리 기반 성능 향상)                        |
| **실험 모드 3종**   | `train/test`, `train/validation/test`, `5-fold cross-validation` |
| **평가지표**       | Accuracy, Precision, Recall, F1-score (macro 평균)                 |
| **그래프 저장**     | k 값에 따른 정확도 변화를 시각화 (`matplotlib`)                               |

## 파일 구성

`knn_cifar10_assignment.py` : 메인 코드 (모든 기능 포함)  
`plot_split_k.png` : train/test 결과 그래프 (자동 생성)  
`plot_val_k.png` : validation 결과 그래프 (자동 생성)  
`plot_cv_k.png` : 5-fold cross-validation 결과 그래프 (자동 생성)

## 의존성
pip install torch torchvision scikit-learn matplotlib numpy  

## 실행 방법
### 1. 단순 train/test split: CIFAR-10 데이터셋을 단순히 학습/테스트로 나누어 평가
 
python knn_cifar10_assignment.py --mode split --k_list 5 \
  --train_size 10000 --test_size 5000
사용 데이터: train 10,000 / test 5,000

결과 그래프: plot_split_k.png

### 2. train / validation / test split: Validation 세트를 사용하여 최적의 k를 선택한 뒤, Test 세트에서 최종 성능 평가

python knn_cifar10_assignment.py --mode split_val --k_list 1 3 5 7 9 \
  --train_size 10000 --val_size 5000 --test_size 5000

Validation set으로 best-k 선택

Test set에서 해당 k로 최종 평가

결과 그래프: plot_val_k.png

### 3. 5-fold cross-validation: StratifiedKFold로 각 fold에서 KNN 학습 후 평균/표준편차 계산

python knn_cifar10_assignment.py --mode cv --k_list 1 3 5 7 9 --folds 5

폴드마다 독립적인 전처리 및 평가 수행

k별 평균 정확도 ± 표준편차 계산

결과 그래프: plot_cv_k.png

## 생성되는 그래프 요약
| 모드          | 파일 이름              | 내용                              |
| ----------- | ------------------ | ------------------------------- |
| `split`     | `plot_split_k.png` | Test Accuracy vs k              |
| `split_val` | `plot_val_k.png`   | Validation Accuracy vs k        |
| `cv`        | `plot_cv_k.png`    | 5-Fold Mean Accuracy ± Std vs k |
