## K-Nearest Neighbors (KNN) on CIFAR-10 — Assignment Version

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
