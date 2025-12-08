import torch
import torch.nn as nn
import torch.nn.functional as F

# inception module
class Inception(nn.Module):
    def __init__(
        self,
        in_channels,
        ch1x1,
        ch3x3_reduce, ch3x3, #chNxN_reduce: 1×1 conv output channels (dimension reduction)
        ch5x5_reduce, ch5x5,
        pool_proj

    ):
        super().__init__()
        #Branch 1 : 1x1 Conv
        self.branch1 = nn.Sequential(
            #nn.Conv2d(입력_채널수, 출력_채널수, 커널크기)
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        #Branch 2 : 1x1 -> 3x3 Conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        #Branch 3 : 1x1 -> 5x5 Conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        #Branch 4 : Maxpool -> 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Concatenate the outputs
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # 채널 방향으로 concat
        return torch.cat([b1, b2, b3, b4], dim=1)

# Auxiliary Classifier
class AuxiliaryClassifier(nn.Module):
    def __init__(
        self, 
        in_channels, 
        num_classes
    ):
        super().__init__()
        #AdaptiveAvgPool2d: NxN ->  4x4
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        # 2) 1x1 Conv로 채널 줄이기 (bottleneck)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # 3) FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 1024) #2048 -> 1024
        self.dropout = nn.Dropout(0.7) 
        self.fc2 = nn.Linear(1024, num_classes) #1024 -> the number of POC dataset class (=4)

    def forward(self, x):
        # x: [Batch size, Channel, Heghit, Weight] (예: [B, 512, 14, 14])
        x = self.avgpool(x)          # → [B, 128, 4, 4] 전에 conv
        x = self.conv(x)             # → [B, 128, 4, 4]
        x = torch.flatten(x, 1)      # → [B, 128*4*4]
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)              # → [B, num_classes]
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=4, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        # ===== Stem =====
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ===== Inception 3a / 3b =====
        self.inception3a = Inception(
            in_channels=192,
            ch1x1=64,
            ch3x3_reduce=96, ch3x3=128,
            ch5x5_reduce=16, ch5x5=32,
            pool_proj=32
        )  # 256

        self.inception3b = Inception(
            in_channels=256,
            ch1x1=128,
            ch3x3_reduce=128, ch3x3=192,
            ch5x5_reduce=32, ch5x5=96,
            pool_proj=64
        )  # 480

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ===== Inception 4a ~ 4e =====
        self.inception4a = Inception(
            in_channels=480,
            ch1x1=192,
            ch3x3_reduce=96, ch3x3=208,
            ch5x5_reduce=16, ch5x5=48,
            pool_proj=64
        )  # 512

        self.inception4b = Inception(
            in_channels=512,
            ch1x1=160,
            ch3x3_reduce=112, ch3x3=224,
            ch5x5_reduce=24, ch5x5=64,
            pool_proj=64
        )  # 512

        self.inception4c = Inception(
            in_channels=512,
            ch1x1=128,
            ch3x3_reduce=128, ch3x3=256,
            ch5x5_reduce=24, ch5x5=64,
            pool_proj=64
        )  # 512

        self.inception4d = Inception(
            in_channels=512,
            ch1x1=112,
            ch3x3_reduce=144, ch3x3=288,
            ch5x5_reduce=32, ch5x5=64,
            pool_proj=64
        )  # 528

        self.inception4e = Inception(
            in_channels=528,
            ch1x1=256,
            ch3x3_reduce=160, ch3x3=320,
            ch5x5_reduce=32, ch5x5=128,
            pool_proj=128
        )  # 832

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ===== Inception 5a / 5b =====
        self.inception5a = Inception(
            in_channels=832,
            ch1x1=256,
            ch3x3_reduce=160, ch3x3=320,
            ch5x5_reduce=32, ch5x5=128,
            pool_proj=128
        )  # 832

        self.inception5b = Inception(
            in_channels=832,
            ch1x1=384,
            ch3x3_reduce=192, ch3x3=384,
            ch5x5_reduce=48, ch5x5=128,
            pool_proj=128
        )  # 1024

        # ===== Auxiliary classifiers =====
        if self.aux_logits:
            self.aux1 = AuxiliaryClassifier(in_channels=512, num_classes=num_classes)  # after 4a
            self.aux2 = AuxiliaryClassifier(in_channels=528, num_classes=num_classes)  # after 4d
        else:
            self.aux1 = None
            self.aux2 = None

        # ===== Classifier head =====
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # ---- Stem ----
        x = F.relu(self.conv1(x), inplace=True)
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = self.maxpool2(x)

        # ---- Inception 3 ----
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # ---- Inception 4 ----
        x = self.inception4a(x)
        aux1 = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)  # after 4a

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)  # after 4d

        x = self.inception4e(x)
        x = self.maxpool4(x)

        # ---- Inception 5 + head ----
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)         # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)     # [B, 1024]
        x = self.dropout(x)
        x = self.fc(x)              # [B, num_classes]

        if self.aux_logits and self.training:
            return x, aux1, aux2    # train 모드
        else:
            return x                # eval 모드

# main
if __name__ == "__main__":

    # test: inception module
    # Dummy input (batch=1, channels=192, size=28×28)
    dummy = torch.randn(1, 192, 28, 28)

    inception3a = Inception(
        in_channels=192,
        ch1x1=64,
        ch3x3_reduce=96, ch3x3=128,
        ch5x5_reduce=16, ch5x5=32,
        pool_proj=32
    )

    # Forward pass
    output = inception3a(dummy)

    print("출력 shape:", output.shape) 

    #test: Auxiliary Classifier
    feat = torch.randn(1, 512, 14, 14)
    aux_clf = AuxiliaryClassifier(in_channels=512, num_classes=4)  # POC_dataset이면 4클래스

    aux_out = aux_clf(feat)
    print("\nAuxiliaryClassifier 출력 shape:", aux_out.shape)  