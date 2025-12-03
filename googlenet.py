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
# GoogLeNet Pipeline
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