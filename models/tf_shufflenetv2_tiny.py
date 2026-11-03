import torch
import torch.nn as nn


def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, k=1, s=1, p=0, g=1, act=True):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, groups=g, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class TFStemLite(nn.Module):
    """
    更轻的 Time-Frequency 双分支输入层
    """
    def __init__(self, out_c=12):
        super().__init__()
        c1 = out_c // 2
        c2 = out_c - c1

        self.branch_t = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=(3, 7), stride=2, padding=(1, 3), bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.branch_f = nn.Sequential(
            nn.Conv2d(1, c2, kernel_size=(7, 3), stride=2, padding=(3, 1), bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        self.fuse = ConvBNAct(out_c, out_c, k=1, s=1, p=0, act=True)

    def forward(self, x):
        xt = self.branch_t(x)
        xf = self.branch_f(x)
        x = torch.cat([xt, xf], dim=1)
        x = self.fuse(x)
        return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        branch_c = out_c // 2

        if stride == 1:
            assert in_c == out_c
            self.branch2 = nn.Sequential(
                ConvBNAct(branch_c, branch_c, k=1, s=1, p=0, act=True),
                ConvBNAct(branch_c, branch_c, k=3, s=1, p=1, g=branch_c, act=False),
                ConvBNAct(branch_c, branch_c, k=1, s=1, p=0, act=True),
            )
        else:
            self.branch1 = nn.Sequential(
                ConvBNAct(in_c, in_c, k=3, s=2, p=1, g=in_c, act=False),
                ConvBNAct(in_c, branch_c, k=1, s=1, p=0, act=True),
            )
            self.branch2 = nn.Sequential(
                ConvBNAct(in_c, branch_c, k=1, s=1, p=0, act=True),
                ConvBNAct(branch_c, branch_c, k=3, s=2, p=1, g=branch_c, act=False),
                ConvBNAct(branch_c, branch_c, k=1, s=1, p=0, act=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, groups=2)
        return out


class ShuffleStage(nn.Module):
    def __init__(self, in_c, out_c, repeat):
        super().__init__()
        layers = [ShuffleUnit(in_c, out_c, stride=2)]
        for _ in range(repeat - 1):
            layers.append(ShuffleUnit(out_c, out_c, stride=1))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class DualPoolHead(nn.Module):
    """
    低 FLOPs 多尺度头：
    不在特征图上做融合，而是先降维再池化，最后拼接
    """
    def __init__(self, c3, c4, p3=32, p4=48):
        super().__init__()
        self.proj3 = ConvBNAct(c3, p3, k=1, s=1, p=0, act=True)
        self.proj4 = ConvBNAct(c4, p4, k=1, s=1, p=0, act=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x3, x4):
        x3 = self.pool(self.proj3(x3))   # [B, p3, 1, 1]
        x4 = self.pool(self.proj4(x4))   # [B, p4, 1, 1]
        x = torch.cat([x3, x4], dim=1)   # [B, p3+p4, 1, 1]
        return x


class TF_ShuffleNetV2_Tiny(nn.Module):
    """
    更低 FLOPs 的时频版 ShuffleNetV2
    输入: [B, 1, 48, 157]
    """
    def __init__(self, num_classes=2):
        super().__init__()

        self.stem = TFStemLite(out_c=12)          # -> [B,12,24,79]

        self.stage2 = ShuffleStage(12, 24, repeat=2)   # -> [B,24,12,40]
        self.stage3 = ShuffleStage(24, 48, repeat=3)   # -> [B,48,6,20]
        self.stage4 = ShuffleStage(48, 96, repeat=2)   # -> [B,96,3,10]

        self.head = DualPoolHead(c3=48, c4=96, p3=32, p4=48)

        self.dropout = nn.Dropout(0.10)
        self.classifier = nn.Linear(80, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x3)

        x = self.head(x3, x4)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = TF_ShuffleNetV2_Tiny(num_classes=2)
    x = torch.randn(2, 1, 48, 157)
    y = model(x)
    print("output shape:", y.shape)