
import torch
import torch.nn as nn


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.size()
    if c % groups != 0:
        raise ValueError(f"channels={c} must be divisible by groups={groups}")
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, k=1, s=1, p=0, g=1, act=True):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, groups=g, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class StandardStem(nn.Module):
    def __init__(self, out_c=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class TimeStem(nn.Module):
    def __init__(self, out_c=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, out_c, kernel_size=(3, 7), stride=2, padding=(1, 3), bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class FreqStem(nn.Module):
    def __init__(self, out_c=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, out_c, kernel_size=(7, 3), stride=2, padding=(3, 1), bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class TFStem(nn.Module):
    def __init__(self, out_c=16):
        super().__init__()
        assert out_c % 2 == 0, "out_c should be divisible by 2"
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
        return self.fuse(x)






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
        return channel_shuffle(out, groups=2)


class ShuffleStage(nn.Module):
    def __init__(self, in_c, out_c, repeat):
        super().__init__()
        layers = [ShuffleUnit(in_c, out_c, stride=2)]
        for _ in range(repeat - 1):
            layers.append(ShuffleUnit(out_c, out_c, stride=1))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class SmallBackbone(nn.Module):
    def __init__(self, stem_type="standard"):
        super().__init__()
        if stem_type == "standard":
            self.stem = StandardStem(out_c=16)
        elif stem_type == "time":
            self.stem = TimeStem(out_c=16)
        elif stem_type == "freq":
            self.stem = FreqStem(out_c=16)
        elif stem_type == "tf":
            self.stem = TFStem(out_c=16)
        else:
            raise ValueError(f"Unsupported stem_type: {stem_type}")
        self.stage2 = ShuffleStage(16, 32, repeat=2)
        self.stage3 = ShuffleStage(32, 64, repeat=4)
        self.stage4 = ShuffleStage(64, 128, repeat=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x3)
        return x3, x4


class GAPHead(nn.Module):
    def __init__(self, in_c=128, num_classes=2, hidden=64, drop1=0.15, drop2=0.10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(drop1),
            nn.Linear(in_c, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x4):
        x = self.pool(x4).flatten(1)
        return self.classifier(x)


class GateFusionHead(nn.Module):
    def __init__(self, c3=64, c4=128, proj_dim=48, hidden_dim=32):
        super().__init__()
        self.proj3 = ConvBNAct(c3, proj_dim, k=1, s=1, p=0, act=True)
        self.proj4 = ConvBNAct(c4, proj_dim, k=1, s=1, p=0, act=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x3, x4):
        v3 = self.pool(self.proj3(x3)).flatten(1)
        v4 = self.pool(self.proj4(x4)).flatten(1)
        g = torch.softmax(self.gate(torch.cat([v3, v4], dim=1)), dim=1)
        return g[:, 0:1] * v3 + g[:, 1:2] * v4


class UnifiedSmallModel(nn.Module):
    def __init__(self, num_classes=2, stem_type="standard", head_type="gap"):
        super().__init__()
        self.stem_type = stem_type
        self.head_type = head_type
        self.backbone = SmallBackbone(stem_type=stem_type)
        if head_type == "gap":
            self.head = GAPHead(in_c=128, num_classes=num_classes)
            self.classifier = None
        elif head_type == "gate":
            self.head = GateFusionHead(c3=64, c4=128, proj_dim=48, hidden_dim=32)
            self.classifier = nn.Sequential(
                nn.Dropout(0.15),
                nn.Linear(48, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.10),
                nn.Linear(64, num_classes),
            )
        else:
            raise ValueError("head_type must be 'gap' or 'gate'")

    def forward(self, x):
        x3, x4 = self.backbone(x)
        if self.head_type == "gap":
            return self.head(x4)
        x = self.head(x3, x4)
        return self.classifier(x)


def Small_Base(num_classes=2):
    return UnifiedSmallModel(num_classes=num_classes, stem_type="standard", head_type="gap")


def Small_TFStemOnly(num_classes=2):
    return UnifiedSmallModel(num_classes=num_classes, stem_type="tf", head_type="gap")


def Small_GateHeadOnly(num_classes=2):
    return UnifiedSmallModel(num_classes=num_classes, stem_type="standard", head_type="gate")


def TF_ShuffleNetV2_Small(num_classes=2):
    return UnifiedSmallModel(num_classes=num_classes, stem_type="tf", head_type="gate")




def Small_TimeStem(num_classes=2):
    return UnifiedSmallModel(num_classes=num_classes, stem_type="time", head_type="gap")


def Small_FreqStem(num_classes=2):
    return UnifiedSmallModel(num_classes=num_classes, stem_type="freq", head_type="gap")


def Small_TFStem(num_classes=2):
    return UnifiedSmallModel(num_classes=num_classes, stem_type="tf", head_type="gap")




MODEL_REGISTRY = {
    "Small-Base": Small_Base,
    "Small-TF-Stem": Small_TFStemOnly,
    "Small-GateFusionHead": Small_GateHeadOnly,
    "TF-ShuffleNetV2-Small": TF_ShuffleNetV2_Small,
    "TimeStem": Small_TimeStem,
    "FreqStem": Small_FreqStem,
    "TF-Stem": Small_TFStem,
}


def build_unified_model(name: str, num_classes=2):
    if name not in MODEL_REGISTRY:
        raise KeyError("Unsupported model: " + name)
    return MODEL_REGISTRY[name](num_classes=num_classes)
