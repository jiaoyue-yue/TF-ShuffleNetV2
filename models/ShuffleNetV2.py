import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    B, C, H, W = x.size()
    x = x.reshape(B, groups, C // groups, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.reshape(B, C, H, W)
    return x

class ShuffleV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 2

        if stride == 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),

                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, 1)
            out = torch.cat((x1, self.branch2(x2)), 1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, stage_out_channels=[24, 48, 96, 192, 1024]):
        super().__init__()
        self.stage_out_channels = stage_out_channels

        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, stage_out_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(stage_out_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 各stage
        self.stage2 = self._make_stage(stage_out_channels[0], stage_out_channels[1], 4)
        self.stage3 = self._make_stage(stage_out_channels[1], stage_out_channels[2], 8)
        self.stage4 = self._make_stage(stage_out_channels[2], stage_out_channels[3], 4)

        # 输出头
        self.conv5 = nn.Sequential(
            nn.Conv2d(stage_out_channels[3], stage_out_channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm2d(stage_out_channels[4]),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_out_channels[4], num_classes)

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = [ShuffleV2Block(in_channels, out_channels, 2)]
        for i in range(repeat - 1):
            layers.append(ShuffleV2Block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 用法示例
# model = ShuffleNetV2(input_channels=1, num_classes=2)
# print(model(torch.randn(2, 1, 48, 157)).shape)  # [2, 2]
