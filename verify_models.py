"""Verify model shapes, parameter counts and the manuscript gate/head settings."""
import torch
import torch.nn as nn
from models import build_model

EXPECTED = {"Small-Base":45314, "Small-TF-Stem":45794, "Small-GateFusionHead":52772,
            "TF-ShuffleNetV2-Small":53252, "TimeStem":45506, "FreqStem":45506,
            "TF-Stem":45794, "TF-ShuffleNetV2-Tiny":27154, "ShuffleNetV2":343410,
            "MobileNetV3-Large":4204306, "DenseNet201":18090498}


def main():
    torch.manual_seed(2026)
    x = torch.randn(2, 1, 48, 157)
    for name, expected in EXPECTED.items():
        model = build_model(name).eval()
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected, name
        with torch.no_grad():
            result = model(x)
        assert tuple(result.shape) == (2, 2) and bool(torch.isfinite(result).all()), name
        if name == "TF-ShuffleNetV2-Small":
            backbone = model.backbone
            with torch.no_grad():
                stem = backbone.stem(x)
                s2 = backbone.stage2(stem)
                s3 = backbone.stage3(s2)
                s4 = backbone.stage4(s3)
            assert [tuple(a.shape[1:]) for a in [stem,s2,s3,s4]] == [(16,24,79),(32,12,40),(64,6,20),(128,3,10)]
            assert [(m.in_features,m.out_features) for m in model.head.gate if isinstance(m,nn.Linear)] == [(96,32),(32,2)]
            assert [m.p for m in model.classifier if isinstance(m,nn.Dropout)] == [.15,.10]
            assert all(m.bias is None for m in model.modules() if isinstance(m,nn.Conv2d))
        if name == "TF-ShuffleNetV2-Tiny":
            assert model.classifier.in_features == 80 and model.dropout.p == .10
        if name == "DenseNet201":
            assert model.model.features.conv0.in_channels == 1
            assert model.model.features.conv0.kernel_size == (7,7)
            assert isinstance(model.model.features.pool0,nn.MaxPool2d)
        print(name, "parameters", expected, "output", tuple(result.shape))


if __name__ == "__main__":
    main()
