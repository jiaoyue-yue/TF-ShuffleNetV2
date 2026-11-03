"""Model factories using the names printed in the manuscript."""
from .tf_shufflenetv2_unified_models import MODEL_REGISTRY, build_unified_model


def build_model(name, num_classes=2):
    if num_classes != 2:
        raise ValueError("The manuscript task has two classes")
    if name in MODEL_REGISTRY:
        return build_unified_model(name, num_classes=num_classes)
    if name == "TF-ShuffleNetV2-Tiny":
        from .tf_shufflenetv2_tiny import TF_ShuffleNetV2_Tiny
        return TF_ShuffleNetV2_Tiny(num_classes=num_classes)
    if name == "ShuffleNetV2":
        from .ShuffleNetV2 import ShuffleNetV2
        return ShuffleNetV2(input_channels=1, num_classes=num_classes)
    if name == "MobileNetV3-Large":
        from .mobilenet_v3 import mobilenet_v3_large
        return mobilenet_v3_large(num_classes=num_classes, reduced_tail=False)
    if name == "DenseNet201":
        from .DenseNet201 import DenseNet201_1ch
        return DenseNet201_1ch(num_classes=num_classes, pretrained=False, keep_stem=True)
    raise KeyError("Model implementation is not included: " + name)
