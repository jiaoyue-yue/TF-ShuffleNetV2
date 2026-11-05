# TF-ShuffleNetV2



## 1. 运行环境

- OS: Linux
- Python: 3.7
- PyTorch: 1.13.1
- GPU: NVIDIA A40

其他依赖见 `requirements.txt`。

## 2. 项目内容

- `models/`：模型结构及对比模型
- `configs/`：实验配置
- `data/`：来源分组与数据划分
- `preprocess.py`：数据预处理
- `train_grouped.py`：模型训练
- `metrics.py`：性能评价
- `recompute_fold_statistics.py`：统计分析

## 3. 数据集

采用公开TreeVibes数据集，共2485条振动声学记录。

数据集地址：
https://www.kaggle.com/datasets/potamitis/treevibes

输入特征为单通道log-Mel时频图，尺寸为1×48×157。

## 4. 模型与实验

包含TF-ShuffleNetV2-Tiny和TF-ShuffleNetV2-Small两种轻量化配置。

提供文件级十折交叉验证、来源分组五折评价、模型对比及模块消融的相关代码。
