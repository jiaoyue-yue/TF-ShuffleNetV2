# TF-ShuffleNetV2 论文配套代码

本代码包整理来源分组比较的五个模型及模块、方向分支消融。统一入口为models.build_model(name)，输入B×1×48×157，输出B×2 logits；clean=0，infested=1。模型名称与论文保持一致。

## 环境与依赖

论文环境为Linux、Python 3.7、PyTorch 1.13.1、NVIDIA A40，列于paper_environment.json。requirements.txt中torchvision 0.14.1与PyTorch 1.13.1配套；依赖配套依据为PyTorch官方旧版本安装页：https://pytorch.org/get-started/previous-versions/ 。其余库列为安装依赖，不作为历史实验版本声明。

python -m pip install -r requirements.txt

## 数据及固定划分

TreeVibes：https://www.kaggle.com/datasets/potamitis/treevibes

数据论文：https://doi.org/10.3390/smartcities4010017

data/records.csv保存2485条记录的相对路径、标签、来源及SHA256。data/assignments.csv为固定17组、5外层折的逐文件角色长表，共12425行，与《来源分组五折划分完整清单.xlsx》一致。保持下载后的field/train目录结构。data/fold_f1.csv和data/fold_counts.csv保存三模型实际逐折统计。

python check_split_manifest.py --records data/records.csv --assignments data/assignments.csv

## 模型与消融

来源分组比较：TF-ShuffleNetV2-Small、TF-ShuffleNetV2-Tiny、ShuffleNetV2、MobileNetV3-Large、DenseNet201。

模块消融：Small-Base、Small-TF-Stem、Small-GateFusionHead、TF-ShuffleNetV2-Small。

方向分支消融：TimeStem、FreqStem、TF-Stem。

DenseNet201采用pretrained=False、keep_stem=True；MobileNetV3-Large使用原主干宽度，输入一通道、输出两类；所有模型均从头训练。

python verify_models.py

verify_models.py检查前向输出、可训练参数量及关键层设置。Small为53252参数（53.25 K），Tiny为27154参数（27.15 K）。这些整数按K单位保留两位小数对应论文。该脚本需在已安装PyTorch的环境中执行。

## 特征与训练

preprocess.py及prepare_features.py根据论文整理WAV到log-Mel的处理入口。train_grouped.py根据固定划分与论文设置整理来源分组训练入口。文件级十折的配置单列在configs/file_tenfold.json；本包训练命令执行来源分组五折。

python prepare_features.py --wav-root /path/to/parent_of_field --output feature_cache

python train_grouped.py --features feature_cache --output results_grouped --device cuda:0

python train_grouped.py --features feature_cache --output results_ablation --models Small-Base Small-TF-Stem Small-GateFusionHead TF-ShuffleNetV2-Small

训练种子为2026；外层每折只由内层验证选择检查点，每2轮验证、最高正类F1、同分保留最早；所选检查点直接外层测试。类别权重仅由当前内层训练记录计算，不重新拟合。外层测试不参与选择。

文件级十折的最大学习率为3e-4、调度起始1.2e-5；来源分组五折分别为3e-3和1.2e-4；其余主要设置按论文配置。更详细的环境、设置和统计定义见回复稿附录A—C。

## 统计复算

python recompute_fold_statistics.py --input data/fold_f1.csv --output fold_statistics.csv

以同一折正类F1差值进行双侧近似配对t检验，自由度4，两项比较用Holm校正。F1输入单位为百分数，差值单位为百分点。AUC由连续受害概率计算，不能根据混淆计数反推。metrics.py提供阈值指标与含并列秩处理的AUC函数。

模型复杂度按论文使用ptflops，输入尺寸为(1,48,157)、批次大小1。完整性哈希见code_checksums.json。
