<div align="center">

# CMDPAD

[**中文**](./README_zh.md) | [**English**](./README.md)

</div>

---


[![Paper](https://img.shields.io/badge/Paper-Pattern%20Recognition-blue)](#) [![Dataset](https://img.shields.io/badge/Dataset-Available-green)](#) 

本仓库包含了发表在 *Pattern Recognition* 上的论文 **"[CMDPAD: A Chinese multimodal dynamic personality and affect dataset for affect prediction in conversations](https://www.sciencedirect.com/science/article/abs/pii/S0031320326007879)"** 的官方 PyTorch 实现代码。

## 💡项目介绍
本项目致力于**多模态对话分析**，重点关注**情感识别**、**人格识别**以及**情感预测**任务。项目包含一个全新的中文多模态数据集，并提供了基线模型代码。

相比于仅停留在识别说话者当前情绪的传统ARC（对话情感识别）任务，APC（对话情感预测）任务实现了视角的转换，将核心聚焦于预测听众在接收信息后可能产生的情感反馈 。这种机制高度契合人类在动态交互中的真实社交认知逻辑，从而为赋予和量化评估AI智能体的类人情感智能提供了一条更具前瞻性的多模态情感预测研究路径。

## 🌟 核心架构
基线方法提出一种多模态注意力 Transformer (Multi-modal Attention Transformer, MAT) 基准模型。特征提取与融合框架如下：
* **多模态特征**：文本 (BERT-Chinese)、音频 (Wav2Vec系列)、视觉 (ViT, ConvNeXt)。
* **单轮编码**：使用 Transformer Encoder 作为全局聚合器，通过 Self-Attention 学习模态间的交叉依赖，提取 `[CLS]` 级表征。
* **双轮预测**：使用 Cross-Attention 机制，结合说话人 A 的历史情感与人格（AR+PR）以及听话人 B 的上下文进行双向交叉注意力交互，预测 B 的未来情感状态（AP）。

## 📂 项目结构
```text
src/
├── config.py                           # 全局参数、路径与模态配置
├── dataset.py                          # PyTorch Dataset 与 Dataloader 实现
├── eval_metrics.py                     # 评估指标 (MAE, F1, Acc-5 等)
├── model.py                            # 核心网络: MAT 单轮多模态基准模型
├── main.py                             # 单轮任务训练脚本 (AR, PR, AP)
├── dialogue_feature_extractor.py       # 双轮特征提取器 (两模型级联)
├── dialogue_feature_extractor_multi.py # 双轮特征提取器 (多模型融合: AR+PR+AP)
└── dialogue_train_from_cls.py          # 最终的对话情感预测级联训练脚本
```

## ⚙️ 环境依赖

建议使用 Python 3.8+ 及 PyTorch 1.12+。

Bash

```
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn
```

## 📊 数据准备

请从 [[数据集链接](https://huggingface.co/datasets/HensonXie/CMDPAD)] 下载数据集，并将其放置在 `./dataset/` 目录下。所需文件包括：

- 单轮数据：`multimodal_features_single.pkl`, `label_single.csv`
- 双轮数据：`multimodal_features_paired.pkl`, `label_paired.csv`

（路径可在 `src/config.py` 中修改）。

## 🏃 快速开始 (Pipeline)

本项目的训练分为三个阶段：

### 阶段 1：训练单轮基准模型

分别训练情感识别 (AR)、动态人格识别 (PR) 和情感预测 (AP) 模型。

Bash

```
# 训练情感识别模型 (Affect Recognition)
python src/main.py --task affect_recognition --modalities bert-base-chinese wav2vec2-large-robust-emotion convnext-base

# 训练人格识别模型 (Personality Recognition)
python src/main.py --task personality_recognition --modalities bert-base-chinese wav2vec2-large-robust-emotion convnext-base

# 训练情感预测模型 (Affect Prediction)
python src/main.py --task affect_prediction --modalities bert-base-chinese wav2vec2-large-xlsr-chinese convnext-base
```

### 阶段 2：提取对话级上下文特征 (Cache CLS)

利用训练好的单轮模型，提取说话人 A 和 B 的 `[CLS]` 融合特征并缓存，以加速后续的双轮级联训练。

Bash

```
# 以 AR + PR + AP 模型提取双轮特征为例
python src/dialogue_feature_extractor_multi.py \
    --features ./dataset/multimodal_features_paired.pkl \
    --labels ./dataset/label_paired.csv \
    --model_A_aff ./saved_models/affect_recognition__bert-base-chinese...pt \
    --model_A_per ./saved_models/personality_recognition__bert-base-chinese...pt \
    --model_B ./saved_models/affect_prediction__bert-base-chinese...pt \
    --out_cache ./dataset/dialogue_cls_multi.pkl
```

### 阶段 3：训练对话情感融合预测器

基于缓存的双方 `[CLS]` 特征，训练最终的 Cross-Attention 融合网络。

Bash

```
python src/dialogue_train_from_cls.py \
    --cls_cache ./dataset/dialogue_cls_multi.pkl \
    --label_cols valence_prediction \
    --epochs 40 \
    --device cuda
```

## 🤝 引用

如果该工作对您的研究有用，请引用我们的论文（**已被 Pattern Recognition 接收**）：

```
@article{ZHOU2026113822,
    title = {CMDPAD: A Chinese multimodal dynamic personality and affect dataset for affect prediction in conversations},
    journal = {Pattern Recognition},
    volume = {179},
    pages = {113822},
    year = {2026},
    issn = {0031-3203},
    doi = {https://doi.org/10.1016/j.patcog.2026.113822},
    author = {Zisen Zhou and Heng Xie and Chang Wen and Xuefei Liu and Jianhua Tao and Zhengqi Wen and Changsheng Li and Zheng Lian and Jinming Zhao and Bingsen Xiong and Shaozheng Qin},
}
```
