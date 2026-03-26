<div align="center">

# CMDPAD

[**中文**](./README_zh.md) | [**English**](./README.md)

</div>

---



### 项目介绍

本项目致力于**多模态对话分析**，重点关注**情感识别**、**人格识别**以及**情感预测**任务。项目包含一个全新的中文多模态数据集，并提供了基线模型代码。

### 🛠️ 环境安装

本项目基于 Python 3.8+ 和 PyTorch 2.0.0 开发。请使用以下命令安装依赖：

```bash
pip install -r requirements.txt

```

**核心依赖库 (`requirements.txt`):**
（请参考英文版中的依赖列表，包含了 PyTorch, Transformers, FunASR, Whisper, Librosa 等关键库）。

### 📂 目录结构说明


* **数据目录 (`./data`)**
* `raw/`: 原始视频/音频数据存放位置
* `label_random.csv`: 标签文件
* `multimodal_features.pkl`: 提取后的多模态特征文件
* `features/features_summary.csv`: 特征格式说明文件


* **预处理目录 (`./preprocessing`)**
* `feature_extract/main.py`: **[核心]** 提取三模态特征并进行视频分帧
* `detection_multimodal.py`: 数据分离、语音转录文本（ASR）
* `test.py`: 数据完整性检查（校验三模态数据量是否匹配）


* **模型目录 (`./model`)**
* `main.py`: **[单轮]** 单轮对话任务主程序
* `dialogue_train_from_cls`: **[多轮]** 单轮对话任务主程序



### 🚀 运行步骤

#### 1. 特征处理

首先在配置文件中设置数据路径、视频分帧数以及预训练模型路径。
运行以下命令提取三模态特征：

```bash
python ./preprocessing/feature_extract/main.py

```

#### 2. 数据筛选与文本转录

进行数据清洗、分离以及音频转文本操作：

```bash
python ./preprocessing/detection_multimodal.py

```

运行测试脚本，确保三模态数据对其且数量一致：

```bash
python ./preprocessing/test.py

```

#### 3. 模型训练

**任务 A：单轮对话任务**
直接运行主程序进行训练/测试：

```bash
python ./model/main.py

```

**任务 B：多轮对话情感预测**

1. 提取上轮对话特征：
```bash
python ./model/dialogue_feature_extractor.py

```


2. 使用 Concat 方式融合并训练：
```bash
python ./model/dialogue_train_from_cls.py

```
