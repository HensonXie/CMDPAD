<div align="center">

# CMDPAD

[**中文**](./README_zh.md) | [**English**](./README.md)

</div>

---
[![Paper](https://img.shields.io/badge/Paper-Pattern%20Recognition-blue)](#) [![Dataset](https://img.shields.io/badge/Dataset-Available-green)](#) 

This repository contains the official PyTorch implementation for the paper **"[CMDPAD: A Chinese multimodal dynamic personality and affect dataset for affect prediction in conversations](https://www.sciencedirect.com/science/article/abs/pii/S0031320326007879)"**, published in *Pattern Recognition*.

## 💡 Project Introduction

This project focuses on **multimodal dialogue analysis**, with an emphasis on **affect recognition**, **personality recognition**, and **affect prediction** tasks. The repository includes a novel Chinese multimodal dataset and provides baseline model code.

Compared to traditional ARC (Affect Recognition in Conversations) tasks that merely identify the speaker's current emotion, the APC (Affect Prediction in Conversations) task shifts the perspective by focusing on predicting the emotional feedback a listener is likely to generate after receiving information. This mechanism highly aligns with the real social cognitive logic of humans in dynamic interactions, thereby providing a more forward-looking, multimodal affect prediction research path for endowing and quantitatively evaluating the human-like emotional intelligence of AI agents.

## 🌟 Core Architecture

The baseline method proposes a Multi-modal Attention Transformer (MAT) benchmark model. The feature extraction and fusion framework is as follows:

* **Multimodal Features**: Text (BERT-Chinese), Audio (Wav2Vec series), Visual (ViT, ConvNeXt).
* **Single-Turn Encoding**: Employs a Transformer Encoder as a global aggregator to learn cross-modal dependencies via Self-Attention and extract `[CLS]` level representations.
* **Dialogue-Based (Two-Turn) Prediction**: Utilizes a Cross-Attention mechanism to conduct bi-directional cross-attention interaction, combining Speaker A's historical affect and personality (AR+PR) with Listener B's context to predict B's future affective state (AP).

## 📂 Project Structure

```text
src/
├── config.py                           # Global parameters, paths, and modality configurations
├── dataset.py                          # PyTorch Dataset and Dataloader implementation
├── eval_metrics.py                     # Evaluation metrics (MAE, F1, Acc-5, etc.)
├── model.py                            # Core network: MAT single-turn multimodal baseline model
├── main.py                             # Single-turn task training script (AR, PR, AP)
├── dialogue_feature_extractor.py       # Two-turn feature extractor (two-model cascade)
├── dialogue_feature_extractor_multi.py # Two-turn feature extractor (multi-model fusion: AR+PR+AP)
└── dialogue_train_from_cls.py          # Final cascaded training script for dialogue affect prediction

```

## ⚙️ Environment Dependencies

Python 3.8+ and PyTorch 1.12+ are recommended.

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn

```

## 📊 Data Preparation

Please download the dataset from [Dataset Link](https://huggingface.co/datasets/HensonXie/CMDPAD) and place it in the `./dataset/` directory. Required files include:

* Single-turn data: `multimodal_features_single.pkl`, `label_single.csv`
* Paired (Dialogue) data: `multimodal_features_paired.pkl`, `label_paired.csv`

*(Paths can be modified in `src/config.py`).*

## 🏃 Quick Start (Pipeline)

The training process of this project is divided into three stages:

### Stage 1: Train Single-Turn Baseline Models

Train the Affect Recognition (AR), Dynamic Personality Recognition (PR), and Affect Prediction (AP) models separately.

```bash
# Train Affect Recognition Model
python src/main.py --task affect_recognition --modalities bert-base-chinese wav2vec2-large-robust-emotion convnext-base

# Train Personality Recognition Model
python src/main.py --task personality_recognition --modalities bert-base-chinese wav2vec2-large-robust-emotion convnext-base

# Train Affect Prediction Model
python src/main.py --task affect_prediction --modalities bert-base-chinese wav2vec2-large-xlsr-chinese convnext-base

```

### Stage 2: Extract Dialogue-Level Context Features (Cache CLS)

Utilize the trained single-turn models to extract and cache the `[CLS]` fused features of Speakers A and B to accelerate subsequent cascaded training for dialogue tasks.

```bash
# Example: Extract two-turn features using the AR + PR + AP models
python src/dialogue_feature_extractor_multi.py \
    --features ./dataset/multimodal_features_paired.pkl \
    --labels ./dataset/label_paired.csv \
    --model_A_aff ./saved_models/affect_recognition__bert-base-chinese...pt \
    --model_A_per ./saved_models/personality_recognition__bert-base-chinese...pt \
    --model_B ./saved_models/affect_prediction__bert-base-chinese...pt \
    --out_cache ./dataset/dialogue_cls_multi.pkl

```

### Stage 3: Train Dialogue Affect Fusion Predictor

Based on the cached `[CLS]` features of both parties, train the final Cross-Attention fusion network.

```bash
python src/dialogue_train_from_cls.py \
    --cls_cache ./dataset/dialogue_cls_multi.pkl \
    --label_cols valence_prediction \
    --epochs 40 \
    --device cuda

```

## 🤝 Citation

If you find this work useful for your research, please cite our paper (**Accepted by Pattern Recognition**):

```bibtex
@article{zhou2026cmdpad,
  title={CMDPAD: A Chinese multimodal dynamic personality and affect dataset for affect prediction in conversations},
  author={Zhou, Zisen and Xie, Heng and Wen, Chang and Liu, Xuefei and Tao, Jianhua and Wen, Zhengqi and Li, Changsheng and Lian, Zheng and Zhao, Jinming and Xiong, Bingsen and Qin, Shaozheng},
  journal={Pattern Recognition},
  pages={113822},
  year={2026},
  publisher={Elsevier}
}

```
