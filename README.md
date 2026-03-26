<div align="center">

# CMDPAD

[**中文**](./README_zh.md) | [**English**](./README.md)

</div>

---

### Introduction

This project focuses on **Multimodal Conversational Analysis**, specifically targeting **Affect Recognition**, **Personality Recognition**, and **Affect Prediction** in multi-turn dialogues. It introduces a novel multimodal dataset and establishes baseline model.

### 🛠️ Environment Requirements

The code is tested with Python 3.8+ and PyTorch 2.0.0. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt

```

**Core Dependencies (`requirements.txt`):**

```text
torch==2.0.0
torchaudio==2.0.1
torchvision==0.15.1
transformers==4.46.3
numpy==1.24.4
pandas==2.0.3
scikit-learn==1.3.2
opencv-python==4.12.0.88
librosa==0.11.0
moviepy==1.0.3
openai-whisper==20250625
funasr==1.2.7
modelscope==1.11.0
hydra-core==1.3.2
einops==0.8.1
tqdm==4.67.1

```

### 📂 Project Structure

```text
.
├── data/                        # Dataset Root
│   ├── raw/                     # Original video/audio data
│   ├── label_random.csv         # Labels
│   ├── multimodal_features.pkl  # Extracted features
│   └── features/
│       └── features_summary.csv # Feature format summary
├── preprocessing/               # Feature Extraction & Data Processing
│   ├── feature_extract/         # Feature extraction scripts
│   ├── detection_multimodal.py  # Transcription & Data splitting
│   └── check.py                  # Data integrity check
└── model/                       # Model Training & Inference
    ├── main.py                  # Single-turn dialogue tasks
    └── ...                      # Multi-turn strategies scripts

```

### 🚀 Usage

#### 1. Feature Processing

Configure data paths, video framing settings, and pre-trained model selection in the config, then run feature extraction:

```bash
python ./preprocessing/feature_extract/main.py

```

#### 2. Data Screening & Transcription

Perform data splitting, audio transcription (ASR), and text generation:

```bash
python ./preprocessing/detection_multimodal.py

```

Check if the data quantity matches across three modalities (Audio/Visual/Text):

```bash
python ./preprocessing/test.py

```

#### 3. Model Training

**A. Single-turn Dialogue Tasks**
Run the baseline model for single-turn tasks directly:

```bash
python ./model/main.py

```

**B. Multi-turn Dialogue (Affect Prediction)**
Extract features from the previous turn (Context):

```bash
# Standard feature selection
python ./model/dialogue_feature_extractor.py

```


Train using direct concatenation fusion:
```bash
python ./model/dialogue_train_from_cls.py

```

