"""
Configuration for experiments (dialogue-enabled).
Edit paths and defaults as needed.
"""
from pathlib import Path

# Base directories
DATA_DIR = Path("./dataset")
SAVE_DIR = Path("./saved_models")

# Single-turn Data paths
FEATURE_PKL = DATA_DIR / "multimodal_features_single_final.pkl"
LABEL_CSV = DATA_DIR / "label_single_final.csv"

# Dialogue Data paths
DIA_FEATURE_PKL = DATA_DIR / "multimodal_features_paired_final.pkl"
DIA_LABEL_CSV = DATA_DIR / "label_paired_final.csv"

# Available modality columns in feature pickle
AVAILABLE_MODALITIES = [
    "bert-base-chinese",
    "wav2vec2-base",
    "wav2vec2-large-robust-emotion",
    "wav2vec2-large-xlsr-chinese",
    "vit-base",
    "convnext-base",
]

# Default experiment hyperparameters
DEFAULTS = {
    "batch_size": 64,
    "epochs": 40,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "proj_dim": 512,        # projection dim for modality -> token
    "transformer_layers": 2,
    "transformer_heads": 8,
    "dropout": 0.2,
    "hidden_dims": [512, 256],  # head MLP hidden dims
    "seed": 42,
    "gpu": 0,
    "device": "cuda",
    "save_dir": str(SAVE_DIR),
    "num_workers": 4,
    "patience": 8,  # early stopping patience on valid MAE
}


# Defaults for easier direct run
# "affect_recognition", "affect_prediction", "personality_recognition", "dialogue_affect_prediction"
DEFAULT_TASK = "affect_recognition"

# "bert-base-chinese", "wav2vec2-base", "wav2vec2-large-robust-emotion", "wav2vec2-large-xlsr-chinese", "vit-base", "convnext-base"
# This supports both unimodal and multimodal runs.
# DEFAULT_MODALITIES = ["wav2vec2-base"]
DEFAULT_MODALITIES = ["bert-base-chinese", "wav2vec2-large-robust-emotion", "convnext-base"]


# Dialogue-specific default modalities (A and B can differ)
# These are lists of base modality names (must exist in the feature pickle).
DIALOGUE_DEFAULT_MODALITIES_A = ["bert-base-chinese", "wav2vec2-large-robust-emotion", "convnext-base"]
DIALOGUE_DEFAULT_MODALITIES_B = ["bert-base-chinese", "wav2vec2-large-xlsr-chinese", "convnext-base"]

# Affect Recognition + Affect Prediction
DIA_FEATURE_DEFAULTS = {
    "features": str(DIA_FEATURE_PKL),
    "labels": str(DIA_LABEL_CSV),
    "modalities_A": DIALOGUE_DEFAULT_MODALITIES_A,
    "modalities_B": DIALOGUE_DEFAULT_MODALITIES_B,
    "model_A": str(SAVE_DIR / "personality_recognition__bert-base-chinese_wav2vec2-large-robust-emotion_convnext-base.pt"),
    "model_B": str(SAVE_DIR / "sentiment_prediction__bert-base-chinese_wav2vec2-large-robust-emotion_convnext-base.pt"),
    "out_cache": str(DATA_DIR / "dialogue_cls_pr_ap.pkl"),
}

# Affect Recognition + Personality Recognition + Affect Prediction
DIA_FEATURE_DEFAULTS_MULTI = {
    "features": str(DIA_FEATURE_PKL),
    "labels": str(DIA_LABEL_CSV),
    "modalities_A_sent": DIALOGUE_DEFAULT_MODALITIES_A,
    "modalities_A_per": DIALOGUE_DEFAULT_MODALITIES_A,
    "modalities_B": DIALOGUE_DEFAULT_MODALITIES_B,
    "model_A_sent": str(SAVE_DIR / "sentiment_recognition__bert-base-chinese_wav2vec2-large-robust-emotion_convnext-base.pt"),
    "model_A_per": str(SAVE_DIR / "personality_recognition__bert-base-chinese_wav2vec2-large-robust-emotion_convnext-base.pt"),
    "model_B": str(SAVE_DIR / "sentiment_prediction__bert-base-chinese_wav2vec2-large-xlsr-chinese_convnext-base.pt"),
    "out_cache": str(DATA_DIR / "dialogue_cls_multi.pkl"),
}