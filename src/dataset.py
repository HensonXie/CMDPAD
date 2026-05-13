"""
Dataset and dataloader helpers.

Loads multimodal features from a pickle and labels from a CSV. Matches on 'name'.
Splits by 'mode' column with values 'train','valid','test'.
Returns PyTorch DataLoaders.

The dataset concatenates selected modalities in a fixed order and returns:

- x: float tensor of shape (total_feature_dim,)
- y: float tensor of shape (label_dim,) where label_dim=1 for affect, 5 for personality
"""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MultimodalFeatureDataset(Dataset):
    def __init__(self, df, modalities, label_cols):
        """
        df: merged DataFrame (labels + modal columns) for one split
        modalities: list of modality column names (must exist in df)
        label_cols: list of label column names
        """
        self.df = df.reset_index(drop=True)
        self.modalities = modalities
        self.label_cols = label_cols

        # Validate
        for m in modalities:
            if m not in self.df.columns:
                raise ValueError(f"Modality '{m}' not found in DataFrame columns")

        # Precompute features and labels to speed up I/O
        self.features = []
        for _, row in self.df.iterrows():
            parts = []
            for m in modalities:
                arr = row[m]
                if isinstance(arr, (list, tuple)):
                    arr = np.array(arr, dtype=np.float32).reshape(-1)
                elif isinstance(arr, np.ndarray):
                    arr = arr.astype(np.float32).reshape(-1)
                else:
                    # fallback
                    arr = np.array([float(arr)], dtype=np.float32)
                parts.append(arr)
            concat = np.concatenate(parts).astype(np.float32)
            self.features.append(concat)
        if label_cols is None or len(label_cols) == 0:
            self.labels = None
        else:
            labs = []
            for _, row in self.df.iterrows():
                vals = row[label_cols].values.astype(np.float32)
                labs.append(vals.reshape(-1))
            self.labels = labs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()
        if self.labels is None:
            return x
        y = torch.from_numpy(self.labels[idx]).float()
        return x, y


def make_dataloaders(feature_pkl, label_csv, modalities, label_cols,
                     batch_size=64, num_workers=4, shuffle_train=True):
    feature_pkl = Path(feature_pkl)
    label_csv = Path(label_csv)
    if not feature_pkl.exists():
        raise FileNotFoundError(f"Feature pickle not found: {feature_pkl}")
    if not label_csv.exists():
        raise FileNotFoundError(f"Label csv not found: {label_csv}")

    feat_df = pd.read_pickle(feature_pkl)
    lab_df = pd.read_csv(label_csv)

    if 'name' not in feat_df.columns or 'name' not in lab_df.columns:
        raise ValueError("Both feature pickle and label csv must contain 'name' column")

    # Merge on name
    if 'mode' in feat_df.columns and 'mode' in lab_df.columns:
        df = pd.merge(lab_df, feat_df.drop(columns=['mode']), on='name', how='inner')
    else:
        df = pd.merge(lab_df, feat_df, on='name', how='inner')
        
    if df.empty:
        raise ValueError("No matching samples after merging features and labels by 'name'")

    # Splits by mode
    splits = {}
    loaders = {}
    for split in ['train', 'valid', 'test']:
        split_df = df[df['mode'] == split].reset_index(drop=True)
        splits[split] = split_df
        if split_df.empty:
            loaders[split] = None
            continue
        dataset = MultimodalFeatureDataset(split_df, modalities, label_cols)
        loaders[split] = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=(split == 'train' and shuffle_train),
                                    num_workers=num_workers, pin_memory=True)
    return loaders, splits