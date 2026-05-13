"""
Extract CLS features for speaker A and B using pretrained single-turn models (AR+AP).

Workflow:
- Load feature pickle and label csv, construct paired dialogue rows (prev -> cur)
- For each pair, build x_dict for speaker A (modalities_A) and for speaker B (modalities_B)
- Load pretrained single-turn models (A_model_path and B_model_path) with their modality lists,
  run .encode(x_dict) to obtain cls_out_A and cls_out_B
- Save a cached pickle file with columns:
    'name' (target current name), 'cls_A' (np.array), 'cls_B' (np.array), label columns and 'mode'
  This cached file can be used to train a lightweight dialogue fusion model.

Usage example:
  python dialogue_feature_extractor.py \
    --features /path/to/multimodal_features.pkl \
    --labels /path/to/label.csv \
    --modalities_A bert-base-chinese wav2vec2-large-robust-emotion convnext-base \
    --modalities_B bert-base-chinese wav2vec2-large-xlsr-chinese convnext-base \
    --model_A /saved_models/affect_recognition__bert-base-chinese.pt \
    --model_B /saved_models/personality_recognition__bert-base-chinese.pt \
    --out_cache /saved_models/cls_cache_A_modelB_model.pkl
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from config import DIA_FEATURE_DEFAULTS
from utils import build_model_and_load, infer_dims_from_feature

def extract_cls_for_row(model, x_dict, device):
    """Convert x_dict arrays to tensors and extract CLS token embedding."""
    x_t = {}
    for k, v in x_dict.items():
        arr = np.array(v).reshape(1, -1).astype(np.float32)
        x_t[k] = torch.from_numpy(arr).to(device)
    with torch.no_grad():
        cls = model.encode(x_t)  # (1, D)
    return cls.cpu().numpy().reshape(-1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, default=DIA_FEATURE_DEFAULTS['features'], help="Path to feature pickle")
    p.add_argument("--labels", type=str, default=DIA_FEATURE_DEFAULTS['labels'], help="Path to label CSV")
    p.add_argument("--modalities_A", nargs='+', default=DIA_FEATURE_DEFAULTS['modalities_A'])
    p.add_argument("--modalities_B", nargs='+', default=DIA_FEATURE_DEFAULTS['modalities_B'])
    p.add_argument("--model_A", type=str, default=DIA_FEATURE_DEFAULTS['model_A'], help="Path to pretrained single-turn model for A (.pt)")
    p.add_argument("--model_B", type=str, default=DIA_FEATURE_DEFAULTS['model_B'], help="Path to pretrained single-turn model for B (.pt)")
    p.add_argument("--out_cache", type=str, default=DIA_FEATURE_DEFAULTS['out_cache'], help="Output pickle path for cached CLS features")
    p.add_argument("--batch_size", type=int, default=64)
    
    # Model instantiation defaults (should match how single-turn models were constructed)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--transformer_layers", type=int, default=2)
    p.add_argument("--transformer_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden", nargs='+', type=int, default=[512, 256])
    p.add_argument("--output_dim_if_needed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    features_pkl = Path(args.features)
    labels_csv = Path(args.labels)
    out_cache = Path(args.out_cache)
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    feat_df = pd.read_pickle(features_pkl)
    lab_df = pd.read_csv(labels_csv)

    # Merge labels and features
    if 'mode' in feat_df.columns and 'mode' in lab_df.columns:
        paired_df = pd.merge(lab_df, feat_df.drop(columns=['mode']), on='name', how='inner')
    else:
        paired_df = pd.merge(lab_df, feat_df, on='name', how='inner')
        
    if paired_df.empty:
        raise RuntimeError("No overlap between features and labels")
    print(f"Loaded {len(paired_df)} dialogue rows")

    # Infer dimensions using A_ and B_ prefixes
    dims_A = infer_dims_from_feature(paired_df, args.modalities_A, prefix="A_")
    dims_B = infer_dims_from_feature(paired_df, args.modalities_B, prefix="B_")
    print("Inferred dims A:", dims_A)
    print("Inferred dims B:", dims_B)

    # Build models and load weights
    modelA = build_model_and_load(args.model_A, {m: dims_A[m] for m in args.modalities_A}, args)
    modelB = build_model_and_load(args.model_B, {m: dims_B[m] for m in args.modalities_B}, args)
    modelA.to(device)
    modelB.to(device)
    modelA.eval()
    modelB.eval()

    # Iterate paired rows and extract cls_A and cls_B
    cache_rows = []
    for idx, row in paired_df.iterrows():
        # Build per-speaker x_dict
        xA = {}
        xB = {}
        for m in args.modalities_A:
            key = f"A_{m}"
            xA[m] = row.get(key, np.zeros(1, dtype=np.float32))
        for m in args.modalities_B:
            key = f"B_{m}"
            xB[m] = row.get(key, np.zeros(1, dtype=np.float32))
            
        # Extract CLS tokens
        clsA = extract_cls_for_row(modelA, xA, device)
        clsB = extract_cls_for_row(modelB, xB, device)
        
        # Prepare output row
        out = {'name': row['name'], 'mode': row.get('mode')}
        
        # Retain original label columns
        for c in lab_df.columns:
            if c in row.index:
                out[c] = row[c]
                
        out['cls_A'] = clsA
        out['cls_B'] = clsB
        cache_rows.append(out)
        
        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1}/{len(paired_df)}")

    cache_df = pd.DataFrame(cache_rows)
    
    # Save cache
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_pickle(out_cache)
    print(f"Saved CLS cache to {out_cache} ({len(cache_df)} rows)")

if __name__ == "__main__":
    main()