"""
Extract contextual dialogue features (AR+AP or AR+PR+AP).
Extract CLS features for speaker A (from multiple pretrained single-turn models)
and speaker B (from one pretrained single-turn model), producing a cache pickle.

Example usage (A uses both affect and personality models; B uses a single model):
  python dialogue_feature_extractor_multi.py \
    --features ./dataset/multimodal_features.pkl \
    --labels ./dataset/label_random.csv \
    --modalities_A_aff bert-base-chinese wav2vec2-large-robust-emotion convnext-base \
    --model_A_aff ./saved_models/affect_recognition__bert-base-chinese.pt \
    --modalities_A_per bert-base-chinese wav2vec2-large-robust-emotion convnext-base \
    --model_A_per ./saved_models/personality_recognition__bert-base-chinese.pt \
    --modalities_B bert-base-chinese wav2vec2-large-xlsr-chinese convnext-base \
    --model_B ./saved_models/affect_prediction__bert-base-chinese.pt \
    --out_cache ./dataset/cls_cache_multi.pkl
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from config import DIA_FEATURE_DEFAULTS_MULTI
from utils import build_model_and_load, infer_dims_from_feature

def extract_cls_for_row_multi(models, modal_lists, row, device):
    """
    Extract and concatenate CLS vectors from multiple models for speaker A.
    
    Args:
        models: List of loaded single-turn models
        modal_lists: Corresponding list of modality lists (base names)
        row: One paired row with A_ and B_ columns
        
    Returns:
        Concatenated CLS vector (np.array)
    """
    cls_parts = []
    for model, modlist in zip(models, modal_lists):
        x_t = {}
        for m in modlist:
            key = f"A_{m}"
            val = row.get(key, np.zeros(1, dtype=np.float32))
            arr = np.array(val).reshape(1, -1).astype(np.float32)
            x_t[m] = torch.from_numpy(arr).to(device)
            
        with torch.no_grad():
            cls = model.encode(x_t)  # (1, D)
        cls_parts.append(cls.cpu().numpy().reshape(-1))
        
    if len(cls_parts) == 0:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(cls_parts).astype(np.float32)

def extract_cls_for_single_model(model, modal_list, prefix, row, device):
    """Extract CLS vector for a single model (used for speaker B)."""
    x_t = {}
    for m in modal_list:
        key = f"{prefix}_{m}"
        val = row.get(key, np.zeros(1, dtype=np.float32))
        arr = np.array(val).reshape(1, -1).astype(np.float32)
        x_t[m] = torch.from_numpy(arr).to(device)
        
    with torch.no_grad():
        cls = model.encode(x_t)
    return cls.cpu().numpy().reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, default=DIA_FEATURE_DEFAULTS_MULTI['features'], help="Path to feature pickle")
    p.add_argument("--labels", type=str, default=DIA_FEATURE_DEFAULTS_MULTI['labels'], help="Path to label CSV")
    
    # A: affect model config
    p.add_argument("--modalities_A_aff", nargs='+', default=DIA_FEATURE_DEFAULTS_MULTI['modalities_A_aff'], help="Modalities for A's affect model")
    p.add_argument("--model_A_aff", type=str, default=DIA_FEATURE_DEFAULTS_MULTI['model_A_aff'], help="Path to A affect pretrained model (.pt)")
    
    # A: personality model config
    p.add_argument("--modalities_A_per", nargs='+', default=DIA_FEATURE_DEFAULTS_MULTI['modalities_A_per'], help="Modalities for A's personality model")
    p.add_argument("--model_A_per", type=str, default=DIA_FEATURE_DEFAULTS_MULTI['model_A_per'], help="Path to A personality pretrained model (.pt)")
    
    # B: single model
    p.add_argument("--modalities_B", nargs='+', default=DIA_FEATURE_DEFAULTS_MULTI['modalities_B'], help="Modalities for B's model")
    p.add_argument("--model_B", type=str, default=DIA_FEATURE_DEFAULTS_MULTI['model_B'], help="Path to B pretrained model (.pt)")
    
    p.add_argument("--out_cache", type=str, default=DIA_FEATURE_DEFAULTS_MULTI['out_cache'], help="Output pickle path for cached CLS features")
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

    # Build paired rows (A may use multiple modal lists)
    modalities_A_list = []
    modelA_paths = []
    
    if args.modalities_A_aff and args.model_A_aff:
        modalities_A_list.append(args.modalities_A_aff)
        modelA_paths.append(args.model_A_aff)
    if args.modalities_A_per and args.model_A_per:
        modalities_A_list.append(args.modalities_A_per)
        modelA_paths.append(args.model_A_per)
        
    if len(modalities_A_list) == 0:
        raise ValueError("At least one A model/modalities pair must be provided (aff or per)")

    # Merge labels and features
    if 'mode' in feat_df.columns and 'mode' in lab_df.columns:
        paired_df = pd.merge(lab_df, feat_df.drop(columns=['mode']), on='name', how='inner')
    else:
        paired_df = pd.merge(lab_df, feat_df, on='name', how='inner')
        
    if paired_df.empty:
        raise RuntimeError("No overlap between features and labels")
    print(f"Loaded {len(paired_df)} dialogue rows")

    # Infer dimensions using A_ and B_ prefixes
    all_A_mods = set()
    for ml in modalities_A_list:
        all_A_mods.update(ml)
        
    dims_A = infer_dims_from_feature(paired_df, list(all_A_mods), prefix="A_")
    dims_B = infer_dims_from_feature(paired_df, args.modalities_B, prefix="B_")
    
    # Combine inferred dimensions
    dims_all = {**dims_A, **dims_B}

    # Build modality_dims per model
    modality_dims_list = []
    for ml in modalities_A_list:
        modality_dims_list.append({m: dims_all[m] for m in ml})
    modality_dims_B = {m: dims_all[m] for m in args.modalities_B}
    
    print("Inferred dims for A-models:", modality_dims_list)
    print("Inferred dims for B-model:", modality_dims_B)

    # Load models for A (multiple) and B (single)
    models_A = []
    for path, mdims in zip(modelA_paths, modality_dims_list):
        print(f"Loading A-model from {path} with dims {mdims} ...")
        m = build_model_and_load(path, mdims, args)
        m.to(device)
        m.eval()
        models_A.append(m)
        
    print(f"Loading B-model from {args.model_B} with dims {modality_dims_B} ...")
    modelB = build_model_and_load(args.model_B, modality_dims_B, args)
    modelB.to(device)
    modelB.eval()

    # Iterate and extract CLS vectors
    cache_rows = []
    for idx, row in paired_df.iterrows():
        # Build combined cls_A from multiple A models
        clsA = extract_cls_for_row_multi(models_A, [ml for ml in modalities_A_list], row, device)
        clsB = extract_cls_for_single_model(modelB, args.modalities_B, prefix="B", row=row, device=device)
        
        out = {'name': row['name'], 'mode': row.get('mode')}
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