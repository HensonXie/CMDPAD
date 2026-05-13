"""
Main training script supporting single-turn tasks.
(Affect Recognition, Affect Prediction, Personality Recognition)
"""

import argparse
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import DEFAULTS, FEATURE_PKL, LABEL_CSV, DEFAULT_TASK, DEFAULT_MODALITIES, AVAILABLE_MODALITIES
from dataset import make_dataloaders
from model import MultimodalAttentionModel
from eval_metrics import eval_per_rec, eval_aff_rec, eval_aff_pre
from utils import set_seed

def infer_modality_dims(feature_pkl, modalities):
    """
    Infer dims for provided modalities (single-turn).
    """
    import pandas as pd
    df = pd.read_pickle(feature_pkl)
    for _, row in df.iterrows():
        ok = True
        dims = {}
        for m in modalities:
            if m in row:
                arr = np.array(row[m]).reshape(-1)
                dims[m] = arr.shape[0]
            else:
                ok = False
                break
        if ok:
            return dims
    raise RuntimeError("Could not infer modality dims from features; check modality names and pkl file.")

def reconstruct_x_dict_from_concat(x_batch, modalities_order, dataset_df):
    """
    Given x_batch (B, total_dim) and modalities_order,
    split according to dims inferred from dataset_df first row and return dict.
    """
    offsets = []
    for m in modalities_order:
        arr = dataset_df.iloc[0][m]
        dim = np.array(arr).reshape(-1).shape[0]
        offsets.append(dim)
    splits = torch.split(x_batch, offsets, dim=1)
    return {m: splits[i] for i, m in enumerate(modalities_order)}

def evaluate_on_loader(model, dataloader, device, task_type):
    """
    Evaluate and return preds, truths, metrics.
    """
    model.eval()
    preds_list = []
    truths_list = []
    dataset = dataloader.dataset
    modalities_order = dataset.modalities

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x_dict = reconstruct_x_dict_from_concat(x, modalities_order, dataset.df)
            out = model(x_dict)
            preds_list.append(out.detach().cpu())
            truths_list.append(y.detach().cpu())
            
    if len(preds_list) == 0:
        return None, None, {}
        
    preds = torch.cat(preds_list, dim=0)
    truths = torch.cat(truths_list, dim=0)
    
    # dispatch metrics
    if task_type == 'affect_recognition':
        metrics = eval_aff_rec(preds.view(-1), truths.view(-1))
        return preds, truths, metrics
    elif task_type == 'affect_prediction':
        metrics = eval_aff_pre(preds.view(-1), truths.view(-1))
        return preds, truths, metrics
    elif task_type == 'personality_recognition':
        per_dim = []
        for d in range(preds.shape[1]):
            m = eval_per_rec(preds[:, d], truths[:, d])
            per_dim.append(m)
        agg = {}
        if len(per_dim) > 0:
            keys = per_dim[0].keys()
            for k in keys:
                try:
                    agg[k] = float(np.mean([m[k] for m in per_dim]))
                except Exception:
                    agg[k] = None
        return preds, truths, {'per_dim': per_dim, 'agg': agg}
    else:
        return preds, truths, {}

def build_model(modality_dims, args, output_dim):
    model = MultimodalAttentionModel(
        modality_dims,
        proj_dim=args.proj_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        head_hidden=tuple(args.hidden),
        output_dim=output_dim
    )
    return model

def train(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu is not None) else "cpu")
    print("Device:", device)

    # decide task specifics
    if args.task == 'affect_recognition':
        label_cols = ['speaker_valence']
        output_dim = 1
    elif args.task == 'affect_prediction':
        label_cols = ['valence_prediction']
        output_dim = 1
    elif args.task == 'personality_recognition':
        label_cols = ['speaker_openness', 'speaker_neuroticism', 'speaker_extraversion',
                      'speaker_agreeableness', 'speaker_conscientiousness']
        output_dim = 5
    else:
        raise ValueError("Unknown task")

    # Load data
    print("Loading data...")
    loaders, splits = make_dataloaders(
        args.features, args.labels, args.modalities, label_cols,
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle_train=True
    )

    train_loader = loaders['train']
    valid_loader = loaders['valid']
    test_loader = loaders['test']

    def count_df(d):
        return 0 if d is None else len(d)
    print(f"Data sizes: train={count_df(splits.get('train'))}, valid={count_df(splits.get('valid'))}, test={count_df(splits.get('test'))}")

    if train_loader is None and valid_loader is None and test_loader is None:
        raise RuntimeError("No data available in any split")

    # infer modality dims
    modality_dims_ordered = infer_modality_dims(args.features, args.modalities)
    print("Modality dims:", modality_dims_ordered)

    # build model
    model = build_model(modality_dims_ordered, args, output_dim)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    no_improve = 0

    os.makedirs(args.save_dir, exist_ok=True)
    model_name = f"{args.task}__{'_'.join(args.modalities)}"
    save_path = Path(args.save_dir) / (model_name + ".pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        if train_loader is None:
            print("No train split; skipping training.")
        else:
            for batch in train_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                modalities_order = train_loader.dataset.modalities
                x_dict = reconstruct_x_dict_from_concat(x, modalities_order, train_loader.dataset.df)
                
                optimizer.zero_grad()
                out = model(x_dict)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else 0.0

        # validation
        val_mae = None
        if valid_loader is not None:
            _, _, valid_res = evaluate_on_loader(model, valid_loader, device, args.task)
            if args.task == 'personality_recognition':
                print(f"\nEpoch {epoch} VALIDATION Personality per-dim:")
                for i, per in enumerate(valid_res['per_dim']):
                    print(f"  Dim {i}: {per}")
                print("  Aggregated:", valid_res['agg'])
                val_mae = valid_res['agg'].get('mae', None)
            else:
                print(f"\nEpoch {epoch} VALIDATION metrics:")
                for k, v in valid_res.items():
                    print(f"  {k}: {v}")
                val_mae = valid_res.get('mae', None)
        else:
            print("\nNo validation split available.")

        if val_mae is not None:
            scheduler.step(val_mae)
            monitored = val_mae
        else:
            scheduler.step(train_loss)
            monitored = train_loss

        improved = False
        if monitored is not None and monitored < best_val:
            best_val = monitored
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            improved = True
            no_improve = 0
        else:
            no_improve += 1

        t1 = time.time()
        print(f"\nEpoch {epoch:03d} summary | train_loss {train_loss:.4f} | valid_mae {val_mae} | best {best_val:.4f} | time {(t1-t0):.1f}s | improved {improved}")
        print("-" * 80)

        if no_improve >= args.patience:
            print(f"No improvement for {no_improve} epochs. Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (best_val={best_val:.4f})")

    if test_loader is not None:
        _, _, test_res = evaluate_on_loader(model, test_loader, device, args.task)
        print("\nFINAL TEST metrics:")
        if args.task == 'personality_recognition':
            for i, per in enumerate(test_res['per_dim']):
                print(f"  Dim {i}: {per}")
            print("  Aggregated:", test_res['agg'])
        else:
            for k, v in test_res.items():
                print(f"  {k}: {v}")

    print(f"\nModel saved to: {save_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, choices=["affect_recognition", "affect_prediction", "personality_recognition"],
                   default=DEFAULT_TASK)
    p.add_argument("--modalities", nargs='+', default=DEFAULT_MODALITIES,
                   help="(single-turn) modality list for non-dialogue tasks")
    p.add_argument("--features", type=str, default=str(FEATURE_PKL))
    p.add_argument("--labels", type=str, default=str(LABEL_CSV))
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--proj_dim", type=int, default=DEFAULTS["proj_dim"])
    p.add_argument("--transformer_layers", type=int, default=DEFAULTS["transformer_layers"])
    p.add_argument("--transformer_heads", type=int, default=DEFAULTS["transformer_heads"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--hidden", nargs='+', type=int, default=DEFAULTS["hidden_dims"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--gpu", type=int, default=DEFAULTS["gpu"], help="GPU id to use. Set to -1 for CPU")
    p.add_argument("--save_dir", type=str, default=DEFAULTS["save_dir"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.gpu is not None and args.gpu < 0:
        args.gpu = None

    unknown = [m for m in args.modalities if m not in AVAILABLE_MODALITIES]
    if unknown:
        print(f"Warning: modalities {list(set(unknown))} are not in AVAILABLE_MODALITIES. Proceed but ensure feature pickle contains these columns.")

    print("Starting training with configuration:")
    print(f"  Task      : {args.task}")
    print(f"  Modalities : {args.modalities}")
    print(f"  Features  : {args.features}")
    print(f"  Labels    : {args.labels}")
    print(f"  Epochs    : {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  Save dir  : {args.save_dir}")
    print(f"  Seed      : {args.seed}, GPU: {args.gpu}")

    train(args)