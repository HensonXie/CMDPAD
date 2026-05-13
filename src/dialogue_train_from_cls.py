"""
Train dialogue fusion predictor using cached cls features for speaker A and B.

Input: a pickle with rows having:
  'name', 'mode', label columns (e.g., 'valence_prediction'), 'cls_A' (np.array), 'cls_B' (np.array)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os

from utils import set_seed
from eval_metrics import eval_aff_pre, eval_aff_rec, eval_per_rec

class CLSDialogueDataset(Dataset):
    def __init__(self, df, label_cols):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        
        # Store features for A and B separately instead of concatenating early
        self.features_A = []
        self.features_B = []
        for _, row in self.df.iterrows():
            a = np.array(row['cls_A'], dtype=np.float32).reshape(-1)
            b = np.array(row['cls_B'], dtype=np.float32).reshape(-1)
            self.features_A.append(a)
            self.features_B.append(b)
            
        if label_cols:
            self.labels = [row[label_cols].values.astype(np.float32).reshape(-1) if hasattr(row[label_cols], '__len__') else np.array([row[label_cols]], dtype=np.float32) for _, row in self.df.iterrows()]
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_a = torch.from_numpy(self.features_A[idx]).float()
        x_b = torch.from_numpy(self.features_B[idx]).float()
        
        if self.labels is None:
            return (x_a, x_b)
        y = torch.from_numpy(self.labels[idx]).float()
        return (x_a, x_b), y

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_A, dim_B, d_model=256, num_heads=4, hidden=(256,128), output_dim=1, dropout=0.2):
        super().__init__()
        
        # 1. Dimensionality alignment layer: map cls_A and cls_B to the same d_model dimension
        self.proj_A = nn.Linear(dim_A, d_model)
        self.proj_B = nn.Linear(dim_B, d_model)
        
        # 2. Cross-attention layer (bidirectional: A queries B, B queries A)
        self.attn_A2B = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_B2A = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Layer normalization for training stability
        self.norm_A = nn.LayerNorm(d_model)
        self.norm_B = nn.LayerNorm(d_model)
        
        # 3. Fused MLP prediction layer
        layers = []
        last = d_model * 2  # Concatenate A2B and B2A outputs
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_A, x_B):
        # Map to common dimension and add seq_len dimension -> [batch_size, 1, d_model]
        h_A = self.proj_A(x_A).unsqueeze(1)
        h_B = self.proj_B(x_B).unsqueeze(1)
        
        # Cross-Attention
        out_A, _ = self.attn_A2B(query=h_A, key=h_B, value=h_B)
        out_B, _ = self.attn_B2A(query=h_B, key=h_A, value=h_A)
        
        # Residual connection and normalization, remove seq_len dimension -> [batch_size, d_model]
        h_A_fused = self.norm_A(h_A + out_A).squeeze(1)
        h_B_fused = self.norm_B(h_B + out_B).squeeze(1)
        
        fused = torch.cat([h_A_fused, h_B_fused], dim=-1)
        return self.mlp(fused)

def load_cache(cache_pkl):
    df = pd.read_pickle(cache_pkl)
    return df

def make_loaders_from_cache(cache_df, label_cols, batch_size=64, num_workers=4):
    loaders = {}
    splits = {}
    for split in ['train', 'valid', 'test']:
        split_df = cache_df[cache_df['mode'] == split].reset_index(drop=True)
        splits[split] = split_df
        if len(split_df) == 0:
            loaders[split] = None
            continue
        ds = CLSDialogueDataset(split_df, label_cols)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers, pin_memory=True)
    return loaders, splits

def train_loop(model, optim, loss_fn, train_loader, valid_loader, test_loader, device, epochs, save_path):
    best_val = float('inf'); best_state=None; no_improve=0
    for epoch in range(1, epochs+1):
        model.train()
        losses=[]
        t0=time.time()
        if train_loader is not None:
            for (x_A, x_B), y in train_loader:
                x_A, x_B, y = x_A.to(device), x_B.to(device), y.to(device)
                optim.zero_grad()
                out = model(x_A, x_B)
                loss = loss_fn(out, y)
                loss.backward()
                optim.step()
                losses.append(loss.item())
        train_loss = float(np.mean(losses)) if losses else 0.0

        val_metrics=None
        if valid_loader is not None:
            preds, truths = [], []
            model.eval()
            with torch.no_grad():
                for (x_A, x_B), y in valid_loader:
                    x_A, x_B, y = x_A.to(device), x_B.to(device), y.to(device)
                    o = model(x_A, x_B)
                    preds.append(o.detach().cpu())
                    truths.append(y.detach().cpu())
            preds = torch.cat(preds, dim=0); truths = torch.cat(truths, dim=0)
            val_metrics = eval_aff_pre(preds.view(-1), truths.view(-1))
            print(f"Epoch {epoch} VALID metrics:")
            for k,v in val_metrics.items():
                print(f"  {k}: {v}")

        val_mae = val_metrics.get('mae', None) if isinstance(val_metrics, dict) else None
        monitored = val_mae if val_mae is not None else train_loss
        
        if monitored is not None and monitored < best_val:
            best_val = monitored
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            torch.save(best_state, save_path)
            no_improve=0
            improved=True
        else:
            no_improve+=1
            improved=False

        t1=time.time()
        print(f"Epoch {epoch} summary | train_loss {train_loss:.4f} | valid_mae {val_mae} | best {best_val:.4f} | time {(t1-t0):.1f}s | improved {improved}")
        if no_improve >= 8:
            print("Early stopping")
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)

    if test_loader is not None:
        preds, truths = [], []
        model.eval()
        with torch.no_grad():
            for (x_A, x_B), y in test_loader:
                x_A, x_B, y = x_A.to(device), x_B.to(device), y.to(device)
                o = model(x_A, x_B)
                preds.append(o.detach().cpu()); truths.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0); truths = torch.cat(truths, dim=0)
        test_metrics = eval_aff_pre(preds.view(-1), truths.view(-1))
        print(f"Best model TEST metrics:")
        for k,v in test_metrics.items():
            print(f"  {k}: {v}")

    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls_cache", type=str, default="./dataset/dialogue_cls_multi.pkl")
    p.add_argument("--label_cols", nargs='+', default=['valence_prediction'])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--d_model", type=int, default=256, help="Common dimension for Cross-Attention")
    p.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--hidden", nargs='+', type=int, default=[256,128])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--save_dir", type=str, default="./saved_models")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    cache_df = load_cache(args.cls_cache)
    print("Loaded cache:", args.cls_cache, "rows:", len(cache_df))
    loaders, splits = make_loaders_from_cache(cache_df, args.label_cols, batch_size=args.batch_size, num_workers=4)

    sample_loader = loaders['train'] or loaders['valid'] or loaders['test']
    if sample_loader is None:
        raise RuntimeError("No data in any split")
    
    sample_batch = next(iter(sample_loader))
    (x_A_sample, x_B_sample) = sample_batch[0]
    dim_A = x_A_sample.shape[1]
    dim_B = x_B_sample.shape[1]
    print(f"Input dims -> dim_A: {dim_A}, dim_B: {dim_B}")

    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    
    model = CrossAttentionFusion(
        dim_A=dim_A, 
        dim_B=dim_B, 
        d_model=args.d_model,
        num_heads=args.num_heads,
        hidden=tuple(args.hidden), 
        output_dim=len(args.label_cols), 
        dropout=args.dropout
    )
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = Path(args.save_dir) / ("dialogue_fusion_attn_" + Path(args.cls_cache).stem + ".pt")
    
    trained = train_loop(model, optim, loss_fn, loaders['train'], loaders['valid'], loaders['test'], device, args.epochs, save_path)
    print("Trained model saved to:", save_path)

if __name__ == "__main__":
    main()