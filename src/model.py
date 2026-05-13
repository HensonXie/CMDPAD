import torch
import torch.nn as nn
from typing import Dict


class ModalityProjector(nn.Module):
    def __init__(self, input_dim, proj_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class CrossModalTransformer(nn.Module):
    def __init__(self, proj_dim, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=n_heads,
                                                   dim_feedforward=proj_dim*4, dropout=dropout,
                                                   activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, seq_tokens):
        return self.encoder(seq_tokens)


class FusionHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), output_dim=1, dropout=0.2):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultimodalAttentionModel(nn.Module):
    """
    modality_dims: dict mapping modality_name -> feature_dim
    The order of keys defines token order.
    """
    def __init__(self,
                 modality_dims: Dict[str, int],
                 proj_dim: int = 512,
                 transformer_layers: int = 2,
                 transformer_heads: int = 8,
                 dropout: float = 0.2,
                 head_hidden=(512, 256),
                 output_dim: int = 1):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.proj_dim = proj_dim
        self.projectors = nn.ModuleDict({
            m: ModalityProjector(modality_dims[m], proj_dim, dropout=dropout)
            for m in self.modalities
        })
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        # modality embeddings (per modality)
        self.modality_embeddings = nn.Parameter(torch.randn(1, len(self.modalities), proj_dim) * 0.02)
        # transformer
        self.transformer = CrossModalTransformer(proj_dim, n_layers=transformer_layers, n_heads=transformer_heads, dropout=dropout)
        # head
        self.head = FusionHead(proj_dim, hidden_dims=head_hidden, output_dim=output_dim, dropout=dropout)

    def forward(self, x_dict: Dict[str, torch.Tensor]):
        """
        Full forward: returns final head output (regression).
        """
        cls_out = self.encode(x_dict)  # (B, proj_dim)
        return self.head(cls_out)

    def encode(self, x_dict: Dict[str, torch.Tensor]):
        """
        Encode inputs and return CLS embedding (before head).
        Useful for extracting features from pretrained single-turn models.
        """
        batch_size = None
        tokens = []
        for i, m in enumerate(self.modalities):
            x = x_dict[m]  # (B, feat_dim)
            if batch_size is None:
                batch_size = x.shape[0]
            proj = self.projectors[m](x)  # (B, proj_dim)
            tokens.append(proj.unsqueeze(1))
        tokens = torch.cat(tokens, dim=1)  # (B, M, D)
        tokens = tokens + self.modality_embeddings  # (1, M, D) broadcast
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+M, D)
        out = self.transformer(seq)  # (B, 1+M, D)
        cls_out = out[:, 0, :]  # (B, D)
        return cls_out