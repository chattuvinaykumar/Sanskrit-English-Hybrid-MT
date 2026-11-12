"""
Hybrid Transformer Model with Paninian Feature Concatenation.
Includes:
- Subword Token Embedding + Positional Encoding
- Categorical Morphological and Karaka Embeddings
- Joint Feature Projection
- Causal Masking on Decoder for Autoregressive Generation
"""

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LinguisticFeatureEmbedder(nn.Module):
    def __init__(self, num_morph_tags: int = 60, num_karaka_roles: int = 16, morph_dim: int = 64, karaka_dim: int = 64):
        super().__init__()
        self.morph_embed = nn.Embedding(num_morph_tags, morph_dim, padding_idx=0)
        self.karaka_embed = nn.Embedding(num_karaka_roles, karaka_dim, padding_idx=0)
        self.total_feature_dim = morph_dim + karaka_dim

    def forward(self, morph_ids: torch.Tensor, karaka_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.morph_embed(morph_ids), self.karaka_embed(karaka_ids)], dim=-1)

class HybridSanskritTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 8000,
        tgt_vocab_size: int = 8000,
        num_morph_tags: int = 60,
        num_karaka_roles: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.src_tok_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_tok_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        self.ling_embedder = LinguisticFeatureEmbedder(num_morph_tags, num_karaka_roles, morph_dim=64, karaka_dim=64)
        self.encoder_proj = nn.Linear(d_model + self.ling_embedder.total_feature_dim, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def encode(self, src_tokens, morph_ids, karaka_ids, src_mask=None, src_key_padding_mask=None):
        tok_repr = self.src_tok_embed(src_tokens) * math.sqrt(self.d_model)
        ling_repr = self.ling_embedder(morph_ids, karaka_ids)
        projected = self.pos_encoder(self.encoder_proj(torch.cat([tok_repr, ling_repr], dim=-1)))
        return self.transformer.encoder(projected, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt_tokens, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_repr = self.pos_encoder(self.tgt_tok_embed(tgt_tokens) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            tgt_repr, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

    def forward(self, src_tokens, morph_ids, karaka_ids, tgt_tokens, src_pad_mask=None, tgt_pad_mask=None):
        device = src_tokens.device
        tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1), device)
        memory = self.encode(src_tokens, morph_ids, karaka_ids, src_key_padding_mask=src_pad_mask)
        out = self.decode(tgt_tokens, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)
        return self.generator(out)
