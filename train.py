"""
Training Pipeline for Hybrid Sanskrit-English Machine Translation.
Executes sequence-to-sequence loss with label smoothing (0.1) and gradient clipping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from hybrid_transformer_model import HybridSanskritTransformer

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        src = batch['src'].to(device)
        morph = batch['morph'].to(device)
        karaka = batch['karaka'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)

        optimizer.zero_grad()
        output = model(src, morph, karaka, tgt_input)
        
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)

def run_training_demo():
    print("=" * 60)
    print("HYBRID SANSKRIT-ENGLISH MODEL INITIALIZATION")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridSanskritTransformer(
        src_vocab_size=8000,
        tgt_vocab_size=8000,
        num_morph_tags=60,
        num_karaka_roles=16,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)

    print(f"Model instantiated with {sum(p.numel() for p in model.parameters()):,} parameters.")
    print(f"Running on hardware: {device}")
    print("Ready to train on 1,600 sentence pairs with 0.1 label smoothing.")
    print("=" * 60)

if __name__ == '__main__':
    run_training_demo()
