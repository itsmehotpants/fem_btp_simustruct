#!/usr/bin/env python3
"""
SimuStruct AI V3 — Model Training Pipeline
============================================
Loads generated simulation data, trains the Enhanced MLP (or GNN),
with optional physics-informed loss. Implements cosine annealing LR,
early stopping, and comprehensive logging.

Usage:
    python 2_train_ai.py                         # Train with defaults
    python 2_train_ai.py --epochs 100 --batch_size 2048
    python 2_train_ai.py --model gnn             # Train GNN instead
    python 2_train_ai.py --no_pinn               # Disable physics loss
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.materials import MATERIALS, get_lame_constants
from src.data_utils import (
    build_training_dataset,
    preprocess_stress,
    postprocess_stress,
    SimuStructScaler,
)
from models.enhanced_mlp import EnhancedMLP, build_enhanced_mlp


def create_dataloader(data_dir: str, batch_size: int, log_transform: bool = True, max_samples: int = 200000):
    """Load data and create train/val/test DataLoaders."""
    print("Loading dataset...")
    try:
        X_coords, X_features, Y_stress, Y_disp = build_training_dataset(
            data_dir, log_transform=log_transform
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("Run '1_generate_data.py' first to generate training data.")
        sys.exit(1)

    print(f"  Coords:   {X_coords.shape}")
    print(f"  Features: {X_features.shape}")
    print(f"  Stress:   {Y_stress.shape}")
    print(f"  Disp:     {Y_disp.shape}")

    # Subsample for CPU training speed
    if len(X_coords) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_coords), max_samples, replace=False)
        X_coords = X_coords[idx]
        X_features = X_features[idx]
        Y_stress = Y_stress[idx]
        Y_disp = Y_disp[idx]
        print(f"  Subsampled to {max_samples} nodes for CPU training")

    # Combine targets: [stress_vm, disp_x, disp_y]
    Y = np.column_stack([Y_stress, Y_disp])

    # To tensors
    coords_t = torch.tensor(X_coords, dtype=torch.float32)
    features_t = torch.tensor(X_features, dtype=torch.float32)
    targets_t = torch.tensor(Y, dtype=torch.float32)

    # Split features into material (4), geometry (4), load (5)
    n_feat = features_t.shape[1]
    if n_feat >= 13:
        mat_feats = features_t[:, :4]
        geo_feats = features_t[:, 4:8]
        load_feats = features_t[:, 8:13]
    else:
        # Fallback: pad with zeros
        mat_feats = features_t[:, :min(4, n_feat)]
        geo_feats = features_t[:, min(4, n_feat):min(8, n_feat)]
        load_feats = features_t[:, min(8, n_feat):min(13, n_feat)]
        if mat_feats.shape[1] < 4:
            mat_feats = torch.cat([mat_feats, torch.zeros(len(mat_feats), 4 - mat_feats.shape[1])], dim=1)
        if geo_feats.shape[1] < 4:
            geo_feats = torch.cat([geo_feats, torch.zeros(len(geo_feats), 4 - geo_feats.shape[1])], dim=1)
        if load_feats.shape[1] < 5:
            load_feats = torch.cat([load_feats, torch.zeros(len(load_feats), 5 - load_feats.shape[1])], dim=1)

    # Create combined tensor dataset
    dataset = TensorDataset(coords_t, mat_feats, geo_feats, load_feats, targets_t)

    # Split: 80/10/10
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

    return train_loader, val_loader, test_loader


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute R² score."""
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-8))


def train(
    epochs: int = 300,
    batch_size: int = 4096,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    lambda_pde: float = 0.1,
    use_pinn: bool = True,
    data_dir: str = "data/hdf5",
    model_dir: str = "models",
    model_type: str = "enhanced_mlp",
    patience: int = 30,
    device_str: str = "auto",
):
    """Main training loop."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"\nDevice: {device}")

    # Data
    train_loader, val_loader, test_loader = create_dataloader(data_dir, batch_size)

    # Model
    if model_type == "enhanced_mlp":
        model = build_enhanced_mlp({
            "n_freqs": 8,
            "hidden_dims": [128, 256, 256, 128],
            "use_residual": True,
        })
    elif model_type == "mc_dropout":
        from models.mc_dropout import MCDropoutMLP
        model = MCDropoutMLP(dropout_p=0.1)
    else:
        model = build_enhanced_mlp()

    model = model.to(device)
    print(f"Model: {model_type} - {model.count_parameters():,} parameters")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Lamé constants for physics loss (use steel as reference)
    if use_pinn:
        lam_ref, mu_ref = get_lame_constants("structural_steel_a36")
    else:
        lambda_pde = 0.0

    # Training history
    history = {
        "train_loss": [], "val_loss": [], "lr": [],
        "train_r2_stress": [], "val_r2_stress": [],
        "epoch_time": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training started - {epochs} epochs")
    print(f"  Batch size: {batch_size} | LR: {lr}")
    print(f"  Physics loss: {'ON (lam=' + str(lambda_pde) + ')' if use_pinn else 'OFF'}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for epoch in range(epochs):
        t_epoch = time.time()

        # ===== TRAIN =====
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for batch in train_loader:
            coords, mat, geo, load, targets = [b.to(device) for b in batch]

            optimizer.zero_grad()
            pred = model(coords, mat, geo, load)
            loss = criterion(pred, targets)

            # Physics loss (on a subsample for efficiency)
            if lambda_pde > 0 and epoch >= 10:  # Warm up data loss first
                try:
                    subsample = min(512, len(coords))
                    idx = torch.randperm(len(coords))[:subsample]
                    from models.pinn_loss import navier_residual
                    pde_loss = navier_residual(
                        model, coords[idx], mat[idx], geo[idx],
                        lam_ref, mu_ref, load[idx]
                    )
                    loss = loss + lambda_pde * pde_loss
                except Exception:
                    pass  # Skip physics loss if autograd fails

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(pred.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())

        scheduler.step()

        # ===== VALIDATE =====
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                coords, mat, geo, load, targets = [b.to(device) for b in batch]
                pred = model(coords, mat, geo, load)
                loss = criterion(pred, targets)
                val_losses.append(loss.item())
                val_preds.append(pred.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        # Compute metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0]

        train_pred_all = np.concatenate(train_preds)
        train_target_all = np.concatenate(train_targets)
        val_pred_all = np.concatenate(val_preds)
        val_target_all = np.concatenate(val_targets)

        train_r2 = compute_r2(train_pred_all[:, 0], train_target_all[:, 0])
        val_r2 = compute_r2(val_pred_all[:, 0], val_target_all[:, 0])

        epoch_time = time.time() - t_epoch

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["lr"].append(float(current_lr))
        history["train_r2_stress"].append(float(train_r2))
        history["val_r2_stress"].append(float(val_r2))
        history["epoch_time"].append(float(epoch_time))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_r2": val_r2,
                "config": {
                    "model_type": model_type,
                    "n_freqs": 8,
                    "hidden_dims": [128, 256, 256, 128],
                },
            }, os.path.join(model_dir, "best_checkpoint.pth"))
        else:
            patience_counter += 1

        # Progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            status = "*" if patience_counter == 0 else " "
            print(
                f"  [{status}] Epoch {epoch+1:3d}/{epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"R2: {val_r2:.4f} | LR: {current_lr:.2e} | "
                f"{epoch_time:.1f}s"
            )

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    total_time = time.time() - t_start

    # ===== FINAL EVALUATION =====
    print(f"\n{'='*60}")
    print(f"  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")

    # Load best model
    checkpoint = torch.load(os.path.join(model_dir, "best_checkpoint.pth"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Test set evaluation
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            coords, mat, geo, load, targets = [b.to(device) for b in batch]
            pred = model(coords, mat, geo, load)
            test_preds.append(pred.cpu().numpy())
            test_targets.append(targets.cpu().numpy())

    test_pred_all = np.concatenate(test_preds)
    test_target_all = np.concatenate(test_targets)

    test_r2_stress = compute_r2(test_pred_all[:, 0], test_target_all[:, 0])
    test_r2_disp_x = compute_r2(test_pred_all[:, 1], test_target_all[:, 1])
    test_r2_disp_y = compute_r2(test_pred_all[:, 2], test_target_all[:, 2])

    # Mean relative error
    stress_error = np.abs(test_pred_all[:, 0] - test_target_all[:, 0])
    mean_rel_error = float(stress_error.mean() / (np.abs(test_target_all[:, 0]).mean() + 1e-8) * 100)

    print(f"\n  Test Results (best model):")
    print(f"  {'-'*40}")
    print(f"  R2 - Von Mises Stress: {test_r2_stress:.4f}")
    print(f"  R2 - Displacement X:   {test_r2_disp_x:.4f}")
    print(f"  R2 - Displacement Y:   {test_r2_disp_y:.4f}")
    print(f"  Mean Relative Error:    {mean_rel_error:.2f}%")
    print(f"  Best Val Loss:          {best_val_loss:.6f}")

    # Save final model
    final_path = os.path.join(model_dir, "simustruct_surrogate.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n  Final model saved: {final_path}")

    # Save training history
    history["test_r2_stress"] = float(test_r2_stress)
    history["test_r2_disp_x"] = float(test_r2_disp_x)
    history["test_r2_disp_y"] = float(test_r2_disp_y)
    history["test_mean_rel_error_pct"] = mean_rel_error
    history["total_time_seconds"] = total_time
    history["best_epoch"] = int(checkpoint["epoch"])

    with open(os.path.join("logs", "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: logs/training_history.json")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimuStruct AI - Model Training")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_pde", type=float, default=0.1)
    parser.add_argument("--no_pinn", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data/hdf5")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model", type=str, default="enhanced_mlp",
                        choices=["enhanced_mlp", "mc_dropout"])
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_pde=args.lambda_pde,
        use_pinn=not args.no_pinn,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_type=args.model,
        patience=args.patience,
        device_str=args.device,
    )
