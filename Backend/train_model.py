import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ======================
# CONFIG
# ======================
DATASET_PATH = "dataset"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD DATASET
# ======================
def load_dataset():
    X_files = sorted(glob.glob(f"{DATASET_PATH}/X*.npy"))
    Y_files = sorted(glob.glob(f"{DATASET_PATH}/Y*.npy"))
    X_all = []
    Y_all = []
    for xf, yf in zip(X_files, Y_files):
        X_all.append(np.load(xf))
        Y_all.append(np.load(yf))
    X = np.concatenate(X_all)
    Y = np.concatenate(Y_all)
    
    # Train/val split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y
    )
    return X_train, X_val, Y_train, Y_val

# ======================
# SIMPLE DATASET (GESTURE ONLY)
# ======================
class EMGDataset(Dataset):
    def __init__(self, X, Y, augment=False):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        
        # Data augmentation
        if self.augment:
            # Random amplitude scaling
            scale = np.random.uniform(0.85, 1.15)
            x = x * scale
            
            # Add noise
            noise = np.random.normal(0, 0.3, x.shape).astype(np.float32)
            x = x + noise
            
            # Random time shift
            shift = np.random.randint(-10, 10)
            x = np.roll(x, shift)
        
        x = x.astype(np.float32)
        
        # Normalization
        mean = x.mean()
        std = x.std()
        if std < 1e-6:
            std = 1.0
        x_norm = (x - mean) / std
        
        gesture = float(self.Y[idx])
        
        return (
            torch.tensor(x_norm, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(gesture, dtype=torch.float32)
        )

# ======================
#  CNN + TRANSFORMER
# ======================
class EMGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN feature extractor (good for local patterns)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        
        # Transformer for temporal dependencies
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, 64))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, T, 1]
        x = x.transpose(1, 2)  # [B, 1, T]
        
        # CNN features
        x = self.conv(x)  # [B, 64, T/4]
        
        # Transformer
        x = x.transpose(1, 2)  # [B, T/4, 64]
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)  # [B, T/4, 64]
        
        # Classification
        x = x.transpose(1, 2)  # [B, 64, T/4]
        x = self.classifier(x)  # [B, 1]
        
        return x

# ======================
# TRAIN LOOP
# ======================
def train():
    X_train, X_val, Y_train, Y_val = load_dataset()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    train_dataset = EMGDataset(X_train, Y_train, augment=True)
    val_dataset = EMGDataset(X_val, Y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = EMGClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion = nn.BCELoss()
    
    best_val_acc = 0
    patience = 0
    max_patience = 15
    
    for epoch in range(EPOCHS):
        # === TRAINING ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(-1)
            
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            pred_class = (pred > 0.5).float()
            train_correct += (pred_class == y).sum().item()
            train_total += len(y)
        
        train_acc = train_correct / train_total * 100
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(-1)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
                pred_class = (pred > 0.5).float()
                val_correct += (pred_class == y).sum().item()
                val_total += len(y)
        
        val_acc = val_correct / val_total * 100
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), "emg_transformer_real.pth")
            print(f"    ✅ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"    ⚠️  Early stopping triggered (patience={max_patience})")
                break
    
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()