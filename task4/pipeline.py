import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = True

class ECGDataset(Dataset):
    def __init__(self, df, masks_dir, is_train=True):
        self.df = df.reset_index(drop=True)
        self.masks_dir = masks_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = row['record_id']
        
        img_path = row['png_path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        mask_path = os.path.join(self.masks_dir, f"{record_id}_mask.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 1. Podbita rozdzielczość (1024x1024)
        img_size = (1024, 1024)
        image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

        # 2. Augmentacje dla zbioru treningowego
        if self.is_train:
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((512, 512), angle, 1.0)
                image = cv2.warpAffine(image, M, img_size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
                mask = cv2.warpAffine(mask, M, img_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    masks_dir = os.path.join(args.output_dir, "masks_train")
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    df = pd.read_csv(args.registry_csv)
    
    # Bierzemy wszystkie dane treningowe z CSV
    full_train_df = df[df['split'] == 'train'].copy()
    
    if full_train_df.empty:
         raise ValueError("❌ Brak danych treningowych w pliku CSV!")

    # DYNAMICZNY PODZIAŁ W PAMIĘCI (90% Train, 10% Val) - to omija błąd z pustym Val
    train_df, val_df = train_test_split(full_train_df, test_size=0.1, random_state=42)

    print(f"[INFO] DataLoaders: TRAIN={len(train_df)}, VAL={len(val_df)} próbek...")
    
    train_dataset = ECGDataset(train_df, masks_dir, is_train=True)
    val_dataset = ECGDataset(val_df, masks_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True)

    print("[INFO] Budowanie modelu U-Net (ResNet34)...")
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    def combined_loss(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"[START] Trening. Device: {device}, Batch: {args.batch_size}, Max Epochs: {args.epochs}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(checkpoints_dir, "unet_best.pth")

    for epoch in range(args.epochs):
        # 1. FAZA TRENINGU
        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        
        for images, masks in pbar_train:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = combined_loss(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar_train.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 2. FAZA WALIDACJI
        model.eval()
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]")
        
        with torch.no_grad():
            for images, masks in pbar_val:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    v_loss = combined_loss(outputs, masks)
                val_loss += v_loss.item()
                pbar_val.set_postfix(val_loss=v_loss.item())

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"--- Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ---")
        
        if avg_val_loss < best_val_loss:
            print(f"🔥 Nowy najlepszy model! Val Loss spadł z {best_val_loss:.4f} na {avg_val_loss:.4f}. Zapisuję...")
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"⚠️ Brak poprawy Val Loss od {patience_counter} epok.")
            
        if patience_counter >= args.patience:
            print(f"🛑 EARLY STOPPING! Przerwano trening po {epoch+1} epokach.")
            break

    print(f"✅ [DONE] Trening zakończony. Najlepsze wagi zapisane w: {best_model_path}")

if __name__ == "__main__":
    main()
