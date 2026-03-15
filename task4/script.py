import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="binarization threshold (overridden by best_threshold.txt if present)")
    args = parser.parse_args()

    # Ścieżki
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    model_path = os.path.join(checkpoints_dir, "unet_best.pth")
    
    masks_out_dir = os.path.join(args.output_dir, "predictions_test")
    os.makedirs(masks_out_dir, exist_ok=True)

    df = pd.read_csv(args.registry_csv)
    test_df = df[df['split'] == 'test']

    if test_df.empty:
        raise ValueError("❌ Brak danych testowych w pliku CSV!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Inicjalizacja modelu U-Net (ResNet34) na urządzeniu: {device}")
    
    # 1. ZBUDOWANIE MODELU (Musi idealnie pasować do treningu!)
    # Uwaga: podczas treningu używaliśmy encoder_weights="imagenet".
    # By uniknąć niespójności, ładowanie modelu powinno używać tych samych ustawień.
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    )
    
    # Wczytanie wag (trenowaliśmy i zapisywaliśmy state_dict)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Nie znaleziono wag modelu: {model_path}")

    # torch.load nie ma parametru weights_only; w trainingu zapisywany jest state_dict
    state = torch.load(model_path, map_location=device)
    # Jeśli ktoś zapisał cały obiekt modelu zamiast state_dict, próbujemy bezpiecznie obsłużyć obie opcje
    if isinstance(state, dict) and any(k.startswith('_') or k in model.state_dict() for k in state.keys()):
        model.load_state_dict(state)
    else:
        # Fallback: jeżeli plik zawiera obiekt modelu, wyładujemy bezpośrednio (rzadkie)
        try:
            model = state
            model.to(device)
            model.eval()
        except Exception:
            raise RuntimeError("Nie udało się załadować wag modelu. Sprawdź czy zapisano state_dict z torch.save(model.state_dict(), path).")
    model.to(device)
    model.eval()

    print(f"[START] Inferencja dla {len(test_df)} obrazów testowych...")

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generowanie masek"):
            record_id = row['record_id']
            img_path = row['png_path']

            out_mask_path = os.path.join(masks_out_dir, f"{record_id}_mask.png")

            # Jeśli maska już istnieje, pomijamy (pozwala na wznowienie po przerwaniu)
            if os.path.exists(out_mask_path):
                continue

            # 2. Wczytanie obrazu w SKALI SZAROŚCI
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h_orig, w_orig = img.shape

            # 3. Zmiana rozmiaru do 1024x1024
            img_resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

            # 4. Normalizacja do [0, 1]
            img_norm = img_resized.astype(np.float32) / 255.0

            # --- Simple TTA: original + horizontal flip (average probabilities) ---
            input_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).to(device)
            img_flipped = cv2.flip(img_resized, 1)
            img_f_norm = img_flipped.astype(np.float32) / 255.0
            input_f = torch.tensor(img_f_norm).unsqueeze(0).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast():
                pred = model(input_tensor)
                pred_f = model(input_f)
                pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
                pred_f_prob = torch.sigmoid(pred_f).squeeze().cpu().numpy()

            # unflip the flipped prediction
            try:
                pred_f_prob = np.fliplr(pred_f_prob)
            except Exception:
                # defensive: if shapes mismatch, fallback to original
                pred_f_prob = pred_prob

            pred_prob_avg = (pred_prob + pred_f_prob) / 2.0

            # threshold: prefer best_threshold.txt in output_dir if present
            t_default = args.threshold
            thr_path = os.path.join(args.output_dir, 'best_threshold.txt')
            if os.path.exists(thr_path):
                try:
                    with open(thr_path, 'r') as tf:
                        t_default = float(tf.read().strip())
                        print(f"[INFO] Loaded threshold from {thr_path}: {t_default}")
                except Exception:
                    pass

            # 6. Binarizacja
            mask_bin = (pred_prob_avg > t_default).astype(np.uint8) * 255

            # 7. SKALOWANIE POWROTNE (Resize do natywnej rozdzielczości)
            mask_native = cv2.resize(mask_bin, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # 8. Zapis gotowej maski dla Agenta 3
            cv2.imwrite(out_mask_path, mask_native)

    print(f"✅ [DONE] Wygenerowano maski testowe w: {masks_out_dir}")

if __name__ == "__main__":
    main()
