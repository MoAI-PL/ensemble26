06 import os
import cv2
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    in_npy_dir = os.path.join(args.output_dir, "signals_1d")
    out_mv_dir = os.path.join(args.output_dir, "signals_mV")
    os.makedirs(out_mv_dir, exist_ok=True)

    df = pd.read_csv(args.registry_csv)
    test_df = df[df['split'] == 'test']

    success_count = 0
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        record_id = row['record_id']
        img_path = row['png_path']
        npy_path = os.path.join(in_npy_dir, f"{record_id}_signals.npy")
        out_path = os.path.join(out_mv_dir, f"{record_id}_mV.npy")
        
        if not os.path.exists(npy_path):
            continue
            
        try:
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
                
            h, w = img_gray.shape
            signals_dict = np.load(npy_path, allow_pickle=True).item()
            signals_mV = {}
            
            # CZYSTA FIZYKA: Szerokość A4 to 297mm. Standardowe EKG to 10mm/mV.
            # Więc 1 mV na obrazie = w / 29.7. 
            px_per_mV = w / 29.7
            
            for lead_name, sig_px in signals_dict.items():
                sig_clean = np.nan_to_num(sig_px, nan=0.0)
                
                if len(sig_clean) < 5:
                    sig_final = np.zeros(1250, dtype=np.float32)
                else:
                    # 1. Centrowanie linii na medianie (izolinii)
                    centered = sig_clean - np.median(sig_clean)
                    
                    # 2. Skalowanie fizyczne do czystych mV
                    sig_scaled = centered / px_per_mV
                    
                    # Brak wygładzania i cudowania! Zostawiamy idealnie ostre igły R!
                    
                    # 3. Bezpieczna interpolacja liniowa do wymaganego 1250 (bez falowania)
                    old_x = np.linspace(0, 1, len(sig_scaled))
                    new_x = np.linspace(0, 1, 1250)
                    f = interp1d(old_x, sig_scaled, kind='linear')
                    sig_final = f(new_x).astype(np.float32)
                    
                clean_name = lead_name.upper().replace("LEAD_", "")
                signals_mV[clean_name] = sig_final
                
            np.save(out_path, signals_mV)
            success_count += 1
        except Exception as e:
            pass
            
    print(f"\n[DONE] Udane konwersje: {success_count} / {len(test_df)}")

if __name__ == "__main__":
    main()
