import os
import cv2
import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import multiprocessing as mp
from functools import partial
from scipy.interpolate import interp1d
from scipy.interpolate import interp1d

# --- ALS i Deskewing zostają bez zmian ---
def deskew_mask(mask):
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None: return mask
    angles = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines if -15 < np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) < 15]
    if not angles: return mask
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5: return mask
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z



class ECGDigitizerDSP:
    def mask_to_1d_signal(self, binary_mask_roi):
        h, w = binary_mask_roi.shape
        if w < 2:
            return np.zeros(1250)
        
        signal_1d = np.full(w, np.nan)
        for x in range(w):
            y_coords = np.where(binary_mask_roi[:, x] > 0)[0]
            if len(y_coords) > 0:
                # ULEPSZENIE 2: Envelope Averaging zamiast Mediany!
                # Bierzemy najwyższy i najniższy piksel linii w danej kolumnie i wyciągamy środek.
                # Chroni to ostre załamki QRS przed spłaszczeniem.
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                signal_1d[x] = h - ((y_min + y_max) / 2.0)
                
        # ULEPSZENIE 1: Smart Time-Cropping (Obcinanie marginesów)
        valid_indices = np.where(~np.isnan(signal_1d))[0]
        if len(valid_indices) < 10:
            return np.zeros(1250)
            
        first_x, last_x = valid_indices[0], valid_indices[-1]
        
        # Wycinamy tylko ten fragment, gdzie fizycznie JEST sygnał
        signal_cropped = signal_1d[first_x:last_x+1]
        
        # Łatanie wewnętrznych dziur (pchip)
        s = pd.Series(signal_cropped).interpolate(method='pchip', limit_direction='both')
        signal_filled = s.fillna(0).to_numpy()
        
        # ALS Baseline Wander Removal
        try:
            baseline = baseline_als(signal_filled)
            signal_flat = signal_filled - baseline
        except:
            signal_flat = signal_filled 
            
        # Lekki filtr wygładzający
        window_length = min(7, len(signal_flat) - 1 if len(signal_flat) % 2 == 0 else len(signal_flat))
        if window_length > 3:
            signal_flat = savgol_filter(signal_flat, window_length=window_length, polyorder=3)
            
        # Zmiana skali czasu dokładnie na 1250 (na "przyciętym" fizycznym sygnale)
        old_x = np.linspace(0, 1, len(signal_flat))
        new_x = np.linspace(0, 1, 1250)
        f_interp = interp1d(old_x, signal_flat, kind='linear')
        signal_1250 = f_interp(new_x)
        
        return signal_1250

class HardGridSlicer:
    def slice_mask(self, full_mask):
        h, w = full_mask.shape
        leads_dict = {}
        row_h, col_w = h // 4, w // 4
        leads_grid = [['I', 'AVR', 'V1', 'V4'], ['II', 'AVL', 'V2', 'V5'], ['III', 'AVF', 'V3', 'V6']]
        
        for row in range(3):
            for col in range(4):
                lead_name = leads_grid[row][col]
                y_start, y_end = row * row_h, (row + 1) * row_h
                x_start, x_end = col * col_w, (col + 1) * col_w
                patch = full_mask[y_start:y_end, x_start:x_end]
                # Marginesy 5% żeby nie łapać liter i ramek
                my, mx = int(row_h * 0.05), int(col_w * 0.05)
                leads_dict[lead_name] = patch[my:-my, mx:-mx]
        return leads_dict

def process_single_record(record, output_dir):
    record_id = record['record_id']
    mask_path = os.path.join(output_dir, "predictions_test", f"{record_id}_mask.png")
    out_npy_path = os.path.join(output_dir, "signals_1d", f"{record_id}_signals.npy")
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = deskew_mask(mask)
        slicer = HardGridSlicer()
        lead_masks = slicer.slice_mask(mask)
        digitizer = ECGDigitizerDSP()
        signals_1d = {name: digitizer.mask_to_1d_signal(m).astype(np.float16) for name, m in lead_masks.items()}
        np.save(out_npy_path, signals_1d)
        return (record_id, True)
    except: return (record_id, False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.output_dir, "signals_1d"), exist_ok=True)
    # CZYŚCIMY POPRZEDNIE BŁĘDNE PLIKI!
    os.system(f"rm -rf {os.path.join(args.output_dir, 'signals_1d')}/*")
    os.system(f"rm -rf {os.path.join(args.output_dir, 'signals_mV')}/*")
    df = pd.read_csv(args.registry_csv)
    test_records = df[df['split'] == 'test'].to_dict('records')
    with mp.Pool(16) as pool:
        pool.map(partial(process_single_record, output_dir=args.output_dir), test_records)
    print("✅ [DONE] DSP zakończone sukcesem.")

if __name__ == "__main__":
    main()
