import os
import gc
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

NUM_WORKERS = min(os.cpu_count(), 32)
print(f"TURBO PIPELINE v3 - Processing dataset... ({NUM_WORKERS} workers)")

# 1. LOAD DEVICES (static features)
devices = pd.read_csv('dataset/devices.csv')
print(f"Loaded {len(devices)} devices")

# 2. AGGREGATE 5-MIN → MONTHLY (parallel, enriched features)
CHUNK_SIZE = 500_000
BASE_COLS = ['x2', 'x1'] + [f't{i}' for i in range(1, 14)]
DERIVED_COLS = ['cos_hour', 'sin_hour', 'delta_source', 'delta_load']
NUMERIC_AGG = BASE_COLS + DERIVED_COLS
VARIANCE_COLS = ['x2', 'x1', 't1', 'delta_source', 'delta_load']

def process_chunk(chunk):
    """Aggregate one chunk with derived physics + diurnal features."""
    # Derived features at 5-min resolution
    hour = chunk['timedate'].str[11:13].astype(float)
    chunk['cos_hour'] = np.cos(2 * np.pi * hour / 24)
    chunk['sin_hour'] = np.sin(2 * np.pi * hour / 24)
    chunk['delta_source'] = chunk['t3'] - chunk['t4']
    chunk['delta_load'] = chunk['t5'] - chunk['t6']

    chunk['month_key'] = chunk['timedate'].str[:4] + '_' + chunk['timedate'].str[5:7].str.lstrip('0')
    group_cols = ['deviceId', 'month_key', 'period']
    grouped = chunk.groupby(group_cols)

    # Dynamic column selection (safe if any columns missing)
    avail_base = [c for c in BASE_COLS if c in chunk.columns]
    avail_agg = [c for c in avail_base + DERIVED_COLS if c in chunk.columns]
    sums = grouped[avail_agg].sum().rename(columns={c: c + '_sum' for c in avail_agg})
    counts = grouped.size().to_frame('_n')

    # Sum of squares for cross-chunk variance
    avail_var = [vc for vc in VARIANCE_COLS if vc in chunk.columns]
    for vc in avail_var:
        chunk[vc + '_sq'] = chunk[vc] ** 2
    sq_sums = grouped[[vc + '_sq' for vc in avail_var]].sum()

    firsts = grouped[['x3', 'deviceType']].first()
    return pd.concat([sums, counts, sq_sums, firsts], axis=1).reset_index()

print(f"Aggregating rows to monthly ({NUM_WORKERS} workers)...")
futures = []
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
    for chunk in pd.read_csv('dataset/data.csv', chunksize=CHUNK_SIZE):
        futures.append(pool.submit(process_chunk, chunk))
        print(f"  Submitted {len(futures)} chunks...", end='\r')
    results = [f.result() for f in futures]
    print(f"\n  Processed {len(results)} chunks")

# Combine partial sums/counts → weighted means + std
combined = pd.concat(results, ignore_index=True)
del results
group_cols = ['deviceId', 'month_key', 'period']
grouped = combined.groupby(group_cols)

# Dynamic column detection from whatever chunks produced
all_sum_cols = [col for col in combined.columns if col.endswith('_sum')]
total_sums = grouped[all_sum_cols].sum()
total_counts = grouped['_n'].sum()
all_sq_cols = [col for col in combined.columns if col.endswith('_sq')]
total_sq = grouped[all_sq_cols].sum()
cats = grouped[['x3', 'deviceType']].first()

# Means
monthly = total_sums.div(total_counts, axis=0)
monthly.columns = [c.replace('_sum', '') for c in all_sum_cols]
for col in monthly.columns:
    monthly[col] = pd.to_numeric(monthly[col], errors='coerce')

# Std from E[X²] - E[X]²
for vc in ['x2', 'x1', 't1', 'delta_source', 'delta_load']:
    if f'{vc}_sq' in all_sq_cols and vc in monthly.columns:
        mean_sq = total_sq[f'{vc}_sq'] / total_counts
        monthly[f'{vc}_std'] = np.sqrt(np.maximum(mean_sq - monthly[vc] ** 2, 0))

monthly = pd.concat([monthly, cats], axis=1).reset_index()
monthly = monthly.merge(devices, on='deviceId', how='left')
del combined
gc.collect()

print(f"Generated {len(monthly):,} monthly records from {monthly['deviceId'].nunique()} devices")

# 3. SPLIT TRAIN/VALID
train = monthly[monthly['period'] == 'train'].copy()
valid = monthly[monthly['period'] == 'valid'].copy()

print(f"Split - Train: {len(train):,}, Valid: {len(valid):,}")

# 4. ENHANCED FEATURES
def create_features(df):
    df['month'] = df['month_key'].str.split('_').str[1].astype(int)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['lat_rad'] = np.deg2rad(df['latitude'])
    df['lon_rad'] = np.deg2rad(df['longitude'])
    
    # Interactions (top impact from physics)
    df['device_month'] = df['deviceType'].astype(str) + '_' + df['month'].astype(str)
    df['curve_month'] = df['x3'].astype(str) + '_' + df['month'].astype(str)
    
    # Temperature physics (use available temp cols)
    temp_cols = [f't{i}' for i in range(1, 14) if f't{i}' in df.columns]
    df['avg_temp'] = df[temp_cols].mean(axis=1)
    df['temp_spread'] = df[temp_cols].max(axis=1) - df[temp_cols].min(axis=1)
    
    # Spatial (distance to Warsaw ~52.23, 21.01)
    cos_arg = (np.sin(df['lat_rad']) * np.sin(0.9115) + 
               np.cos(df['lat_rad']) * np.cos(0.9115) * np.cos(df['lon_rad'] - 0.3665))
    df['dist_warsaw'] = np.arccos(np.clip(cos_arg, -1, 1)) * 6371
    
    # COP proxy (freq-based, avoids using target x2)
    df['cop_proxy'] = df['x1'] / np.maximum(df['t1'] - df['avg_temp'], 1e-6)
    
    return df

train = create_features(train)
valid = create_features(valid)

# 5. TARGET ENCODING (device-specific baselines)
device_means = train.groupby('deviceId')['x2'].mean()
train['device_baseline'] = train['deviceId'].map(device_means)
valid['device_baseline'] = valid['deviceId'].map(device_means)

# 6. FEATURE LIST
features = [
    'avg_temp', 'temp_spread', 'x1', 'dist_warsaw',
    'month', 'cos_month', 'sin_month', 'deviceType', 'x3',
    'device_baseline', 'latitude', 'longitude',
    'delta_source', 'delta_load', 'cos_hour', 'sin_hour',
    'x2_std', 'x1_std', 't1_std', 'delta_source_std', 'delta_load_std',
    'cop_proxy',
]

X_train = train[features].fillna(0)
y_train = train['x2']
X_valid = valid[features].fillna(0)
y_valid = valid['x2']

print(f"Training on {len(features)} features...")

# 7. ENHANCED MODEL
lgb = LGBMRegressor(
    n_estimators=5000, learning_rate=0.01, num_leaves=128,
    max_depth=10, min_child_samples=30, subsample=0.85,
    colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1,
    early_stopping_rounds=200
)

lgb.fit(
    X_train, y_train, 
    eval_set=[(X_valid, y_valid)], 
    eval_metric='mae'
)

print(f"Valid MAE: {mean_absolute_error(y_valid, lgb.predict(X_valid)):.4f}")

# 8. PREDICT TEST (Jul-Oct 2025)
test = monthly[monthly['period'] == 'test'].copy()
test = create_features(test)
test['device_baseline'] = test['deviceId'].map(device_means)
X_test = test[features].fillna(0)

test['prediction'] = lgb.predict(X_test)

# 9. FORMAT SUBMISSION
submission = test[['deviceId', 'month_key', 'prediction']].copy()
submission[['year', 'month']] = submission['month_key'].str.split('_', expand=True).astype(int)
submission = submission[['deviceId', 'year', 'month', 'prediction']].sort_values(['deviceId', 'year', 'month'])
submission.columns = ['deviceId', 'year', 'month', 'prediction']

submission.to_csv('submission_turbo.csv', index=False)
print("submission_turbo.csv successfully generated.")
print(submission.head())
print(f"Submission shape: {submission.shape}")
