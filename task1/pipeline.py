"""
EnsembleAI Hackathon 2026 — Task 1: ChEBI Ontology Prediction
Full pipeline: data loading, feature extraction, training, prediction, submission.
"""


import os
import io
import re
import warnings
import time
import pickle
from collections import defaultdict, deque
from pathlib import Path


import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


warnings.filterwarnings("ignore")


# Thread control
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[v] = "32"


load_dotenv()


DATA_DIR = Path(__file__).parent / "Instruction" / "task1"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ─── Phase 1: Load Data & Parse OBO ──────────────────────────────────────────


def parse_obo(obo_path):
    """Parse chebi_classes.obo → dict mapping child_class → list of parent_classes."""
    parent_map = defaultdict(list)
    current_id = None
    with open(obo_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: class_"):
                current_id = line.split("id: ")[1]
            elif line.startswith("is_a: class_"):
                parent = line.split("is_a: ")[1].strip()
                if current_id:
                    parent_map[current_id].append(parent)
    return dict(parent_map)


def get_all_ancestors(cls, parent_map, cache=None):
    """Recursively get all ancestors of a class."""
    if cache is None:
        cache = {}
    if cls in cache:
        return cache[cls]
    ancestors = set()
    for parent in parent_map.get(cls, []):
        ancestors.add(parent)
        ancestors |= get_all_ancestors(parent, parent_map, cache)
    cache[cls] = ancestors
    return ancestors


def dag_propagate_labels(labels_df, class_cols, parent_map):
    """Propagate labels upward: if child=1, all ancestors=1."""
    print("  DAG propagating labels upward...")
    labels = labels_df[class_cols].values.copy()
    col_to_idx = {c: i for i, c in enumerate(class_cols)}

    ancestor_cache = {}
    for cls in class_cols:
        get_all_ancestors(cls, parent_map, ancestor_cache)

    for cls_idx, cls in enumerate(class_cols):
        ancestors = ancestor_cache.get(cls, set())
        if not ancestors:
            continue
        anc_idxs = [col_to_idx[a] for a in ancestors if a in col_to_idx]
        if anc_idxs:
            positive_mask = labels[:, cls_idx] == 1
            if positive_mask.any():
                for ai in anc_idxs:
                    labels[positive_mask, ai] = 1

    result = labels_df.copy()
    result[class_cols] = labels
    return result


def soft_dag_propagate(probs, class_cols, parent_map):
    """Bidirectional probability-level DAG propagation using topological sort.
    Top-down: child prob = min(child, parent)
    Bottom-up: parent prob = max(parent, child)
    """
    result = probs.copy()
    col_to_idx = {c: i for i, c in enumerate(class_cols)}

    # Build children map and in-degree for topological sort
    children_map = defaultdict(list)
    in_degree = {c: 0 for c in class_cols}
    for child, parents in parent_map.items():
        if child not in col_to_idx:
            continue
        for parent in parents:
            if parent in col_to_idx:
                children_map[parent].append(child)
                in_degree[child] = in_degree.get(child, 0) + 1

    # Kahn's algorithm for topological sort
    queue = deque(c for c in class_cols if in_degree.get(c, 0) == 0)
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for child in children_map.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Top-down: child prob = min(child, parent)
    for node in topo_order:
        node_idx = col_to_idx[node]
        for child in children_map.get(node, []):
            if child in col_to_idx:
                child_idx = col_to_idx[child]
                result[:, child_idx] = np.minimum(result[:, child_idx], result[:, node_idx])

    # Bottom-up: parent prob = max(parent, child)
    for node in reversed(topo_order):
        node_idx = col_to_idx[node]
        for parent in parent_map.get(node, []):
            if parent in col_to_idx:
                parent_idx = col_to_idx[parent]
                result[:, parent_idx] = np.maximum(result[:, parent_idx], result[:, node_idx])

    return result


def scaffold_splits(smiles_list, n_splits=3, test_size=0.2, random_state=42):
    """Murcko scaffold-based splits → yields (train_idx, val_idx) for each fold."""
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    scaffolds = []
    for smi in smiles_list:
        try:
            scaf = MurckoScaffoldSmiles(smi, includeChirality=False)
        except Exception:
            scaf = smi
        scaffolds.append(scaf)

    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    indices = np.arange(len(smiles_list))
    for train_idx, val_idx in gss.split(indices, groups=scaffolds):
        yield train_idx, val_idx


# ─── Phase 2: Feature Extraction ───


def extract_features(smiles_list):
    """Extract concatenated molecular fingerprints."""
    from rdkit import Chem
    from skfp.fingerprints import (
        ECFPFingerprint,
        MACCSFingerprint,
        TopologicalTorsionFingerprint,
        AtomPairFingerprint,
    )
    from skfp.preprocessing import MolFromSmilesTransformer

    print("  Converting SMILES to mols...")
    mol_transformer = MolFromSmilesTransformer()
    mols = mol_transformer.transform(smiles_list)

    fps = []

    print("  ECFP4 (2048, count)...")
    ecfp = ECFPFingerprint(fp_size=2048, radius=2, count=True, n_jobs=-1)
    fps.append(ecfp.transform(mols))

    print("  MACCS (166)...")
    maccs = MACCSFingerprint(n_jobs=-1)
    fps.append(maccs.transform(mols))

    print("  TopologicalTorsion (2048, count)...")
    tt = TopologicalTorsionFingerprint(fp_size=2048, count=True, n_jobs=-1)
    fps.append(tt.transform(mols))

    print("  AtomPair (2048, count)...")
    ap = AtomPairFingerprint(fp_size=2048, count=True, n_jobs=-1)
    fps.append(ap.transform(mols))

    print("  RDKit 2D descriptors...")
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

    desc_names = [d[0] for d in Descriptors._descList]
    calc = MolecularDescriptorCalculator(desc_names)

    def _calc_desc(idx, mol):
        try:
            return idx, list(calc.CalcDescriptors(mol))
        except Exception:
            return idx, [0.0] * len(desc_names)

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_calc_desc)(idx, mol) for idx, mol in enumerate(mols)
    )
    # Preserve original order from parallel results
    results.sort(key=lambda x: x[0])
    desc_arr = np.array([r[1] for r in results], dtype=np.float32)
    desc_arr = np.nan_to_num(desc_arr, nan=0.0, posinf=0.0, neginf=0.0)
    fps.append(desc_arr)

    X = np.hstack([np.asarray(f, dtype=np.float32) for f in fps])
    print(f"  Total feature dimension: {X.shape[1]}")
    return X


# ─── Phase 3: Train LightGBM ─────────────────────────────────────────────────


def train_lgbm_per_class(X_train, Y_train, X_val, Y_val, class_cols):
    """Train one LGBMClassifier per class. Returns val probabilities + models."""
    import lightgbm as lgb

    n_classes = len(class_cols)
    val_probs = np.zeros((X_val.shape[0], n_classes), dtype=np.float32)
    models = [None] * n_classes

    def _fit_one(i, cls):
        y_tr = Y_train[:, i]
        y_vl = Y_val[:, i]
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        pw = n_neg / max(n_pos, 1)

        model = lgb.LGBMClassifier(
            scale_pos_weight=pw,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            device="cpu",
            verbose=-1,
            random_state=42,
        )
        model.fit(X_train, y_tr)
        val_prob = model.predict_proba(X_val)[:, 1]
        return i, val_prob, model

    results = Parallel(n_jobs=16, prefer="threads")(
        delayed(_fit_one)(i, cls) for i, cls in enumerate(class_cols)
    )

    for i, val_prob, model in results:
        val_probs[:, i] = val_prob
        models[i] = model

    preds_so_far = (val_probs > 0.5).astype(int)
    f1s = []
    for j in range(n_classes):
        f1s.append(f1_score(Y_val[:, j], preds_so_far[:, j], zero_division=0))
    print(f"  [all/{n_classes}] running macro-F1 = {np.mean(f1s):.4f}")

    return val_probs, models


# ─── Phase 4: Threshold Tuning ───────────


def tune_thresholds(val_probs, Y_val, class_cols):
    """Per-class F1-optimal thresholds using PR-curve fallback for rare classes."""
    from sklearn.metrics import precision_recall_curve

    n_classes = len(class_cols)
    thresholds = np.full(n_classes, 0.5)
    grid = np.linspace(0.05, 0.95, 50)

    def _best_threshold(i):
        y_true = Y_val[:, i]
        y_prob = val_probs[:, i]
        if y_true.sum() < 10:  # PR-curve fallback for rare classes
            if y_true.sum() == 0:
                return i, 0.5
            precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob)
            if len(pr_thresh) == 0:
                return i, 0.5
            f1s = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = np.argmax(f1s[:-1])  # exclude last (threshold=0)
            return i, float(np.clip(pr_thresh[best_idx], 0.05, 0.95))
        best_f1, best_t = -1, 0.5
        for t in grid:
            f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        return i, best_t

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_best_threshold)(i) for i in range(n_classes)
    )
    for i, t in results:
        thresholds[i] = t
    return thresholds


# ─── Phase Post-processing 5: ───────────────────────────


def dag_propagate_predictions(pred_probs, pred_binary, class_cols, parent_map):
    """Enforce DAG consistency: if child=1, parent must=1."""
    result = pred_binary.copy()
    col_to_idx = {c: i for i, c in enumerate(class_cols)}

    ancestor_cache = {}
    for cls in class_cols:
        get_all_ancestors(cls, parent_map, ancestor_cache)

    for cls_idx, cls in enumerate(class_cols):
        ancestors = ancestor_cache.get(cls, set())
        if not ancestors:
            continue
        anc_idxs = [col_to_idx[a] for a in ancestors if a in col_to_idx]
        if anc_idxs:
            positive_mask = result[:, cls_idx] == 1
            if positive_mask.any():
                for ai in anc_idxs:
                    result[positive_mask, ai] = 1

    # Hardcode class_0 = 1
    if "class_0" in col_to_idx:
        result[:, col_to_idx["class_0"]] = 1

    return result


def count_inconsistencies(pred_probs, class_cols, parent_map):
    """Count cases where child prob > parent prob."""
    col_to_idx = {c: i for i, c in enumerate(class_cols)}
    count = 0
    for cls in class_cols:
        for parent in parent_map.get(cls, []):
            if parent in col_to_idx:
                ci = col_to_idx[cls]
                pi = col_to_idx[parent]
                count += (pred_probs[:, ci] > pred_probs[:, pi]).sum()
    return count


# ─── Main Pipeline ──────────────


def main():
    t0 = time.time()

    # --- Phase 1: Load ---
    print("=" * 60)
    print("PHASE 1: Loading data")
    print("=" * 60)

    train_df = pd.read_parquet(DATA_DIR / "chebi_dataset_train.parquet")
    test_df = pd.read_parquet(DATA_DIR / "chebi_dataset_test_empty.parquet")

    class_cols = [c for c in train_df.columns if c.startswith("class_")]
    # Sort numerically
    class_cols = sorted(class_cols, key=lambda x: int(x.split("_")[1]))
    # Drop class_0 from training targets (always 1)
    target_cols = [c for c in class_cols if c != "class_0"]

    print(f"  Train: {train_df.shape[0]} molecules, {len(class_cols)} classes")
    print(f"  Test:  {test_df.shape[0]} molecules")
    print(f"  Target cols (excl class_0): {len(target_cols)}")

    # Parse OBO
    parent_map = parse_obo(DATA_DIR / "chebi_classes.obo")
    print(f"  OBO: {len(parent_map)} classes with parents")

    # --- Phase 1b: DAG label propagation ---
    print("\nPHASE 1b: DAG label propagation on training labels")
    train_df = dag_propagate_labels(train_df, target_cols, parent_map)

    # --- Phase 1c: Scaffold splits (3-fold) ---
    print("\nPHASE 1c: Scaffold splits (3-fold)")
    splits = list(scaffold_splits(train_df["SMILES"].tolist(), n_splits=3, test_size=0.2, random_state=42))
    for i, (tr, vl) in enumerate(splits, 1):
        print(f"  Fold {i}: train {len(tr)}, val {len(vl)}")

    # --- Phase 2: Features ---
    print("\n" + "=" * 60)
    print("PHASE 2: Feature extraction")
    print("=" * 60)

    all_smiles = pd.concat([train_df["SMILES"], test_df["SMILES"]], ignore_index=True).tolist()
    n_train = len(train_df)
    n_test = len(test_df)

    # Check for cached features
    feat_cache = ARTIFACTS_DIR / "features_all.npy"
    if feat_cache.exists():
        print("  Loading cached features...")
        X_all = np.load(feat_cache)
    else:
        X_all = extract_features(all_smiles)
        np.save(feat_cache, X_all)
        print(f"  Cached features to {feat_cache}")

    X_train_full = X_all[:n_train]
    X_test = X_all[n_train:]
    Y_full = train_df[target_cols].values.astype(np.int8)

    print(f"  X_train_full: {X_train_full.shape}, X_test: {X_test.shape}")

    # --- Phase 3: Train (3-fold CV) ---
    print("\n" + "=" * 60)
    print("PHASE 3: Training LightGBM (per-class, 3-fold CV)")
    print("=" * 60)

    fold_thresholds = []
    fold_test_probs = []
    fold_f1 = []

    for fold_id, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n-- Fold {fold_id}/3 --")
        X_train = X_train_full[train_idx]
        X_val = X_train_full[val_idx]
        Y_train = Y_full[train_idx]
        Y_val = Y_full[val_idx]

        print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

        val_probs, models = train_lgbm_per_class(X_train, Y_train, X_val, Y_val, target_cols)

        # Soft DAG propagation on val probabilities
        val_probs = soft_dag_propagate(val_probs, target_cols, parent_map)

        # --- Phase 4: Threshold tuning ---
        thresholds = tune_thresholds(val_probs, Y_val, target_cols)
        fold_thresholds.append(thresholds)
        np.save(ARTIFACTS_DIR / f"thresholds_fold{fold_id}.npy", thresholds)

        val_preds = (val_probs >= thresholds).astype(int)
        f1_before = f1_score(Y_val, val_preds, average="macro", zero_division=0)

        # --- Phase 5: Post-processing on val ---
        val_preds_pp = dag_propagate_predictions(val_probs, val_preds, target_cols, parent_map)
        f1_after = f1_score(Y_val, val_preds_pp, average="macro", zero_division=0)
        incons = count_inconsistencies(val_probs, target_cols, parent_map)
        fold_f1.append((f1_before, f1_after))

        print(f"  Fold {fold_id} macro-F1 (before postproc): {f1_before:.4f}")
        print(f"  Fold {fold_id} macro-F1 (after DAG postproc): {f1_after:.4f}")
        print(f"  Fold {fold_id} inconsistencies: {incons}")

        # Predict test set for this fold
        test_probs_fold = np.zeros((X_test.shape[0], len(target_cols)), dtype=np.float32)
        for i, model in enumerate(models):
            test_probs_fold[:, i] = model.predict_proba(X_test)[:, 1]
        test_probs_fold = soft_dag_propagate(test_probs_fold, target_cols, parent_map)
        fold_test_probs.append(test_probs_fold)

    # Average thresholds and test probabilities across folds
    avg_thresholds = np.mean(fold_thresholds, axis=0)
    np.save(ARTIFACTS_DIR / "thresholds.npy", avg_thresholds)
    print("\nAveraged thresholds saved.")
    print("Fold F1 (before → after):", [f"{b:.4f}→{a:.4f}" for b, a in fold_f1])

    # --- Phase 6: Generate test predictions and submit (3-fold ensemble) ---
    print("\n" + "=" * 60)
    print("PHASE 6: Test prediction & submission (3-fold ensemble)")
    print("=" * 60)

    mean_test_probs = np.mean(fold_test_probs, axis=0)
    test_preds = (mean_test_probs >= avg_thresholds).astype(int)
    test_preds = dag_propagate_predictions(mean_test_probs, test_preds, target_cols, parent_map)

    # Build submission DataFrame
    sub = test_df[["mol_id", "SMILES"]].copy()
    sub["class_0"] = 1
    for i, col in enumerate(target_cols):
        sub[col] = test_preds[:, i]

    # Ensure column order matches example
    example_cols = pd.read_parquet(DATA_DIR / "chebi_submission_example.parquet").columns.tolist()
    sub = sub[example_cols]

    sub_path = ARTIFACTS_DIR / "submission_val_cv.parquet"
    sub.to_parquet(sub_path, index=False)
    print(f"  Saved submission: {sub_path}")
    print(f"  Shape: {sub.shape}")

    submit_to_server(sub_path)

    # --- Phase 6b: Retrain on full data ---
    print("\n" + "=" * 60)
    print("PHASE 6b: Retrain on full training data")
    print("=" * 60)

    retrain_and_submit(X_train_full, Y_full, X_test, test_df, target_cols,
                       avg_thresholds, parent_map, example_cols)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE — total time: {elapsed/60:.1f} min")


def retrain_and_submit(X_train_full, Y_full, X_test, test_df, target_cols,
                       thresholds, parent_map, example_cols):
    """Retrain on all training data using same hyperparams, then submit."""
    import lightgbm as lgb

    n_classes = len(target_cols)
    test_probs = np.zeros((X_test.shape[0], n_classes), dtype=np.float32)
    models_full = [None] * n_classes

    def _fit_full(i, cls):
        y_tr = Y_full[:, i]
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        pw = n_neg / max(n_pos, 1)

        model = lgb.LGBMClassifier(
            scale_pos_weight=pw,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            device="cpu",
            verbose=-1,
            random_state=42,
        )
        model.fit(X_train_full, y_tr)
        test_prob = model.predict_proba(X_test)[:, 1]
        return i, test_prob, model

    results = Parallel(n_jobs=16, prefer="threads")(
        delayed(_fit_full)(i, cls) for i, cls in enumerate(target_cols)
    )

    for i, test_prob, model in results:
        test_probs[:, i] = test_prob
        models_full[i] = model

    test_probs = soft_dag_propagate(test_probs, target_cols, parent_map)

    with open(ARTIFACTS_DIR / "models_full.pkl", "wb") as f:
        pickle.dump(models_full, f)

    test_preds = (test_probs >= thresholds).astype(int)
    test_preds = dag_propagate_predictions(test_probs, test_preds, target_cols, parent_map)

    sub = test_df[["mol_id", "SMILES"]].copy()
    sub["class_0"] = 1
    for i, col in enumerate(target_cols):
        sub[col] = test_preds[:, i]
    sub = sub[example_cols]

    sub_path = ARTIFACTS_DIR / "submission_full_retrain.parquet"
    sub.to_parquet(sub_path, index=False)
    print(f"  Saved full-retrain submission: {sub_path}")

    submit_to_server(sub_path)


def submit_to_server(parquet_path):
    """Submit parquet to the leaderboard server."""
    import requests

    api_token = os.getenv("TEAM_TOKEN")
    server_url = os.getenv("SERVER_URL")

    if not api_token or not server_url:
        print("  ⚠  TEAM_TOKEN or SERVER_URL not set — skipping submission")
        return

    df = pd.read_parquet(parquet_path)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    headers = {"X-API-Token": api_token}
    try:
        resp = requests.post(
            f"{server_url}/task1",
            files={"parquet_file": buffer},
            headers=headers,
            timeout=120,
        )
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        print(f"  Submission response: {resp.status_code} — {data}")
    except Exception as e:
        print(f"  Submission failed: {e}")


if __name__ == "__main__":
    main()
