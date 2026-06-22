import argparse
import os 
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from dotenv import load_dotenv
from datasets import load_dataset

# Custom metric functions
from flower.evaluation.metrics import bootstrap_summary, print_bootstrap_stats

load_dotenv() 

# --- Configuration ---
N_BOOT = 1000
N_FILTER = 300000
N_TRAIN = 200000
TARGET_ATTRIBUTES = ["z", "logM*", "logSFR", "A_v"]

SPENDER_MAP = {
    "spender_I": "spender_I_flow_v2/embeddings/7526202_0",
    "spender_II": "spender_II_flow_v2/embeddings/7527549_0"
}

ARCHITECTURES = {
    "1-Layer": (64,),
    "2-Layer": (64, 64)
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spender", type=str, required=True, choices=["spender_I", "spender_II"])
    parser.add_argument("--split", type=str, required=True, help="e.g., cond, orig, or uncond")
    args = parser.parse_args()

    DATA_ROOT = os.getenv("DATA_ROOT")
    spender_path = SPENDER_MAP[args.spender]
    embed_path = f"{DATA_ROOT}/sdss_II/{spender_path}"

    # 1. Load Datasets
    print(f"Loading {args.spender} | Split: {args.split}")
    ds = load_dataset("parquet", data_files={
        "train": f"{embed_path}/train/*.parquet", 
        "test": f"{embed_path}/test/*.parquet"
    })
    ds_attributes = load_dataset("Birr001/spectra_catalog")

    # 2. Data Alignment & Masking
    # Load raw embeddings for the specific split requested
    X_train_raw = np.array(ds["train"][args.split], dtype=float)[:N_FILTER]
    X_test_raw = np.array(ds["test"][args.split], dtype=float)

    # We use redshift (z) as the primary alignment key to mask out invalid rows
    y_z_train_raw = pd.to_numeric(ds_attributes["train"]["z"], errors='coerce')[:N_FILTER]
    y_z_test_raw = pd.to_numeric(ds_attributes["test"]["z"], errors='coerce')

    mask_train = (np.isfinite(y_z_train_raw)) & (y_z_train_raw != -99.) & (np.isfinite(X_train_raw).all(axis=1))
    mask_test = (np.isfinite(y_z_test_raw)) & (y_z_test_raw != -99.) & (np.isfinite(X_test_raw).all(axis=1))

    X_train = X_train_raw[mask_train][:N_TRAIN]
    X_test = X_test_raw[mask_test]

    # 3. Scaling
    # Essential for MLP performance on raw embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Evaluation Loop
    results = []
    
    for attr in TARGET_ATTRIBUTES:
        print(f"\n--- Target Attribute: {attr} ---")
        
        # Extract target values and apply alignment mask
        y_attr_tr = pd.to_numeric(ds_attributes["train"][attr], errors='coerce')[:N_FILTER][mask_train][:N_TRAIN]
        y_attr_te = pd.to_numeric(ds_attributes["test"][attr], errors='coerce')[mask_test]

        # Ensure no NaNs in the target attribute itself
        m_tr = np.isfinite(y_attr_tr) & (y_attr_tr != -99.)
        m_te = np.isfinite(y_attr_te) & (y_attr_te != -99.)

        for arch_name, layers in ARCHITECTURES.items():
            print(f"  Training {arch_name}...")
            reg = MLPRegressor(hidden_layer_sizes=layers, max_iter=1000, random_state=42)
            reg.fit(X_train_scaled[m_tr], y_attr_tr[m_tr])
            
            y_true = y_attr_te[m_te].values if hasattr(y_attr_te[m_te], 'values') else y_attr_te[m_te]
            y_pred = reg.predict(X_test_scaled[m_te])

            # --- Bootstrap Analysis ---
            boot_scores = []
            for i in range(N_BOOT):
                y_true_resamp, y_pred_resamp = resample(y_true, y_pred, random_state=i)
                boot_scores.append(r2_score(y_true_resamp, y_pred_resamp))
            
            stats = bootstrap_summary(boot_scores)
            print_bootstrap_stats(f"{args.split}_{attr}_{arch_name}", stats)
            
            results.append({
                "Spender": args.spender,
                "Split": args.split,
                "Attribute": attr,
                "Layers": arch_name,
                "R2_Mean": round(stats['mean'], 4),
                "R2_Median": round(stats['median'], 4),
                "CI_95_Low": round(stats['ci_95'][0], 4),
                "CI_95_High": round(stats['ci_95'][1], 4),
                "Err_95": round(stats['err_95'], 4)
            })

    # 5. Save Results
    out_dir = "./results_raw_embeddings"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/raw_{args.spender}_{args.split}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")

if __name__ == "__main__":
    main()