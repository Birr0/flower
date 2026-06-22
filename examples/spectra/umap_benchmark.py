import argparse
import os 
import numpy as np
import pandas as pd
import umap
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from dotenv import load_dotenv
from datasets import load_dataset

from flower.evaluation.metrics import bootstrap_summary, print_bootstrap_stats

load_dotenv() 

# Constants
N_BOOT = 1000
N_filter = 300000
N_train = 200000
MANIFOLD_ATTR = "z"  # The attribute used to condition the UMAP
TARGET_ATTRIBUTES = ["z", "logM*", "logSFR", "A_v"]
catalog_dir = ... # This is from a non-anonymous source. It will be made avaialbe after release.

# Spender Mapping
SPENDER_MAP = {
    "spender_I": "spender_I_flow_v2/embeddings/7526202_0",
    "spender_II": "spender_II_flow_v2/embeddings/7527549_0"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spender", type=str, required=True, choices=["spender_I", "spender_II"])
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    # 1. Setup Paths and Load Data
    DATA_ROOT = os.getenv("DATA_ROOT")
    spender_embed_path = SPENDER_MAP[args.spender]
    embed_path = f"{DATA_ROOT}/sdss_II/{spender_embed_path}"

    data_files = {
        "train": f"{embed_path}/train/*.parquet",
        "test": f"{embed_path}/test/*.parquet",
    }
    
    print(f"Loading datasets for {args.spender} | Split: {args.split}...")
    ds = load_dataset("parquet", data_files=data_files)
    ds_attributes = load_dataset(catalog_dir)

    # 2. Data Preparation for Manifold (Conditioned on Redshift 'z')
    # Use N_filter for initial slice to ensure alignment
    X_train_raw = np.array(ds["train"][args.split])[:N_filter]
    y_z_train_raw = pd.to_numeric(ds_attributes["train"][MANIFOLD_ATTR], errors='coerce')[:N_filter]
    
    X_test_raw = np.array(ds["test"][args.split])
    y_z_test_raw = pd.to_numeric(ds_attributes["test"][MANIFOLD_ATTR], errors='coerce')

    # Mask for Z-alignment (Must have valid Z to build the manifold)
    mask_train = (np.isfinite(y_z_train_raw)) & (y_z_train_raw != -99.)
    mask_test = (np.isfinite(y_z_test_raw)) & (y_z_test_raw != -99.)

    # Apply mask and slice to N_train
    X_train = X_train_raw[mask_train][:N_train]
    y_train_z = y_z_train_raw[mask_train][:N_train]
    
    X_test = X_test_raw[mask_test]

    # Scaling Features for UMAP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Fit Z-Conditioned UMAP Manifold
    print(f"Fitting UMAP manifold conditioned on {MANIFOLD_ATTR}...")
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        transform_seed=42
    )
    
    reducer.fit(X_train_scaled, y=y_train_z)
    X_train_emb = reducer.transform(X_train_scaled)
    X_test_emb = reducer.transform(X_test_scaled)

    # 4. Iterate through Physical Attributes
    architectures = {
        "1-Layer": (64,),
        "2-Layer": (64, 64)
    }

    # 5. Iterate through Physical Attributes and Architectures
    results = []

    for attr in TARGET_ATTRIBUTES:
        # 1. Get raw labels and align with the N_filter slice
        y_attr_tr_raw = pd.to_numeric(ds_attributes["train"][attr], errors='coerce')[:N_filter]
        y_attr_tr_raw = y_attr_tr_raw[mask_train][:N_train]
        
        y_attr_te_raw = pd.to_numeric(ds_attributes["test"][attr], errors='coerce')
        y_attr_te_raw = y_attr_te_raw[mask_test]

        # 2. Final finite mask (Specific to this attribute)
        m_tr = np.isfinite(y_attr_tr_raw) & (y_attr_tr_raw != -99.)
        m_te = np.isfinite(y_attr_te_raw) & (y_attr_te_raw != -99.)

        # 3. Apply mask to the ALREADY-MASKED embeddings
        X_tr_final = X_train_emb[m_tr]
        y_tr_final = y_attr_tr_raw[m_tr]
        
        X_te_final = X_test_emb[m_te]
        y_te_final = y_attr_te_raw[m_te]

        for arch_name, layers in architectures.items():
            reg = MLPRegressor(hidden_layer_sizes=layers, max_iter=1000, random_state=42)
            reg.fit(X_tr_final, y_tr_final)
        
            y_pred = reg.predict(X_te_final)
            y_true = y_te_final.values if hasattr(y_te_final, 'values') else y_te_final
                
            # --- Bootstrap Analysis ---
            boot_scores = []
            for i in range(N_BOOT):
                # Resample the test set indices (with replacement)
                y_true_resamp, y_pred_resamp = resample(y_true, y_pred, random_state=i)
                boot_scores.append(r2_score(y_true_resamp, y_pred_resamp))
            
            stats = bootstrap_summary(boot_scores)
            label = f"{attr}_{arch_name}"
            print_bootstrap_stats(label, stats)
            
            results.append({
                "Spender": args.spender,
                "Split": args.split,
                "Feature_Type": attr,
                "Attribute": attr,
                "Layers": arch_name,
                "R2_Mean": round(stats['mean'], 4),
                "R2_Median": round(stats['median'], 4),
                "CI_95_Low": round(stats['ci_95'][0], 4),
                "CI_95_High": round(stats['ci_95'][1], 4),
                "Err_95": round(stats['err_95'], 4)
            })

    # 5. Save Results
    '''
    os.makedirs("./results_umap", exist_ok=True)
    out_name = f"./results_umap/umap_{args.spender}_{args.split}.csv"
    pd.DataFrame(umap_results).to_csv(out_name, index=False)
    print(f"Job Complete. Results saved to {out_name}")
    '''
    os.makedirs("./results_umap_w_err", exist_ok=True)
    out_name = f"./results_umap_w_err/resid_{args.spender}_{args.split}.csv"
    pd.DataFrame(results).to_csv(out_name, index=False)
    print(f"\nResults saved to {out_name}")

if __name__ == "__main__":
    main()