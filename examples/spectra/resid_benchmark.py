import argparse
import os 
import numpy as np
import pandas as pd
import umap
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
MANIFOLD_ATTR = "z"  
TARGET_ATTRIBUTES = ["z", "logM*", "logSFR", "A_v"]
catalog_dir = ... # This is from a non-anonymous source. It will be made available after release.


SPENDER_MAP = {
    "spender_I": "spender_I_flow_v2/embeddings/7526202_0",
    "spender_II": "spender_II_flow_v2/embeddings/7527549_0"
}

def get_residuals(X_train, z_train, X_test, z_test, method="linear"):
    """
    Predicts X from z and returns (X_scaled - predicted_X_scaled).
    """
    # 1. Scale the raw embeddings before residualization
    # This ensures the regressor treats all embedding dimensions equally.
    embed_scaler = StandardScaler()
    X_tr_scaled = embed_scaler.fit_transform(X_train)
    X_te_scaled = embed_scaler.transform(X_test)

    # 2. Reshape z for sklearn (needs 2D array)
    z_tr = z_train.values.reshape(-1, 1) if hasattr(z_train, 'values') else z_train.reshape(-1, 1)
    z_te = z_test.values.reshape(-1, 1) if hasattr(z_test, 'values') else z_test.reshape(-1, 1)
    
    if method == "linear":
        # A true linear baseline
        print(f"  Fitting Linear residualizer (z -> Scaled Embedding)...")
        model = LinearRegression()
    elif method == "mlp":
        # Non-linear residualizer: Small MLP to capture complex z-dependencies
        print(f"  Fitting Non-Linear (MLP) residualizer (z -> Scaled Embedding)...")
        model = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=1000, random_state=42)
    elif method == "rf":
         # Non-linear residualizer: Small MLP to capture complex z-dependencies
        print(f"  Fitting RF residualizer (z -> Scaled Embedding)...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        msg = f"{method} not avaiable. Select either linear, mlp or rf"
        raise NameError(msg)

    model.fit(z_tr, X_tr_scaled)
    
    # 3. Calculate residuals in the scaled space
    X_train_resid = X_tr_scaled - model.predict(z_tr)
    X_test_resid = X_te_scaled - model.predict(z_te)
    
    return X_train_resid, X_test_resid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spender", type=str, required=True, choices=["spender_I", "spender_II"])
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Data
    DATA_ROOT = os.getenv("DATA_ROOT")
    spender_embed_path = SPENDER_MAP[args.spender]
    embed_path = f"{DATA_ROOT}/sdss_II/{spender_embed_path}"
    ds = load_dataset("parquet", data_files={"train": f"{embed_path}/train/*.parquet", "test": f"{embed_path}/test/*.parquet"})
    ds_attributes = load_dataset(catalog_dir)

    # 2. Alignment and Masking
    X_train_raw = np.array(ds["train"][args.split], dtype=float)[:N_filter]
    y_z_train_raw = pd.to_numeric(ds_attributes["train"][MANIFOLD_ATTR], errors='coerce')[:N_filter]
    
    X_test_raw = np.array(ds["test"][args.split], dtype=float)
    y_z_test_raw = pd.to_numeric(ds_attributes["test"][MANIFOLD_ATTR], errors='coerce')

    mask_train = (np.isfinite(y_z_train_raw)) & (y_z_train_raw != -99.) & (np.isfinite(X_train_raw).all(axis=1))
    mask_test = (np.isfinite(y_z_test_raw)) & (y_z_test_raw != -99.) & (np.isfinite(X_test_raw).all(axis=1))

    X_train = X_train_raw[mask_train][:N_train]
    y_train_z = y_z_train_raw[mask_train][:N_train]
    X_test = X_test_raw[mask_test]
    y_test_z = y_z_test_raw[mask_test]

    # 3. Perform Residualization
    # We create two versions of our feature set
    X_train_lin, X_test_lin = get_residuals(X_train, y_train_z, X_test, y_test_z, method="linear")
    X_train_mlp, X_test_mlp = get_residuals(X_train, y_train_z, X_test, y_test_z, method="mlp")
    X_train_rf, X_test_rf = get_residuals(X_train, y_train_z, X_test, y_test_z, method="rf")

    feature_sets = {
        "Linear-Resid": (X_train_lin, X_test_lin),
        "MLP-Resid": (X_train_mlp, X_test_mlp),
        "RF-Resid": (X_train_rf, X_test_rf)
    }

    # 4. Evaluation Loop
    architectures = {"1-Layer": (64,), "2-Layer": (64, 64)}
    results = []

    for feat_name, (X_tr, X_te) in feature_sets.items():
        print(f"Evaluating Feature Set: {feat_name}")
        
        # Scale the residuals
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        for attr in TARGET_ATTRIBUTES:
            y_attr_tr = pd.to_numeric(ds_attributes["train"][attr], errors='coerce')[:N_filter][mask_train][:N_train]
            y_attr_te = pd.to_numeric(ds_attributes["test"][attr], errors='coerce')[mask_test]

            # Final check for target NaNs
            m_tr = np.isfinite(y_attr_tr) & (y_attr_tr != -99.)
            m_te = np.isfinite(y_attr_te) & (y_attr_te != -99.)

            for arch_name, layers in architectures.items():
                reg = MLPRegressor(hidden_layer_sizes=layers, max_iter=1000, random_state=42)
                reg.fit(X_tr_sc[m_tr], y_attr_tr[m_tr])
                
                '''
                r2 = r2_score(y_attr_te[m_te], reg.predict(X_te_sc[m_te]))
                
                results.append({
                    "Spender": args.spender,
                    "Split": args.split,
                    "Feature_Type": feat_name,
                    "Attribute": attr,
                    "Layers": arch_name,
                    "Test R2": round(r2, 4)
                })
                '''
                # Get base predictions
                y_true = y_attr_te[m_te].values if hasattr(y_attr_te[m_te], 'values') else y_attr_te[m_te]
                y_pred = reg.predict(X_te_sc[m_te])

                # --- Bootstrap Analysis ---
                boot_scores = []
                for i in range(N_BOOT):
                    # Resample the test set indices (with replacement)
                    y_true_resamp, y_pred_resamp = resample(y_true, y_pred, random_state=i)
                    boot_scores.append(r2_score(y_true_resamp, y_pred_resamp))
                
                stats = bootstrap_summary(boot_scores)
                label = f"{feat_name}_{attr}_{arch_name}"
                print_bootstrap_stats(label, stats)
                
                results.append({
                    "Spender": args.spender,
                    "Split": args.split,
                    "Feature_Type": feat_name,
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
    os.makedirs("./results_resid", exist_ok=True)
    out_name = f"./results_resid/resid_{args.spender}_{args.split}_without_std_scaler.csv"
    pd.DataFrame(results).to_csv(out_name, index=False)
    print(f"Results saved to {out_name}")
    '''
    os.makedirs("./results_resid_w_err", exist_ok=True)
    out_name = f"./results_resid_w_err/resid_{args.spender}_{args.split}.csv"
    pd.DataFrame(results).to_csv(out_name, index=False)
    print(f"\nResults saved to {out_name}")

if __name__ == "__main__":
    main()