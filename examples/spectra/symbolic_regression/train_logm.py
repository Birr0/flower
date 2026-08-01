import os 
import subprocess
import csv 
import argparse
import glob

import processing_fun
import scienceplots
import matplotlib.pyplot as plt 
from astroML.datasets import fetch_sdss_specgals
import numpy as np
from datasets import load_dataset
from natsort import natsorted
import pandas as pd
from astropy.table import Table

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from utils import fit_sym_fn, fit_model_fn, export_and_visualise
# ---------------------------------------------------------
# 1. ARGUMENT PARSING (For Node Submission)
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Run SDSS models on a compute node.")
parser.add_argument('--feature', type=str, required=True, help="Target feature (e.g., LGM_FIB_P50)")
parser.add_argument('--embed_type', type=str, required=True, help="Embedding type (e.g., cond+z, z, uncond)")
parser.add_argument('--seed', type=int, required=True, help="Random seed for this run")
args = parser.parse_args()

FEATURE = args.feature
EMBED_TYPE = args.embed_type
SEED = args.seed

print(f"=== Job Start | Feature: {FEATURE} | Embed: {EMBED_TYPE} | Seed: {SEED} ===")

# ---------------------------------------------------------
# 2. SETUP & DATA LOADING
# ---------------------------------------------------------
plt.style.use("science")

DATA_ROOT = os.getenv("DATA_ROOT")
if DATA_ROOT:
    latex_bin_path = f"{DATA_ROOT}/texlive_store/texlive/bin/x86_64-linux"
    os.environ["PATH"] = latex_bin_path + os.pathsep + os.environ["PATH"]

try:
    version = subprocess.check_output(["latex", "--version"]).decode().splitlines()[0]
    print(f"Using LaTeX version: {version}")
except Exception as e:
    print(f"Error finding LaTeX: {e}")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})


# Update this path if needed for your environment
fp = "/data/dtce-schmidt/phys2526/sdss_II/spender_I_flow_v2/embeddings/7655991_0"
train_files = natsorted(glob.glob(f"{fp}/train/*.parquet"))
test_files = natsorted(glob.glob(f"{fp}/test/*.parquet"))
val_files = natsorted(glob.glob(f"{fp}/val/*.parquet"))

data_files = {"train": train_files, "test": test_files, "val": val_files}
ds = load_dataset("parquet", data_files=data_files)


# download data
data = fetch_sdss_specgals(
    data_home=f"{DATA_ROOT}/sdss",
    download_if_missing=True
)
df_sdss = pd.DataFrame(data)


# 1. Point directly to the already-downloaded file
#sdss_fits_path = f"{DATA_ROOT}/sdss/.fit.gz"

# 2. Load it as an Astropy Table (completely parallel-safe)
#sdss_table = Table.read(sdss_fits_path, format='fits')

# 3. Convert to Pandas
# (FITS files use big-endian memory; Pandas prefers native byte order, so we force conversion here to prevent future warnings)
#df_sdss = sdss_table.to_pandas()


df_sdss['merge_id'] = df_sdss['specObjID'].astype('int64')

merged_dfs = {}
for split in ['train', 'test', 'val']:
    df_spender = ds[split].to_pandas()
    df_spender['merge_id'] = df_spender['id'].astype(str).str.extract(r'(\d+)')[0].astype('int64')
    matched_df = pd.merge(df_spender, df_sdss, on='merge_id', how='inner').drop(columns=['merge_id'])
    
    if "mask_ratio" in matched_df.columns:
        matched_df = matched_df[(matched_df["mask_ratio"] <= 0.5) & (matched_df["z_x"] <= 0.3)]
        
    merged_dfs[split] = matched_df.reset_index(drop=True)

galspec = Table.read(f"{DATA_ROOT}/sdss/galSpecExtra-dr8.fits", format='fits')
galspec_df = galspec.to_pandas().dropna(subset=['SPECOBJID']).copy()
galspec_df['specObjID'] = galspec_df['SPECOBJID'].astype(str).str.extract(r'(\d+)')[0].astype('int64')

for split in ["train", "test", "val"]:
    merged_dfs[split] = pd.merge(merged_dfs[split], galspec_df, on=['specObjID'], how='inner', suffixes=["_", ""])

print("Data successfully loaded and merged.")

# ---------------------------------------------------------
# 4. SINGLE EXECUTION (Driven by SLURM)
# ---------------------------------------------------------
summary_results = []
pareto_data = []

# 1. Fit Symbolic Regression
results_sym = fit_sym_fn(FEATURE, EMBED_TYPE, merged_dfs, SEED)
sym_model = results_sym["Model"]

best_len = None
best_sym_mse = float('inf')

for m in sym_model.pareto_front_:
    tree = m['tree']
    ypred_test = sym_model.evaluate_model(tree, results_sym["X_test"])
    ypred_train = sym_model.evaluate_model(tree, results_sym["X_train"])

    finite_test = np.isfinite(ypred_test)
    finite_train = np.isfinite(ypred_train)

    if not finite_test.all() or not finite_train.all():
        n_bad = int((~finite_test).sum()) 
        print(f"[warn] length={tree.Length}: {n_bad}/{finite_test.size} non-finite preds in test -> excluded from best")
        mse_test = float('inf')
        r2_test = float('-inf')
    else:
        mse_test = mean_squared_error(results_sym["y_test"], ypred_test)
        r2_test = r2_score(results_sym["y_test"], ypred_test)

        mse_train = mean_squared_error(results_sym["y_train"], ypred_train)
        r2_train = r2_score(results_sym["y_train"], ypred_train)

    pareto_data.append({
        "Feature": FEATURE,
        "Embed_Type": EMBED_TYPE,
        "Seed": SEED,
        "Length": tree.Length,
        "Train_MSE": mse_train,
        "Train_R2": r2_train,
        "Test_MSE": mse_test,
        "Test_R2": r2_test,
        "Equation": m['model'],
    })

    
    if mse_test < best_sym_mse:
        best_sym_mse = mse_test
        best_len = tree.Length

# Export specific to this run
#out_dir = f'./output_{FEATURE}_{EMBED_TYPE}_seed{SEED}'
#export_and_visualise(results_sym, out_dir, ilen=best_len)

# 2. Fit MLP Regressor 
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, random_state=SEED)
mlp_results = fit_model_fn(FEATURE, EMBED_TYPE, mlp_model, merged_dfs)

# 3. Fit Linear Regression 
linear_model = LinearRegression()
linear_results = fit_model_fn(FEATURE, EMBED_TYPE, linear_model, merged_dfs)

summary_results.append({
    "Feature": FEATURE,
    "Embed_Type": EMBED_TYPE,
    "Seed": SEED,
    "MLP_Test_R2": mlp_results["Test_R2"],
    "MLP_Test_MSE": mlp_results["Test_MSE"],
    "LR_Test_R2": linear_results["Test_R2"],
    "LR_Test_MSE": linear_results["Test_MSE"],
    "SYM_Best_Test_MSE": best_sym_mse
})

# Save results uniquely for this specific job to prevent overwrite collisions
df_summary = pd.DataFrame(summary_results)
df_pareto = pd.DataFrame(pareto_data)

job_dir = f"./job_results_{FEATURE}_{EMBED_TYPE}_seed{SEED}"
os.makedirs(job_dir, exist_ok=True)

df_summary.to_csv(f"{job_dir}/summary_metrics.csv", index=False)
df_pareto.to_csv(f"{job_dir}/pareto_fronts.csv", index=False)

print(f"\n✅ Job completed. Results saved in {job_dir}/")