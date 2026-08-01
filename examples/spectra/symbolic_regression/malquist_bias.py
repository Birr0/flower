import os 
import subprocess
import csv 
import math


import processing_fun

import scienceplots
import matplotlib.pyplot as plt 
from astroML.datasets import fetch_sdss_specgals
import glob
import numpy as np
from datasets import load_dataset
from natsort import natsorted
import pandas as pd
from astropy.table import Table

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pyoperon.sklearn import SymbolicRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import numpy as np


plt.style.use("science")

DATA_ROOT = os.getenv("DATA_ROOT")

latex_bin_path = f"{DATA_ROOT}/texlive_store/texlive/bin/x86_64-linux"
os.environ["PATH"] = latex_bin_path + os.pathsep + os.environ["PATH"]

try:
    version = subprocess.check_output(["latex", "--version"]).decode().splitlines()[0]
    print(f"Using LaTeX version: {version}")
except Exception as e:
    print(f"Error finding LaTeX: {e}")

# Now enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

# Downloads and caches a robust subset of the MPA-JHU catalog
data = fetch_sdss_specgals()

fp = "/data/dtce-schmidt/phys2526/sdss_II/spender_I_flow_v2/embeddings/7655991_0"
train_files = natsorted(glob.glob(f"{fp}/train/*.parquet"))
test_files = natsorted(glob.glob(f"{fp}/test/*.parquet"))
val_files = natsorted(glob.glob(f"{fp}/val/*.parquet"))

# 2. Define the file path mapping using the explicitly sorted lists
data_files = {
    "train": train_files,
    "test": test_files,
    "val": val_files
}

# 3. Load the files into a single DatasetDict
ds = load_dataset("parquet", data_files=data_files)

import pandas as pd
import numpy as np

df_sdss = pd.DataFrame(data)
df_sdss['merge_id'] = df_sdss['specObjID'].astype('int64')

merged_dfs = {}
aligned_arrays = {}

splits = ['train', 'test', 'val']
for split in splits:
    print(f"--- Processing '{split}' split ---")
    # Convert the current Hugging Face split to a Pandas DataFrame
    df_spender = ds[split].to_pandas()
    
    # Clean the Hugging Face IDs (strip the 'b', quotes, and spaces)
    df_spender['merge_id'] = df_spender['id'].astype(str).str.extract(r'(\d+)')[0].astype('int64')
    
    # Perform the master merge
    matched_df = pd.merge(df_spender, df_sdss, on='merge_id', how='inner')
    
    # Clean up by dropping the temporary merge column
    matched_df = matched_df.drop(columns=['merge_id'])
    
    initial_len = len(matched_df)
    
    # Apply the mask ratio filter (drop any rows where mask_ratio is 1.0)
    if "mask_ratio" in matched_df.columns:
        matched_df = matched_df[(matched_df["mask_ratio"] <= 0.5) & (matched_df["z_x"] <= 0.3)]
        
    # Reset index for clean dataframes
    matched_df = matched_df.reset_index(drop=True)
    
    # Store the full pandas dataframe in our dictionary
    merged_dfs[split] = matched_df
    
    dropped_count = initial_len - len(matched_df)
    print(f"Successfully merged {initial_len} galaxies.")
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with mask_ratio == 1.0. Final count: {len(matched_df)}")

print("All splits successfully matched, masked, and aligned!")
# 1. Clean the data: Drop missing values and non-positive fluxes
# (You cannot take the log of 0 or negative numbers)

# Load the FITS file into an Astropy Table
galspec = Table.read('galSpecExtra-dr8.fits', format='fits')
# Print the column names so you know exactly what IDs you are working with
print(galspec.colnames)

# Convert the loaded FITS table to a pandas DataFrame
galspec_df = galspec.to_pandas()

# Drop any rows where SPECOBJID is NaN
galspec_df = galspec_df.dropna(subset=['SPECOBJID']).copy()
# Now run your original extraction
galspec_df['specObjID'] = galspec_df['SPECOBJID'].astype(str).str.extract(r'(\d+)')[0].astype('int64')

for split in ["train", "test", "val"]:
    merged_dfs[split] = pd.merge(merged_dfs[split], galspec_df, on=['specObjID'], how='inner', suffixes=["", ""])
    

# Convert the loaded FITS table to a pandas DataFrame
galspec_df = galspec.to_pandas()

# Drop any rows where SPECOBJID is NaN
galspec_df = galspec_df.dropna(subset=['SPECOBJID']).copy()
# Now run your original extraction
galspec_df['specObjID'] = galspec_df['SPECOBJID'].astype(str).str.extract(r'(\d+)')[0].astype('int64')

for split in ["train", "test", "val"]:
    merged_dfs[split] = pd.merge(merged_dfs[split], galspec_df, on=['specObjID'], how='inner', suffixes=["_", ""])

symbols = 'add,sub,mul,div,exp,log,constant,variable,pow'
N = 5000

def get_even_z_samples(z_array, n_total=5000, z_min=0.05, z_max=0.3, n_bins=10, seed=42):
    """
    Selects evenly distributed indices across redshift bins.
    Targets (n_total / n_bins) samples per bin. If a bin has fewer items 
    than the target, it takes all available items in that bin.
    """
    rng = np.random.default_rng(seed=seed)
    
    # Create bin edges
    bins = np.linspace(z_min, z_max, n_bins + 1)
    
    # np.digitize with right=True mimics 0.05 < z <= z_max behavior
    bin_indices = np.digitize(z_array, bins, right=True)
    
    samples_per_bin = n_total // n_bins
    selected_indices = []
    
    for i in range(1, n_bins + 1):
        in_bin = np.where(bin_indices == i)[0]
        
        if len(in_bin) == 0:
            continue
            
        # Take the required amount, or all available if the bin is sparse
        n_to_select = min(len(in_bin), samples_per_bin)
        selected = rng.choice(in_bin, size=n_to_select, replace=False)
        selected_indices.extend(selected)
        print(len(selected_indices))
        
    # Return shuffled indices so the bins aren't grouped sequentially in the arrays
    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)
    
    return selected_indices


def fit_sym_fn(feature, embed_type, merged_dfs):
    is_flux = "flux" in feature.lower()

    feat_root = feature.split("_P50")[0]

    is_mass = feat_root == "LGM_FIB"

    # 1. Clean the data
    train_clean = merged_dfs["train"].dropna(subset=[feature])
    test_clean = merged_dfs["test"].dropna(subset=[feature])

    train_clean = train_clean[(train_clean[feature] != -9999.0) & (train_clean["z_x"] >= 0.075)]
    test_clean = test_clean[(test_clean[feature] != -9999.0) & (test_clean["z_x"] >= 0.075)]

    if is_flux:
        train_clean = train_clean[train_clean[feature] > 0]
        test_clean = test_clean[test_clean[feature] > 0]
    elif is_mass:
        train_clean = train_clean[(train_clean[feature] > 9.5) & (train_clean[feature] < 11.5)]
        test_clean = test_clean[(test_clean[feature] > 9.5) & (test_clean[feature] < 11.5)]
    print(f"\n--- Running: {embed_type} ---")
    
    # 2. Extract arrays
    # Universally grab the 1D redshift array for our sampling logic
    z_train_sampling = train_clean["z_x"].values 
    
    if embed_type == "cond+z":
        X_train = np.stack(train_clean["cond"].values)
        y_train = train_clean[feature].values

        X_test = np.stack(test_clean["cond"].values)
        y_test = test_clean[feature].values

        z_train = train_clean["z_x"].values.reshape(-1, 1)
        z_test = test_clean["z_x"].values.reshape(-1, 1)
        
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        X_train = np.hstack((X_train, z_train))
        X_test = np.hstack((X_test, z_test))

    elif embed_type == "z":
        y_train = train_clean[feature].values
        y_test = test_clean[feature].values

        z_train = train_clean["z_x"].values.reshape(-1, 1)
        z_test = test_clean["z_x"].values.reshape(-1, 1)

        X_train = z_train 
        X_test = z_test 

    else:
        X_train = np.stack(train_clean[embed_type].values)
        y_train = train_clean[feature].values

        X_test = np.stack(test_clean[embed_type].values)
        y_test = test_clean[feature].values

        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. Conditionally Log10 the Target Variable
    if is_flux:
        y_train = np.log10(y_train)
        y_test = np.log10(y_test)
        
    y_train = y_train.reshape(-1, 1).flatten() 
    y_test = y_test.reshape(-1, 1).flatten()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

    # --- NEW: Stratified sampling across redshift bins ---
    # Will grab evenly from 0.05 < z <= 0.3 across 10 bins
    '''
    random_indices = get_even_z_samples(
        z_array=z_train_sampling, 
        n_total=N, 
        z_min=0.05, 
        z_max=0.3, 
        n_bins=10 
    )
    '''
    
    # Generate random indices with seed 42
    n_samples = min(N, len(X_train_scaled))
    rng = np.random.default_rng(seed=42)
    random_indices = rng.choice(len(X_train_scaled), size=n_samples, replace=False)
    
    # Apply random sampling
    X_train_fit = X_train_scaled[random_indices]
    y_train_fit = y_train[random_indices]

    # 4. Train the Symbolic Regression Model
    print(f"Training on {X_train_fit.shape[0]} samples, {X_train_fit.shape[1]} features")
    
    objectives = ['mse', 'length']
    #uncert = (train_clean[f"{feat_root}_P84"][random_indices] - train_clean[f"{feat_root}_P16"][random_indices])/2

    reg = SymbolicRegressor(
        allowed_symbols=symbols,
        offspring_generator='basic',
        optimizer_iterations=100,
        max_length=15,
        initialization_method='btc',
        n_threads=8,
        objectives=objectives,
        epsilon=1e-6,
        random_state=42, 
        reinserter='keep-best',
        max_evaluations=int(1e6),
        symbolic_mode=False,
        tournament_size=3,
        #uncertainty=uncert
    )
    
    reg.fit(X_train_fit, y_train_fit)

    # 4b. Print the symbolic form of the best model
    best_model_str = reg.get_model_string(reg.model_, precision=3)
    print(f"\nBest symbolic expression ({embed_type} -> {feature}):")
    print(f"  y = {best_model_str}\n")

    print("Pareto front (all candidate expressions):")
    for entry in reg.pareto_front_:
        expr_str = reg.get_model_string(entry['tree'], precision=3)
        mse_val, length_val = entry['objective_values']
        # 5. Make predictions on the full test set
        predictions = reg.predict(X_test_scaled)

        # 6. Calculate Metrics
        test_r2 = r2_score(y_test, predictions)
        print(f"  [mse={mse_val:.5g}, r2={test_r2:.5g}, length={length_val:.0f}]  y = {expr_str}")
    print()

    # 5. Make predictions on the full test set
    predictions = reg.predict(X_test_scaled)

    # 6. Calculate Metrics
    test_r2 = r2_score(y_test, predictions)
    test_mse = mean_squared_error(y_test, predictions)
    pareto_front_values = [t['objective_values'] for t in reg.pareto_front_]

    return {
        "Feature": feature,
        "Embed_Type": embed_type,
        "Test_R2": test_r2,
        "Test_MSE": test_mse,
        "Pareto_Front_Objectives": pareto_front_values,
        "Model": reg,                   
        "X_train": X_train_fit,         
        "y_train": y_train_fit,         
        "X_test": X_test_scaled,        
        "y_test": y_test,
        "best_model": best_model_str             
    }

def fit_model_fn(
    feature,
    embed_type,
    estimator,
    merged_dfs
):
    """
    Fits a given scikit-learn estimator to the specified feature and embedding type.
    """
    is_flux = "flux" in feature.lower()

    # 1. Clean the data: Drop missing values
    train_clean = merged_dfs["train"].dropna(subset=[feature])
    test_clean = merged_dfs["test"].dropna(subset=[feature])

    # --- Remove standard SDSS missing data flags (-9999.0) ---
    train_clean = train_clean[train_clean[feature] != -9999.0]
    test_clean = test_clean[test_clean[feature] != -9999.0]

    # --- Only enforce strictly positive values if we are doing a log10 transform ---
    if is_flux:
        train_clean = train_clean[train_clean[feature] > 0]
        test_clean = test_clean[test_clean[feature] > 0]
        

    model_name = estimator.__class__.__name__
    print(f"\n--- Running {model_name}: {embed_type} ---")
    
    # 2. Extract arrays
    if embed_type == "cond+z":
        X_train = np.stack(train_clean["cond"].values)
        y_train = train_clean[feature].values

        X_test = np.stack(test_clean["cond"].values)
        y_test = test_clean[feature].values

        z_train = train_clean["z_x"].values.reshape(-1, 1)
        z_test = test_clean["z_x"].values.reshape(-1, 1)

        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        X_train = np.hstack((X_train, z_train))
        X_test = np.hstack((X_test, z_test))
    
    elif embed_type == "z":
        # --- Fit using only redshift as the input feature ---
        y_train = train_clean[feature].values
        y_test = test_clean[feature].values

        z_train = train_clean["z_x"].values.reshape(-1, 1)
        z_test = test_clean["z_x"].values.reshape(-1, 1)

        X_train = z_train 
        X_test = z_test 
        
    else:
        X_train = np.stack(train_clean[embed_type].values)
        y_train = train_clean[feature].values

        X_test = np.stack(test_clean[embed_type].values)
        y_test = test_clean[feature].values

        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. Conditionally Log10 the Target Variable
    if is_flux:
        y_train = np.log10(y_train)
        y_test = np.log10(y_test)
        
    y_train = y_train.reshape(-1, 1).flatten() 
    y_test = y_test.reshape(-1, 1).flatten()
    
    # --- Scaling ---
    # Crucial for MLPs; optional but harmless (and sometimes helpful for numerical stability) for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the Model
    print(f"Training {model_name} on {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
    estimator.fit(X_train_scaled, y_train)

    # 5. Make predictions on the full test set
    predictions = estimator.predict(X_test_scaled)

    # 6. Calculate Metrics
    test_r2 = r2_score(y_test, predictions)
    test_mse = mean_squared_error(y_test, predictions)

    # Return the results
    return {
        "Feature": feature,
        "Embed_Type": embed_type,
        "Test_R2": test_r2,
        "Test_MSE": test_mse,
        "Model": estimator,
        "Scaler": scaler,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test_scaled,
        "y_test": y_test
    }

def export_and_visualise(fit_results, ilen=11):
    # Extract the objects from our fitting function
    reg = fit_results["Model"]
    X_train = fit_results["X_train"]
    y_train = fit_results["y_train"]
    X_test = fit_results["X_test"]
    y_test = fit_results["y_test"]
    feature_name = fit_results["Feature"]
    embed_type = fit_results["Embed_Type"] 

    out_dir = f'output_{feature_name}_{embed_type}'

    # File names
    outname = f'{out_dir}/fun.csv'
    outname_pred_train = f'{out_dir}/train'
    outname_pred_test = f'{out_dir}/test'

    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # List to store metrics for ranking the best equations
    model_metrics = []

    with open(outname, "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Equation", "Length", "MSE_train", "MSE_test", "R2_train", "R2_test"])

        for m in reg.pareto_front_:
            tree = m['tree']
            
            # Evaluate on train
            ypred_train = reg.evaluate_model(tree, X_train)
            mse_train = mean_squared_error(y_train, ypred_train)
            r2_train = r2_score(y_train, ypred_train)
            
            # Evaluate on test
            ypred_test = reg.evaluate_model(tree, X_test)
            mse_test = mean_squared_error(y_test, ypred_test)
            r2_test = r2_score(y_test, ypred_test)
            
            # Write all metrics to CSV
            writer.writerow([m['model'], tree.Length, mse_train, mse_test, r2_train, r2_test])

            # Store metrics in memory to sort later
            model_metrics.append({
                "equation": m['model'],
                "length": tree.Length,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "mse_train": mse_train,
                "mse_test": mse_test
            })

            # Save arrays
            output = np.vstack([X_train.T, y_train, ypred_train]).T
            output_test = np.vstack([X_test.T, y_test, ypred_test]).T
            np.savetxt(f'{outname_pred_train}_{tree.Length}.csv', output)
            np.savetxt(f'{outname_pred_test}_{tree.Length}.csv', output_test)

    names = [f'x{i}' for i in range(X_train.shape[1])]
    
    # Sort the models by Test R2 (descending) to find the best performers
    sorted_models = sorted(model_metrics, key=lambda x: x['r2_test'], reverse=True)
    
    print("\n" + "="*70)
    print("Top 3 Best Performing Equations (Sorted by Test R2):")
    print("="*70)
    for i, m in enumerate(sorted_models[:3], 1):
        print(f"Rank {i} (Length {m['length']}):")
        print(f"Equation:  {m['equation']}")
        print(f"Test R2:   {m['r2_test']:.6f} | Train R2:   {m['r2_train']:.6f}")
        print(f"Test MSE:  {m['mse_test']:.6f} | Train MSE:  {m['mse_train']:.6f}")
        print("-" * 70)
    
    # Check that 'ilen' exists and print the metrics for the plotted equation
    target_m = next((m for m in reg.pareto_front_ if m['tree'].Length == ilen), None)
    
    if target_m is None:
        available_lengths = [m['length'] for m in model_metrics]
        print(f"\nWarning: Tree length {ilen} not found in Pareto front. Available lengths: {available_lengths}")
    else:
        # Retrieve the already-calculated metrics for the target equation
        target_metrics = next(m for m in model_metrics if m['length'] == ilen)
        print("\n" + "="*70)
        print(f"Metrics for Plotted Equation (Length {ilen}):")
        print(f"Train R2: {target_metrics['r2_train']:.6f}")
        print(f"Test R2:  {target_metrics['r2_test']:.6f}")
        print("="*70 + "\n")
    
    # Changed objective to 'MSE' to match our current setup
    fig, ax = processing_fun.plot_pareto(out_dir, names, ilen=ilen, objective='MSE')
    fig, axs = processing_fun.prediction_plots(out_dir, ilen=ilen)
    
    return fig, ax, axs

feature = "LGM_FIB_P50"
embed_types = ["z", "cond", "cond+z", "uncond", "orig"]

# Data structures to store our results
summary_results = []
pareto_data = []

print("Starting model training loop...")

for embed_type in embed_types:
    print(f"\n{'='*50}")
    print(f"Processing Embedding Type: {embed_type}")
    print(f"{'='*50}")

    # 1. Fit Symbolic Regression
    results_sym = fit_sym_fn(feature=feature, embed_type=embed_type, merged_dfs=merged_dfs)
    sym_model = results_sym["Model"]
    
    # --- FIX: Initialize tracking variables ---
    best_len = None
    best_sym_mse = float('inf')
    best_sym_r2 = -float('inf')

    # Iterate through the Pareto front to find the lowest Test MSE
    for m in sym_model.pareto_front_:
        tree = m['tree']
        ypred_test = sym_model.evaluate_model(tree, results_sym["X_test"])
        
        print(np.isnan(results_sym["y_test"]).sum(), np.isnan(ypred_test).sum())

        mse_test = mean_squared_error(results_sym["y_test"], ypred_test)
        r2_test = r2_score(results_sym["y_test"], ypred_test)
        
        # Re-added: Store data for the aggregate Pareto plot
        pareto_data.append({
            "Embed_Type": embed_type,
            "Length": tree.Length,
            "Test_MSE": mse_test,
            "Test_R2": r2_test,
            "Equation": m['model']
        })
        
        # Track the absolute best performing equation
        if mse_test < best_sym_mse:
            best_sym_mse = mse_test
            best_sym_r2 = r2_test # Capture the corresponding R2
            best_len = tree.Length

    print(f"Found best equation for {embed_type}: {results_sym['best_model']}")
    # 2. Run export_and_visualise using the BEST length
    try:
        export_and_visualise(results_sym, ilen=best_len)
    except Exception as e:
        print(f"Could not run export_and_visualise for {embed_type}: {e}")

    # 3. Fit MLP Regressor
    mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
    mlp_results = fit_model_fn(
        feature=feature, 
        embed_type=embed_type, 
        estimator=mlp_model, 
        merged_dfs=merged_dfs
    )

    # 4. Fit Linear Regression
    linear_model = LinearRegression()
    linear_results = fit_model_fn(
        feature=feature, 
        embed_type=embed_type, 
        estimator=linear_model, 
        merged_dfs=merged_dfs
    )

    # Store top-level summary metrics
    summary_results.append({
        "Embed_Type": embed_type,
        
        "MLP_Test_R2": mlp_results["Test_R2"],
        "MLP_Test_MSE": mlp_results["Test_MSE"],
        "LR_Test_R2": linear_results["Test_R2"],
        "LR_Test_MSE": linear_results["Test_MSE"]
    })

# Convert to pandas DataFrames for easy viewing and plotting
df_summary = pd.DataFrame(summary_results)
df_pareto = pd.DataFrame(pareto_data)

print("\n--- Summary Results ---")
print(df_summary.to_string(index=False))