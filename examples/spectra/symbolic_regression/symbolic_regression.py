import os 

from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
import numpy as np
from sklearn.preprocessing import StandardScaler
from astroML.datasets import fetch_sdss_specgals

# Downloads and caches a robust subset of the MPA-JHU catalog
data = fetch_sdss_specgals()

import glob
import numpy as np
from datasets import load_dataset
from natsort import natsorted

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
        matched_df = matched_df[matched_df["mask_ratio"] <= 0.5]
        
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

def fit_pysr_fn(feature, embed_type, n_samples=10000):
    # Check if the feature is a flux early on so we can use it in data cleaning
    is_flux = "flux" in feature.lower()

    # 1. Clean the data: Drop missing values
    train_clean = merged_dfs["train"].dropna(subset=[feature])
    test_clean = merged_dfs["test"].dropna(subset=[feature])

    # --- NEW: Remove standard SDSS missing data flags (-9999.0) ---
    train_clean = train_clean[train_clean[feature] != -9999.0]
    test_clean = test_clean[test_clean[feature] != -9999.0]

    # --- UPDATED: Only enforce strictly positive values if we are doing a log10 transform ---
    if is_flux:
        train_clean = train_clean[train_clean[feature] > 0]
        test_clean = test_clean[test_clean[feature] > 0]

    print(f"\n--- Running PySR: {embed_type} ---")
    
    # 2. Extract arrays and handle 'cond+z' logic
    if embed_type == "cond+z":
        X_train = np.stack(train_clean["cond"].values)
        y_train = train_clean[feature].values

        X_test = np.stack(test_clean["cond"].values)
        y_test = test_clean[feature].values

        z_train = train_clean["z_x"].values.reshape(-1, 1)
        z_test = test_clean["z_x"].values.reshape(-1, 1)
        
        #z_scaler = StandardScaler()
        #z_train_scaled = z_scaler.fit_transform(z_train)
        #z_test_scaled = z_scaler.transform(z_test)

        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        # Append scaled redshift to the embedding
        X_train = np.hstack((X_train, z_train))
        X_test = np.hstack((X_test, z_test))
        
    else:
        X_train = np.stack(train_clean[embed_type].values)
        y_train = train_clean[feature].values

        X_test = np.stack(test_clean[embed_type].values)
        y_test = test_clean[feature].values

        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. Conditionally Log10 AND Standard Scale the Target Variable
    #y_scaler = None

    if is_flux:
        y_train = np.log10(y_train)
        y_test = np.log10(y_test)
        
    #y_scaler = StandardScaler()
    #y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    #y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
    # Standardize Features
    #x_scaler = StandardScaler()
    #X_train_scaled = x_scaler.fit_transform(X_train) 
    #X_test_scaled = x_scaler.transform(X_test) 

    # 4. Subsample the training data for PySR
    np.random.seed(42)
    # Ensure we don't try to sample more rows than exist
    actual_samples = min(n_samples, X_train.shape[0])
    sample_indices = np.random.choice(X_train.shape[0], size=actual_samples, replace=False)
    
    X_train = X_train[sample_indices]
    y_train = y_train[sample_indices]

    # 5. Train the PySR Model
    model = PySRRegressor(
        niterations=50,             
        binary_operators=["+", "*", "/", "-"],
        unary_operators=["square", "cube"],
        maxsize=20,                 
        model_selection="best",     
        random_state=42,
    )
    
    print(f"Training on {X_train.shape[0]} samples, {X_train.shape[1]} features...")
    model.fit(X_train, y_train)

    # Extract the best equation as a string 
    best_equation_str = str(model.sympy())

    # 6. Make predictions on the FULL test set
    predictions = model.predict(X_test)

    # 7. Evaluate
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # 8. Bootstrapping for Errors
    n_iterations = 1000
    bootstrapped_r2 = []
    bootstrapped_mse = []

    for _ in range(n_iterations):
        y_test_resampled, preds_resampled = resample(y_test, predictions)
        bootstrapped_r2.append(r2_score(y_test_resampled, preds_resampled))
        bootstrapped_mse.append(mean_squared_error(y_test_resampled, preds_resampled))

    # Calculate statistics 
    r2_mean = np.mean(bootstrapped_r2)
    r2_median = np.median(bootstrapped_r2)
    r2_std = np.std(bootstrapped_r2)
    r2_ci_lower = np.percentile(bootstrapped_r2, 2.5)
    r2_ci_upper = np.percentile(bootstrapped_r2, 97.5)

    mse_mean = np.mean(bootstrapped_mse)
    mse_median = np.median(bootstrapped_mse)
    mse_std = np.std(bootstrapped_mse)
    mse_ci_lower = np.percentile(bootstrapped_mse, 2.5)
    mse_ci_upper = np.percentile(bootstrapped_mse, 97.5)

    # Return the results including the equation
    return {
        "Feature": feature,
        "Embed_Type": embed_type,
        "R2_Mean": r2_mean,
        "R2_Median": r2_median,
        "R2_Std": r2_std,
        "R2_95%_CI_Lower": r2_ci_lower,
        "R2_95%_CI_Upper": r2_ci_upper,
        "MSE_Mean": mse_mean,
        "MSE_Median": mse_median,
        "MSE_Std": mse_std,
        "MSE_95%_CI_Lower": mse_ci_lower,
        "MSE_95%_CI_Upper": mse_ci_upper,
        "Best_Equation": best_equation_str 
    }

# --- Execution Loop ---
feature_to_predict = "h_alpha_flux" #"sfr_tot_p50" #"lgm_tot_p50" #"h_alpha_flux" #"lgm_tot_p50" # "sfr_tot_p50" #"lgm_tot_p50" #"h_alpha_flux"
embed_types = ["orig", "cond", "uncond", "cond+z"]

output_dir = "pysr_results"
os.makedirs(output_dir, exist_ok=True)

for embed in embed_types:
    # Run the function
    result_dict = fit_pysr_fn(feature=feature_to_predict, embed_type=embed)
    
    # Convert to DataFrame
    df_result = pd.DataFrame([result_dict])
    
    # Clean the filename (replace '+' to avoid file system issues)
    safe_embed_name = embed.replace("+", "_")
    save_path = os.path.join(output_dir, f"pysr_metrics_{feature_to_predict}_{safe_embed_name}.csv")
    
    # Save to file
    df_result.to_csv(save_path, index=False)
    print(f"Successfully saved results for '{embed}' to {save_path}")