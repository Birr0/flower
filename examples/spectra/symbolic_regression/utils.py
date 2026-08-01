import os
import csv 

import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from pyoperon.sklearn import SymbolicRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

import processing_fun

# ---------------------------------------------------------
# 3. FUNCTIONS
# ---------------------------------------------------------
symbols = 'add,sub,mul,div,exp,log1p,logabs,constant,variable,pow' 
N = 10000

def fit_sym_fn(feature, embed_type, merged_dfs, seed):
    is_flux = "flux" in feature.lower()
    #feat_root = feature.split("_P50")[0]
    #is_mass = feat_root == "LGM_FIB"

    train_clean = merged_dfs["train"].dropna(subset=[feature])
    test_clean = merged_dfs["test"].dropna(subset=[feature])

    train_clean = train_clean[(train_clean[feature] != -9999.0)] #& (train_clean["z_x"] >= 0.075)
    test_clean = test_clean[(test_clean[feature] != -9999.0)] # & (test_clean["z_x"] >= 0.075)

    if is_flux:
        train_clean = train_clean[train_clean[feature] > 0]
        test_clean = test_clean[test_clean[feature] > 0]
    #elif is_mass:
        #train_clean = train_clean[(train_clean[feature] > 9.5) & (train_clean[feature] < 11.5)]
        #test_clean = test_clean[(test_clean[feature] > 9.5) & (test_clean[feature] < 11.5)]
    
    if embed_type == "cond+z":
        X_train = np.stack(train_clean["cond"].values)
        X_test = np.stack(test_clean["cond"].values)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = train_clean[feature].values
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
        X_train = train_clean["z_x"].values.reshape(-1, 1)
        X_test = test_clean["z_x"].values.reshape(-1, 1)

    else:
        X_train = np.stack(train_clean[embed_type].values)
        X_test = np.stack(test_clean[embed_type].values)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = train_clean[feature].values
        y_test = test_clean[feature].values

        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

    if is_flux:
        y_train = np.log10(y_train)
        y_test = np.log10(y_test)
        
    y_train = y_train.reshape(-1, 1).flatten() 
    y_test = y_test.reshape(-1, 1).flatten()
    
    X_train_scaled = X_train 
    X_test_scaled = X_test 

    # --- SEED INTEGRATION HERE ---
    n_samples = min(N, len(X_train_scaled))
    rng = np.random.default_rng(seed=seed)
    random_indices = rng.choice(len(X_train_scaled), size=n_samples, replace=False)
    
    X_train_fit = X_train_scaled[random_indices]
    y_train_fit = y_train[random_indices]

    '''
    reg = SymbolicRegressor(
        allowed_symbols=symbols,
        offspring_generator='basic',
        optimizer_iterations=500,
        max_length=25,
        initialization_method='btc',
        n_threads=8,
        objectives=['mse', 'length'],
        epsilon=1e-6,
        random_state=seed,  
        reinserter='keep-best',
        max_evaluations=int(1e7),
        symbolic_mode=False,
        tournament_size=3,
    )
    '''

    objectives=['mse', 'length']
    reg = SymbolicRegressor(
        allowed_symbols=symbols,
        offspring_generator='basic',
        optimizer_iterations=1000,
        max_length=25,
        initialization_method='btc',
        n_threads=8,
        objectives=objectives,
        epsilon=1e-6,
        random_state=seed,
        reinserter='keep-best',
        max_evaluations=int(1e7),
        symbolic_mode=False,
        tournament_size=3
    )
    
    reg.fit(X_train_fit, y_train_fit)
    best_model_str = reg.get_model_string(reg.model_, precision=12)

    return {
        "Feature": feature,
        "Embed_Type": embed_type,
        "Seed": seed,
        "Model": reg,                   
        "X_train": X_train_fit,         
        "y_train": y_train_fit,         
        "X_test": X_test_scaled,        
        "y_test": y_test,
        "best_model": best_model_str             
    }

def fit_model_fn(feature, embed_type, estimator, merged_dfs):
    is_flux = "flux" in feature.lower()
    train_clean = merged_dfs["train"].dropna(subset=[feature])
    test_clean = merged_dfs["test"].dropna(subset=[feature])
    train_clean = train_clean[train_clean[feature] != -9999.0]
    test_clean = test_clean[test_clean[feature] != -9999.0]
    if is_flux:
        train_clean = train_clean[train_clean[feature] > 0]
        test_clean = test_clean[test_clean[feature] > 0]
        
    if embed_type == "cond+z":
        X_train = np.stack(train_clean["cond"].values)
        X_test = np.stack(test_clean["cond"].values)
        y_train = train_clean[feature].values
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
        X_train = train_clean["z_x"].values.reshape(-1, 1)
        X_test = test_clean["z_x"].values.reshape(-1, 1)
    else:
        X_train = np.stack(train_clean[embed_type].values)
        X_test = np.stack(test_clean[embed_type].values)
        y_train = train_clean[feature].values
        y_test = test_clean[feature].values
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

    if is_flux:
        y_train = np.log10(y_train)
        y_test = np.log10(y_test)
        
    y_train = y_train.reshape(-1, 1).flatten() 
    y_test = y_test.reshape(-1, 1).flatten()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    estimator.fit(X_train_scaled, y_train)
    
    train_predictions = estimator.predict(X_train_scaled)
    test_predictions = estimator.predict(X_test_scaled)

    return {
        "Train_R2": r2_score(y_train, train_predictions),
        "Test_R2": r2_score(y_test, test_predictions),
        "Train_MSE": mean_squared_error(y_train, train_predictions),
        "Test_MSE": mean_squared_error(y_test, test_predictions)
    }

def export_and_visualise(fit_results, out_dir):
    reg = fit_results["Model"]
    X_train, y_train = fit_results["X_train"], fit_results["y_train"]
    X_test, y_test = fit_results["X_test"], fit_results["y_test"]

    os.makedirs(out_dir, exist_ok=True)
    outname = f'{out_dir}/fun.csv'
    
    with open(outname, "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Equation", "Length", "MSE_train", "MSE_test", "R2_train", "R2_test"])

        for m in reg.pareto_front_:
            tree = m['tree']
            ypred_train = reg.evaluate_model(tree, X_train)
            ypred_test = reg.evaluate_model(tree, X_test)
            
            writer.writerow([
                m['model'], tree.Length, 
                mean_squared_error(y_train, ypred_train), mean_squared_error(y_test, ypred_test), 
                r2_score(y_train, ypred_train), r2_score(y_test, ypred_test)
            ])

            np.savetxt(f'{out_dir}/train_{tree.Length}.csv', np.vstack([X_train.T, y_train, ypred_train]).T)
            np.savetxt(f'{out_dir}/test_{tree.Length}.csv', np.vstack([X_test.T, y_test, ypred_test]).T)

    '''
    names = [f'x{i}' for i in range(X_train.shape[1])]
    try:
        fig, ax = processing_fun.plot_pareto(out_dir, names, ilen=ilen, objective='MSE')
        fig, axs = processing_fun.prediction_plots(out_dir, ilen=ilen)
    except Exception as e:
        print(f"Plotting failed (expected if headless node): {e}")
    '''
