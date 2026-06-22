import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, r2_score
from sklearn.utils import resample

def bootstrap_summary(scores):
    """
    Returns summary statistics for bootstrap scores.
    """
    scores = np.array(scores)

    mean_s = np.mean(scores)
    median_s = np.median(scores)

    ci_95 = (
        np.percentile(scores, 2.5),
        np.percentile(scores, 97.5)
    )

    ci_68 = (
        np.percentile(scores, 16),
        np.percentile(scores, 84)
    )

    err_95 = (ci_95[1] - ci_95[0]) / 2

    return {
        "mean": mean_s,
        "median": median_s,
        "ci_95": ci_95,
        "ci_68": ci_68,
        "err_95": err_95,
        "scores": scores
    }

def print_bootstrap_stats(name, stats):
    """
    Prints bootstrap summary statistics.
    """
    print(f"{name} Mean: {stats['mean']:.4f}")
    print(f"{name} Median: {stats['median']:.4f}")
    print(f"   95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")
    print(f"   68% CI: [{stats['ci_68'][0]:.4f}, {stats['ci_68'][1]:.4f}]")
    print(f"   Err 95: [{stats['err_95']:.4f}]")


def evaluate_embedding_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    n_iterations=1000,
    random_state=42
):
    """
    Trains a classifier on training data, evaluates it on test data,
    and performs bootstrap testing on the test F1 and accuracy scores.
    """

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    pipeline = Pipeline([
        ("mlp", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)

    boot_f1 = []
    boot_acc = []

    indices = np.arange(len(y_test))
    rng = np.random.RandomState(random_state)

    for _ in range(n_iterations):
        resample_idx = resample(
            indices,
            replace=True,
            random_state=rng
        )

        y_true_boot = y_test[resample_idx]
        y_pred_boot = y_pred[resample_idx]

        boot_f1.append(
            f1_score(
                y_true_boot,
                y_pred_boot,
                average="weighted"
            )
        )

        boot_acc.append(
            accuracy_score(
                y_true_boot,
                y_pred_boot
            )
        )

    boot_f1_stats = bootstrap_summary(boot_f1)
    boot_acc_stats = bootstrap_summary(boot_acc)

    print("--- Classification Results ---")
    print(f"Test F1: {f1:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print()
    print_bootstrap_stats("Bootstrap F1", boot_f1_stats)
    print()
    print_bootstrap_stats("Bootstrap Accuracy", boot_acc_stats)
    print("-" * 40)

    return {
        "test_f1": f1,
        "test_accuracy": acc,
        "bootstrap_f1": boot_f1_stats,
        "bootstrap_accuracy": boot_acc_stats,
        "pipeline": pipeline
    }

def evaluate_embedding_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    n_iterations=1000,
    random_state=42
):
    """
    Trains a regressor on training data, evaluates it on test data,
    and performs bootstrap testing on the test R2 score.
    """

    # Reshape if data is 1D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    pipeline = Pipeline([
        ("mlp", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Original test evaluation
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # Bootstrap evaluation on test set
    boot_r2 = []
    indices = np.arange(len(y_test))

    rng = np.random.RandomState(random_state)

    for _ in range(n_iterations):
        resample_idx = resample(
            indices,
            replace=True,
            random_state=rng
        )

        boot_score = r2_score(
            y_test[resample_idx],
            y_pred[resample_idx]
        )

        boot_r2.append(boot_score)

    boot_stats = bootstrap_summary(boot_r2)

    # Report results
    print("--- Regression Results ---")
    print(f"Test R2: {r2:.4f}")
    print_bootstrap_stats("Bootstrap R2", boot_stats)
    print("-" * 40)

    return {
        "test_r2": r2,
        "bootstrap": boot_stats,
        "pipeline": pipeline
    }

def prepare_data(ds, embed_type, factor):
    # 1. Prepare Training and Testing Data
    X_train = np.array(ds["train"][embed_type])
    y_train = np.array(ds["train"][factor]).ravel()
    X_test = np.array(ds["test"][embed_type])
    y_test = np.array(ds["test"][factor]).ravel()
    return X_train, y_train, X_test, y_test