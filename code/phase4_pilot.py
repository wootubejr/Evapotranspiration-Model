"""
Phase 4 pilot: train/evaluate GBDT, RF, and Logistic Regression on the Phase 3
zonal-stat feature table (phase3_pilot_parcel_level.csv).

RandomForest is tuned FIRST, per feature set, before the holdout/CV evaluation --
the pilot run showed RF badly overfitting on ET-only (train_F1=0.987, CV F1=0.423),
so a small RandomizedSearchCV over regularization params (max_depth, min_samples_leaf,
min_samples_split, max_features) runs before eval_holdout, and the tuned
hyperparameters are used everywhere downstream. Logistic and GBDT are left as-is --
GBDT's gap was small and didn't trigger CV; Logistic has few enough parameters that
overfitting isn't the concern there.

For the PILOT (not the final national run), 5-fold CV is SKIPPED BY DEFAULT to save
time -- it only runs for a given model if that model's holdout train/test F1 gap
exceeds CV_TRIGGER_GAP (0.2), since a big gap is exactly the case where you can't
trust the holdout split alone and need CV to tell you if it's real overfitting or
just an unlucky split. Set FORCE_CV = True (below) to always run CV on every model/
feature-set combo regardless of the gap -- flip this on for the final Spain-wide run.

Two feature sets, matching your ET-only vs ET+crop ablation:
  - ET-only: the 12 zonal-stat columns (mean/std of ET_inst, ET24h, LE, EF,
    NDVI, NDWI across the July date trajectory)
  - ET+crop: ET-only plus one-hot encoded parc_producto (declared crop code)

For every evaluation, a summary table is printed with accuracy, misclassification
rate, balanced accuracy, AUC, and F1 per model, alongside the confusion matrix.

Class imbalance: sample_weight='balanced' (inverse class frequency) for all
three models, via sklearn's compute_sample_weight -- chosen over SMOTE/
oversampling for now since it's leakage-free across CV folds and costs nothing
extra at national scale. If class-weighted holdout F1 on the irrigated class
comes back weak, that's the trigger to revisit oversampling, not a default.

NOTE: I can't run this without your actual phase3_pilot_parcel_level.csv --
syntax/logic-checked against a synthetic table with the same schema, but not
against your real numbers. Run it and paste back the printed output.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score,
)

FEATURE_TABLE = r"C:\VSCODE_Research\ET_Model\phase3_national\phase3_national_parcel_level_ALL.csv"
RANDOM_STATE = 42
N_FOLDS = 5
CV_TRIGGER_GAP = 0.2   # holdout train_F1 - test_F1 above this triggers CV for that model
FORCE_CV = True        # True = always run CV on every model/feature-set (final national run)

# Pre-tuned RF params from the completed ET-only RandomizedSearchCV run -- set to None
# to make main() call tune_random_forest() again instead of reusing these.
RF_TUNED_PARAMS = {
    "n_estimators": 300,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": None,
    "max_depth": 12,
}

# Which feature set(s) to actually process this run -- subset of {"ET-only", "ET+crop"}.
# ET-only tuning/holdout/CV already complete, so this defaults to ET+crop only, reusing
# RF_TUNED_PARAMS above rather than re-tuning.
FEATURE_SETS_TO_RUN = ["ET+crop"]

# Where fitted models + train/test predictions & probabilities get saved (one .joblib
# per model per feature set, from eval_holdout only -- not per CV fold). This is what
# lets you regenerate ROC curves, confusion matrices, etc. later without refitting.
MODEL_OUTPUT_DIR = r"C:\VSCODE_Research\ET_Model\phase4_national\model_artifacts"

# RF regularization search space -- targets the exact overfitting the pilot showed
# (train_F1=0.987 vs CV F1=0.423 on ET-only). RandomizedSearchCV rather than an
# exhaustive grid so this stays fast regardless of how large the space gets.
RF_PARAM_DIST = {
    "n_estimators": [200, 300, 400],       
    "max_depth": [3, 5, 8, 12],       
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "min_samples_split": [2, 5, 10, 20],
    "max_features": ["sqrt", "log2", 0.5, None],
}
RF_SEARCH_N_ITER = 40  # number of random param combos to try
# Controls how many candidate/fold combinations RandomizedSearchCV runs simultaneously.
# -1 (all cores) at BOTH this level AND inside the RandomForestClassifier passed into it
# is a classic nested-parallelism trap -- each of N parallel search workers separately
# tries to parallelize its own tree-building across all N cores again, multiplying peak
# memory (each worker holds its own partial ensemble in memory at once). Fixed here:
# the search parallelizes (N_JOBS), the inner estimator does not (forced to 1 below).
# If you're still hitting memory limits at national scale, lower this further (e.g. 2)
# to directly bound how many full model fits exist in memory at once -- the most
# reliable lever regardless of whether nesting was the actual culprit.
N_JOBS = 8
THRESHOLD_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)  # probability cutoffs to report per model


def print_threshold_sweep(y_true, y_proba, model_name: str):
    """Precision/recall/accuracy at several probability cutoffs, computed post-hoc from
    predict_proba -- no retraining needed. Default .predict() uses 0.5; a lower
    threshold trades precision for recall (good for 'find every irrigated parcel'
    use cases), a higher threshold trades recall for precision (good for tallying/monitoring use cases). Pick per use case, or report a couple."""
    rows = []
    for t in THRESHOLD_SWEEP:
        pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        rows.append({
            "threshold": t,
            "precision_irrigated": precision_score(y_true, pred, zero_division=0),
            "recall_irrigated": recall_score(y_true, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "accuracy": acc,
            "misclass_rate": 1 - acc,
        })
    sweep_df = pd.DataFrame(rows).set_index("threshold").round(3)
    print(f"  {model_name} threshold sweep (test set, default .predict() = 0.5):\n"
          f"{sweep_df.to_string()}")


def load_feature_table():
    df = pd.read_csv(FEATURE_TABLE)
    et_cols = [c for c in df.columns if c.endswith("_mean_mean") or c.endswith("_mean_std")]
    print(f"{len(df)} parcels loaded, {len(et_cols)} ET-only features: {et_cols}")

    before = len(df)
    df = df.dropna(subset=et_cols)
    print(f"Dropped {before - len(df)} parcels with missing ET features (tile/date "
          f"failures or all-nodata) -- kept {len(df)}")
    print(f"Class balance: {df['irrigated_label'].sum()} irrigated / "
          f"{(df['irrigated_label'] == 0).sum()} rainfed")
    return df, et_cols


def build_feature_sets(df: pd.DataFrame, et_cols: list[str]):
    X_et = df[et_cols].copy()
    # NOTE: one-hot vocabulary is built from whichever crop codes appear in THIS
    # dataframe -- fine for offline pilot evaluation, but at national scale you'd
    # want a fixed crop-code vocabulary (or target encoding) so rare/unseen crops
    # don't silently create high-cardinality or train/test column mismatches.
    crop_dummies = pd.get_dummies(df["parc_producto"].astype(str), prefix="crop")
    X_et_crop = pd.concat([X_et, crop_dummies], axis=1)
    y = df["irrigated_label"].values
    return {"ET-only": X_et, "ET+crop": X_et_crop}, y


def get_models(rf_params: dict = None):
    rf_kwargs = dict(n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    if rf_params:
        rf_kwargs.update(rf_params)
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(**rf_kwargs),
        "GBDT": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def tune_random_forest(X: pd.DataFrame, y: np.ndarray, feature_set_name: str) -> dict:
    """RandomizedSearchCV over RF_PARAM_DIST, scored on F1 via the same stratified
    N_FOLDS CV used elsewhere, sample-weighted 'balanced' for the imbalance. Returns
    the best param dict to feed into get_models(rf_params=...) downstream."""
    print(f"\n=== TUNING RandomForest ({feature_set_name}) -- {RF_SEARCH_N_ITER} candidates x {N_FOLDS} folds "
          f"= {RF_SEARCH_N_ITER * N_FOLDS} total fits ===")
    t0 = time.time()
    sw_full = compute_sample_weight("balanced", y)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=2),  # bounded, not N_JOBS -- see note above.
        # N_JOBS(8) x inner(2) = 16 concurrent threads, still comfortably under this
        # machine's 20 logical processors (i9-13900H: 6 P-cores hyperthreaded=12 threads
        # + 8 E-cores=8 threads). Different from the original bug: that was UNBOUNDED on
        # both sides (-1 x -1), which could demand far more threads than exist at once.
        # This is a deliberate, bounded step up from fully-serial inner fits (n_jobs=1),
        # not a return to the original oversubscription problem. Watch memory when you
        # first restart with this -- if it climbs toward the ceiling again, drop back to
        # n_jobs=1 here rather than push forward.
        param_distributions=RF_PARAM_DIST,
        n_iter=RF_SEARCH_N_ITER,
        scoring="f1",
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=False,  # we refit ourselves downstream via get_models(rf_params=...)
        verbose=2,    # prints one line per fit with timing -- without this, this step
                      # (your longest one at national scale) runs completely silently
    )
    search.fit(X, y, sample_weight=sw_full)
    print(f"Tuning finished in {time.time()-t0:.0f}s")

    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1 (mean across {N_FOLDS} folds, at search time): {search.best_score_:.3f}")
    return search.best_params_


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": acc,
        "misclass_rate": 1 - acc,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
    }


def _fit_predict(name, model, X_train, y_train, X_test, sw_train):
    """Logistic gets scaled features (distance-based/regularized linear model needs
    it); tree ensembles (RF, GBDT) don't -- scaling is a no-op for them but wastes
    time/complexity, so skip it. Returns predictions AND positive-class probabilities
    (needed for AUC) for both train and test."""
    if name == "LogisticRegression":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    model.fit(X_train, y_train, sample_weight=sw_train)
    train_pred, test_pred = model.predict(X_train), model.predict(X_test)
    train_proba, test_proba = model.predict_proba(X_train)[:, 1], model.predict_proba(X_test)[:, 1]
    return model, train_pred, test_pred, train_proba, test_proba


def eval_holdout(X: pd.DataFrame, y: np.ndarray, feature_set_name: str, rf_params: dict = None):
    """Returns (summary_df, models_to_cv) -- models_to_cv is the list of model names
    whose train/test F1 gap exceeded CV_TRIGGER_GAP (or all models, if FORCE_CV)."""
    print(f"\n=== HOLDOUT SPLIT ({feature_set_name}) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    sw_train = compute_sample_weight("balanced", y_train)

    summary_rows = []
    models_to_cv = []
    for name, model in get_models(rf_params=rf_params).items():
        t0 = time.time()
        print(f"  Training {name}...")
        model, train_pred, test_pred, train_proba, test_proba = _fit_predict(
            name, model, X_train, y_train, X_test, sw_train
        )
        print(f"  {name} trained in {time.time()-t0:.0f}s")

        test_metrics = compute_metrics(y_test, test_pred, test_proba)
        train_f1 = f1_score(y_train, train_pred)
        gap = train_f1 - test_metrics["F1"]
        triggered = gap > CV_TRIGGER_GAP
        flag = f"  <-- F1 gap={gap:.3f} > {CV_TRIGGER_GAP} -- will run CV to check" if triggered else ""

        print(f"{name:20s} train_F1={train_f1:.3f} test_F1={test_metrics['F1']:.3f}{flag}")
        print(f"  confusion matrix (test, rows=true[rainfed,irrigated], cols=pred):\n"
              f"{confusion_matrix(y_test, test_pred)}")
        print_threshold_sweep(y_test, test_proba, name)

        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        artifact_path = os.path.join(
            MODEL_OUTPUT_DIR, f"{feature_set_name.replace('+', '_plus_')}_{name}_holdout.joblib"
        )
        joblib.dump({
            "model": model,
            "y_train": y_train, "y_test": y_test,
            "train_pred": train_pred, "test_pred": test_pred,
            "train_proba": train_proba, "test_proba": test_proba,
        }, artifact_path)
        print(f"  Saved model + predictions -> {artifact_path}")

        summary_rows.append({"model": name, **test_metrics})
        if triggered or FORCE_CV:
            models_to_cv.append(name)

        if name == "GBDT":
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print(f"  GBDT feature importances (top 10):\n{importances.head(10).to_string()}")

    summary_df = pd.DataFrame(summary_rows).set_index("model").round(3)
    print(f"\n--- Test-set metrics summary ({feature_set_name}) ---")
    print(summary_df.to_string())
    return summary_df, models_to_cv


def eval_stratified_cv(X: pd.DataFrame, y: np.ndarray, feature_set_name: str, model_names: list[str], rf_params: dict = None) -> pd.DataFrame:
    print(f"\n=== STRATIFIED {N_FOLDS}-FOLD CV ({feature_set_name}) -- {model_names} ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    summary_rows = []
    for name in model_names:
        t0 = time.time()
        print(f"  {name}: fold ", end="", flush=True)
        fold_metrics = {"accuracy": [], "misclass_rate": [], "balanced_accuracy": [], "AUC": [], "F1": []}
        for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            print(f"{fold_i+1}...", end="", flush=True)
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            sw = compute_sample_weight("balanced", y_tr)

            model = get_models(rf_params=rf_params)[name]  # fresh, unfit model per fold
            _, _, pred, _, proba = _fit_predict(name, model, X_tr, y_tr, X_te, sw)
            m = compute_metrics(y_te, pred, proba)
            for k in fold_metrics:
                fold_metrics[k].append(m[k])
        print(f" done in {time.time()-t0:.0f}s")

        row = {"model": name}
        for k, vals in fold_metrics.items():
            row[k] = f"{np.mean(vals):.3f}\u00b1{np.std(vals):.3f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("model")
    print(f"--- {N_FOLDS}-fold CV metrics summary ({feature_set_name}) ---")
    print(summary_df.to_string())
    return summary_df


def main():
    df, et_cols = load_feature_table()
    feature_sets, y = build_feature_sets(df, et_cols)

    for name in FEATURE_SETS_TO_RUN:
        X = feature_sets[name]
        if RF_TUNED_PARAMS is not None:
            print(f"\n[{name}] Reusing pre-tuned RF params (skipping tune_random_forest): "
                  f"{RF_TUNED_PARAMS}")
            rf_params = RF_TUNED_PARAMS
        else:
            rf_params = tune_random_forest(X, y, name)

        _, models_to_cv = eval_holdout(X, y, name, rf_params=rf_params)
        if models_to_cv:
            eval_stratified_cv(X, y, name, model_names=models_to_cv, rf_params=rf_params)
        else:
            print(f"\n[{name}] No model's F1 gap exceeded {CV_TRIGGER_GAP} -- skipping "
                  f"CV for this feature set. Set FORCE_CV=True to run it anyway.")


if __name__ == "__main__":
    main()

