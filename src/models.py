import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Suppressing LightGBM depracation warning
warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def train_evaluate_lgbm(X: pd.DataFrame, y: pd.DataFrame, n_splits = 5) -> tuple:
    """Trains LightGBM regressors across target columns using K-Fold CV.

    Returns:
        models (dict): Trained model per target column (trained on 100% data)
        oof_predictions (pd.DataFrame): Out-of-fold predictions
        cv_scores (dict): RMSE per target column
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    target_cols = y.columns.to_list()

    oof_predictions = pd.DataFrame(0.0, index=X.index, columns=target_cols)
    cv_scores = {}
    final_models = {}

    print(f"Running {n_splits}-Fold cross validation...")
    for target in target_cols:
        target_oof = np.zeros(len(X))
        best_iterations = []

        is_spin = (target == "launch_spin_rate")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y[target].iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y[target].iloc[val_idx]

            model = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01 if is_spin else 0.02,
                max_depth=5,
                num_leaves=15,
                min_child_samples=30 if is_spin else 20,
                reg_alpha=1.0 if is_spin else 0.0,
                reg_lambda=1.0 if is_spin else 0.0,
                random_state=(42 + fold),
                verbosity=-1,
                colsample_bytree=0.8,
                subsample=0.8,
            )

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

            target_oof[val_idx] = model.predict(X_val)
            best_iterations.append(model.best_iteration_)

        oof_predictions[target] = target_oof
        rmse = np.sqrt(mean_squared_error(y[target], target_oof))
        cv_scores[target] = rmse
        print(f"Target: {target:20s} | OOF RMSE: {rmse:.4f}")

        # Capture average best_iteration per target from CV folds
        avg_best_iteration = int(np.mean(best_iterations))

        # Retrain final models on entire data set for test set predictions
        final_model = lgb.LGBMRegressor(
            n_estimators=avg_best_iteration,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=15,
            random_state=42,
            verbosity=-1,
        )
        final_model.fit(X, y[target])
        final_models[target] = final_model

    return final_models, oof_predictions, cv_scores
