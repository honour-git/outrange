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
    fold_models_dict = {}

    print(f"Running {n_splits}-Fold cross validation...")
    for target in target_cols:
        target_oof = np.zeros(len(X))
        fold_models = []

        is_spin = (target == "launch_spin_rate")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y[target].iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y[target].iloc[val_idx]

            model = lgb.LGBMRegressor(
                n_estimators=2000 if is_spin else 1200,
                learning_rate=0.02,
                max_depth=7 if is_spin else 6,
                num_leaves=31 if is_spin else 20,
                min_child_samples=15 if is_spin else 20,
                colsample_bytree=0.7 if is_spin else 0.8,
                subsample=0.8,
                reg_alpha=0.1,
                reg_lambda=0.5,
                random_state=(42 + fold),
                verbosity=-1,
            )

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

            target_oof[val_idx] = model.predict(X_val)
            fold_models.append(model)

        oof_predictions[target] = target_oof
        rmse = np.sqrt(mean_squared_error(y[target], target_oof))
        cv_scores[target] = rmse
        print(f"Target: {target:20s} | OOF RMSE: {rmse:.4f}")

        fold_models_dict[target] = fold_models


    return fold_models_dict, oof_predictions, cv_scores
