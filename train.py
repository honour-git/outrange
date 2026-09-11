import os
import numpy as np
import pandas as pd
from src.features import extract_features, get_target_columns
from src.models import train_evaluate_lgbm

def main():
    os.makedirs("outputs", exist_ok=True)

    print("Loading raw data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    target_cols = get_target_columns()

    print("Extracting Kinematic features...")
    X_train = extract_features(train_df)
    y_train = train_df[target_cols]
    X_test = extract_features(test_df)

    print("Training LightGBM models...")
    fold_models_dict, oof_predictions, cv_scores = train_evaluate_lgbm(X_train, y_train)

    print("\nGenerating submission predictions...")
    submission = pd.DataFrame({"track_id": test_df["track_id"]})

    for target in target_cols:
        models = fold_models_dict[target]

        # Average predictions from CV fold models
        predictions = np.column_stack([model.predict(X_test) for model in models])

        submission[target] = np.mean(predictions, axis=1)

    submission.to_csv("outputs/submission.csv", index=False)
    print("Submissions saved successfully to \"outputs/submission.csv\"!")

if __name__ == "__main__":
    main()
