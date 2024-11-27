import numpy as np
from catboost import CatBoostRegressor
import sage
from sage import plotting
from df_loader import load_df
import matplotlib.pyplot as plt

class BaseImputer:
    def __init__(self, model, num_features):
        """
        Base imputer for PermutationEstimator.

        Args:
            model: Trained machine learning model.
            num_features: Number of features in the dataset.
        """
        self.model = model
        self.num_groups = num_features

    def __call__(self, X, mask):
        """
        Predict with features masked.

        Args:
            X: Input data (numpy array), shape (batch_size, num_features).
            mask: Boolean mask array where True indicates the feature is included,
                  shape (batch_size, num_features).

        Returns:
            Predictions as numpy array.
        """
        # Ensure X and mask have compatible shapes
        if len(mask.shape) == 2:
            masked_X = X * mask
        elif len(mask.shape) == 3:
            masked_X = np.einsum('ij,ijk->ik', X, mask)

        return self.model.predict(masked_X)

def main():
    # Load data
    feature_names, X_train, X_val, X_test, Y_train, Y_val = load_df(
        train_path="data/train.csv", test_path="data/test.csv"
    )

    # Train CatBoost Regressor
    print("Training the CatBoost Regressor...")
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        verbose=200,
    )
    model.fit(X_train, Y_train, eval_set=(X_val, Y_val), use_best_model=True)

    # Apply SAGE
    print("Applying SAGE...")

    # Initialize Base Imputer
    num_features = X_train.shape[1]
    imputer = BaseImputer(model, num_features)

    # Initialize PermutationEstimator
    estimator = sage.PermutationEstimator(imputer, loss="mse")

    # Compute SAGE values
    explanation = estimator(X_val, Y_val, batch_size=100)

    # Display all features
    feature_importances = explanation.values
    sorted_indices = np.argsort(-feature_importances)
    print("\nFeature Importances (All):")
    for idx in sorted_indices:
        print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")

    plot(
        explanation,
        feature_names=feature_names,
        sort_features=True,
        max_features=len(feature_names),
        title="Feature Importance via SAGE",
        figsize=(12, 8),
        color="tab:green",
        title_size=18,
        label_size=14,
        tick_size=12,
        return_fig=False
    )
    plt.show()

if __name__ == "__main__":
    main()
