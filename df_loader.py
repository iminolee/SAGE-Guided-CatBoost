import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_df(train_path="data/train.csv", test_path="data/test.csv"):
    # Load the data
    train_df = pd.read_csv(train_path, encoding="cp949")
    test_df = pd.read_csv(test_path, encoding="cp949")

    # Drop the 'ID' column
    if "ID" in train_df.columns:
        train_df = train_df.drop(columns=["ID"])
    if "ID" in test_df.columns:
        test_df = test_df.drop(columns=["ID"])

    # Define target column
    target_column = "Temperature"

    # Separate features and target in train_df
    X_train_df = train_df.drop(columns=[target_column])
    Y_train = train_df[target_column].values

    # Split train_df into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_df.values, Y_train, test_size=0.2, random_state=42
    )

    # Align test_df columns with train_df columns
    X_test = test_df[X_train_df.columns.tolist()].values

    # Get feature names
    feature_names = X_train_df.columns.tolist()

    return feature_names, X_train, X_val, X_test, Y_train, Y_val
