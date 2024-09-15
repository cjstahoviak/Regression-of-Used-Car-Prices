import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
data = pd.read_csv("./data/raw/train.csv")
print("Data loaded successfully.")

# Separate features (X) and target (y). Assume 'price' is the target column.
X = data.drop(columns=["price"])
y = data["price"]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Create a preprocessing pipeline for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Combine preprocessing into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define hyperparameter grid for manual iteration
param_grid = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split into training and test sets.")

# Track the number of iterations and total combinations
total_iterations = (
    len(param_grid["max_depth"])
    * len(param_grid["min_samples_split"])
    * len(param_grid["min_samples_leaf"])
)
iteration_count = 0
best_score = float("inf")
best_params = None
best_model = None

# Start timing the entire process
start_time = time.time()

# Iterate over all combinations of hyperparameters
print("Starting hyperparameter search...")
for max_depth in param_grid["max_depth"]:
    for min_samples_split in param_grid["min_samples_split"]:
        for min_samples_leaf in param_grid["min_samples_leaf"]:
            iteration_start = time.time()  # Start timing for this iteration

            # Create a DecisionTreeRegressor model pipeline with the current set of hyperparameters
            model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        DecisionTreeRegressor(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42,
                        ),
                    ),
                ]
            )

            # Fit the model
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Evaluate the model's performance
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Check if this is the best model so far
            if rmse < best_score:
                best_score = rmse
                best_params = {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                }
                best_model = model  # Store the best model

            # Timing for this iteration
            iteration_end = time.time()
            iteration_time = iteration_end - iteration_start
            iteration_count += 1

            # Estimate the remaining time
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / iteration_count) * total_iterations
            remaining_time = estimated_total_time - elapsed_time

            # Output RMSE, time taken, and estimated remaining time
            print(
                f"Iteration {iteration_count}/{total_iterations}: RMSE = {rmse:.2f}, Time = {iteration_time:.2f} seconds, Estimated remaining time = {remaining_time:.2f} seconds"
            )

# Save the best model to a file
joblib.dump(best_model, "./saved_models/decision_tree_best_model.pkl")

print("\nBest Model Hyperparameters:", best_params)
print(f"Best Model RMSE: {best_score:.2f}")
print(f"Total time taken: {time.time() - start_time:.2f} seconds")
print("Best model saved successfully.")
