import os

import joblib
import pandas as pd

# Specify the model file path
model_path = "./saved_models/decision_tree_best_model.pkl"

# Extract the model name (without the extension) for naming the output file
model_name = os.path.basename(model_path).replace(".pkl", "")

# Load the trained model
model = joblib.load(model_path)
print(f"Model '{model_name}' loaded successfully.")

# Load the new dataset (update the file path as needed)
new_data = pd.read_csv("./data/raw/test.csv")

# Ensure 'id' column exists in the new dataset for output purposes
if "id" not in new_data.columns:
    raise ValueError("The input dataset must contain an 'id' column.")

# Prepare the features for prediction (assuming the target column is not present in the new data)
X_new = new_data.drop(columns=["price"], errors="ignore")  # Drop 'price' if it exists

# Make predictions
predictions = model.predict(X_new)

# Create a DataFrame with 'id' and 'predicted' columns
output_df = pd.DataFrame({"id": new_data["id"], "predicted": predictions})

# Create the submissions directory if it does not exist
os.makedirs("./submissions", exist_ok=True)

# Save the predictions to a CSV file with the model name
output_file_path = f"./submissions/{model_name}_predictions.csv"
output_df.to_csv(output_file_path, index=False)
print(f"Predictions saved to '{output_file_path}'.")
