import os
import numpy as np

# 1. Load the experiment_data.npy file
working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(data_path, allow_pickle=True).item()

# 2 & 3 & 4. Extract and print metrics
# The structure is: experiment_data['ensemble_audit']['metrics']
dataset_name = "ensemble_audit"
print(f"Dataset: {dataset_name}")

metrics = experiment_data[dataset_name]["metrics"]

# Effective Ensemble Size
if "effective_size" in metrics and len(metrics["effective_size"]) > 0:
    # The list has one element appended at the end of training/evaluation
    effective_size_val = metrics["effective_size"][-1]
    print(f"Effective Ensemble Size: {effective_size_val:.4f}")

# Mean Pairwise Failure Correlation
if "mean_correlation" in metrics and len(metrics["mean_correlation"]) > 0:
    mean_corr_val = metrics["mean_correlation"][-1]
    print(f"Mean Pairwise Failure Correlation: {mean_corr_val:.4f}")
