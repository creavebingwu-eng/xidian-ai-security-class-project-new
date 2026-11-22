import os
import numpy as np

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")

try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Define metric labels mapping for clarity
metric_labels = {
    "neff": "Effective Independent Ensemble Size",
    "avg_acc": "Average Accuracy",
}

# Iterate through datasets/configurations
for dataset_name, data in experiment_data.items():
    print(f"\nDataset: {dataset_name}")

    if "metrics" in data:
        metrics = data["metrics"]
        for metric_key, metric_values in metrics.items():
            # Ensure there are values to process
            if not isinstance(metric_values, list) or len(metric_values) == 0:
                continue

            # Get the final/best value (last in list usually implies final result in this context)
            final_value = metric_values[-1]

            # Format output based on known keys
            label = metric_labels.get(metric_key, metric_key)

            # Special formatting for accuracy vs size
            if "acc" in metric_key:
                print(f"{label}: {final_value:.4f}")
            else:
                print(f"{label}: {final_value:.4f}")
    else:
        print("No metrics found for this dataset.")
