"""Small, type-safe starter experiment for auditing ensemble failure dependence."""

import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 17
NUM_MODELS = 5
TRAIN_SIZE = 700
TEST_SIZE = 500
EPOCHS = 18
BATCH_SIZE = 64


def make_dataset(size, rng):
    """Generate labels from noisy features without encoding labels in X."""
    x = rng.normal(size=(size, 2)).astype(np.float32)
    logits = 1.2 * x[:, 0] - 0.8 * x[:, 1]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(size) < probabilities).astype(np.int64)
    return x, y


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 2)
        )

    def forward(self, x):
        return self.layers(x)


def train_member(x, y, seed, device):
    torch.manual_seed(seed)
    model = Classifier().to(device)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model


def predict(model, x, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=256)
    predictions = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def summarize_failures(failure_matrix):
    """Return empirical metrics; undefined edge cases are reported as None."""
    failures = np.asarray(failure_matrix, dtype=np.float64)
    if failures.ndim != 2 or failures.shape[0] == 0 or failures.shape[1] < 2:
        raise ValueError("failure_matrix must have shape (samples, >=2 members)")

    num_samples, num_members = failures.shape
    member_rates = failures.mean(axis=0)
    p_hat = float(member_rates.mean())
    pair_correlations = []
    for left in range(num_members):
        for right in range(left + 1, num_members):
            a, b = failures[:, left], failures[:, right]
            if np.std(a) > 0 and np.std(b) > 0:
                pair_correlations.append(float(np.corrcoef(a, b)[0, 1]))

    mean_pair_correlation = (
        float(np.mean(pair_correlations)) if pair_correlations else None
    )
    ensemble_failure_rate = failures.mean(axis=1)
    observed_variance = float(np.var(ensemble_failure_rate))
    independent_variance = p_hat * (1.0 - p_hat)
    n_eff_var = (
        independent_variance / observed_variance
        if independent_variance > 0 and observed_variance > 0
        else None
    )

    p_all_wrong = float(np.all(failures == 1, axis=1).mean())
    n_eff_all = (
        float(np.log(p_all_wrong) / np.log(p_hat))
        if 0 < p_all_wrong < 1 and 0 < p_hat < 1
        else None
    )

    return {
        "num_samples": num_samples,
        "num_members": num_members,
        "member_accuracy": (1.0 - member_rates).tolist(),
        "mean_failure_rate": p_hat,
        "mean_pair_failure_correlation": mean_pair_correlation,
        "n_eff_var": n_eff_var,
        "p_all_wrong": p_all_wrong,
        "n_eff_all": n_eff_all,
        "undefined_metric_note": (
            "None means the empirical data do not identify this metric "
            "(for example, zero variance or no observed all-member failures)."
        ),
    }


def run():
    rng = np.random.default_rng(SEED)
    train_x, train_y = make_dataset(TRAIN_SIZE, rng)
    test_x, test_y = make_dataset(TEST_SIZE, rng)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    failures_by_condition = {}
    for condition in ("shared_data", "bootstrap_data"):
        member_failures = []
        for member_idx in range(NUM_MODELS):
            if condition == "shared_data":
                member_x, member_y = train_x, train_y
            else:
                sample_idx = rng.integers(0, TRAIN_SIZE, size=TRAIN_SIZE)
                member_x, member_y = train_x[sample_idx], train_y[sample_idx]

            model = train_member(
                member_x, member_y, SEED + member_idx + 1, device
            )
            member_failures.append((predict(model, test_x, device) != test_y).astype(np.int8))

        failures_by_condition[condition] = np.stack(member_failures, axis=1)

    results = {
        condition: summarize_failures(matrix)
        for condition, matrix in failures_by_condition.items()
    }

    # Exercise mathematically undefined cases explicitly; these are not crashes.
    edge_cases = {
        "all_correct": summarize_failures(np.zeros((8, 3), dtype=np.int8)),
        "constant_member": summarize_failures(
            np.column_stack((np.zeros(8), [0, 1] * 4, [1, 0] * 4))
        ),
    }
    output = {"conditions": results, "metric_edge_cases": edge_cases}
    os.makedirs("working", exist_ok=True)
    with open(os.path.join("working", "experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)

    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
