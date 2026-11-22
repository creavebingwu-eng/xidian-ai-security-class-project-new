To implement the "Effective Failure Size" audit, I will construct a controlled experiment using synthetic data where we simulate an ensemble of classifiers trained on overlapping subsets of the same underlying distribution. This allows us to precisely measure pairwise failure correlations and compute the effective independent ensemble size (K_eff) based on the variance of the average failure indicator. I will use simple logistic regression models as baseline classifiers, generating predictions on a held-out test set to derive binary failure indicators (0 for correct, 1 for wrong). By comparing ensembles with high overlap in training data versus those with low overlap, we can demonstrate how correlation reduces K_eff below nominal N. The code will calculate rho_avg (average pairwise correlation of failures), then derive K_eff = N / (1 + (N-1)*rho_avg), along with tail probability estimates. Finally, it will visualize the relationship between training subset overlap and K_eff, confirming that without decorrelated errors, added models yield diminishing returns in independence.

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import warnings
warnings.filterwarnings('ignore')

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def generate_synthetic_data(n_samples=5000, n_features=20, n_classes=3, random_state=42):
    """Generates synthetic classification data."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state
    )
    # Normalize features for stability
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_normalized = (X - mean) / std
    return X_normalized, y

def compute_failure_indicators(predictions, labels):
    """Computes binary failure indicators: 1 if prediction is wrong, 0 if correct."""
    return (predictions != labels).astype(int)

def estimate_rho_avg(failures_matrix):
    """
    Estimates average pairwise Pearson correlation of failure indicators.
    failures_matrix: shape (n_members, n_samples) where each row is a member's failure vector.
    """
    n_members, _ = failures_matrix.shape
    correlations = []
    for i in range(n_members):
        for j in range(i + 1, n_members):
            corr = np.corrcoef(failures_matrix[i], failures_matrix[j])[0, 1]
            # Handle NaNs from constant vectors (all zeros or all ones)
            if not np.isnan(corr):
                correlations.append(corr)
    
    if len(correlations) == 0:
        return 0.0
    return np.mean(correlations)

def compute_k_eff_variance(nominal_size, rho_avg):
    """
    Computes Effective Independent Ensemble Size using variance-based estimator.
    K_eff = N / (1 + (N-1) * rho_avg)
    """
    denominator = 1 + (nominal_size - 1) * rho_avg
    if denominator <= 0:
        return nominal_size # Cap at nominal if negative due to extreme correlation artifacts
    return nominal_size / denominator

def compute_k_eff_tail(failures_matrix):
    """
    Computes K_eff using tail-probability method: inverse log-probability of unanimous failure.
    If P(all fail) is p_all, then K_eff approx = ln(p_all) / ln(p_single_mean) assuming independence? 
    Actually, simpler metric: K_eff_tail = 1 / P(all fail) is too crude. 
    Using: K_eff = ln(P(unanimous failure)) / ln(P(single failure average)) -- this assumes independence structure.
    Let's stick to the variance definition for primary metric, but report unanimity rate.
    """
    n_members, n_samples = failures_matrix.shape
    # Check if ALL members failed on each sample
    all_failed = np.all(failures_matrix == 1, axis=0)
    p_unanimous = np.mean(all_failed)
    
    # Mean single failure rate
    p_single_mean = np.mean(failures_matrix)
    
    # Estimate K_eff based on ratio of probabilities under independence assumption vs observed
    # Under independence: P(all fail) = p^k. Observed p_all > p^k implies k < nominal.
    # Solve for k: k = ln(p_all) / ln(p_single_mean)
    if p_single_mean <= 0 or p_single_mean >= 1 or p_unanimous <= 0:
        return float('nan')
        
    try:
        k_est = np.log(p_unanimous) / np.log(p_single_mean)
        return max(1.0, min(k_est, n_members))
    except Exception:
        return float('nan')

def run_experiment():
    print("Generating synthetic data...")
    X, y = generate_synthetic_data()
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Experiment parameters
    ensemble_sizes = [5, 10, 20]
    overlaps = [0.1, 0.5, 1.0] # Fraction of shared training data
    
    results = []
    plot_data = {'overlap': [], 'nominal_n': [], 'k_eff_var': [], 'k_eff_tail': [], 'rho_avg': []}
    
    print("Running ensemble simulations...")
    
    for N in ensemble_sizes:
        for overlap_frac in overlaps:
            failures_list = []
            
            # Determine unique pool size and shared portion
            total_train_size = len(X_train_full)
            shared_size = int(total_train_size * overlap_frac)
            unique_share_per_member = max(1, int((total_train_size - shared_size) / N))
            
            # We need enough data for unique parts
            # Adjust if necessary
            if shared_size + (unique_share_per_member * N) > total_train_size:
                unique_share_per_member = max(1, (total_train_size - shared_size) // N)
            
            rng = np.random.RandomState(42)
            
            # Fixed shared subset indices
            shared_indices = rng.choice(total_train_size, size=shared_size, replace=False)
            
            for i in range(N):
                # Sample unique indices for this member
                remaining_indices = np.setdiff1d(np.arange(total_train_size), shared_indices)
                if len(remaining_indices) < unique_share_per_member:
                    # Fallback if not enough data, reuse some with replacement or just use shared
                    unique_indices = rng.choice(remaining_indices, size=max(1, len(remaining_indices)), replace=False)
                else:
                    unique_indices = rng.choice(remaining_indices, size=unique_share_per_member, replace=False)
                
                member_train_idx = np.concatenate([shared_indices, unique_indices])
                X_train_i = X_train_full[member_train_idx]
                y_train_i = y_train_full[member_train_idx]
                
                # Train Logistic Regression
                model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
                model.fit(X_train_i, y_train_i)
                
                # Predict on test set
                preds = model.predict(X_test)
                failures = compute_failure_indicators(preds, y_test)
                failures_list.append(failures)
            
            failures_matrix = np.vstack(failures_list)
            
            # Metrics
            rho_avg = estimate_rho_avg(failures_matrix)
            k_eff_var = compute_k_eff_variance(N, rho_avg)
            k_eff_tail = compute_k_eff_tail(failures_matrix)
            
            results.append({
                'N': N,
                'overlap': overlap_frac,
                'rho_avg': rho_avg,
                'k_eff_var': k_eff_var,
                'k_eff_tail': k_eff_tail
            })
            
            plot_data['overlap'].append(overlap_frac)
            plot_data['nominal_n'].append(N)
            plot_data['k_eff_var'].append(k_eff_var)
            plot_data['k_eff_tail'].append(k_eff_tail)
            plot_data['rho_avg'].append(rho_avg)
            
            print(f"N={N}, Overlap={overlap_frac:.2f}: Rho={rho_avg:.3f}, K_eff(var)={k_eff_var:.2f}, K_eff(tail)={k_eff_tail:.2f}")

    # Save Data
    experiment_data = {
        'ensemble_audit': {
            'metrics': {
                'rho_avg': plot_data['rho_avg'],
                'k_eff_var': plot_data['k_eff_var'],
                'k_eff_tail': plot_data['k_eff_tail']
            },
            'config': {
                'overlaps': plot_data['overlap'],
                'nominal