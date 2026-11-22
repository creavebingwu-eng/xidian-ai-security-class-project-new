The previous implementation failed primarily because natural language explanations were included outside the code block, causing a syntax error during execution. Additionally, the `run_experiment_configuration` function was incomplete, missing the handling for 'weighted_soup' configurations and lacking a proper main execution loop to train models, evaluate failures, and compute metrics for all ensemble types. I will provide a complete, self-contained script that implements synthetic data generation with shared latent biases, defines four distinct ensemble strategies (Independent Bagging, Same-Base Adapters, Different-Base Models, and Weighted Soups), and correctly calculates the effective independent ensemble size using the specified variance-based formula. The script will include robust device handling, training loops, evaluation routines, and save results to ensure full executability and adherence to experimental goals.

```python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

# Setup device and working directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Hyperparameters
N_SAMPLES_TRAIN = 5000
N_SAMPLES_TEST = 1000
FEATURE_DIM = 20
NUM_ENSEMBLE_MEMBERS = 10
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.01
EPSILON = 1e-8

def generate_synthetic_data(n_samples, feature_dim, seed_offset=0):
    """Generate synthetic data with a shared latent bias."""
    rng = np.random.RandomState(seed_offset + SEED)
    
    # Shared latent factor causing systematic errors
    z_shared = rng.randn(n_samples, 1) * 2.0
    
    # Individual noise/features
    X_noise = rng.randn(n_samples, feature_dim - 1) * 1.0
    
    # Combine: The first dimension is heavily influenced by the shared bias
    X = np.concatenate([z_shared, X_noise], axis=1)
    
    # Ground truth logic: Linear combination plus non-linear interaction with shared bias
    w_true = rng.randn(feature_dim) * 0.5
    y_logits = X @ w_true + 3.0 * z_shared[:, 0] * X[:, 1] # Interaction term amplifies shared bias impact
    probs = 1 / (1 + np.exp(-y_logits))
    y = (probs > 0.5).astype(np.float32)
    
    return X.astype(np.float32), y

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_model(model, loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}: training_loss = {avg_loss:.4f}')
    return model

def evaluate_failures(models, test_X_tensor, test_y_np):
    """
    Returns a matrix F where F[i,j] = 1 if model j fails on sample i, else 0.
    Also returns predictions for accuracy check.
    """
    n_models = len(models)
    n_samples = test_X_tensor.shape[0]
    failure_matrix = np.zeros((n_samples, n_models), dtype=np.int32)
    
    preds_list = []
    
    for j, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            logits = model(test_X_tensor)
            preds = (logits > 0.5).float().cpu().numpy()
            preds_list.append(preds)
            
            # Failure is when prediction != ground truth
            failures = (preds != test_y_np).astype(np.int32)
            failure_matrix[:, j] = failures
            
    return failure_matrix, np.array(preds_list)

def calculate_effective_size(failure_matrix, epsilon=EPSILON):
    """
    Calculate N_eff based on the variance of the average failure indicator.
    Formula from spec: N_nominal / (N_nominal * Var(avg_failure_indicator) + epsilon)
    
    Let f_i be the fraction of models failing on sample i.
    f_i = (1/M) * sum_j I_{ij}
    We compute Variance of f_i across samples.
    """
    M = failure_matrix.shape[1] # Nominal ensemble size
    # Average failure indicator per sample: shape (N,)
    avg_failure_per_sample = np.mean(failure_matrix, axis=1)
    
    # Variance of this average across samples
    var_avg_failure = np.var(avg_failure_per_sample)
    
    # Apply formula
    n_eff = M / (M * var_avg_failure + epsilon)
    return n_eff, var_avg_failure

def get_weighted_soup_model(base_model, perturbation_scale=0.05, num_members=1):
    """Create a single model representing a weighted soup by averaging weights of perturbed copies."""
    souped_model = copy.deepcopy(base_model)
    accumulated_state = {}
    
    # Initialize accumulated state
    for name, param in base_model.named_parameters():
        accumulated_state[name] = torch.zeros_like(param, device=device)
        
    for _ in range(num_members):
        variant = copy.deepcopy(base_model)
        with torch.no_grad():
            for param in variant.parameters():
                param.add_(torch.randn_like(param) * perturbation_scale)
        
        for name, param in variant.named_parameters():
            accumulated_state[name] += param.detach()
            
    # Average the accumulated weights
    with torch.no_grad():
        for name, param in souped_model.named_parameters():
            param.copy_(accumulated_state[name] / num_members)
            
    return souped_model

def run_experiment_configuration(config_name, X_train, y_train, X_test, y_test, config_type):
    print(f"\nRunning Configuration: {config_name}")
    models = []
    
    dataset_full = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader_full = DataLoader(dataset_full, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = nn.BCELoss()
    
    if config_type == 'independent':
        # Train K independent models on random subsets (bagging style but fully independent init/data split)
        for i in range(NUM_ENSEMBLE_MEMBERS):
            idx = np.random.choice(len(X_train), int(0.9*len(X_train)), replace=False)
            sub_X, sub_y = X_train[idx], y_train[idx]
            
            dataset_sub = TensorDataset(torch.from_numpy(sub_X), torch.from_numpy(sub_y))
            loader_sub = DataLoader(dataset_sub, batch_size=BATCH_SIZE, shuffle=True)
            
            model = SimpleClassifier(FEATURE_DIM).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            
            model = train_model(model, loader_sub, criterion, optimizer)
            models.append(model)
            
    elif config_type == 'same_base_adapters':
        # Simulate adapters: Share base weights, only vary final layer or slight perturbation
        base_model = SimpleClassifier(FEATURE_DIM).to(device)
        optimizer = optim.Adam(base_model.parameters(), lr=LR)
        base_model = train_model(base_model, loader_full, criterion, optimizer)
        
        # Create variants by copying base and adding small noise/adapters
        for i in range(NUM_ENSEMBLE_MEMBERS):
            variant = copy.deepcopy(base_model)
            with torch.no_grad():
                for param in variant.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            models.append(variant)
            
    elif config_type == 'different_base':
        # Train models with different random initializations but same data/objective
        for i in range(NUM_ENSEMBLE_MEMBERS):
            model = SimpleClassifier(FEATURE_DIM).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            model = train_model(model, loader_full, criterion, optimizer)
            models.append(model)

    elif config_type == 'weighted_soup':
        # Train one base model, then create members by perturbing it significantly before souping? 
        # Actually, usually soup IS the ensemble member in some contexts,