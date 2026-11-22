## Name

effective_failure_size

## Title

Adapters Are Not Independent Experts: Measuring the Effective Failure Size of Applied Deep Learning Ensembles

## Short Hypothesis

Deployed deep learning systems increasingly use ensembles of fine-tuned foundation-model variants, adapters, checkpoints, or model soups under the assumption that functional variety implies independent errors. We hypothesize that these variants often fail together because they share pretraining representations, optimization biases, training data, spurious features, and evaluation distributions; therefore their effective number of independent members is far smaller than the nominal ensemble size, especially on rare slices and distribution shifts. Average accuracy or calibration cannot answer this question, because two ensembles can have identical mean error but very different joint failure structure. Directly measuring failure dependence is the minimal and best setting for testing whether ensemble breadth translates into deployment reliability, and it can be audited with public models and modest parameter-efficient fine-tuning.

## Related Work

Classical ensemble learning and deep ensembles show that averaging predictions can improve accuracy and uncertainty estimation, e.g., Dietterich on ensemble methods, Lakshminarayanan et al. on deep ensembles, and Ovadia et al. on uncertainty in deep learning. Diversity measures in ensembles usually study output disagreement, decision-boundary variety, or error correlation as proxies for average-case improvement. Recent work such as LEVI proposes layer-wise ensembles of different views to improve fine-tuning generalization, and heterogeneous ensemble uncertainty metrics have been used in scientific foundation models to track prediction error. LLM juries, model soups, LoRA adapters, and checkpoint ensembles are also increasingly used in applied systems. Our proposal is not another ensemble architecture. It introduces a deployment-oriented audit of joint failure dependence, defines effective failure size from the variance of ensemble failure counts and the probability that all members fail, and tests the negative result that nominal diversity from adapters, checkpoints, or same-family foundation models may provide little independent tail-risk reduction. This distinguishes the work from prior ensemble diversity literature, which rarely focuses on rare catastrophic failures, shared foundation-model biases, or parameter-efficient variant ensembles.

## Abstract

Deep ensembles are a default recipe for making applied machine learning safer, but modern deployments often form diverse systems from fine-tuned adapters, checkpoints, model soups, and prompt variants derived from the same foundation models, data, and objectives. This creates a tempting illusion: more models should mean more independent chances to avoid a costly error. We ask how many independent errors an applied ensemble actually contains. Our hypothesis is that failure events across such variants are strongly positively correlated because they share pretraining representations, optimization biases, spurious features, and evaluation distributions; therefore the effective failure size is far smaller than the nominal ensemble size, especially on rare slices and distribution shifts. We propose a simple audit that treats each member's wrong or unsafe prediction as a binary failure, estimates pairwise failure correlation, fits the distribution of the number of failing members, and computes an effective independent ensemble size from the variance of the average failure indicator and from the probability that all members fail. We will instantiate the audit on vision, language-safety, and healthcare-shift benchmarks using public backbones and parameter-efficient fine-tuning, comparing same-base adapters, different-base models, bagged data, different losses, weight-averaged soups, and voting ensembles. The expected negative result is that nominal architectural or adapter diversity often yields little tail-risk reduction, while only interventions that explicitly decorrelate failure modes increase effective independence. The contribution is a reusable diagnostic that separates average-case ensemble gains from deployment-reliable independence.

## Experiments

- Construct model banks across three feasible applied domains: biomedical image classification using MedMNIST PathMNIST/DermaMNIST with scanner or corruption shifts; text safety using BOLD/ToxiGen-style labels with small open language models such as Pythia-410M/1B/2.8B and DistilBERT/RoBERTa-small; and general vision OOD using WILDS FMoW or CIFAR-100-C with ViT/CLIP/ConvNeXt backbones. For each domain, train or collect m=5-10 members under conditions: same base with different seeds, same base with LoRA adapters on different related tasks, same base with LoRA adapters on the same task but different rank/dropout/optimizer, different base families, weight-averaged model soups, and standard voting ensembles. Use LoRA or small full fine-tuning so the study fits academic-lab compute.
- Define binary failure indicators f_i(x)=1 when member i is incorrect or unsafe, where unsafe means a false negative on critical classes such as malignancy, toxicity, or safety-violating content. Compute pairwise phi/Pearson correlations among failure indicators, mean pairwise correlation, and the distribution of K(x)=sum_i f_i(x). Estimate effective failure size in two complementary ways: n_eff_var = p_hat(1-p_hat) / Var(K/m), where p_hat is the average marginal failure rate, and n_eff_all = log(P_all_wrong) / log(p_hat), the number of independent members that would reproduce the observed probability that all members fail. Use bootstrap confidence intervals and permutation nulls that preserve marginal failure rates to test whether observed all-fail rates exceed independence.
- Evaluate under i.i.d. test data and deployment-stress conditions: natural site/scanner shifts, corruption severity, temporal or dialect shift for text, and rare subgroups. Report marginal accuracy, macro F1, worst-group F1, expected calibration error, average failure rate, pairwise failure correlation, n_eff_var, n_eff_all, P_all_wrong, majority-vote error, and selective risk-coverage curves when abstaining if at least k members fail. The key predicted pattern is that ensembles look better on average i.i.d. metrics but their effective failure size collapses on rare or shifted examples.
- Test simple interventions that should, if anything, create independent failures: bagging or subsampling training data, different objective functions such as cross-entropy, focal loss, group-DRO, and adversarial perturbation training, LEVI-style layer-wise mixing, full fine-tuning versus LoRA, different base pretraining families, and explicit failure-decorrelating regularization. For the regularizer, add a calibration-set penalty lambda times the squared covariance or correlation of soft failure indicators, using differentiable surrogates such as sigmoid(loss_i - tau) or smoothed hinge losses on member margins. Also test greedy ensemble selection that maximizes estimated n_eff_all subject to a minimum average accuracy. Measure whether tail risk decreases more than average risk.
- Perform a small human or secondary-label audit on 200-500 examples sampled from four categories: all members wrong, majority wrong, only one member wrong, and permuted independent-null cases. Use available expert labels, second annotators, or trusted auxiliary labels where possible to estimate how much observed failure correlation is caused by label noise versus genuine shared model failure. Recompute the main dependence metrics after removing or reweighting suspected noisy labels.
- Release an audit toolkit that takes as input member predictions and gold labels and outputs pairwise failure correlation, n_eff_var, n_eff_all, all-fail probability versus independence, bootstrap uncertainty, and plots of failure-count distributions. Include a cross-domain meta-analysis asking whether same-base adapters, same-data checkpoints, or same-pretraining families produce higher failure correlation than genuinely different data, objectives, or base models.

## Risk Factors And Limitations

- Tail events such as all members failing may be rare, making estimates noisy; we will mitigate this with bootstrap intervals, beta-binomial or permutation models, and by pooling related conditions while reporting domain-specific results.
- Label noise can artificially inflate failure correlation; the human/secondary-label audit is intended to separate shared data errors from shared model errors.
- The proposal does not claim that ensembles are useless; ensembles may still improve average accuracy or calibration. The focused claim is that nominal diversity can overstate independent tail-risk reduction.
- Some foundation models are too large for some labs; we will restrict the main study to small open models, LoRA fine-tuning, and public checkpoints, with optional larger-model replication if compute allows.
- Defining unsafe failure is domain-specific and partly subjective; we will use multiple failure definitions, including plain error, critical-class false negative, and high-margin confident error.
- Measured correlation may reflect shared evaluation distribution rather than shared internal failure mechanisms; intervention experiments and failure-example clustering will help distinguish correlation from causal sources of dependence.
- The effective failure size estimators assume approximate exchangeability within an ensemble condition; we will report diagnostics for heterogeneous marginal accuracies and use stratified estimators when members differ substantially.

## Code To Potentially Use

Use the following code as context for your experiments:

```python
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

```

