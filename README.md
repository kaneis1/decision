# Decision Transformer for Iterated Prisoner’s Dilemma (IPD)

This project uses a **Decision Transformer (DT)** to model IPD behavior and then applies **SHAP** to interpret the learned policy. The workflow is: train DT → run SHAP → aggregate explanations → interpret feature importance.

## 1) Use Decision Transformer to process IPD

### Data and feature design

The DT consumes a sequence of **Return‑to‑Go (R), State (S), Action (A)** tokens.  
State is built from game structure and decision history:

- **Game structure**: `risk`, `error`, `delta`, `infin`, `contin`
- **Payoff parameters**: `r`, `s`, `t`, `p`
- **Derived payoff indices**: `r1`, `r2`
- **Decision history**: `my.decision1`, `other.decision1`, … (up to `history_k`)

**Action**: `coop = 1`, `defect = 0`  
**Reward**: `my.payoff1`

### Train DT

```bash
python ipd/experiment_ipd.py
```

This trains DT with 5‑fold CV and saves the checkpoint to:
`models/ipd_dt_model/ipd_decision_transformer.pt`

## 2) Use SHAP to analyze the DT

### Run SHAP in batches

```bash
python ipd/analyze_ipd_with_shap.py --no_visualize
```

Defaults:
- **1000 trajectories** total
- **10 batches** of 100
- Fixed normalization from the checkpoint (`state_mean`, `state_std`)

### Aggregate feature importance

```bash
python ipd/aggregate_shap_results.py
```

Outputs:
- `Figures/ipd_shap/shap_feature_importance.csv`
- `Figures/ipd_shap/shap_feature_importance.png`

## 3) Decision‑Transformer Results (from `log.md`)

From `log.md`:
- **Train loss (mean)**: 0.1039  
- **Train loss (std)**: 0.0225  
- **Action error**: 0.1543  
- **5‑fold CV validation MSE**: 0.1337

Other models logged for comparison:
- **BERT DT**: Acc.t=1 0.842, Acc.t>1 0.926  
- **GPT2 model**: Acc.t=1 0.726, Acc.t>1 0.871  
- **SRA model**: Acc.t=1 0.734, Acc.t>1 0.820

## 4) SHAP Results and Feature Meaning

Top features by mean |SHAP| (from the latest run):

1. `other.decision1`  
2. `r2`  
3. `my.decision1`  
4. `p`  
5. `error`  
6. `r`  
7. `r1`  
8. `t`  
9. `delta`  
10. `infin`

### Feature meaning (from the payoff table image)

The IPD payoff matrix uses:
- **R**: reward for mutual cooperation  
- **S**: sucker’s payoff (you cooperate, other defects)  
- **T**: temptation payoff (you defect, other cooperates)  
- **P**: punishment for mutual defection  

Game structure variables:
- **risk**: stochasticity in payoffs  
- **error**: probability a player’s action is flipped  
- **delta**: continuation probability  
- **infin**: interaction is indefinite  
- **contin**: interaction happens in continuous time  

Derived indices:
- **r1**: normalized difference between `R` and `P`  
- **r2**: normalized difference between `R` and `S`

## Folder Architecture (Updated)

```
decision/
├── README.md
├── log.md
├── ipd/
│   ├── experiment_ipd.py               # Train DT on IPD CSV and save checkpoint
│   ├── analyze_ipd_with_shap.py        # Run SHAP on 1000 trajs (10 batches)
│   ├── aggregate_shap_results.py       # Aggregate SHAP results → CSV + plot
│   ├── experiment_ipd2.py              # (legacy) MO vector model
│   ├── experiment_ipd3.py              # (legacy) MO scalar model
│   └── experiment_ipd_gpt.py           # (legacy) GPT2 model
├── decision_transformer/
│   ├── models/
│   │   ├── decision_transformer.py     # DT architecture (S, A, R)
│   │   ├── decision_transformer2.py    # (legacy) MO vector
│   │   ├── decision_transformer3.py    # (legacy) MO scalar
│   │   └── ...
│   ├── training/                       # Training utilities
│   ├── evaluation/                     # Evaluation metrics
│   └── explainability/
│       ├── shap_explainer_ipd.py       # IPD SHAP pipeline (main)
│       └── __init__.py
├── data/
│   ├── all_data.csv                    # Main IPD dataset
│   └── all_data_subset.csv             # Smaller subset
├── models/
│   └── ipd_dt_model/
│       └── ipd_decision_transformer.pt # Saved checkpoint
└── Figures/
    └── ipd_shap/
        ├── shap_explanation_*.png      # Per‑trajectory plots
        ├── shap_results.pkl            # Saved SHAP values
        ├── shap_feature_importance.csv # Aggregated importance
        └── shap_feature_importance.png # Aggregated plot
```

## File Usage (Quick Guide)

- `ipd/experiment_ipd.py`: train DT and save checkpoint  
- `decision_transformer/explainability/shap_explainer_ipd.py`: SHAP pipeline  
- `ipd/analyze_ipd_with_shap.py`: run SHAP batches  
- `ipd/aggregate_shap_results.py`: aggregate feature importance  

## Installation

```bash
conda env create -f conda_env2.yml
conda activate <env_name>
pip install torch shap pandas numpy
```





