# Decision Transformer for Iterated Prisoner's Dilemma

This project implements Decision Transformer models to predict human decision-making behavior in the Iterated Prisoner's Dilemma (IPD) game. The models learn from human experimental data to understand and predict cooperative and defection strategies.

## Overview

The project explores different architectures for modeling human decision-making in IPD:

1. **Basic Decision Transformer** (`experiment_ipd.py`): Uses State (S), Action (A), and Reward (R) modalities
2. **Extended Decision Transformer** (`experiment_ipd2.py`): Incorporates Modus Operandi (MO) as a fourth modality to capture behavioral patterns
3. **GPT2-based Model** (`experiment_ipd_gpt.py`): Uses GPT2 architecture to model decision history sequences

## Features

### State (S) Features
The state feature vector includes:
- Game parameters: `risk`, `error`, `delta`, `infin`, `contin`
- Payoff parameters: `r1`, `r2`, `r`, `s`, `t`, `p`
- Decision history: `my.decision1`, `other.decision1` (and potentially more history)

### Action (A) Features
- Binary action: Cooperate (1) or Defect (0)

### Reward (R) Features
- Scalar payoff received after taking action A in state S (`my.payoff1`)

### Modus Operandi (MO) Features
The MO feature represents the behavioral strategy or "operational style" of a player:
- Base features: `r1`, `r2`, `risk`, `error`, `delta`, `r1*delta`, `r2*delta`, `infin`, `contin`, `delta*infin`
- Decision history: `my.decision1`, `other.decision1` (and additional history based on `history_k`)

The sequence format for the extended model is: **R₁, S₁, MO₁, A₁, R₂, S₂, MO₂, A₂, ...**

This structure reflects human thinking: we first observe the reward, then see the state, think about our strategy (MO), and finally make a decision.

## Project Structure

```
ipd/
├── experiment_ipd.py          # Basic Decision Transformer (S, A, R)
├── experiment_ipd2.py         # Extended Decision Transformer (S, A, R, MO)
├── experiment_ipd_gpt.py      # GPT2-based model for IPD
├── log.md                     # Project log with ideas, tasks, and results
├── data/
│   ├── all_data.csv          # Main dataset
│   └── all_data_subset.csv   # Subset of data
├── decision_transformer/
│   ├── models/               # Model architectures
│   ├── training/             # Training scripts
│   └── evaluation/           # Evaluation metrics
└── models/                   # Saved model checkpoints
```

## Installation

1. Create a conda environment from the provided environment files:
```bash
conda env create -f conda_env.yml
# or
conda env create -f conda_env2.yml
```

2. Activate the environment:
```bash
conda activate <env_name>
```

3. Install additional dependencies if needed:
```bash
pip install torch wandb pandas numpy
```

## Usage

### Basic Decision Transformer

Train the basic model with State, Action, and Reward:

```bash
python experiment_ipd.py \
    --csv_path data/all_data.csv \
    --batch_size 8 \
    --num_steps_per_iter 50 \
    --pct_traj 0.1 \
    --warmup_steps 1 \
    --num_folds 5
```

### Extended Decision Transformer with MO

Train the extended model with Modus Operandi:

```bash
python experiment_ipd2.py \
    --csv_path data/all_data.csv \
    --batch_size 32 \
    --num_steps_per_iter 1000 \
    --num_folds 5 \
    --history_k 1
```

### GPT2-based Model

Train the GPT2 model for decision history prediction:

```bash
python experiment_ipd_gpt.py \
    --csv_path data/all_data.csv \
    --batch_size 32 \
    --embed_dim 64 \
    --n_layer 2 \
    --n_head 1 \
    --learning_rate 1e-4 \
    --max_iters 10 \
    --num_steps_per_iter 1000 \
    --num_folds 5 \
    --history_k 5
```

## Evaluation Metrics

The models are evaluated using the following metrics (aligned with "Predicting Human Cooperation" paper):

- **Accuracy at t=1**: Accuracy of predictions at the first timestep
- **Accuracy at t>1**: Accuracy of predictions for timesteps after the first
- **Log-Likelihood (LL)**: Log-likelihood of predictions at t=1 and t>1
- **Correlation-Time**: Correlation between predicted and actual actions across time
- **Correlation-Average**: Average correlation across trajectories
- **RMSE-Time**: Root Mean Squared Error across time
- **RMSE-Average**: Average RMSE across trajectories

For the extended model with MO, additional metrics include:
- **MO MSE**: Mean Squared Error for Modus Operandi prediction
- **MO R²**: R-squared for Modus Operandi prediction

## Key Design Decisions

1. **5-Fold Cross-Validation**: Instead of using top 10% reward trajectories, the project uses 5-fold cross-validation for robust evaluation.

2. **Trajectory Division**: Trajectories are divided by period to maintain temporal structure.

3. **Modus Operandi**: The MO feature captures psychological/behavioral patterns beyond observable states, enabling the model to learn human-like decision strategies.

4. **Sequence Format**: The extended model uses R, S, MO, A sequence to mirror human decision-making: observe reward → see state → think strategy → act.

## Results

Example results from `experiment_ipd.py`:
- Training loss: ~0.104 (mean), ~0.022 (std)
- Action error: ~0.154
- Average validation MSE (5-fold): ~0.134

## Weights & Biases Integration

The project supports logging to Weights & Biases. To enable:

```bash
python experiment_ipd2.py \
    --log_to_wandb \
    --wandb_api_key <your_api_key> \
    ...
```

Or set the environment variable:
```bash
export WANDB_API_KEY=<your_api_key>
```

## Future Work & Ideas

From `log.md`:

- [ ] Use SHAP to weight different features
- [ ] Put history decision outside as a special feature
- [ ] Adjust RTG calculation to account for previous reward outcomes
- [ ] Learn both game policy and psychological policy
- [ ] Create a model that predicts actions given state and reward

## Questions & Notes

- The Decision Transformer's action prediction should be based on predicted state (following Markov chain principles), not just current state
- Previous rewards represent outcomes of chosen actions, which should be considered in RTG calculation

## Data Format

The CSV file should contain the following columns:
- `data_id`: Unique identifier for each trajectory
- `period`: Period number within a trajectory
- `my.decision`: Player's decision (coop/defect)
- `my.decision1`, `other.decision1`: Previous decisions
- `my.payoff1`: Payoff received
- Game parameters: `risk`, `error`, `delta`, `infin`, `contin`, `r1`, `r2`, `r`, `s`, `t`, `p`





