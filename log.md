experiment_ipd.py 
In `experiment_ipd.py`, the features contained in S (state), A (action), and R (reward) are as follows:

- **S (State):**  
  The state feature vector typically contains the recent history relevant to the agent's decision. For the iterated prisoner's dilemma (IPD) experiments, the state often includes:
  - The player's own previous actions over a fixed window (history).
  - The opponent's previous actions over a similar window.
  - The player's previous received rewards (payoffs), again over a history window.
  - Optionally, additional contextual information like timestep or episode progress.

  This encoding enables the model to reason about how past interactions and outcomes might influence the agent's current decision.

- **A (Action):**  
  The action feature is usually a single value indicating:
  - The action chosen by the agent at that timestep, e.g., cooperate (0) or defect (1) in the binary IPD setting.
  - For experiments considering multi-agent or extended action spaces, this might be one-hot encoded or expanded accordingly.

- **R (Reward):**  
  The reward feature is:
  - The scalar payoff the agent received after taking the action A in state S, usually dictated by the IPD payoff matrix for the round.

In summary, S encodes the agent's and opponent's previous decisions and payoffs over a defined history, A is the chosen action at the current step, and R is the scalar reward (payoff) resulting from that action.


experiment_ipd2.py, the features contained in S (state), A (action), MO and R (reward)

**MO (Modus Operandi):**  
The "MO" feature stands for "modus operandi," which represents the manner or strategy by which a player typically operates in the iterated prisoner's dilemma. In the context of this experiment, MO serves as a behavioral or psychological variable. It can encode aspects such as a player's overall tendency or bias (e.g., propensity to cooperate, defect, reciprocate, or employ specific strategies like tit-for-tat). Unlike state, which captures the explicit recent historical context, MO is intended to abstractly represent the underlying behavioral policy or operational "style" of the agent or observed human.

Including MO as a fourth modality in the model allows the decision transformer to reason not just from observable states, actions, and rewards, but also from inferred or given psychological traits, thereby enriching the model's ability to emulate or predict nuanced, human-like decision patterns.






log

-- run the `experiment_ipd.py` with --batch_size 8 --num_steps_per_iter 50 --pct_traj 0.1 --warmup_steps 1 parameters and get this results
training/train_loss_mean: 0.10394519180059433
training/train_loss_std: 0.022497849022460978
training/action_error: 0.15428277850151062

I have updated experiment_ipd.py with 5-fold instead of choose highest 20% reward as input, then the results is
Average validation MSE over 5 fold(s): 0.133717

I am updating model to have 4 modality and the model will be R1,s1,MO1,A1,R2,S2,MO2,A2.... etc
this is more like a human thinking way like we meet a question, and we saw reward we get first, then we see state and with states we can think about a strategy, at last we move.

bert_dt_model:
======================================================================
BERT IPD Eval → Acc.t=1 0.842, Acc.t>1 0.926, LL.t=1 -4357, LL.t>1 -3932, Cor-Time 0.967, RMSE-Time 0.060, Cor-Avg 0.000, RMSE-Avg 0.000

======================================================================
Test Set (10% held-out) - Performance Metrics
======================================================================
Metric               Value
----------------------------------------------------------------------
Acc. t = 1           0.842
Acc. t > 1           0.926
LL t = 1             -4357
LL t > 1             -3932
Cor-Time             0.967
Cor-Avg.             0.000
RMSE-Time            0.060
RMSE-Avg.            0.000======================================================================

gpt2_model:
======================================================================
GPT2 IPD Eval → Acc.t=1 0.726, Acc.t>1 0.871, LL.t=1 -1836, LL.t>1 -4299, Cor-Time 0.883, RMSE-Time 0.097, Cor-Avg 0.000, RMSE-Avg 0.000

======================================================================
Test Set (10% held-out) - Performance Metrics
======================================================================
Metric               Value
----------------------------------------------------------------------
Acc. t = 1           0.726
Acc. t > 1           0.871
LL t = 1             -1836
LL t > 1             -4299
Cor-Time             0.883
Cor-Avg.             0.000
RMSE-Time            0.097
RMSE-Avg.            0.000
======================================================================

main.py - Baseline LSTM/AR/LR Models Results
======================================================================
Recreated baseline models from "Predicting Human Cooperation" paper using main.py
Model Configuration: LSTM with 10 hidden nodes, 2 layers, 5-fold cross-validation

IPD (Iterated Prisoner's Dilemma) Task Results:
----------------------------------------------------------------------
Task Type: Binary classification (cooperate/defect)
Loss Function: BCEWithLogitsLoss (changed from MSE for binary classification)
Evaluation Metric: Accuracy (changed from MSE)

Models Compared:
- LSTM: LSTM-based sequence model
- AR: AutoRegressive (VAR) model
- LR: Logistic Regression model
- BERT: Transformer-based model with multi-head attention (NEW)

Key Results (from Figures/):
1. Action Prediction Accuracy (ipd_accuracy_nodes_10_layers_2.png):
   - Shows accuracy over 8 prediction time steps for LSTM, AR, LR, and BERT models
   - Accuracy calculated using binary classification (threshold at 0.5)
   - Results printed: lstm_acc, ar_acc, lr_acc, bert_acc
   - BERT model added with same architecture (10 hidden nodes, 2 layers, 2 attention heads)

2. Cooperation Rate Predictions (ipd_coop_nodes_10_layers_2.png):
   - Comparison of cooperation rates across all models vs human data
   - Shows mean cooperation rates with standard error bands
   - Models: LSTM (red), AR (blue), LR (green), BERT (magenta), Human (black)

3. Individual Model Cooperation Predictions:
   - ipd_lstm_coop_nodes_10_layers_2.png: LSTM predictions vs real cooperation rates
   - ipd_ar_coop_nodes_10_layers_2.png: AR predictions vs real cooperation rates
   - ipd_lr_coop_nodes_10_layers_2.png: LR predictions vs real cooperation rates

4. Training Loss (ipd_lstm_loss_nodes_10_layers_2.png):
   - LSTM training loss over batches during training
   - Training parameters: 10 epochs, batch size 100, learning rate 1e-2

IGT (Iowa Gambling Task) Results:
----------------------------------------------------------------------
Task Type: Multi-class classification (4 deck choices)
Loss Function: MSE (maintained for multi-class setting)
Evaluation Metric: MSE for action prediction, correct deck choice rates

Models Compared:
- LSTM: LSTM-based sequence model
- AR: AutoRegressive (VAR) model
- BERT: Transformer-based model with multi-head attention (NEW)

Key Results (from Figures/):
1. Action Prediction MSE (igt_mse_nodes_10_layers_2.png):
   - MSE comparison between LSTM, AR, and BERT models over 94 time steps
   - Results: LSTM MSE = 0.0149, AR MSE = 0.0202, BERT MSE (to be recorded)
   - LSTM performs better (lower MSE) than AR model

2. Action Prediction Accuracy (igt_accuracy_nodes_10_layers_2.png):
   - Accuracy comparison between LSTM, AR, and BERT models over 94 time steps
   - Results: LSTM accuracy, AR accuracy, BERT accuracy (to be recorded)
   - NEW: Added accuracy metric for IGT task

3. Correct Deck Choice Rates (igt_corr_nodes_10_layers_2.png):
   - Percentage of choosing better decks (C and D) over time
   - Comparison: LSTM (red), AR (blue), BERT (magenta), Human (black)
   - Shows learning curve for selecting advantageous decks

4. Individual Deck Choice Predictions (igt_pred_nodes_10_layers_2.png):
   - 2x2 subplot showing choice rates for each deck (A, B, C, D)
   - Comparison of LSTM, AR, BERT predictions vs human choices for all decks

4. Model-Specific Deck Predictions:
   - igt_ar_pred_nodes_10_layers_2.png: AR model predictions for each deck vs real
   - igt_lstm_pred_nodes_10_layers_2.png: LSTM predictions for each deck vs real
   - 4 subplots (one per deck) showing prediction accuracy

5. Correct Deck Predictions:
   - igt_ar_corr_nodes_10_layers_2.png: AR correct deck predictions vs real
   - igt_lstm_corr_nodes_10_layers_2.png: LSTM correct deck predictions vs real

6. Training Loss (igt_lstm_loss_nodes_10_layers_2.png):
   - LSTM training loss over batches
   - Training parameters: 100 epochs, batch size 10, learning rate 1e-2

Key Findings:
- IPD: Binary classification approach with BCEWithLogitsLoss and accuracy metrics
- IGT: LSTM outperforms AR model (MSE: 0.0149 vs 0.0202)
- Both tasks use 5-fold cross-validation for robust evaluation
- All models show learning curves that can be compared to human behavior patterns
- BERT model added to both IPD and IGT tasks for comparison
- IGT task now includes accuracy metric in addition to MSE

Data Preprocessing Comparison: main.py vs experiment_ipd_bert_dt.py
======================================================================
Comparison of data preprocessing methods between baseline (main.py) and Decision Transformer (experiment_ipd_bert_dt.py):

main.py - Baseline Data Preprocessing:
----------------------------------------------------------------------
1. Data Loading:
   - Reads from "./data/IPD/all_data.csv"
   - Filters for period == 10
   - Extracts trajectories: columns 9:27 (18 columns) → reshaped to (8258, 2, 9)
   - Extracts regression data: columns 3:51 (48 columns)
   - Random shuffle with fixed seed

2. Data Format:
   - Trajectories: (batch, 2, 9) where 2 = [my_action, other_action], 9 = time steps
   - Actions encoded: 0→2, then -1 (cooperate=-1, defect=0)
   - No state normalization
   - No RTG (Return-To-Go) computation
   - Direct sequence prediction: predict next action given history

3. Train/Test Split:
   - 5-fold cross-validation
   - 20% test, 80% train per fold
   - Simple random split

4. Model Input:
   - Direct action sequences: (batch, seq_len, action_dim)
   - For LSTM: (batch, 2, 9) transposed to (batch, 9, 2)
   - For BERT: same format, uses transformer architecture
   - No state features, only action history

experiment_ipd_bert_dt.py - Decision Transformer Data Preprocessing:
----------------------------------------------------------------------
1. Data Loading:
   - Reads from CSV (default: "decision/data/all_data.csv")
   - Groups by period (trajectory segmentation)
   - Extracts state features: ["risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p"]
   - Adds decision history: my.decision1-k, other.decision1-k (history_k parameter, default=3)
   - State dimension: 11 base + 2*history_k decision columns

2. Data Format:
   - Trajectories: dict with 'observations', 'actions', 'rewards', 'terminals', 'lens'
   - Observations: (T, state_dim) - normalized state features
   - Actions: (T, 1) - binary (coop=1, defect=0)
   - Rewards: (T,) - my.payoff1 values
   - Computes RTG (Return-To-Go): discounted cumulative sum of rewards

3. Train/Test Split:
   - 90% for 5-fold cross-validation
   - 10% held-out test set
   - Stratified by trajectory returns

4. Model Input:
   - Decision Transformer format: (R, s, a, R, s, a, ...)
   - States normalized: (states - state_mean) / state_std
   - RTG normalized by scale (95th percentile of returns)
   - Padded sequences to max_length=K (default 40)
   - Attention masks for valid timesteps

Key Differences:
----------------------------------------------------------------------
1. State Representation:
   - main.py: Only action sequences (2D: my_action, other_action)
   - experiment_ipd_bert_dt.py: Rich state features (11+ features) + action history

2. Normalization:
   - main.py: No normalization
   - experiment_ipd_bert_dt.py: State normalization + RTG scaling

3. Sequence Format:
   - main.py: Simple action sequences
   - experiment_ipd_bert_dt.py: Interleaved (R, s, a) tokens with RTG conditioning

4. History Handling:
   - main.py: Implicit in sequence (9 time steps)
   - experiment_ipd_bert_dt.py: Explicit history_k parameter (default 3) in state features

5. Reward Integration:
   - main.py: No explicit reward modeling
   - experiment_ipd_bert_dt.py: RTG (Return-To-Go) as conditioning signal

6. Padding Strategy:
   - main.py: No padding (fixed length sequences)
   - experiment_ipd_bert_dt.py: Left-padding with zeros/masks for variable-length trajectories

======================================================================

idea

-- use Shap to weight different features
-- maybe we need put history decision outside and put it as a special 
-- we can't simply calculate RTG using previous reward cuz previous reward represent the outcome that we choose
-- we want model to learn policy which not just game policy but also some psychology policy so that we can help model to understand why people choose this
-- put MO as reward, the metrics is MSE and R^2
-- the player decision based on the history payoff and history decision and game policy
-- adjust decision-transformer so that it can encode 4 more classes feature
done! -- use 5-fold instead of top 10% reward   
-- I want to create a model that give the state and reward, the model can give its prediction
-- use choice 13k as dataset
-- find out why dt can work with 3 order input as R,S,A
task
done! -- run the original one @experiment_ipd.py which set payoff as reward     
-- learn how to input MO in the this model as the fourth modality  
-- write another file use Gpt2-model to run the ipd task
done! -- instead of using top 10% reward, we use ramdonly 20% data to prove another 80% data accracy 
-- use `Predicting Human Cooperation` paper metric to find out and also need to learn how to take trajectories
done! -- divide trajectories by period

-- recreate lstm and all model in your paper
-- find another good dataset
-- 

question:

-- In the decision-transformer model # 
get predictions return_preds = self.predict_return(x[:,2]) # predict next return given state and action state_preds = self.predict_state(x[:,2]) # predict next state given state and action action_preds = self.predict_action(x[:,1]) # predict next action given state 

we get action_preds from current state which against markov chain that current action only based on current state, so we need to change it action_preds = self.predict_actgion(state_preds)