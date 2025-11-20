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

bert_model:
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
RMSE-Avg.            0.000
======================================================================

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

task
done! -- run the original one @experiment_ipd.py which set payoff as reward     
-- learn how to input MO in the this model as the fourth modality  
-- write another file use Gpt2-model to run the ipd task
done! -- instead of using top 10% reward, we use ramdonly 20% data to prove another 80% data accracy 
-- use `Predicting Human Cooperation` paper metric to find out and also need to learn how to take trajectories
done! -- divide trajectories by period

question:

-- In the decision-transformer model # 
get predictions return_preds = self.predict_return(x[:,2]) # predict next return given state and action state_preds = self.predict_state(x[:,2]) # predict next state given state and action action_preds = self.predict_action(x[:,1]) # predict next action given state 

we get action_preds from current state which against markov chain that current action only based on current state, so we need to change it action_preds = self.predict_actgion(state_preds)