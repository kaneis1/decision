# Pipeline Explanation: experiment_splendor_dt.py

## Overview
This script trains a **Decision Transformer** model to learn how to play the Splendor board game by learning from expert/random game trajectories.

---

## Pipeline Stages

### **Stage 1: Environment Setup** (Lines 45-451)

**Components:**
- `Card`: Represents a game card (level, color, cost, points)
- `Noble`: Represents a noble card with requirements
- `SplendorEnv`: Gym environment implementing the Splendor game rules

**Key Features:**
- Game state tracking: player gems, gem supply, player cards, nobles, board state
- Action space: Buy cards, reserve cards, buy reserved cards, take gems
- Reward calculation: Positive reward for winning in fewer turns, negative for exceeding 60 turns
- Legal action filtering: Only allows valid moves based on game rules

---

### **Stage 2: Data Loading** (Lines 497-529)

**`load_card_data(csv_path)`:**
- Loads card definitions from CSV file
- Parses card properties (tier, color, cost, points)
- Creates card objects and organizes them into 3 tiers
- Creates noble cards with predefined requirements
- Returns `card_supply` (3-tier structure) and `nobles` list

---

### **Stage 3: State Encoding** (Lines 453-589)

**Purpose:** Convert complex game state dictionary into fixed-size vector

**`flatten_state(state)`:**
- Converts nested state dict to flat dictionary format
- Extracts: player gems, gem supply, reserved cards, score, nobles, player cards, board cards

**`encode_state(state, hasher=None)`:**
- **Two encoding modes:**
  1. **Feature Hashing** (if `hasher` provided): Uses sklearn's FeatureHasher for sparse states
  2. **Manual Encoding** (default): Creates fixed-size vector with:
     - Player gems (6 values: 5 colors + joker)
     - Gem supply (6 values)
     - Player score (1 value)
     - Card counts (2 values: owned + reserved)
     - Player cards summary: color distribution (5) + total points (1)
     - Reserved cards summary: color distribution (5) + total points (1)
     - Board cards: 12 cards × 8 features (level, color, points, 5 cost values) = 96 features
     - **Total: ~118 features** (without hashing)

---

### **Stage 4: Trajectory Generation** (Lines 592-669)

**`generate_trajectories(env, card_supply, nobles, num_episodes, agent_type, hasher)`:**
- Generates game episodes by playing Splendor with a specified agent
- **Current agents:** `'random'` (random legal actions)
- For each episode:
  1. Resets environment
  2. While game not done:
     - Gets valid legal actions
     - Selects action (random or future: DQN)
     - Encodes current state
     - Records: (state_encoded, action_idx, reward)
     - Steps environment
  3. Creates trajectory dictionary:
     ```python
     {
         'observations': np.array([state_vectors...]),
         'actions': np.array([action_indices...]),
         'rewards': np.array([rewards...]),
         'terminals': np.array([0,0,...,1]),  # 1 at episode end
         'lens': episode_length
     }
     ```

**Returns:** List of trajectories + maximum episode length

---

### **Stage 5: Trajectory Filtering** (Lines 714-726)

**Purpose:** Keep only high-performing trajectories (offline RL best practice)

**Process:**
1. Calculate returns (sum of rewards) for all trajectories
2. Filter to top N% (default: top 10% via `--top_percentile`)
3. Keep only trajectories with highest returns
4. Split remaining trajectories into train/val sets (90/10 split)

**Rationale:** Decision Transformers work better when trained on expert demonstrations rather than random play

---

### **Stage 6: Data Preprocessing** (Lines 758-815)

**Normalization:**
- Compute `state_mean` and `state_std` from training states
- Normalize states: `(state - mean) / std`

**Return-to-Go (RTG) Computation:**
- `discount_cumsum()`: Computes cumulative discounted returns (gamma=1.0 = undiscounted)
- RTG represents "remaining reward to achieve" - key input for Decision Transformer

**Padded Sequence Creation (`make_padded_sample`):**
- Creates sequences of length K (context window, default: 40)
- For each trajectory segment:
  - States: padded to length K, normalized
  - Actions: padded with -10.0
  - Rewards: padded with 0
  - RTG: padded with 0, normalized by scale factor
  - Terminals: padded with 2 (special value)
  - Timesteps: relative timestep indices
  - Mask: binary mask indicating valid vs padded positions

**Sampling Strategy:**
- Weighted sampling based on trajectory returns (via `return_weights`)
- Higher-return trajectories sampled more frequently
- Power weighting (`--return_weight_power`, default: 2.0) emphasizes top performers

---

### **Stage 7: Model Architecture** (Lines 859-875)

**Decision Transformer:**
- **Input:** Sequences of (States, Actions, Rewards, RTG, Timesteps)
- **Architecture:**
  - Embedding dimension: `--embed_dim` (default: 256)
  - Layers: `--n_layer` (default: 4)
  - Attention heads: `--n_head` (default: 4)
  - Activation: `--activation_function` (default: "relu")
  - Dropout: `--dropout` (default: 0.1)
- **Output:** Action predictions (continuous logits, converted to discrete action indices)

**Key Principle:** Model conditions on desired RTG (Return-to-Go) to predict actions that achieve that return level

---

### **Stage 8: Training Loop** (Lines 877-961)

**Setup:**
- Optimizer: AdamW with learning rate scheduling
- Scheduler: Linear warmup (`--warmup_steps`, default: 100)
- Loss function: `loss_fn_weighted()` - MSE loss weighted by trajectory returns

**Training Process:**
1. For each iteration (`--max_iters`, default: 10):
  2. For each training step (`--num_steps_per_iter`, default: 10,000):
     - Sample batch using weighted sampling (higher returns = more likely)
     - Forward pass: Model predicts actions given states and target RTG
     - Compute loss: MSE between predicted and actual actions
     - Weight loss by trajectory returns (emphasize high-scoring trajectories)
     - Backward pass and optimizer step
  3. Log metrics (optionally to wandb)

**Loss Weighting:**
- Trajectories with higher returns contribute more to loss
- Encourages model to learn patterns from successful games

---

### **Stage 9: Evaluation** (Lines 962-1087)

**Evaluation Process:**
1. For each evaluation episode (`--eval_episodes`, default: 100):
   1. Reset environment with new card setup
   2. Initialize: empty state/action/reward history
   3. While game not done:
      - Encode current state
      - Build context window (last K states/actions/rewards)
      - **Set target RTG** using percentile of training returns (`--target_rtg_percentile`, default: 90th)
      - Model predicts action given context + target RTG
      - Convert predicted action to discrete action index
      - Filter to valid legal actions
      - Choose valid action closest to prediction
      - Execute action, record results
   4. Track metrics: turns taken, final score

**Metrics Computed:**
- Average score
- Number of losses (episodes > 60 turns)
- Average turns to win (excluding losses)
- Win rate (% of episodes won in ≤60 turns)

---

### **Stage 10: Model Saving** (Lines 1089-1092)

If `--save_model` flag is set:
- Saves model state dict to `--model_path` (default: "models/splendor_dt_model.pt")

---

## Key Design Decisions

### 1. **Offline RL Approach**
- No online interaction during training
- Learn from pre-collected trajectories
- Filters to top performers (top 10%)

### 2. **Return-to-Go Conditioning**
- Model sees "desired return" (RTG) as input
- During inference, set high target RTG to encourage good play
- Key innovation of Decision Transformer architecture

### 3. **Weighted Sampling & Loss**
- Sample high-return trajectories more frequently
- Weight loss by trajectory quality
- Power parameter controls emphasis strength

### 4. **Context Window (K)**
- Uses last K=40 steps as context
- Similar to transformer attention window
- Must be ≥ maximum trajectory length

### 5. **Action Space**
- Discrete actions mapped to indices
- Model outputs continuous logits, clamped to valid range
- At inference, maps to nearest valid legal action

---

## Command Line Arguments Summary

**Data:**
- `--card_data_path`: Path to card CSV file
- `--num_trajectories`: Number of episodes to generate
- `--top_percentile`: Top % of trajectories to keep (default: 10%)

**Model:**
- `--K`: Context length (default: 40)
- `--embed_dim`: Hidden dimension (default: 256)
- `--n_layer`: Number of transformer layers (default: 4)
- `--n_head`: Attention heads (default: 4)

**Training:**
- `--batch_size`: Batch size (default: 128)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--max_iters`: Training iterations (default: 10)
- `--num_steps_per_iter`: Steps per iteration (default: 10,000)
- `--return_weight_power`: Power for return weighting (default: 2.0)

**Evaluation:**
- `--eval_episodes`: Episodes to evaluate (default: 100)
- `--target_rtg_percentile`: Target RTG percentile (default: 90th)

**Other:**
- `--device`: cuda/cpu
- `--log_to_wandb`: Enable wandb logging
- `--save_model`: Save trained model

---

## Flow Diagram

```
Load Card Data
    ↓
Create Environment
    ↓
Generate Trajectories (Random Agent)
    ↓
Filter Top 10% by Return
    ↓
Split Train/Val
    ↓
Preprocess: Normalize States, Compute RTG, Create Padded Sequences
    ↓
Create Decision Transformer Model
    ↓
Train: Sample weighted batches → Forward → Weighted Loss → Backward
    ↓
Evaluate: Play games with model, track metrics
    ↓
Save Model (optional)
```

---

## Dependencies

- `torch`: PyTorch for model
- `gym`: Gym environment interface
- `numpy`, `pandas`: Data handling
- `sklearn`: Optional, for state hashing
- `wandb`: Optional, for experiment tracking
- `decision_transformer`: Custom package with DecisionTransformer model and SequenceTrainer

