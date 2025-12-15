# Improvements to experiment_splendor_dt.py for Better Performance

## Goal
Prioritize successful trajectories that win quickly (reach 15 points in fewer turns) to improve model performance compared to the baseline DQN agent from the Reinforcement Learning Report (average 36.6 turns, 95.3% win rate).

## Key Changes Made

### 1. **More Aggressive Trajectory Filtering** (Line ~717-725)
   - **Before**: Filtered to top 10% by return only
   - **After**: 
     - Filters to top 5% by default (more selective)
     - Uses a **combined scoring metric**:
       - 70% weight on return (total reward)
       - 30% weight on efficiency (return / turns)
     - This prioritizes trajectories that **both win quickly AND have high returns**

### 2. **Enhanced Sampling Weights** (Line ~767-783)
   - **Before**: Basic exponential weighting with power 2.0
   - **After**:
     - Increased default power from 2.0 to **3.5** (much more aggressive)
     - Better normalization that avoids zeros
     - Added minimum weight floor to ensure all trajectories have some probability
     - Results in **significantly more sampling** of top-performing trajectories

### 3. **Improved Loss Weighting** (Line ~936-975)
   - **Before**: Simple multiplication by (1.0 + avg_weight)
   - **After**:
     - More aggressive normalization of trajectory returns
     - Higher power exponent (3.5) for stronger emphasis
     - Loss scaling: `loss * (1.0 + return_weight_power * weight_scale)`
     - This makes high-return trajectories contribute **much more** to gradient updates

### 4. **Higher Target RTG During Evaluation** (Line ~1101)
   - **Before**: 90th percentile of training returns
   - **After**: **95th percentile** by default
   - Encourages the model to aim for faster wins during inference

### 5. **Better Reporting** (Line ~742-750)
   - Added reporting of average turns for filtered trajectories
   - Shows turns range to understand trajectory quality
   - Added note about focusing on fastest high-return trajectories

## Updated Default Parameters

```python
--top_percentile 5.0          # Was 10.0 - keep only top 5% of trajectories
--return_weight_power 3.5     # Was 2.0 - much stronger emphasis on high returns
--target_rtg_percentile 95.0  # Was 90.0 - aim for faster wins
```

## Expected Impact

Based on the Reinforcement Learning Report:
- **DQN baseline**: 36.6 avg turns (w/o losses), 95.3% win rate
- **Random baseline**: 42.46 avg turns, 0.8% loss rate

With these improvements:
1. **Better trajectory selection**: Top 5% with efficiency metric focuses on fastest wins
2. **Stronger learning signal**: Higher weighting ensures model learns from best examples
3. **More aggressive exploration of good strategies**: 3.5x power means top trajectories are sampled much more frequently
4. **Higher aspirations**: 95th percentile target RTG encourages aiming for the best performance

## Usage

Run with default improved parameters:
```bash
python experiment_splendor_dt.py --card_data_path data/card_data.csv --save_model
```

Or customize for even more aggressive prioritization:
```bash
python experiment_splendor_dt.py \
    --top_percentile 3.0 \
    --return_weight_power 4.0 \
    --target_rtg_percentile 98.0 \
    --card_data_path data/card_data.csv
```

## Technical Details

### Combined Scoring Metric
```python
efficiency = returns / (turns + 1.0)
normalized_returns = (returns - min) / (max - min)
normalized_efficiency = (efficiency - min) / (max - min)
combined_score = 0.7 * normalized_returns + 0.3 * normalized_efficiency
```

This ensures we keep trajectories that:
- Have high total returns (win the game)
- Win in fewer turns (fast wins)

### Sampling Probability
```python
normalized_returns = (returns - min + eps) / (max - min + eps)
weights = normalized_returns^3.5  # Higher power = more skewed
weights = weights + 0.01 * max(weights)  # Floor to avoid zero probability
p_sample = weights / sum(weights)
```

With power 3.5, if one trajectory has 2x the return of another, it will be sampled approximately 11x more frequently (2^3.5 ≈ 11.3).

## Comparison to Report

The report mentions that better state representation could improve results. While we're using the same state encoding, we're now:
1. Training only on the **best** trajectories (top 5% vs top 10%)
2. Learning **much more** from those trajectories (3.5x power vs 2.0x)
3. Aiming for **faster wins** (95th percentile RTG vs 90th)

This should help the Decision Transformer achieve better performance than the baseline DQN agent.

