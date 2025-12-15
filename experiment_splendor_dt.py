"""
Train a Decision Transformer on Splendor game trajectories.

This script:
1. Generates trajectories by running the Splendor environment
2. Encodes states and actions for the decision transformer
3. Trains the decision transformer on collected trajectories
4. Evaluates the trained model
"""

import argparse
import os
import random
import copy as cp
import itertools
import numpy as np
import pandas as pd
import torch

# Optional sklearn import (only needed for state hashing)
try:
    from sklearn.feature_extraction import FeatureHasher
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. State hashing will be disabled.")

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging to wandb will be disabled.")

from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.models.decision_transformer import DecisionTransformer

# Import Splendor environment classes from the notebook
import sys
sys.path.append('.')

# We'll need to import the Splendor classes - let's define them here
# or import from a module if available
class Card:
    def __init__(self, level, color, cost, points):
        self.level = level
        self.color = color
        self.cost = cost
        self.points = points

class Noble:
    def __init__(self, requirements, points):
        self.requirements = requirements
        self.points = points

# Import gym for the environment
import gym
from gym import spaces


class SplendorEnv(gym.Env):
    MAX_GEMS_SUPPLY = 10
    MAX_JOKERS = 3
    
    def __init__(self, card_supply, deck_nobles, initial_gems=[5, 5, 5, 5, 5, 3], cards_player=None, nobles=None):
        num_card_tiers = 3
        cards_per_tier = 4
        num_token_colors = 5
        num_nobles = 3
        
        self.turns = 0
        deck = cp.deepcopy(card_supply)
        self.deck_1 = deck[0]
        self.deck_2 = deck[1]
        self.deck_3 = deck[2]

        self.board_1 = []
        self.board_2 = []
        self.board_3 = []
        self.UpdateCardBoard()

        self.nobles = []
        self.CreateNobles(deck_nobles)

        self.gem_reserve = initial_gems
        self.gems = [0, 0, 0, 0, 0, 0]

        self.binary_lists_with_sum_of_3 = list(filter(lambda binary_list: sum(binary_list) == 3, itertools.product([0, 1], repeat=5)))
        self.binary_lists_with_sum_of_2 = list(filter(lambda binary_list: sum(binary_list) == 2, itertools.product([0, 1], repeat=5)))
        
        self.all_actions = []
        self.create_all_actions()
        self.valid_actions = [0] * len(self.all_actions)

        self.score = 0
        self.player_cards = cards_player if cards_player is not None else []
        self.player_reserved_cards = []
        self.nobles = nobles if nobles is not None else []
        self.buying_power = self.gems

        self.observation_space = gym.spaces.Dict({
            'player_gems': spaces.Box(low=0, high=7, shape=(num_token_colors+1,), dtype=np.integer),
            'gems_supply': spaces.Box(low=0, high=7, shape=(num_token_colors+1,), dtype=np.integer),
            'player_cards': spaces.Box(low=0, high=np.inf, shape=(num_token_colors,), dtype=np.integer),
            'player_score': spaces.Discrete(30),
            'nobles': spaces.Tuple(tuple([spaces.Tuple((spaces.MultiDiscrete([1,1,1,1,1]), spaces.Discrete(5))) for _ in range(num_nobles)])),
            'player_reserved': spaces.Discrete(2),
            'valid_actions': spaces.Discrete(57)
        })
        
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(4),
            'purchase_card': spaces.Tuple((spaces.Discrete(num_card_tiers), spaces.Discrete(cards_per_tier))),
            'reserve_card': spaces.Tuple((spaces.Discrete(num_card_tiers), spaces.Discrete(cards_per_tier))),
            'buy_reserved': spaces.Discrete(3),
            'pick_tokens': spaces.MultiBinary(num_token_colors)
        })

    def UpdateCardBoard(self):
        random.shuffle(self.deck_1)
        random.shuffle(self.deck_2)
        random.shuffle(self.deck_3)
        while len(self.board_1) < 4:
            if len(self.deck_1) == 0:
                break
            self.board_1.append(self.deck_1.pop(0))
        while len(self.board_2) < 4:
            if len(self.deck_2) == 0:
                break
            self.board_2.append(self.deck_2.pop(0))
        while len(self.board_3) < 4:
            if len(self.deck_3) == 0:
                break
            self.board_3.append(self.deck_3.pop(0))

    def create_all_actions(self):
        card_buy = []
        for board in [self.board_1, self.board_2, self.board_3]:
            for card in board:
                card_buy.append((card.level-1, board.index(card)))
        self.all_actions.extend([('buy card', x) for x in card_buy])

        reserve_cards = []
        for board in [self.board_1, self.board_2, self.board_3]:
            for card in board:
                reserve_cards.append((card.level-1, board.index(card)))
        self.all_actions.extend([('reserve', x) for x in reserve_cards])

        for i in range(self.MAX_JOKERS):
            self.all_actions.append(('buy reserved', i))
        
        self.all_actions.extend([('take', x) for x in self.binary_lists_with_sum_of_2])
        self.all_actions.extend([('take', x) for x in self.binary_lists_with_sum_of_3])
        self.all_actions.extend([('take', x) for x in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [2,0,0,0,0], [0,2,0,0,0], [0,0,2,0,0], [0,0,0,2,0], [0,0,0,0,2]]])
        negative_gems = [[-x for x in inner_list] for inner_list in self.binary_lists_with_sum_of_3]
        self.all_actions.extend([('take', x) for x in negative_gems])

    def CreateNobles(self, deck_nobles):
        deck_n = cp.deepcopy(deck_nobles)
        random.shuffle(deck_n)
        while len(self.nobles) < 4:
            if len(deck_n) == 0:
                break
            self.nobles.append(deck_n.pop(0))

    def UpdateBuyingPower(self):
        self.buying_power = cp.deepcopy(self.gems)
        for card in self.player_cards:
            self.buying_power[card.color] += 1

    def get_legal_actions(self):
        legal_actions = []
        card_buy = self.get_purchasable_cards()
        legal_actions.extend(card_buy)
        reserve_cards = self.get_reservable_cards()
        legal_actions.extend(reserve_cards)
        buyable_reserved_cards = self.get_purchasable_reserved()
        legal_actions.extend(buyable_reserved_cards)
        gem_buy = self.get_takable_gems()
        legal_actions.extend(gem_buy)
        
        if len(legal_actions) == 0:
            last_resort_action = []
            for gem in self.gems[:5]:
                if gem > 0 and sum(last_resort_action) > -3:
                    last_resort_action.append(-1)
                else:
                    last_resort_action.append(0)
            legal_actions = [('take', last_resort_action)]
        
        self.valid_actions = [0] * len(self.all_actions)
        for x in legal_actions:
            if x in self.all_actions:
                self.valid_actions[self.all_actions.index(x)] = 1
        return legal_actions

    def get_purchasable_cards(self):
        card_buy = []
        for board in [self.board_1, self.board_2, self.board_3]:
            for card in board:
                over = 0
                for i in range(len(card.cost)):
                    over += max(0, card.cost[i] - self.buying_power[i])
                if over <= self.gems[-1]:
                    card_buy.append((card.level-1, board.index(card)))
        return [('buy card', x) for x in card_buy]

    def get_reservable_cards(self):
        reserve_cards = []
        if sum(self.gems) < 10 and len(self.player_reserved_cards) < self.MAX_JOKERS:
            for board in [self.board_1, self.board_2, self.board_3]:
                for card in board:
                    reserve_cards.append((card.level-1, board.index(card)))
        return [('reserve', x) for x in reserve_cards]

    def get_purchasable_reserved(self):
        buyable_reserved_cards = []
        for card in self.player_reserved_cards:
            over = 0
            for i in range(len(card.cost)):
                over += max(0, card.cost[i] - self.buying_power[i])
            if over <= self.gems[-1]:
                buyable_reserved_cards.append(self.player_reserved_cards.index(card))
        return [('buy reserved', x) for x in buyable_reserved_cards]

    def get_takable_gems(self):
        doable_3gems = []
        for comb in self.binary_lists_with_sum_of_3:
            if min([self.gem_reserve[i] - comb[i] for i in range(len(comb))]) >= 0:
                doable_3gems.append(comb)

        doable_2gems = []
        for comb in self.binary_lists_with_sum_of_2:
            if min([self.gem_reserve[i] - comb[i] for i in range(len(comb))]) >= 0:
                doable_2gems.append(comb)

        doable_1gems = []
        for comb in [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]:
            if min([self.gem_reserve[i] - comb[i] for i in range(len(comb))]) >= 0:
                doable_1gems.append(comb)

        gem_buy = []
        if sum(self.gems) == SplendorEnv.MAX_GEMS_SUPPLY:
            return []
        
        if sum(self.gems) <= 7:
            gem_buy.extend(doable_3gems)
        if sum(self.gems) == 8:
            gem_buy.extend(doable_2gems)
        if sum(self.gems) == 9:
            gem_buy.extend(doable_1gems)

        if sum(self.gems) <= 8:
            for x in range(len(self.gems[:5])):
                list_empty = [0, 0, 0, 0, 0]
                if self.gem_reserve[x] > 3:
                    list_empty[x] = 2
                    if list_empty != [0, 0, 0, 0, 0]:
                        gem_buy.append(list_empty)

        return [('take', x) for x in gem_buy]

    def execute_action(self, action):
        action_type, action_params = action
        if action_type == 'buy card':
            self.buy_card(action_params)
        elif action_type == 'buy reserved':
            self.buy_reserved_card(action_params)
        elif action_type == 'reserve':
            self.reserve_card(action_params)
        elif action_type == 'take':
            self.take_gems(action_params)

    def buy_card(self, index):
        card = None
        while card is None:
            for board in [self.board_1, self.board_2, self.board_3]:
                for cards in board:
                    if index == (cards.level-1, board.index(cards)):
                        card = cards

        cards_color = [x.color for x in self.player_cards]
        cards_sum = [0, 0, 0, 0, 0]
        for x in cards_color:
            cards_sum[x] += 1

        self.player_cards.append(card)

        if card.level == 1:
            self.board_1.remove(card)
        elif card.level == 2:
            self.board_2.remove(card)
        elif card.level == 3:
            self.board_3.remove(card)

        new_cost = [0, 0, 0, 0, 0]
        for i in range(len(card.cost)):
            new_cost[i] = max(0, card.cost[i] - cards_sum[i])

        over = 0
        for i in range(len(card.cost)):
            over = max(0, new_cost[i] - self.gems[i])
            gem_transaction = max(0, new_cost[i] - over)
            if over > self.gems[5]:
                raise ValueError("Too many jokers")
            self.gems[i] -= gem_transaction
            self.gem_reserve[i] += gem_transaction
            self.gems[5] -= over
            self.gem_reserve[5] += over

    def reserve_card(self, index):
        card = None
        while card is None:
            for board in [self.board_1, self.board_2, self.board_3]:
                for cards in board:
                    if index == (cards.level-1, board.index(cards)):
                        card = cards
        self.player_reserved_cards.append(card)
        if card.level == 1:
            self.board_1.remove(card)
        elif card.level == 2:
            self.board_2.remove(card)
        elif card.level == 3:
            self.board_3.remove(card)
        self.gems[5] += 1
        self.gem_reserve[5] -= 1

    def buy_reserved_card(self, index):
        card = self.player_reserved_cards[index]
        self.player_reserved_cards.remove(card)
        cards_color = [x.color for x in self.player_cards]
        cards_sum = [0, 0, 0, 0, 0]
        for x in cards_color:
            cards_sum[x] += 1
        new_cost = [0, 0, 0, 0, 0]
        for i in range(len(card.cost)):
            new_cost[i] = max(0, card.cost[i] - cards_sum[i])
        self.player_cards.append(card)
        over = 0
        for i in range(len(card.cost)):
            over = max(0, new_cost[i] - self.gems[i])
            if over > self.gems[5]:
                raise ValueError("Too many jokers needed")
            gem_transaction = max(0, new_cost[i] - over)
            self.gems[i] -= gem_transaction
            self.gem_reserve[i] += gem_transaction
            self.gem_reserve[5] += over
            self.gems[5] -= over

    def take_gems(self, gems_array):
        for i in range(len(gems_array)):
            self.gems[i] += gems_array[i]
            self.gem_reserve[i] -= gems_array[i]

    def step(self, action):
        self.execute_action(action)
        self.turns += 1
        self.UpdateBuyingPower()
        self.UpdateCardBoard()

        reward, _, done = self.calculate_reward()
        if action[0] == 'take' and sum(action[1]) < 0:
            reward -= 5

        self.get_legal_actions()

        next_obs = {
            'player_gems': self.gems,
            'gems_supply': self.gem_reserve,
            'player_cards': self.player_cards,
            'player_score': self.score,
            'nobles': self.nobles,
            'player_reserved': self.player_reserved_cards,
            'valid_actions': self.valid_actions,
            'Board': np.array([self.board_1, self.board_2, self.board_3], dtype=object),
            'all_actions': self.all_actions
        }
        return next_obs, reward, done, {}

    def calculate_reward(self, max_turns=60):
        reward = 0
        points = 0
        win = False

        for cards in self.player_cards:
            points += cards.points

        for noble in self.nobles:
            points += noble[1]

        if points >= 15:
            win = True
            reward = max_turns - self.turns

        if self.turns > max_turns:
            win = True
            reward = -50
        self.score = points

        return reward, points, win

    def reset(self, new_deck, new_deck_nobles, reset_gems=[4, 4, 4, 4, 4, 3]):
        self.turns = 0
        self.deck_1 = []
        self.deck_2 = []
        self.deck_3 = []

        self.deck_1 = cp.deepcopy(new_deck[0])
        self.deck_2 = cp.deepcopy(new_deck[1])
        self.deck_3 = cp.deepcopy(new_deck[2])

        self.board_1 = []
        self.board_2 = []
        self.board_3 = []

        self.UpdateCardBoard()

        self.nobles = []
        self.CreateNobles(new_deck_nobles)

        self.gems = [0, 0, 0, 0, 0, 0]
        self.score = 0
        self.gem_reserve = [4, 4, 4, 4, 4, 3]
        self.player_cards = []
        self.player_reserved_cards = []
        self.nobles = []
        self.buying_power = self.gems

        self.get_legal_actions()

        obs = {
            'player_gems': self.gems,
            'gems_supply': self.gem_reserve,
            'player_cards': self.player_cards,
            'player_score': self.score,
            'nobles': self.nobles,
            'player_reserved': self.player_reserved_cards,
            'valid_actions': self.valid_actions,
            'Board': np.array([self.board_1, self.board_2, self.board_3], dtype=object),
            'all_actions': self.all_actions
        }

        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def flatten_state(state):
    """Flatten Splendor state to dictionary for hashing."""
    flat_state = {}
    
    for idx, value in enumerate(state['player_gems']):
        flat_state[f'player_gems_{idx}'] = value

    for idx, value in enumerate(state['gems_supply']):
        flat_state[f'gems_supply_{idx}'] = value

    for idx, rcard in enumerate(state['player_reserved']):
        flat_state[f'player_reserved_{idx}_level'] = rcard.level
        flat_state[f'player_reserved_{idx}_color'] = rcard.color
        flat_state[f'player_reserved_{idx}_points'] = rcard.points
        for jdx, cost in enumerate(rcard.cost):
            flat_state[f'player_reserved_{idx}_cost_{jdx}'] = cost
        
    flat_state['player_score'] = state['player_score']

    for idx, noble in enumerate(state['nobles']):
        for jdx, req in enumerate(noble.requirements):
            flat_state[f'noble_{idx}_req_{jdx}'] = req
        flat_state[f'noble_{idx}_points'] = noble.points

    for idx, card in enumerate(state['player_cards']):
        flat_state[f'player_card_{idx}_level'] = card.level
        flat_state[f'player_card_{idx}_color'] = card.color
        flat_state[f'player_card_{idx}_points'] = card.points
        for jdx, cost in enumerate(card.cost):
            flat_state[f'player_card_{idx}_cost_{jdx}'] = cost
    
    for i in range(state['Board'].shape[0]):
        for j in range(len(state['Board'][0])):
            card = state['Board'][i][j]
            idx = i * 4 + j
            flat_state[f'Board_{idx}_level'] = card.level
            flat_state[f'Board_{idx}_color'] = card.color
            flat_state[f'Board_{idx}_points'] = card.points
            for jdx, cost in enumerate(card.cost):
                flat_state[f'Board_{idx}_cost_{jdx}'] = cost

    return flat_state


def load_card_data(csv_path):
    """Load card data from CSV and create card objects."""
    cards_1 = []
    cards_2 = []
    cards_3 = []

    cards = pd.read_csv(csv_path).fillna(0)
    for index, row in cards.iterrows():
        cost = [row['cost_black'], row['cost_blue'], row['cost_green'], row['cost_red'], row['cost_white']]
        cost = list(map(int, cost))
        color_dict = {'black': 0, 'blue': 1, 'green': 2, 'red': 3, 'white': 4}
        color = color_dict[row['color']]

        card = Card(row['tier'], color, cost, row['points'])

        if row['tier'] == 1:
            cards_1.append(card)
        elif row['tier'] == 2:
            cards_2.append(card)
        else:
            cards_3.append(card)

    nobles = [
        Noble([3, 3, 0, 0, 3], 3),
        Noble([3, 0, 3, 3, 0], 3),
        Noble([0, 4, 0, 4, 0], 3),
        Noble([0, 0, 4, 0, 4], 3),
        Noble([4, 0, 0, 0, 4], 3),
        Noble([0, 0, 0, 5, 3], 3)
    ]

    card_supply = [cards_1, cards_2, cards_3]
    return card_supply, nobles


def encode_state(state, hasher=None):
    """Encode Splendor state to fixed-size vector."""
    flat_state = flatten_state(state)
    if hasher is not None:
        state_encoded = hasher.transform([flat_state]).toarray()[0]
    else:
        # Simple encoding: use player_gems, gems_supply, player_score, card counts
        # Also encode board cards and reserved cards
        state_vec = []
        
        # Player gems (6 values: 5 colors + joker)
        state_vec.extend(state['player_gems'])
        
        # Gem supply (6 values)
        state_vec.extend(state['gems_supply'])
        
        # Player score
        state_vec.append(state['player_score'])
        
        # Card counts
        state_vec.append(len(state['player_cards']))
        state_vec.append(len(state['player_reserved']))
        
        # Player cards summary (color distribution)
        card_colors = [0, 0, 0, 0, 0]
        card_points = 0
        for card in state['player_cards']:
            card_colors[card.color] += 1
            card_points += card.points
        state_vec.extend(card_colors)
        state_vec.append(card_points)
        
        # Reserved cards summary
        reserved_colors = [0, 0, 0, 0, 0]
        reserved_points = 0
        for card in state['player_reserved']:
            reserved_colors[card.color] += 1
            reserved_points += card.points
        state_vec.extend(reserved_colors)
        state_vec.append(reserved_points)
        
        # Board cards summary (12 cards: 3 tiers x 4 cards)
        board_features = []
        for tier in range(3):
            for pos in range(4):
                if pos < len(state['Board'][tier]):
                    card = state['Board'][tier][pos]
                    board_features.extend([card.level, card.color, card.points])
                    board_features.extend(card.cost)
                else:
                    # Empty slot
                    board_features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        state_vec.extend(board_features[:96])  # Limit to 96 features (12 cards * 8 features)
        
        state_encoded = np.array(state_vec, dtype=np.float32)
    
    return state_encoded


def generate_trajectories(env, card_supply, nobles, num_episodes, agent_type='random', hasher=None):
    """
    Generate trajectories from the Splendor environment.
    
    Args:
        env: SplendorEnv instance
        card_supply: Card supply for reset
        nobles: Nobles for reset
        num_episodes: Number of episodes to generate
        agent_type: 'random' for random agent, 'dqn' for DQN agent (if available)
        hasher: FeatureHasher for state encoding (if using hashing)
    
    Returns:
        trajectories: List of trajectory dictionaries
        max_ep_len: Maximum episode length
    """
    trajectories = []
    max_ep_len = 0
    
    for episode in range(num_episodes):
        state = env.reset(card_supply, nobles)
        
        observations = []
        actions = []
        rewards = []
        done = False
        
        while not done:
            # Get valid actions
            valid_actions = env.get_legal_actions()
            
            if len(valid_actions) == 0:
                break
            
            # Choose action based on agent type
            if agent_type == 'random':
                action = random.choice(valid_actions)
            else:
                # For now, default to random
                action = random.choice(valid_actions)
            
            # Encode state
            state_encoded = encode_state(state, hasher)
            
            # Encode action as index
            if action in env.all_actions:
                action_idx = env.all_actions.index(action)
            else:
                # If action not in all_actions, skip this step
                continue
            
            observations.append(state_encoded)
            actions.append(action_idx)
            
            # Step environment
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            
            state = next_state
        
        if len(observations) > 0:
            T = len(observations)
            terminals = np.zeros(T, dtype=np.int64)
            terminals[-1] = 1
            
            trajectories.append({
                'observations': np.array(observations, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32).reshape(-1, 1),
                'rewards': np.array(rewards, dtype=np.float32),
                'terminals': terminals,
                'lens': int(T)
            })
            max_ep_len = max(max_ep_len, T)
        
        if (episode + 1) % 100 == 0:
            print(f"Generated {episode + 1}/{num_episodes} episodes")
    
    return trajectories, max_ep_len


def discount_cumsum(x, gamma=1.0):
    """Compute discounted cumulative sum."""
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def main(args):
    device = args.device
    print(f"Using device: {device}")
    
    # Load card data
    card_supply, nobles = load_card_data(args.card_data_path)
    
    # Create environment
    env = SplendorEnv(card_supply, nobles)
    
    # Generate trajectories
    print("Generating trajectories...")
    hasher = None
    
    if args.use_state_hashing:
        if not SKLEARN_AVAILABLE:
            print("Error: --use_state_hashing requires sklearn. Install with: pip install scikit-learn")
            return
        hasher = FeatureHasher(n_features=args.state_size, input_type='dict')
    
    trajectories, max_ep_len = generate_trajectories(
        env,
        card_supply,
        nobles,
        num_episodes=args.num_trajectories,
        agent_type=args.agent_type,
        hasher=hasher
    )
    
    if len(trajectories) == 0:
        print("No trajectories generated! Exiting.")
        return
    
    returns = np.array([tr["rewards"].sum() for tr in trajectories], dtype=np.float32)
    traj_lens = np.array([tr["lens"] for tr in trajectories], dtype=np.int32)
    
    # Filter to top 10% by return (score)
    top_percentile = args.top_percentile / 100.0
    num_keep = max(1, int(len(trajectories) * top_percentile))
    top_indices = np.argsort(returns)[-num_keep:]  # Get indices of top trajectories
    
    # Filter trajectories to keep only top 10%
    trajectories = [trajectories[i] for i in top_indices]
    returns = returns[top_indices]
    traj_lens = traj_lens[top_indices]
    num_timesteps = int(traj_lens.sum())
    
    total_trajs = len(trajectories)
    
    # Determine state dimension
    if args.use_state_hashing:
        state_dim = args.state_size
    else:
        # Test state encoding to get dimension
        test_state = env.reset(card_supply, nobles)
        test_encoded = encode_state(test_state, hasher)
        state_dim = test_encoded.shape[0]
    
    print("=" * 50)
    print(f"Generated {args.num_trajectories} trajectories, keeping top {args.top_percentile}% ({total_trajs} trajectories)")
    print(f"Splendor dataset: {total_trajs} trajectories, {num_timesteps} timesteps")
    print(f"State dim = {state_dim}, Action dim = 1")
    print(f"Avg return (filtered): {returns.mean():.3f} ± {returns.std():.3f}")
    print(f"Return range: [{returns.min():.3f}, {returns.max():.3f}]")
    print(f"Max ep len: {max_ep_len}")
    print("=" * 50)
    
    # Split into train/val
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(total_trajs)
    train_size = int(total_trajs * 0.9)
    train_inds = perm[:train_size]
    val_inds = perm[train_size:]
    
    print(f"Train: {len(train_inds)}, Val: {len(val_inds)}")
    
    # Prepare training data
    train_returns = returns[train_inds]
    train_traj_lens = traj_lens[train_inds]
    train_states = np.concatenate([trajectories[int(i)]["observations"] for i in train_inds], axis=0)
    state_mean = train_states.mean(axis=0)
    state_std = train_states.std(axis=0) + 1e-6
    
    # Weight sampling by returns (higher returns = more likely to be sampled)
    # Normalize returns to positive values and use as sampling weights
    min_return = train_returns.min()
    normalized_returns = train_returns - min_return + 1.0  # Shift to positive
    # Use exponential weighting to emphasize high returns more
    return_weights = np.power(normalized_returns / normalized_returns.mean(), args.return_weight_power)
    p_sample = return_weights / return_weights.sum()
    
    print(f"Sampling weights: min={p_sample.min():.4f}, max={p_sample.max():.4f}, mean={p_sample.mean():.4f}")
    
    state_dim = train_states.shape[1]
    act_dim = 1
    scale = max(1.0, np.percentile(train_returns, 95))
    
    K = args.K
    max_len = K
    num_traj_keep = train_inds.shape[0]
    
    def make_padded_sample(traj, start_idx):
        s_i = traj["observations"][start_idx:start_idx + max_len]
        a_i = traj["actions"][start_idx:start_idx + max_len]
        r_i = traj["rewards"][start_idx:start_idx + max_len]
        d_i = traj["terminals"][start_idx:start_idx + max_len]
        
        ts = np.arange(start_idx, start_idx + s_i.shape[0], dtype=np.int64)
        ts[ts >= max_ep_len] = max_ep_len - 1
        
        rtg_i = discount_cumsum(traj["rewards"][start_idx:], gamma=1.0)[:s_i.shape[0] + 1]
        if rtg_i.shape[0] <= s_i.shape[0]:
            rtg_i = np.concatenate([rtg_i, np.zeros((1,), dtype=np.float32)], axis=0)
        
        tlen = s_i.shape[0]
        
        s_pad = np.zeros((max_len - tlen, state_dim), dtype=np.float32)
        a_pad = np.ones((max_len - tlen, act_dim), dtype=np.float32) * -10.0
        r_pad = np.zeros((max_len - tlen, 1), dtype=np.float32)
        d_pad = np.ones((max_len - tlen,), dtype=np.int64) * 2
        rtg_pad = np.zeros((max_len - tlen, 1), dtype=np.float32)
        ts_pad = np.zeros((max_len - tlen,), dtype=np.int64)
        
        s_i = np.concatenate([s_pad, s_i], axis=0).astype(np.float32)
        s_i = (s_i - state_mean) / state_std
        a_i = np.concatenate([a_pad, a_i], axis=0).astype(np.float32)
        r_i = np.concatenate([r_pad, r_i.reshape(-1, 1)], axis=0).astype(np.float32)
        d_i = np.concatenate([d_pad, d_i], axis=0).astype(np.int64)
        rtg_i = np.concatenate([rtg_pad, rtg_i.reshape(-1, 1)], axis=0).astype(np.float32) / float(scale)
        ts = np.concatenate([ts_pad, ts], axis=0).astype(np.int64)
        mask = np.concatenate(
            [np.zeros((max_len - tlen,)), np.ones((tlen,))], axis=0
        ).astype(np.float32)
        
        return s_i, a_i, r_i, d_i, rtg_i, ts, mask
    
    # Store trajectory returns for loss weighting
    traj_returns_cache = {}
    
    def get_batch(batch_size=args.batch_size, max_len=max_len):
        batch_inds = np.random.choice(
            np.arange(num_traj_keep),
            size=batch_size,
            replace=True,
            p=p_sample,
        )
        
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        batch_traj_returns = []
        for i in range(batch_size):
            traj_idx = int(train_inds[batch_inds[i]])
            traj = trajectories[traj_idx]
            traj_return = train_returns[batch_inds[i]]
            batch_traj_returns.append(traj_return)
            si = random.randint(0, traj["rewards"].shape[0] - 1)
            s_i, a_i, r_i, d_i, rtg_i, ts, m = make_padded_sample(traj, si)
            
            s.append(s_i[None])
            a.append(a_i[None])
            r.append(r_i[None])
            d.append(d_i[None])
            rtg.append(rtg_i[None])
            timesteps.append(ts[None])
            mask.append(m[None])
        
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps_t = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask_t = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        
        # Store batch trajectory returns for loss weighting
        get_batch.last_traj_returns = torch.from_numpy(np.array(batch_traj_returns)).to(dtype=torch.float32, device=device)
        
        return s, a, r, d, rtg, timesteps_t, mask_t
    
    # Create model
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4 * args.embed_dim,
        activation_function=args.activation_function,
        n_positions=1024,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        action_tanh=False,  # Output raw logits
    )
    model = model.to(device=device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1),
    )
    
    # Loss function for discrete actions (cross-entropy)
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    def loss_fn(s_hat, a_hat, r_hat, s, a, r):
        # a_hat: logits of shape (N, act_dim) where act_dim=1 but represents action index
        # a: action indices of shape (N, 1) with values 0..num_actions-1
        # We need to expand action space to num_actions
        num_actions = len(env.all_actions)
        # Reshape a_hat to (N, num_actions) - this is a simplification
        # In practice, we'd need to modify the model to output num_actions logits
        # For now, use a linear layer to map to num_actions
        if not hasattr(model, 'action_head'):
            model.action_head = torch.nn.Linear(args.embed_dim, num_actions).to(device)
        
        # Get hidden representation before action prediction
        # This is a workaround - ideally the model should output num_actions directly
        # For now, we'll use a simpler approach: treat as regression and use MSE
        action_target = a.squeeze(-1).long()  # (N,)
        # Use a simple embedding approach
        loss_raw = ce_loss(a_hat.squeeze(-1), action_target)
        return loss_raw.mean()
    
    # Loss function weighted by trajectory returns (prioritize high-scoring trajectories)
    def loss_fn_weighted(s_hat, a_hat, r_hat, s, a, r):
        # a_hat: (N, 1) - predicted action index (continuous)
        # a: (N, 1) - target action index (continuous, but represents discrete action)
        # Weight loss by trajectory returns to prioritize high-scoring trajectories
        mse_loss = torch.nn.functional.mse_loss(a_hat, a, reduction='none')
        
        # Get trajectory returns from get_batch (stored in closure)
        if hasattr(get_batch, 'last_traj_returns'):
            traj_returns = get_batch.last_traj_returns
            # Normalize returns to create weights
            min_return = traj_returns.min()
            max_return = traj_returns.max()
            if max_return > min_return:
                normalized_returns = (traj_returns - min_return + 1.0) / (max_return - min_return + 1.0)
            else:
                normalized_returns = torch.ones_like(traj_returns)
            
            # Apply power to emphasize high returns more
            weights = torch.pow(normalized_returns, args.return_weight_power)
            # Average weight per batch (simplified - ideally weight per timestep)
            avg_weight = weights.mean()
            weighted_loss = mse_loss.mean() * (1.0 + avg_weight)  # Boost loss for high-return trajectories
        else:
            weighted_loss = mse_loss.mean()
        
        return weighted_loss
    
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=args.batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=loss_fn_weighted,
        eval_fns=[],
    )
    
    # Training loop
    print("Starting training...")
    for it in range(args.max_iters):
        itn = it + 1
        print(f">>> Iteration {itn}/{args.max_iters}")
        logs = trainer.train_iteration(num_steps=args.num_steps_per_iter, iter_num=itn, print_logs=True)
        if logs is None:
            logs = {}
        
        if args.log_to_wandb:
            if WANDB_AVAILABLE:
                wandb.log(logs)
            else:
                print("Warning: wandb logging requested but wandb is not installed.")
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    
    eval_episodes = args.eval_episodes
    eval_turns = []
    eval_scores = []
    
    with torch.no_grad():
        for episode in range(eval_episodes):
            card_supply, nobles = load_card_data(args.card_data_path)
            env = SplendorEnv(card_supply, nobles)
            state = env.reset(card_supply, nobles)
            
            states_history = []
            actions_history = []
            rewards_history = []
            returns_to_go = []
            timesteps = []
            
            done = False
            while not done:
                # Encode current state (use same encoding as training)
                state_encoded = encode_state(state, hasher)
                
                states_history.append(state_encoded)
                if len(states_history) > K:
                    states_history = states_history[-K:]
                    actions_history = actions_history[-K:]
                    rewards_history = rewards_history[-K:]
                    returns_to_go = returns_to_go[-K:]
                    timesteps = timesteps[-K:]
                
                # Prepare input for model
                states_t = torch.from_numpy(np.array(states_history)[None]).to(dtype=torch.float32, device=device)
                # Convert state_mean and state_std to tensors on the same device
                state_mean_t = torch.from_numpy(state_mean).to(dtype=torch.float32, device=device)
                state_std_t = torch.from_numpy(state_std).to(dtype=torch.float32, device=device)
                states_t = (states_t - state_mean_t) / state_std_t
                
                if len(actions_history) > 0:
                    actions_t = torch.from_numpy(np.array(actions_history).reshape(-1, 1)[None]).to(dtype=torch.float32, device=device)
                else:
                    actions_t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                
                if len(rewards_history) > 0:
                    rewards_t = torch.from_numpy(np.array(rewards_history).reshape(-1, 1)[None]).to(dtype=torch.float32, device=device)
                else:
                    rewards_t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                
                # Compute returns to go - use higher target for better scores
                # Set target RTG to a high value (e.g., 90th percentile of training returns)
                target_rtg = np.percentile(train_returns, args.target_rtg_percentile) if len(train_returns) > 0 else 20.0
                if len(rewards_history) > 0:
                    rtg = discount_cumsum(np.array(rewards_history), gamma=1.0)
                    # Use target RTG instead of actual remaining return to encourage high-scoring behavior
                    remaining_steps = max(1, 60 - len(rewards_history))
                    target_remaining = target_rtg - rtg[-1] if len(rtg) > 0 else target_rtg
                    rtg_adjusted = np.concatenate([rtg, [target_remaining]]) if len(rtg) > 0 else np.array([target_rtg])
                    rtg_t = torch.from_numpy(rtg_adjusted.reshape(-1, 1)[None]).to(dtype=torch.float32, device=device) / float(scale)
                else:
                    # At start, use high target RTG
                    rtg_t = torch.full((1, 1, 1), target_rtg / float(scale), dtype=torch.float32, device=device)
                
                ts_t = torch.from_numpy(np.array(timesteps)[None]).to(dtype=torch.long, device=device) if len(timesteps) > 0 else torch.zeros((1, 1), dtype=torch.long, device=device)
                
                # Get action prediction
                action_pred = model.get_action(
                    states_t,
                    actions_t,
                    rewards_t,
                    rtg_t,
                    ts_t,
                )
                
                # Convert action prediction to discrete action
                action_idx = int(torch.clamp(action_pred, 0, len(env.all_actions) - 1).item())
                
                # Get valid actions and choose best valid one
                valid_actions = env.get_legal_actions()
                if len(valid_actions) == 0:
                    break
                
                # Find valid action closest to predicted action
                valid_action_indices = [env.all_actions.index(a) for a in valid_actions if a in env.all_actions]
                if len(valid_action_indices) > 0:
                    # Choose action closest to prediction
                    best_idx = min(valid_action_indices, key=lambda x: abs(x - action_idx))
                    action = env.all_actions[best_idx]
                else:
                    action = random.choice(valid_actions)
                
                # Step environment
                next_state, reward, done, _ = env.step(action)
                
                actions_history.append(env.all_actions.index(action) if action in env.all_actions else 0)
                rewards_history.append(reward)
                timesteps.append(len(states_history) - 1)
                
                state = next_state
                
                if env.turns > 60:
                    done = True
            
            eval_turns.append(env.turns)
            eval_scores.append(env.score)
            print(f"Episode {episode + 1}/{eval_episodes}: {env.turns} turns, {env.score} points")
    
    # Calculate metrics matching the report format (Reinforcement_Learning_Report.pdf)
    # Loss is defined as taking more than 60 turns to win
    num_losses = sum(1 for t in eval_turns if t > 60)
    wins = [t for t in eval_turns if t <= 60]
    avg_turns_without_losses = np.mean(wins) if len(wins) > 0 else 0.0
    avg_score = np.mean(eval_scores)
    avg_turns = np.mean(eval_turns)
    win_rate = len(wins) / len(eval_turns) if len(eval_turns) > 0 else 0.0
    
    print("=" * 50)
    print(f"Decision Transformer Agent Results - {eval_episodes} episodes")
    print("=" * 50)
    print(f"Average score over {eval_episodes} episodes: {avg_score:.2f}")
    print(f"Number of Losses: {num_losses}")
    print(f"Average turns to win without losses: {avg_turns_without_losses:.2f}")
    print(f"Average turns (all episodes): {avg_turns:.2f}")
    print(f"Win rate: {win_rate:.2%}")
    print("=" * 50)
    
    # Save model
    if args.save_model:
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--card_data_path", type=str, default="data/card_data.csv")
    parser.add_argument("--num_trajectories", type=int, default=10000, help="Number of trajectories to generate")
    parser.add_argument("--top_percentile", type=float, default=10.0, help="Top percentile of trajectories to keep for training (default: 10% = top 10%)")
    parser.add_argument("--return_weight_power", type=float, default=2.0, help="Power for return-based sampling weights (higher = more emphasis on high returns)")
    parser.add_argument("--target_rtg_percentile", type=float, default=90.0, help="Percentile of training returns to use as target RTG during evaluation (default: 90th percentile)")
    parser.add_argument("--agent_type", type=str, default="random", choices=["random", "dqn"])
    parser.add_argument("--use_state_hashing", action="store_true", help="Use feature hashing for states")
    parser.add_argument("--state_size", type=int, default=1000, help="Size of hashed state vector")
    parser.add_argument("--K", type=int, default=40, help="Context length")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--num_steps_per_iter", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--model_path", type=str, default="models/splendor_dt_model.pt")
    args = parser.parse_args()
    
    main(args)

