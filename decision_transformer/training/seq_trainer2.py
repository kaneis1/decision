import numpy as np
import torch
import inspect

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        # get_batch returns: s, a, r, d, mo, rtg, timesteps_t, mask_t
        states, actions, rewards, dones, mos, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        # Check model forward signature to handle both DT and GPT2 models
        forward_sig = inspect.signature(self.model.forward)
        forward_params = list(forward_sig.parameters.keys())
        
        # DecisionTransformer expects: states, actions, rewards, returns_to_go, timesteps, attention_mask=None, mo=None
        # GPT2BCModel expects: states, actions, rewards, attention_mask=None, target_return=None
        if 'mo' in forward_params:
            # DecisionTransformer with MO support
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, mo=mos,
            )
        elif 'returns_to_go' in forward_params or 'target_return' in forward_params:
            # DecisionTransformer without MO or GPT2BCModel
            # Check if it's GPT2BCModel (has timesteps as optional parameter)
            if 'returns_to_go' not in forward_params:
                # GPT2BCModel: forward(states, actions, rewards, timesteps=None, attention_mask=None, target_return=None)
                state_preds, action_preds, reward_preds = self.model.forward(
                    states, actions, rewards, timesteps=timesteps, attention_mask=attention_mask,
                )
            else:
                # DecisionTransformer without MO: forward(states, actions, rewards, returns_to_go, timesteps, attention_mask=None)
                state_preds, action_preds, reward_preds = self.model.forward(
                    states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
                )
        else:
            # Fallback: try standard DT signature without MO
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
            )

        # Handle different output shapes: DT returns (B, T, act_dim), GPT2 returns (B, 1, act_dim)
        if action_preds.shape[1] == 1:
            # GPT2BCModel: (B, 1, act_dim) - predicts next action after the sequence
            # Target should be the action at the next timestep after the sequence
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)  # (B, act_dim)
            # Get the last valid action from each sequence as target
            # (GPT2 predicts next action, so we use the last action in the window as target)
            batch_size = action_target.shape[0]
            mask_bool = attention_mask.bool()  # (B, T)
            # Find last valid position for each sequence (index of last True in mask)
            last_valid_indices = (mask_bool.sum(dim=1) - 1).clamp(min=0)  # (B,)
            action_target = action_target[torch.arange(batch_size, device=action_target.device), last_valid_indices]  # (B, act_dim)
        else:
            # DecisionTransformer: (B, T, act_dim) - predict action for each timestep
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            # Calculate accuracy as diagnostic metric (apply sigmoid and threshold at 0.5)
            action_probs = torch.sigmoid(action_preds)
            action_preds_binary = (action_probs >= 0.5).float()
            accuracy = (action_preds_binary == action_target).float().mean().detach().cpu().item()
            self.diagnostics['training/action_accuracy'] = accuracy

        return loss.detach().cpu().item()
