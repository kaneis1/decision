import torch
import torch.nn as nn
import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class GPT2BCModel(TrajectoryModel):
    """
    GPT-2-based behavior cloning model.

    It treats the sequence of past states as tokens and predicts the next action.
    No returns-to-go, no Decision Transformer structure, just sequence modeling.
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        n_layer,
        n_head,
        max_length,
        dropout=0.1,
        action_tanh=True,
        max_ep_len=4096,
        **kwargs,
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.max_length = max_length

        # GPT2 config (we ignore vocab since we pass inputs_embeds)
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=max_length,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            **kwargs,
        )

        # GPT2 backbone without positional embeddings (trajectory_gpt2.py)
        self.transformer = GPT2Model(config)

        # Embed states into hidden_size
        self.embed_state = nn.Linear(state_dim, hidden_size)
        # Embed timesteps (like DecisionTransformer)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # Predict next action from last token hidden state
        head = [nn.Linear(hidden_size, act_dim)]
        if action_tanh:
            head.append(nn.Tanh())
        self.predict_action = nn.Sequential(*head)

    def forward(self, states, actions, rewards, timesteps=None,
                attention_mask=None, target_return=None):
        """
        states: (B, T, state_dim)
        timesteps: (B, T) timestep indices
        returns: None, actions_pred (B, 1, act_dim), None
        """
        batch_size, seq_len, _ = states.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=states.device)

        # embed states
        state_embeddings = self.embed_state(states)                # (B, T, H)
        
        # Add timestep embeddings if provided (like DecisionTransformer)
        if timesteps is not None:
            time_embeddings = self.embed_timestep(timesteps)       # (B, T, H)
            state_embeddings = state_embeddings + time_embeddings
        
        x = self.embed_ln(state_embeddings)                        # (B, T, H)

        # GPT2 expects (B, T, H) as inputs_embeds
        transformer_outputs = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
        )
        hidden = transformer_outputs["last_hidden_state"]          # (B, T, H)

        # use last time step hidden state to predict next action
        last_hidden = hidden[:, -1, :]                             # (B, H)
        action_pred = self.predict_action(last_hidden)             # (B, act_dim)
        action_pred = action_pred.view(batch_size, 1, self.act_dim)

        return None, action_pred, None

    def get_action(self, states, actions, rewards, timesteps=None, **kwargs):
        """
        states: (T, state_dim) for a single episode prefix
        timesteps: (T,) timestep indices (optional)
        returns: (act_dim,) for next action
        """
        states = states.reshape(1, -1, self.state_dim)
        
        # Handle timesteps if provided
        if timesteps is not None:
            timesteps = timesteps.reshape(1, -1)

        # clip/pad to max_length like MLPBC and DT do
        if states.shape[1] > self.max_length:
            states = states[:, -self.max_length:]
            if timesteps is not None:
                timesteps = timesteps[:, -self.max_length:]
        elif states.shape[1] < self.max_length:
            pad = torch.zeros(
                (1, self.max_length - states.shape[1], self.state_dim),
                dtype=states.dtype,
                device=states.device,
            )
            states = torch.cat([pad, states], dim=1)
            if timesteps is not None:
                # Pad timesteps with zeros (will be masked out anyway)
                timesteps_pad = torch.zeros(
                    (1, self.max_length - timesteps.shape[1]),
                    dtype=timesteps.dtype,
                    device=timesteps.device,
                )
                timesteps = torch.cat([timesteps_pad, timesteps], dim=1)

        attention_mask = torch.ones((1, self.max_length), dtype=torch.long, device=states.device)

        _, action_pred, _ = self.forward(states, None, None, timesteps=timesteps, 
                                         attention_mask=attention_mask, **kwargs)
        return action_pred[0, -1]
