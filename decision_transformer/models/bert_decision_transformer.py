import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertDecisionTransformer(nn.Module):
    """
    BERT encoder used as the trajectory transformer in a Decision-Transformer-style model.

    Sequence layout per trajectory:
        (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K)
    where each token is embedded to d_model (BERT hidden size) and
    passed to a BERT encoder via `inputs_embeds`.

    
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        K: int,
        max_ep_len: int = 4096,
        bert_config: BertConfig = None,
        use_action_ids: bool = False,
        action_tanh: bool = False,
        n_layer: int = None,
        n_head: int = None,
        n_inner: int = None,
        activation_function: str = "relu",
        dropout: float = 0.1,
        max_position_embeddings: int = 1024,
        binary_classification: bool = True,
    ):
        super().__init__()

        # --- BERT backbone ---
        # Use provided config or create one
        if bert_config is None:
            bert_config = BertConfig(
                vocab_size=1,  # Dummy, we use inputs_embeds
                hidden_size=hidden_size,
                num_hidden_layers=n_layer if n_layer is not None else 2,
                num_attention_heads=n_head if n_head is not None else 4,
                intermediate_size=n_inner if n_inner is not None else 4 * hidden_size,
                hidden_act=activation_function,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                max_position_embeddings=max_position_embeddings,
                type_vocab_size=1,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
            )
        
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.d_model = hidden_size
        self.K = K
        self.act_dim = act_dim
        self.use_action_ids = use_action_ids
        self.binary_classification = binary_classification

        # --- Modality embeddings (R, s, a) ---
        # Returns-to-go is usually scalar.
        self.embed_rtg = nn.Linear(1, self.d_model)
        # State could be any feature vector: last actions, MO features, etc.
        self.embed_state = nn.Linear(state_dim, self.d_model)

        if use_action_ids:
            # Actions are discrete indices -> embed like tokens
            self.action_embedding = nn.Embedding(act_dim, self.d_model)
            self.action_feature_dim = None
        else:
            # Actions are continuous feature vectors
            self.embed_action = nn.Linear(act_dim, self.d_model)
            self.action_feature_dim = act_dim

        # --- Timestep embedding (like DT) ---
        # We learn a timestep embedding of size d_model for timesteps [0..max_ep_len-1].
        self.embed_timestep = nn.Embedding(max_ep_len, self.d_model)

        # --- Action prediction head ---
        # Given BERT hidden state at an action token, output logits over actions.
        if binary_classification:
            # For binary classification: output 1 logit
            self.predict_action = nn.Linear(self.d_model, 1)
        else:
            # For multi-class: output act_dim logits
            self.predict_action = nn.Linear(self.d_model, act_dim)
        
        if action_tanh and not binary_classification:
            # Only apply tanh for multi-class continuous actions
            self.predict_action = nn.Sequential(
                nn.Linear(self.d_model, act_dim),
                nn.Tanh()
            )

    def forward(
        self,
        states: torch.Tensor,        # (B, K, state_dim)
        actions: torch.Tensor,       # (B, K) if ids, or (B, K, act_dim) if continuous
        returns_to_go: torch.Tensor, # (B, K, 1)
        timesteps: torch.Tensor,     # (B, K) integer timestep indices in [0, K-1]
        attention_mask: torch.Tensor = None,  # (B, K) mask over timesteps (1 = valid, 0 = pad)
    ):
        B, K, _ = states.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"

        # ----- Embed each modality -----
        rtg_emb = self.embed_rtg(returns_to_go)            # (B, K, d)
        state_emb = self.embed_state(states)               # (B, K, d)

        if self.use_action_ids:
            # actions: (B, K) -> (B, K, d)
            action_emb = self.action_embedding(actions.long())
        else:
            # actions: (B, K, act_dim)
            action_emb = self.embed_action(actions)

        # ----- Add timestep embeddings -----
        # timesteps: (B, K) -> (B, K, d)
        time_emb = self.embed_timestep(timesteps.long())
        rtg_emb = rtg_emb + time_emb
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb

        # ----- Interleave as (R1, s1, a1, R2, s2, a2, ... ) -----
        # Stack along a "modality" axis = 3, then reshape.
        # stacked: (B, K, 3, d)  -> view: (B, 3K, d)
        tokens = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
        tokens = tokens.view(B, 3 * K, self.d_model)       # (B, 3K, d_model)

        # ----- Build attention mask for BERT -----
        if attention_mask is None:
            # If no mask: all timesteps valid.
            # Need shape (B, 3K): repeat each timestep's mask for R, s, a.
            attention_mask = torch.ones(B, K, device=states.device, dtype=torch.long)

        # attention_mask: (B, K) -> (B, 3K)
        attn_mask_3k = attention_mask.repeat_interleave(3, dim=1)

        # ----- Run BERT on our embeddings -----
        # We bypass word embeddings by using `inputs_embeds`.
        bert_outputs = self.bert(
            inputs_embeds=tokens,
            attention_mask=attn_mask_3k,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = bert_outputs.last_hidden_state            # (B, 3K, d_model)

        # ----- Take only the hidden states at action positions -----
        # Sequence indices: 0:R1, 1:s1, 2:a1, 3:R2, 4:s2, 5:a2, ...
        # So action indices are 2,5,8,... => slice with step 3 starting at 2.
        action_hidden = hidden[:, 2::3, :]                 # (B, K, d_model)

        # ----- Predict next action logits for each timestep -----
        if self.binary_classification:
            action_logits = self.predict_action(action_hidden)  # (B, K, 1)
        else:
            action_logits = self.predict_action(action_hidden)  # (B, K, act_dim)

        return action_logits
