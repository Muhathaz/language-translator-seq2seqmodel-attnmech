"""Attention mechanism for the NMT system."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Scaled Dot-Product Attention mechanism.

    Args:
        hidden_dim (int): Dimension of the hidden states
        attention_dim (int): Dimension of the attention vectors
    """

    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()

        self.attention_dim = attention_dim

        # Linear layers to transform encoder and decoder states
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate attention weights and weighted sum of encoder outputs.

        Args:
            decoder_hidden (torch.Tensor): Current decoder hidden state [batch_size, hidden_dim]
            encoder_outputs (torch.Tensor): All encoder outputs [batch_size, src_len, hidden_dim * 2]
            encoder_mask (torch.Tensor): Source padding mask [batch_size, src_len]

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Context vector [batch_size, hidden_dim * 2]
                - Attention weights [batch_size, src_len]
        """
        batch_size, src_len, encoder_dim = encoder_outputs.shape

        # Repeat decoder hidden state src_len times
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Concatenate encoder outputs and decoder hidden state
        energy = torch.tanh(
            self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2))
        )

        # Calculate attention scores
        attention = self.v(energy).squeeze(2)

        # Mask out padding tokens
        attention = attention.masked_fill(encoder_mask == 0, float("-inf"))

        # Calculate attention weights
        attention_weights = F.softmax(attention, dim=1)

        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights
