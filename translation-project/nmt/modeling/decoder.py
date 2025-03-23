"""Decoder module for the NMT system."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention


class Decoder(nn.Module):
    """LSTM Decoder with attention for the NMT system.

    Args:
        vocab_size (int): Size of the target vocabulary
        embedding_dim (int): Dimension of the embedding layer
        hidden_dim (int): Number of hidden units in the LSTM
        num_layers (int): Number of LSTM layers
        attention_dim (int): Dimension of the attention vectors
        dropout (float): Dropout rate
        encoder_hidden_dim (int): Dimension of encoder hidden states (needed for attention)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        attention_dim: int,
        dropout: float = 0.3,
        encoder_hidden_dim: int = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # If encoder is bidirectional, hidden states are doubled
        encoder_hidden_dim = encoder_hidden_dim or hidden_dim * 2

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim, attention_dim)

        # Input to LSTM: embedding + context vector
        self.lstm = nn.LSTM(
            embedding_dim + encoder_hidden_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.output = nn.Linear(
            hidden_dim + encoder_hidden_dim + embedding_dim, vocab_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass of the decoder.

        Args:
            input (torch.Tensor): Input tokens [batch_size, 1]
            hidden (tuple[torch.Tensor, torch.Tensor]): Previous hidden state and cell state
            encoder_outputs (torch.Tensor): Encoder outputs [batch_size, src_len, hidden_dim * 2]
            encoder_mask (torch.Tensor): Source padding mask [batch_size, src_len]

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                - Output distribution over vocabulary [batch_size, vocab_size]
                - New hidden state and cell state
                - Attention weights [batch_size, src_len]
        """
        input = input.unsqueeze(1)  # Add sequence length dimension
        embedded = self.dropout(self.embedding(input))

        # Calculate attention
        hidden_state = hidden[0][-1].unsqueeze(0)  # Use last layer's hidden state
        context, attention_weights = self.attention(
            hidden_state.squeeze(0), encoder_outputs, encoder_mask
        )

        # Combine embedding and context vector
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)

        # Pass through LSTM
        output, hidden = self.lstm(lstm_input, hidden)

        # Combine LSTM output, context, and embedding for prediction
        output = torch.cat((output.squeeze(1), context, embedded.squeeze(1)), dim=1)

        # Get vocabulary distribution
        prediction = self.output(output)

        return prediction, hidden, attention_weights

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state and cell state.

        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Initial hidden state and cell state
        """
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
        )
