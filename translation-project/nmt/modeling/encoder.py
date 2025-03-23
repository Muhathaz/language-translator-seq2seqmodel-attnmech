"""Encoder module for the NMT system."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Bidirectional LSTM Encoder for the NMT system.

    Args:
        vocab_size (int): Size of the source vocabulary
        embedding_dim (int): Dimension of the embedding layer
        hidden_dim (int): Number of hidden units in the LSTM
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        bidirectional (bool): Whether to use bidirectional LSTM
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the encoder.

        Args:
            src (torch.Tensor): Source sequences [batch_size, seq_len]
            src_lengths (torch.Tensor): Length of each sequence [batch_size]

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                - Encoder outputs [batch_size, seq_len, hidden_dim * num_directions]
                - Tuple of final hidden state and cell state
        """
        embedded = self.dropout(self.embedding(src))

        # Pack padded sequence for efficient computation
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack the sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # If bidirectional, combine forward and backward states
        if self.num_directions == 2:
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)
            cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
            cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)

        return outputs, (hidden, cell)
