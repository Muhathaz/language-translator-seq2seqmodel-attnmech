"""Neural Machine Translation model combining encoder, decoder, and attention."""

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class NMTModel(nn.Module):
    """Neural Machine Translation model with attention.

    Args:
        src_vocab_size (int): Size of the source vocabulary
        tgt_vocab_size (int): Size of the target vocabulary
        embedding_dim (int): Dimension of the embedding layers
        hidden_dim (int): Number of hidden units in LSTM layers
        num_layers (int): Number of LSTM layers
        attention_dim (int): Dimension of attention vectors
        dropout (float): Dropout rate
        pad_idx (int): Index of padding token
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        attention_dim: int,
        dropout: float = 0.3,
        pad_idx: int = 1,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            attention_dim=attention_dim,
            dropout=dropout,
            encoder_hidden_dim=hidden_dim * 2,  # bidirectional
        )

        self.pad_idx = pad_idx

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask for source sequences.

        Args:
            src (torch.Tensor): Source sequences [batch_size, seq_len]

        Returns:
            torch.Tensor: Binary mask [batch_size, seq_len]
        """
        mask = (src != self.pad_idx).float()
        return mask

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            src (torch.Tensor): Source sequences [batch_size, src_len]
            src_lengths (torch.Tensor): Length of source sequences [batch_size]
            tgt (torch.Tensor): Target sequences [batch_size, tgt_len]
            teacher_forcing_ratio (float): Probability of using teacher forcing

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output distributions [batch_size, tgt_len, vocab_size]
                - Attention weights [batch_size, tgt_len, src_len]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size

        # Initialize outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=src.device)
        attentions = torch.zeros(batch_size, tgt_len, src.shape[1], device=src.device)

        # Create source padding mask
        encoder_mask = self.create_mask(src)

        # Encode source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # First input to decoder is start token
        decoder_input = tgt[:, 0]

        for t in range(1, tgt_len):
            # Pass through decoder
            output, hidden, attention = self.decoder(
                decoder_input, hidden, encoder_outputs, encoder_mask
            )

            # Store predictions and attention weights
            outputs[:, t] = output
            attentions[:, t] = attention

            # Teacher forcing
            teacher_force = torch.rand(1) < teacher_forcing_ratio

            # Get next input token
            if teacher_force:
                decoder_input = tgt[:, t]
            else:
                decoder_input = output.argmax(1)

        return outputs, attentions

    def translate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_length: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Translate source sequences.

        Args:
            src (torch.Tensor): Source sequences [batch_size, src_len]
            src_lengths (torch.Tensor): Length of source sequences [batch_size]
            max_length (int): Maximum length of generated translations

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Generated translations [batch_size, max_length]
                - Attention weights [batch_size, max_length, src_len]
        """
        batch_size = src.shape[0]

        # Initialize outputs
        outputs = torch.zeros(
            batch_size, max_length, dtype=torch.long, device=src.device
        )
        attentions = torch.zeros(
            batch_size, max_length, src.shape[1], device=src.device
        )

        # Create source padding mask
        encoder_mask = self.create_mask(src)

        # Encode source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # First input to decoder is start token (assumed to be index 2)
        decoder_input = torch.full(
            (batch_size,), 2, dtype=torch.long, device=src.device
        )

        for t in range(max_length):
            # Pass through decoder
            output, hidden, attention = self.decoder(
                decoder_input, hidden, encoder_outputs, encoder_mask
            )

            # Store predictions and attention weights
            prediction = output.argmax(1)
            outputs[:, t] = prediction
            attentions[:, t] = attention

            # Stop if all sequences have generated end token (assumed to be index 3)
            if (prediction == 3).all():
                break

            decoder_input = prediction

        return outputs, attentions
