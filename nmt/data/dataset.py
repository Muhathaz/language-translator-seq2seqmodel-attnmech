"""Dataset module for handling translation data."""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    """Dataset for machine translation.

    Args:
        source_texts (List[List[int]]): List of source sequences (as token indices)
        target_texts (List[List[int]]): List of target sequences (as token indices)
        pad_idx (int): Index of padding token
    """

    def __init__(
        self,
        source_texts: List[List[int]],
        target_texts: List[List[int]],
        pad_idx: int = 1,
    ):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.pad_idx = pad_idx

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.source_texts[idx], self.target_texts[idx]


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    pad_idx: int = 1,
) -> Dict[str, torch.Tensor]:
    """Collate function for creating batches.

    Args:
        batch (List[Tuple[List[int], List[int]]]): List of (source, target) pairs
        pad_idx (int): Index of padding token

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - src: Padded source sequences
            - tgt: Padded target sequences
            - src_lengths: Original lengths of source sequences
            - tgt_lengths: Original lengths of target sequences
    """
    # Separate source and target sequences
    src_seqs, tgt_seqs = zip(*batch)

    # Get sequence lengths
    src_lengths = torch.tensor([len(s) for s in src_seqs])
    tgt_lengths = torch.tensor([len(t) for t in tgt_seqs])

    # Convert to tensors and pad
    src_padded = pad_sequence(
        [torch.tensor(s) for s in src_seqs],
        batch_first=True,
        padding_value=pad_idx,
    )
    tgt_padded = pad_sequence(
        [torch.tensor(t) for t in tgt_seqs],
        batch_first=True,
        padding_value=pad_idx,
    )

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_lengths": src_lengths,
        "tgt_lengths": tgt_lengths,
    }
