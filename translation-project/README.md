# German-English Neural Machine Translation

A PyTorch-based Neural Machine Translation system with attention mechanism for German-English translation.

## Project Structure

```
.
├── data/                      # Data directory
│   ├── raw/                  # Original untouched datasets
│   └── processed/            # Preprocessed and tokenized data
├── nmt/                      # Main package directory
│   ├── data/                # Data processing modules
│   ├── modeling/            # Neural network model components
│   └── utils/               # Utility functions and helpers
├── scripts/                  # Training and preprocessing scripts
├── notebooks/                # Jupyter notebooks for analysis
├── configs/                  # Configuration files
└── tests/                   # Unit tests
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

2. Prepare the data:
```bash
make download_data
make preprocess_data
```

3. Train the model:
```bash
make train
```

## Project Components

- **Data Processing**: Handles tokenization, batching, and dataset management
- **Model Architecture**: 
  - Encoder: Bidirectional LSTM with embeddings
  - Decoder: LSTM with attention mechanism
  - Attention: Scaled dot-product attention
- **Training Pipeline**: Includes logging, checkpointing, and validation
- **Inference**: Beam search decoding for translation generation

## Usage

See the Makefile for common commands and the configs directory for model configuration. 