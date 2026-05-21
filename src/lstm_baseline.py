"""
BiLSTM floor baseline with pre-trained word embeddings (GloVe 300d recommended).
Config-driven via configs/lstm.yaml.

  1. Build a word vocabulary from the training set texts
  2. Load GloVe embeddings (download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip)
  3. Define a Bidirectional LSTM model in PyTorch
  4. Write a training loop with early stopping (patience is in the config)
  5. Use class weights from src.data.class_weights() — severe is only 6% of the data
  6. Call src.eval.evaluate() on the test set when done

Usage:
    python src/lstm_baseline.py --config configs/lstm.yaml

See notebooks/03_lstm_run.ipynb for step-by-step guidance.
"""

import argparse
import yaml
from src.data import load_splits, class_weights
from src.eval import evaluate
from src.utils import set_seed, get_logger

log = get_logger(__name__)

# TODO: implement vocabulary building
# TODO: implement GloVe loading
# TODO: implement PyTorch Dataset class
# TODO: implement BiLSTM model (nn.Module)
# TODO: implement train() function
# TODO: call evaluate() on test set at the end

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    # TODO: call your train() function here
