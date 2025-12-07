# 🤖 ARC-AGI: From-Scratch Autoregressive Transformer for the Abstraction and Reasoning Corpus

## 🌟 Overview

This repository presents a complete, **from-scratch** implementation of an **Autoregressive Transformer** pipeline specifically designed to tackle the **Abstraction and Reasoning Corpus (ARC)** challenge.

The ARC challenge requires an agent to generalize abstract concepts from very few examples and apply that reasoning to novel tasks. This project approaches ARC as a sequence-to-sequence prediction problem, where the input and output grids are serialized into tokens and processed by a custom-built autoregressive transformer model, leveraging the power of attention mechanisms for complex pattern recognition.

### Key Features ✨

  * **Custom Transformer Implementation:** A complete, modular implementation of the standard Transformer architecture (`model.py`), built without reliance on high-level libraries like `transformers`.
  * **ARC Data Pipeline:** Scripts to generate, tokenize, and augment training data from the original ARC tasks (`create_training_data.py`, `get_dataloaders.py`).
  * **Custom Tokenizer & Serialization:** Specialized tokenizer (`get_tokenizer.py`) designed to convert 2D ARC grids (10 colors + grid structure) into a linear sequence of tokens suitable for autoregressive modeling.
  * **Modular Training Engine:** Separate scripts for training, configuration, and evaluation (`train.py`, `engine.py`, `configurations.py`).
  * **Evaluation Framework:** Dedicated scripts for running inference and evaluating model performance against the ARC metrics (`eval.py`).
