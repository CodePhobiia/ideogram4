"""Training spine: QLoRA flow-matching fine-tune of Ideogram 4's conditional transformer.

Runs on a CUDA GPU (Vast) — the 9.3B model can't load on the local CPU box.
Entry point: `python -m rtie.training.train_sft` (see train_sft.py).
"""
