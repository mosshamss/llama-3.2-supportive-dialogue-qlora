# Experiment notes and quality review

This document records observations about the uploaded notebook. The notebook is copied unchanged, including all evaluation code, graphs and saved generated answers. No new experiment was run.

## Original training run

- Input file: `llama_therapist.ipynb`.
- 4-bit Llama 3.2 3B Instruct model and LoRA configuration, `r=16`, `lora_alpha=16`.
- Initial 99,086 conversation-row dataset listed on the dataset's Hugging Face page. The notebook samples `frac=0.2`.
- Extracted examples: **161,466 input-response pairs** (this is a count of individual turns, not source conversations).
- Train/validation: **160,658 / 808** input-response pairs, split using `test_size=0.005`, seed 42.
- Training: **48 optimizer steps**, 2 per-device batch, accumulation 4, effective batch 8, learning rate 2e-4, maximum token length 150.
- Recorded environment: Colab Tesla T4 GPU; the notebook reported Unsloth 2024.12.12, Transformers 4.47.1, Torch 2.5.1+cu121.

## Observed experimental artifacts

- Training loss visualised over 48 steps (see `../assets/training_loss.png`).
- Evaluation cell displayed approximately **ROUGE-1 0.1164, ROUGE-2 0.0162, ROUGE-L 0.0652, METEOR 0.2156**. BERTScore was printed as a long list of example-level values, not an aggregated score.
- A later cell displayed an inconsistent dictionary with a `bleu` entry although the current evaluation function does not compute BLEU. This indicates that notebook execution state may be stale.
- Qualitative inference tried prompts about anxiety, motivation, friendships, hobbies, loneliness, and time management.

### Why these evaluation figures are not valid benchmarks

1. `apply_chat_template(messages, ...)` is called on a list of user-message dictionaries representing different examples, which creates **one multiround conversation**, not a batch of independent prompts. For genuine batched evaluation, tokenize a list of separate conversations with correct padding and an attention mask.
2. `model.generate(...)` returns full prompt-plus-continuation token sequences for a decoder-only model. The code decodes entire sequences and compares them to reference replies. The prompt tokens must be sliced off before decoding predictions.
3. The split is performed after extraction into turn pairs; related turns from the same original synthetic conversation can leak across train and validation splits. Instead split *conversation IDs before flattening* and preferably remove near duplicates.
4. Long outputs can be truncated by the 150-token setting during training. Measure truncation rates before training again.
5. The dataset card explicitly warns synthetic conversations do not represent actual therapy, and some samples include recurring names. Model responses copied a recurring name even when it was absent from the prompt.
6. There is no holdout comparison to the base model, no reliable out-of-domain testing, and no mental-health safety evaluation.

## Suggested next experiment

1. Filter, inspect, de-duplicate and document synthetic records, preserving source conversation IDs.
2. Split by conversation ID into separate train/validation/test partitions, before pairing turns.
3. Apply the correct Llama chat template, account for input/output length truncation, and train beyond 48 steps with learning-curve monitoring.
4. Build a held-out evaluation with independent prompts. Decode generated completions only, with correct padding and attention masks.
5. Compare the same prompts using base and adapted models. Report aggregate ROUGE/METEOR/BERTScore with sample count and limitations.
6. Conduct separate domain-specific human reviews, risk evaluations for harmful advice and crisis scenarios, and bias tests prior to any real-world use.

## Reproducibility and data distribution

The uploaded notebook included multiple version-specific installation cells and a compatibility monkey patch. They were retained as part of the historical record, but a fresh execution in a current Colab environment may require updated installation guidance and code. The model checkpoint and dataset are not redistributed. Refer to the source links and applicable model/dataset terms.
