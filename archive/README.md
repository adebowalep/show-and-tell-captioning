# Original Udacity Assignment Submission

This directory contains the original assignment submission for the **Image Captioning** project
in Udacity's Computer Vision Nanodegree, preserved unmodified for provenance and reference.

## Contents

### Jupyter Notebooks (with saved outputs)

1. **0_Dataset.ipynb** — Dataset exploration and setup
   - Load COCO dataset metadata
   - Display sample images and captions

2. **1_Preliminaries.ipynb** — Preliminary architecture setup
   - Implement vocabulary from captions
   - Build data loaders
   - Verify ResNet-50 encoder shapes

3. **2_Training.ipynb** — Full training pipeline (completed)
   - Hyperparameter selection with written justifications (Questions 1-4)
   - Training loop with checkpoint resuming
   - Loss and perplexity logging over 3 epochs
   - All training output saved to notebook

4. **3_Inference.ipynb** — Validation and inference (completed)
   - Load trained encoder and decoder checkpoints
   - Generate captions for validation images
   - Compute BLEU-1/2/3/4 scores
   - Show 4 example predictions (2 good, 2 poor) with images

4. **4_Zip Your Project Files and Submit.ipynb** — Submission helper
   - Zip project files for upload

### Python Files

- **model.py** — Original `DecoderRNN` class with greedy decoding
- **data_loader.py** — Training data loader for COCO captions
- **data_loader_val.py** — Validation data loader (1000 images)
- **vocabulary.py** — Vocabulary class and builder

### Supporting Files

- **requirements.txt** — Original course dependencies
- **LICENSE** — MIT license

## Key Results (from 3_Inference.ipynb)

| Metric | Value |
|--------|-------|
| BLEU-1 | 0.656 |
| BLEU-2 | 0.483 |
| BLEU-3 | 0.326 |
| BLEU-4 | 0.207 |

Training completed all 3 epochs (Steps 1–12942), with stable loss reduction from ~6.5 to ~3.1.

## Notes

This original submission met all Udacity rubric requirements:
- ✓ `model.py` contains working `EncoderCNN` and `DecoderRNN`
- ✓ `2_Training.ipynb` shows full training with answers to design questions
- ✓ `3_Inference.ipynb` shows BLEU scores and example captions
- ✓ All outputs saved in notebook cells

The refactored `image_captioning/` package in the parent directory builds on this work,
adding type hints, docstrings, CLI tools, tests, and modern Python structure while
preserving the exact same trained model weights and core logic.
