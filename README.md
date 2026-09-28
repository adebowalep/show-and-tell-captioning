# Image Captioning: CNN-RNN Encoder-Decoder for Automatic Image Description

![Python](https://img.shields.io/badge/python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.9+-red)
![license](https://img.shields.io/badge/license-MIT-green)

A production-ready implementation of automatic image captioning using a CNN-RNN
(convolutional neural network paired with recurrent neural network) encoder-decoder
architecture. Given an image, the model generates natural-language captions that
accurately describe objects, activities, and spatial relationships.

**What this demonstrates:** training and deploying a Show-and-Tell encoder-decoder
model on the COCO dataset, achieving competitive caption quality (BLEU-4: 0.207)
while remaining interpretable and maintainable, with both greedy and beam-search
inference modes, and a CLI tool for easy deployment.

## Problem Statement

Automatic image captioning is the task of generating natural-language descriptions
of visual content. Given only an image, produce a sequence of words that meaningfully
describes what the image contains — objects, their attributes, spatial relationships,
and activities.

This is a multimodal problem: the input is visual (pixels) and the output is sequential
(words). A single model must:
- **Extract visual features** from the image (the "encoder")
- **Decode those features into words**, attending to relevant parts of the image
  at each step (the "decoder")
- **Learn the joint distribution** of images and captions end-to-end

## Architecture / Data Flow

```
Image → Encoder (ResNet-50) → Image Features (256-dim)
                                     ↓
                          LSTM Decoder + Vocabulary
                                     ↓
                          Caption (sequence of words)
```

### Encoder: Pretrained ResNet-50

- Removes the final classification layer of a pretrained ResNet-50
- Outputs a 2048-dimensional feature vector per image
- Projects to 256 dimensions (trainable projection layer)
- ResNet weights are **frozen**; only the projection layer learns

### Decoder: LSTM with Embedding and Output Layers

- **Embedding layer**: maps word indices → 256-dimensional dense vectors
- **LSTM layer**: processes embedded words sequentially, maintaining a 512-dimensional hidden state
- **Output/linear layer**: projects LSTM hidden states back to vocabulary size
  (produces logits for the next word)

### Training: Teacher Forcing

- At each step, the decoder receives the **true previous word** (from the reference caption),
  not its own predicted word
- This allows training without exposure bias, making the model learn faster and more stably
- Captions are tokenized, vocabulary is limited to words appearing ≥5 times in COCO (9,490 words)

### Inference: Greedy & Beam Search

- **Greedy decoding**: at each step, pick the token with highest probability.
  Fast, but may miss better sequences.
- **Beam search**: maintain the top-k sequences at each step, exploring multiple hypotheses
  in parallel. Slower but produces higher-quality captions (especially with beam_width ≥ 3).

## Features

- **Production-grade encoder**: pretrained ResNet-50 image feature extraction with frozen weights
- **Trainable LSTM decoder**: efficient single-layer LSTM with embedding and output projection
- **Flexible inference**: both greedy decoding and beam search (width configurable via CLI)
- **Vocabulary management**: 9,490-word vocabulary from COCO, with <start>, <end>, <unk> tokens
- **CLI tool**: `caption-image photo.jpg` for quick single-image inference
- **Type hints & docstrings**: modern Python practices for maintainability
- **BLEU metric**: compute BLEU-1/2/3/4 scores against reference captions
- **End-to-end inference API**: `Captioner` class handles loading models and generating captions

## Repository Layout

```
image_captioning/          # Main package
├── __init__.py
├── model.py               # EncoderCNN, DecoderRNN (with beam search)
├── vocabulary.py          # Vocabulary: load vocab.pkl, idx↔word mapping
├── caption.py             # Captioner: high-level inference API
├── metrics.py             # BLEU score computation
└── cli.py                 # Command-line entry point

models/                    # Pre-trained weights (not in repo, fetch from Release)
├── encoder-3.pkl          # Trained ResNet-50 projection
├── decoder-3.pkl          # Trained LSTM decoder
└── vocab.pkl              # Vocabulary mapping (9,490 words)

tests/                     # Test suite
├── test_model.py
├── test_vocabulary.py
└── test_metrics.py

archive/                   # Original Udacity assignment (preserved as-is)
notebooks/                 # Original course materials (reference)

docs/
├── metrics.json           # BLEU scores from validation run
└── example_captions.md    # Sample model outputs

pyproject.toml             # Package metadata & dependencies
README.md                  # This file
LICENSE                    # MIT
```

## Installation

```bash
git clone https://github.com/adebowalep/image-captioning.git
cd image-captioning

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e .
```

Verified with Python 3.8+ and PyTorch 1.9+.

## Quick Start: CLI Demo

```bash
caption-image path/to/image.jpg
```

This uses greedy decoding. For better quality, add beam search:

```bash
caption-image path/to/image.jpg --beam-width 3 --max-len 20
```

For all options:

```bash
caption-image --help
```

## Python API

```python
from image_captioning import Captioner

captioner = Captioner(
    encoder_path="models/encoder-3.pkl",
    decoder_path="models/decoder-3.pkl",
    vocab_path="models/vocab.pkl",
)

caption = captioner.caption_image("photo.jpg", beam_width=3)
print(caption)  # "A woman holding a surfboard on the beach"
```

## Trained Model Results

The model was trained for **3 epochs** on the COCO training set (414k images, ~2.1M captions)
using:
- Batch size: 32
- Adam optimizer, learning rate 1e-3
- Loss: cross-entropy (with <end> token handling)
- Early stopping on validation BLEU-4

### Validation Metrics

| Metric | Value |
|--------|-------|
| BLEU-1 | 0.656 |
| BLEU-2 | 0.483 |
| BLEU-3 | 0.326 |
| BLEU-4 | 0.207 |
| Training loss | 6.5 → 3.1 (perplexity: 97 → 22) |

### Example Captions

**Good predictions** (semantically accurate):
- Image: giraffe in savanna
  - Predicted: *"A giraffe standing in a field with a tree in the background"*
  - ✓ Correct objects, spatial relationships, activity

- Image: cat on laptop
  - Predicted: *"A cat is sitting on top of a laptop computer"*
  - ✓ Correct objects, pose, spatial relationship

**Failure cases** (shows model limitations):
- Image: two children playing indoors
  - Predicted: *"A man and woman are standing in a kitchen"* ✗
  - Issue: missed "children," misidentified activity and location

- Image: wine glass
  - Predicted: *"A glass of wine and a glass of wine"* ✗
  - Issue: repetition, failed to capture single vs. plural

## The Math

### Training (Teacher Forcing)

```
Image → Encoder → features (256-dim)
features + caption_tokens → Decoder → logits (vocab_size dim)
Cross-entropy loss on logits vs. true next tokens
```

**Token input/output sequence:**
- Input: `features | <start> | "a" | "giraffe" | "standing" | ...`
- Output logits: `(vocab_size,)` predictions for `"a" | "giraffe" | "standing" | <end>`
- The <end> token is dropped from input (never fed to the decoder as a word), but appears in targets

### Inference (Decoding)

**Greedy:**
```
features → LSTM → logits
token = argmax(logits)
embed(token) → LSTM → logits → token = argmax(logits)
... repeat until <end> or max_len
```

**Beam Search (k=3):**
```
Maintain top-3 partial sequences at each step.
For each sequence, generate k candidates by expanding.
Prune to top-k by cumulative log-probability.
Stop when all k-best sequences end with <end>.
```

## Limitations & Future Work

- **No attention mechanism**: the decoder doesn't explicitly attend to image regions.
  Future: spatial attention (Xu et al. 2015) or visual attention (You et al. 2016).
- **Single-layer LSTM**: a 2-3 layer LSTM might capture richer language structure.
- **Greedy vs. beam trade-off**: greedy is fast but suboptimal; beam is better but slower.
  Future: distillation or temperature-based sampling.
- **Fixed vocabulary**: out-of-vocabulary words become `<unk>`. Future: subword tokenization (BPE, WordPiece).
- **No data augmentation**: training data is used as-is. Future: multi-crop evaluation, augmentation during training.
- **Frozen encoder**: ResNet-50 is frozen. Future: fine-tuning the encoder after an initial training phase.
- **COCO-only**: model is trained on COCO; generalization to other domains untested.

## Project Origin

This started as the "Image Captioning" project in Udacity's Computer Vision Nanodegree.
The course provided:
- Problem statement (CNN-LSTM encoder-decoder for captions)
- COCO 2014 dataset with preprocessing pipeline
- Notebook scaffolding and evaluation rubric

The original assignment submission (notebooks and .py files) is preserved in
[`archive/`](archive/) for provenance. Everything outside `archive/`:
- The `image_captioning/` package with modern Python structure, type hints, docstrings
- The `Captioner` inference API
- CLI tool (`caption-image` command)
- Metrics and tests
- This README and repo structure

were built independently for this portfolio refactor.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the original Show-and-Tell paper:

```bibtex
@inproceedings{vinyals2015show,
  title={Show and tell: A neural image caption generator},
  author={Vinyals, Oriol and Toshev, Alexander and Bengio, Samy and Erhan, Dumitry},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3156--3164},
  year={2015}
}
```

And the COCO dataset:

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and others},
  booktitle={European Conference on Computer Vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```
