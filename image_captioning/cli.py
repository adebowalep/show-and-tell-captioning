"""Command-line interface for image captioning."""
import argparse
from pathlib import Path

from .caption import Captioner


def main():
    """CLI entry point: caption-image [image_path] [options]."""
    parser = argparse.ArgumentParser(
        description="Generate captions for images using a trained CNN-RNN model."
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--encoder",
        default="models/encoder-3.pkl",
        help="Path to encoder checkpoint (default: models/encoder-3.pkl)",
    )
    parser.add_argument(
        "--decoder",
        default="models/decoder-3.pkl",
        help="Path to decoder checkpoint (default: models/decoder-3.pkl)",
    )
    parser.add_argument(
        "--vocab",
        default="models/vocab.pkl",
        help="Path to vocabulary file (default: models/vocab.pkl)",
    )
    parser.add_argument(
        "--embed-size",
        type=int,
        default=256,
        help="Embedding dimension (default: 256)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="LSTM hidden dimension (default: 512)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=20,
        help="Maximum caption length (default: 20)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="Beam width for beam search; if not specified, use greedy decoding",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    captioner = Captioner(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        vocab_path=args.vocab,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
    )

    caption = captioner.caption_image(
        str(image_path),
        max_len=args.max_len,
        beam_width=args.beam_width,
    )

    print(f"\n📸 {image_path.name}")
    print(f"   {caption}\n")


if __name__ == "__main__":
    main()
