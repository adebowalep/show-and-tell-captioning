"""High-level API for image captioning inference."""
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms

from .model import DecoderRNN, EncoderCNN
from .vocabulary import Vocabulary


class Captioner:
    """End-to-end image captioner: encode image, decode caption."""

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        vocab_path: str,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        device: Optional[str] = None,
    ) -> None:
        """Load trained encoder and decoder.

        Args:
            encoder_path: Path to encoder checkpoint (encoder-3.pkl).
            decoder_path: Path to decoder checkpoint (decoder-3.pkl).
            vocab_path: Path to vocabulary file (vocab.pkl).
            embed_size: Embedding dimension (must match training).
            hidden_size: LSTM hidden dimension (must match training).
            num_layers: Number of LSTM layers (must match training).
            device: 'cpu' or 'cuda'; auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.vocab = Vocabulary(vocab_path)

        self.encoder = EncoderCNN(embed_size).to(self.device)
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.encoder.eval()

        self.decoder = DecoderRNN(embed_size, hidden_size, len(self.vocab), num_layers).to(self.device)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        self.decoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256,)),
            transforms.CenterCrop((224,)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def to(self, device: str) -> "Captioner":
        """Move encoder and decoder to a device (e.g. for ZeroGPU's dynamic GPU attach).

        Args:
            device: 'cpu' or 'cuda'.

        Returns:
            self, for chaining.
        """
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        return self

    def caption_image(
        self,
        image_path: str,
        max_len: int = 20,
        beam_width: Optional[int] = None,
    ) -> str:
        """Generate a caption for an image.

        Args:
            image_path: Path to image file.
            max_len: Maximum caption length.
            beam_width: If specified, use beam search instead of greedy.

        Returns:
            Caption string.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encoder(image).unsqueeze(1)

        with torch.no_grad():
            if beam_width is not None:
                tokens = self.decoder.sample_beam(features, max_len=max_len, beam_width=beam_width)
            else:
                tokens = self.decoder.sample(features, max_len=max_len)

        caption = self.vocab.decode(tokens)
        return caption.strip()
