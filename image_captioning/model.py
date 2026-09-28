"""CNN-RNN encoder-decoder architecture for image captioning."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """ResNet-50-based image feature encoder.

    Extracts features from images using a pretrained ResNet-50, then projects
    to embedding size. ResNet weights are frozen; only the projection layer is trainable.
    """

    def __init__(self, embed_size: int) -> None:
        """Initialize the encoder.

        Args:
            embed_size: Dimension of the output feature vector.
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Batch of images, shape (batch_size, 3, 224, 224).

        Returns:
            Image features, shape (batch_size, embed_size).
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    """LSTM-based caption decoder with greedy and beam-search sampling.

    Decodes image features into natural-language captions using an LSTM
    with embedding and output layers.
    """

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
    ) -> None:
        """Initialize the decoder.

        Args:
            embed_size: Dimension of word embeddings.
            hidden_size: Dimension of LSTM hidden state.
            vocab_size: Size of vocabulary.
            num_layers: Number of stacked LSTM layers.
        """
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training with teacher forcing.

        Args:
            features: Image features, shape (batch_size, embed_size).
            captions: Token indices, shape (batch_size, seq_len).
                     Last token is assumed to be <end> and is dropped.

        Returns:
            Logits for each output position, shape (batch_size, seq_len-1, vocab_size).
        """
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def sample(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_len: int = 20,
    ) -> list:
        """Greedy decoding: generate caption by always picking the max-probability token.

        Args:
            inputs: Image features, shape (1, embed_size).
            states: Initial LSTM states (h, c), or None.
            max_len: Maximum caption length (stops early if <end> is predicted).

        Returns:
            List of token indices.
        """
        output = []
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            scores = self.linear(hiddens.squeeze(1))
            predicted = scores.argmax(dim=1)
            output.append(predicted.item())
            if predicted.item() == 1:  # <end> token
                break
            inputs = self.embed(predicted).unsqueeze(1)
        return output

    def sample_beam(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_len: int = 20,
        beam_width: int = 3,
    ) -> list:
        """Beam search decoding: maintain top-k candidate sequences.

        Args:
            inputs: Image features, shape (1, embed_size).
            states: Initial LSTM states, or None.
            max_len: Maximum caption length.
            beam_width: Number of top candidates to keep at each step.

        Returns:
            List of token indices for the best-scoring sequence.
        """
        device = inputs.device
        candidates = [([], 0.0)]

        for step in range(max_len):
            next_candidates = []

            for sequence, score in candidates:
                if sequence and sequence[-1] == 1:
                    next_candidates.append((sequence, score))
                    continue

                if not sequence:
                    step_input = inputs
                    step_states = states
                else:
                    last_token = torch.tensor([sequence[-1]], device=device)
                    step_input = self.embed(last_token).unsqueeze(0)
                    step_states = states

                hiddens, new_states = self.lstm(step_input, step_states)
                scores = self.linear(hiddens.squeeze(1))
                log_probs = torch.log_softmax(scores, dim=1)

                top_k = torch.topk(log_probs[0], min(beam_width, log_probs.size(1)))

                for log_prob, token_id in zip(top_k.values, top_k.indices):
                    new_seq = sequence + [token_id.item()]
                    new_score = score + log_prob.item()
                    next_candidates.append((new_seq, new_score))

                states = new_states

            candidates = sorted(next_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(c[0][-1] == 1 for c in candidates if c[0]):
                break

        return candidates[0][0] if candidates[0][0] else [1]
