"""Tests for model architecture and shapes."""
import torch
import pytest
from image_captioning.model import EncoderCNN, DecoderRNN


def test_encoder_output_shape():
    """Test encoder produces correct output dimension."""
    embed_size = 256
    batch_size = 4

    encoder = EncoderCNN(embed_size)
    images = torch.randn(batch_size, 3, 224, 224)

    features = encoder(images)

    assert features.shape == (batch_size, embed_size)


def test_decoder_forward_shape():
    """Test decoder forward pass produces correct logit shape."""
    embed_size = 256
    hidden_size = 512
    vocab_size = 1000
    batch_size = 4
    seq_len = 10

    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    features = torch.randn(batch_size, embed_size)
    captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    outputs = decoder(features, captions)

    assert outputs.shape == (batch_size, seq_len - 1, vocab_size)


def test_decoder_sample_greedy():
    """Test greedy decoding produces a sequence."""
    embed_size = 256
    hidden_size = 512
    vocab_size = 1000
    max_len = 20

    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    features = torch.randn(1, embed_size)

    output = decoder.sample(features, max_len=max_len)

    assert isinstance(output, list)
    assert len(output) > 0
    assert len(output) <= max_len
    assert all(isinstance(t, int) for t in output)


def test_decoder_sample_ends_at_eos():
    """Test greedy decoding stops at <end> token (index 1)."""
    embed_size = 256
    hidden_size = 512
    vocab_size = 1000
    max_len = 20

    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    features = torch.randn(1, embed_size)

    output = decoder.sample(features, max_len=max_len)

    if output:
        assert output[-1] == 1 or len(output) == max_len


def test_decoder_sample_beam():
    """Test beam search produces a sequence."""
    embed_size = 256
    hidden_size = 512
    vocab_size = 1000
    max_len = 20
    beam_width = 3

    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    features = torch.randn(1, embed_size)

    output = decoder.sample_beam(features, max_len=max_len, beam_width=beam_width)

    assert isinstance(output, list)
    assert len(output) > 0
    assert len(output) <= max_len
    assert all(isinstance(t, int) for t in output)


def test_encoder_weights_frozen():
    """Test that ResNet-50 weights are frozen."""
    encoder = EncoderCNN(256)

    for name, param in encoder.named_parameters():
        if 'resnet' in name:
            assert not param.requires_grad, f"ResNet param {name} should be frozen"
        elif 'embed' in name:
            assert param.requires_grad, f"Embedding param {name} should be trainable"


def test_model_on_gpu_if_available():
    """Test model can be moved to GPU if available."""
    encoder = EncoderCNN(256)
    decoder = DecoderRNN(256, 512, 1000)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

        images = torch.randn(1, 3, 224, 224).cuda()
        features = encoder(images)

        assert features.device.type == 'cuda'
