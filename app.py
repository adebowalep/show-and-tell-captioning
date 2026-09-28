"""Gradio web demo for the CNN-RNN image captioning model.

Run locally:
    pip install -e ".[app]"
    python app.py

Deploy: push this repo to a Hugging Face Space (SDK: Gradio). See README
"Web Demo" section for step-by-step instructions.
"""
import os

import gradio as gr
import torch

from image_captioning import Captioner

try:
    import spaces
    GPU = spaces.GPU
except ImportError:
    # Not running on a Hugging Face Space (or ZeroGPU not applicable): no-op decorator.
    def GPU(fn):
        return fn

ENCODER_PATH = os.environ.get("ENCODER_PATH", "models/encoder-3.pkl")
DECODER_PATH = os.environ.get("DECODER_PATH", "models/decoder-3.pkl")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "models/vocab.pkl")

print("Loading model...")
# Load on CPU: ZeroGPU only attaches a GPU for the duration of a @spaces.GPU call,
# so the model must start on CPU and move to CUDA inside that call.
captioner = Captioner(
    encoder_path=ENCODER_PATH,
    decoder_path=DECODER_PATH,
    vocab_path=VOCAB_PATH,
    device="cpu",
)
print("Model loaded.")


@GPU
def generate_caption(image, beam_width, max_len):
    """Gradio callback: save the uploaded image, run the captioner, return text."""
    if image is None:
        return "Please upload an image first."

    if torch.cuda.is_available():
        captioner.to("cuda")

    tmp_path = "/tmp/gradio_input.jpg"
    image.convert("RGB").save(tmp_path)

    width = None if beam_width == 1 else int(beam_width)
    caption = captioner.caption_image(tmp_path, max_len=int(max_len), beam_width=width)

    mode = "greedy" if width is None else f"beam search (width={width})"
    return f"{caption}\n\n— generated with {mode}"


demo = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="pil", label="Upload an image"),
        gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Beam width (1 = greedy decoding)"),
        gr.Slider(minimum=5, maximum=30, step=1, value=20, label="Max caption length"),
    ],
    outputs=gr.Textbox(label="Generated caption"),
    title="Show and Tell: Image Captioning",
    description=(
        "CNN-RNN encoder-decoder (ResNet-50 + LSTM) trained on COCO. "
        "Upload an image and the model will generate a natural-language caption. "
        "Try beam width 1 for fast greedy decoding, or 3-5 for higher-quality beam search."
    ),
    article=(
        "Model details, architecture, and training results: "
        "[GitHub repo](https://github.com/adebowalep/show-and-tell-captioning)"
    ),
)

if __name__ == "__main__":
    demo.launch()
