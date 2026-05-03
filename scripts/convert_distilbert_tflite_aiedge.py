"""
Convert DistilBERT SST-2 to TFLite using Google's ai-edge-torch.

This is the alternative to convert_distilbert_tflite_int32.py — that one uses
the TF Keras → TFLite path, which emits 2D GATHER and shrink_axis_mask
STRIDED_SLICE that LiteRT.js's WebGPU delegate cannot lower. ai-edge-torch
converts directly from PyTorch with op patterns specifically chosen for
LiteRT WebGPU compatibility.

Usage:
    pip install ai-edge-torch transformers torch
    python scripts/convert_distilbert_tflite_aiedge.py

Output: public/distilbert-base-uncased-finetuned-sst-2-english/tflite/model.tflite
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import ai_edge_torch
from transformers import AutoModelForSequenceClassification

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "public", "distilbert-base-uncased-finetuned-sst-2-english", "tflite", "model.tflite",
)
SEQ_LEN = 128


class DistilBertForLiteRT(torch.nn.Module):
    """Thin wrapper that returns just the logits tensor.

    HF returns a SequenceClassifierOutput dataclass; ai-edge-torch's tracer
    needs a plain tensor (or tuple of tensors) as output, so we unwrap here.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


def main():
    print(f"Loading {MODEL_ID}...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    model = DistilBertForLiteRT(hf_model).eval()

    # Sample inputs for graph tracing. Shapes/dtypes here become the .tflite
    # signature, so they must match what the JS adapter feeds at runtime.
    sample = (
        torch.zeros(1, SEQ_LEN, dtype=torch.int32),   # input_ids
        torch.ones(1, SEQ_LEN, dtype=torch.int32),    # attention_mask
    )

    print("Converting via ai-edge-torch...")
    edge_model = ai_edge_torch.convert(model, sample)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    edge_model.export(OUTPUT_PATH)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"Saved TFLite model ({size_mb:.1f} MB) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
