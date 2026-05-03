"""
Convert DistilBERT SST-2 to TFLite with INT32 inputs (LiteRT.js doesn't support INT64).

Usage:
    pip install transformers tensorflow
    python scripts/convert_distilbert_tflite_int32.py

Output: public/distilbert-base-uncased-finetuned-sst-2-english/tflite/model.tflite
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tempfile
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "public", "distilbert-base-uncased-finetuned-sst-2-english", "tflite", "model.tflite"
)
SEQ_LEN = 128

def main():
    print(f"Loading {MODEL_ID}...")
    # use_safetensors=False forces loading from pytorch_model.bin instead of
    # model.safetensors. Necessary because transformers 4.57's PT->TF cross-loader
    # has a bug iterating the safe_open object (TypeError: not iterable).
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_ID, use_safetensors=False)

    # Create a concrete function with INT32 input signatures
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, SEQ_LEN], dtype=tf.int32, name="input_ids"),
        tf.TensorSpec(shape=[1, SEQ_LEN], dtype=tf.int32, name="attention_mask"),
    ])
    def serve(input_ids, attention_mask):
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

    concrete_func = serve.get_concrete_function()

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # Force the converter to emit ONLY canonical TFLite builtins (drop
    # SELECT_TF_OPS). With SELECT_TF_OPS, the converter is happy to leave raw
    # TF ops in the graph — including 2D GATHER and STRIDED_SLICE with
    # shrink_axis_mask, both of which LiteRT.js's WebGPU delegate refuses.
    # Builtins-only forces the converter to lower those into webgpu-friendly
    # equivalents (or fail, in which case we fall back to plan B/C).
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_new_converter = True              # MLIR-based path (default in recent TF, set explicitly)
    converter._experimental_lower_tensor_list_ops = True     # simplify control flow
    converter.allow_custom_ops = False                       # don't let custom ops sneak in
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model ({len(tflite_model) / 1024 / 1024:.1f} MB) to {OUTPUT_PATH}")

    # Verify
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    for inp in interpreter.get_input_details():
        print(f"  Input: {inp['name']} shape={inp['shape']} dtype={inp['dtype']}")
    for out in interpreter.get_output_details():
        print(f"  Output: {out['name']} shape={out['shape']} dtype={out['dtype']}")

if __name__ == "__main__":
    main()
