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
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_ID)

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
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
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
