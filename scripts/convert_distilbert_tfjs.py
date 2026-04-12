"""
Convert DistilBERT SST-2 to TensorFlow.js graph model format.

Usage:
    pip install transformers tensorflow tensorflowjs
    python scripts/convert_distilbert_tfjs.py

Output goes to public/distilbert-base-uncased-finetuned-sst-2-english/model_tfjs/
"""

import os

# Disable Intel MKL/oneDNN optimizations BEFORE importing TensorFlow.
# MKL ops like _MklLayerNorm are not supported by TF.js.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import shutil
import tempfile
from transformers import TFDistilBertForSequenceClassification
import tensorflowjs as tfjs

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "public", "distilbert-base-uncased-finetuned-sst-2-english", "model_tfjs"
)

def main():
    print(f"Loading {MODEL_ID} from HuggingFace...")
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_ID)

    # Save as TF SavedModel first
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = os.path.join(tmpdir, "saved_model")
        print(f"Saving TF SavedModel to {saved_model_path}...")
        model.save_pretrained(saved_model_path, saved_model=True)

        # Clear old output
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"Converting to TF.js graph model in {OUTPUT_DIR}...")
        tfjs.converters.convert_tf_saved_model(
            os.path.join(saved_model_path, "saved_model", "1"),
            OUTPUT_DIR
        )

    print("Done! Model saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
