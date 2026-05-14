# Local Models Benchmark

Browser-based benchmark comparing ML inference frameworks running in Chromium. Measures latency, RAM, and CPU across frameworks and hardware backends — part of a master's thesis.

## Frameworks & Backends

| Framework | Backends |
|---|---|
| TensorFlow.js | WASM, WebGL, WebGPU |
| ONNX Runtime Web | WASM, WebGL, WebGPU, WebNN |
| LiteRT.js | WASM, WebGPU |
| Transformers.js | WASM, WebGPU, WebNN |
| [TFLite Native](https://github.com/amalolepszy/chromium-tflite-native) | CPU, GPU |

## Tasks

- **Image classification** — MobileNet v2 1.0 224×224, ImageNet labels
- **Text classification** — sentiment analysis

## Stack

- Vite + vanilla TypeScript
- Playwright for automated benchmark runs
- Python profiler for system-level metrics

## Getting Started

```bash
npm install
npm run dev
```

> Requires Chrome with cross-origin isolation (`COOP`/`COEP` headers, set in `vite.config.ts`).  
> For `performance.memory` metrics, enable `chrome://flags/#enable-precise-memory-info`.

## Running Tests

The Playwright benchmark suite requires the Vite dev server to be running first:

```bash
npm run dev          # keep this running in a separate terminal
```

Then in another terminal:

```bash
# Run all benchmarks (headless, image classification by default)
npm run bench

# With browser window visible
npm run bench:headed

# Text classification task
BENCHMARK_TASK=text npm run bench

# Custom number of sessions per combo (default: 10)
BENCHMARK_SESSIONS=5 npm run bench

# With Python profiler for CPU/GPU/RAM sampling
pip install playwright psutil gputil
python -m playwright install chromium
npm run bench:profile
```

Results are written to `benchmark_results/` as both JSON and CSV.

> The Playwright spec launches a custom Chromium build (`D:/chr-build/...`). Update `executablePath` in `e2e/benchmark.spec.ts` if your build is in a different location.

## Models

All models are MobileNet v2 1.0 224×224 served locally from `public/`:

- `mobilenet_v2_tfjs/model.json` — TF.js graph model
- `mobilenet_v2_1.0_224.onnx` — ONNX (static batch dim)
- `mobilenet_v2-1-0-224.tflite` — LiteRT / TFLite Native

Transformers.js fetches `onnx-community/mobilenet_v2_1.0_224` remotely.

## Project Structure

```
src/
  main.ts                  # UI + benchmark orchestration
  benchmarks/              # Per-framework adapters
    types.ts               # Shared types and framework×backend matrix
  utils/
    metrics.ts             # Timing, memory measurement, CSV export
    image-input.ts         # Image preprocessing (NHWC/NCHW, 224×224)
    imagenet-labels.ts     # 1001 ImageNet class labels
```
