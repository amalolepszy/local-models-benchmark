import type { BackendId, FrameworkBenchmark } from './types';
import { loadLiteRt, loadAndCompile, Tensor, type CompiledModel } from '@litertjs/core';

/**
 * LiteRT.js benchmark adapter.
 * Uses MobileNet v2 (.tflite) for image classification.
 */
let liteRtLoaded = false;

export class LiteRTBenchmark implements FrameworkBenchmark {
  name = 'LiteRT.js';
  supportedBackends: BackendId[] = ['wasm', 'webgpu'];
  private model: CompiledModel | null = null;
  private inputData: Float32Array | null = null;
  private backend: BackendId = 'wasm';

  async initFramework(backend: BackendId): Promise<void> {
    this.backend = backend;
    if (!liteRtLoaded) {
      await loadLiteRt('/litert-wasm/');
      liteRtLoaded = true;
    }
  }

  async loadModel(): Promise<void> {
    // MobileNet v2 1.0 224x224 classification (served locally)
    const MODEL_URL = '/mobilenet_v2-1-0-224.tflite';

    const accelerator = this.backend === 'webgpu' ? 'webgpu' : 'wasm';

    this.model = await loadAndCompile(MODEL_URL, { accelerator });

    // EfficientNet-Lite0 expects (1, 224, 224, 3) float32 input
    this.inputData = new Float32Array(1 * 224 * 224 * 3);
    for (let i = 0; i < this.inputData.length; i++) {
      this.inputData[i] = Math.random();
    }
  }

  async runInference(): Promise<number> {
    const inputTensor = new Tensor(this.inputData!, [1, 224, 224, 3]);

    // Move to GPU if using WebGPU backend
    const tensor = this.backend === 'webgpu'
      ? await inputTensor.moveTo('webgpu')
      : inputTensor;

    const start = performance.now();
    const results = await this.model!.run(tensor);
    // Force data readback to ensure computation completes
    await results[0]!.data();
    const elapsed = performance.now() - start;

    // Cleanup tensors
    if (this.backend === 'webgpu' && tensor !== inputTensor) {
      tensor.delete();
    } else {
      inputTensor.delete();
    }
    results[0]!.delete();

    return elapsed;
  }

  async dispose(): Promise<void> {
    this.model?.delete();
    this.model = null;
    this.inputData = null;
  }
}
