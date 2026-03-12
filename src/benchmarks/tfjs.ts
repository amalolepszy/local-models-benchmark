import type { BackendId, FrameworkBenchmark } from './types';
import * as tf from '@tensorflow/tfjs';

/**
 * TensorFlow.js benchmark adapter.
 * Uses MobileNet v2 (small classification model) for inference.
 */
export class TfjsBenchmark implements FrameworkBenchmark {
  name = 'TensorFlow.js';
  supportedBackends: BackendId[] = ['wasm', 'webgl', 'webgpu'];
  private model: tf.GraphModel | null = null;
  private inputTensor: tf.Tensor | null = null;

  async initFramework(backend: BackendId): Promise<void> {
    if (backend === 'wasm') {
      const wasmModule = await import('@tensorflow/tfjs-backend-wasm');
      wasmModule.setWasmPaths('/');
      await tf.setBackend('wasm');
    } else if (backend === 'webgl') {
      await import('@tensorflow/tfjs-backend-webgl');
      await tf.setBackend('webgl');
    } else if (backend === 'webgpu') {
      await import('@tensorflow/tfjs-backend-webgpu');
      await tf.setBackend('webgpu');
    }
    await tf.ready();
  }

  async loadModel(): Promise<void> {
    // MobileNet v2 1.0 224x224 classification (served locally)
    const MODEL_URL = '/mobilenet_v2_tfjs/model.json';
    this.model = await tf.loadGraphModel(MODEL_URL);
    // MobileNet v2 1.0_224 expects (1, 224, 224, 3) with values in [0, 1]
    this.inputTensor = tf.randomUniform([1, 224, 224, 3]);
  }

  async runInference(): Promise<number> {
    const start = performance.now();
    const result = this.model!.predict(this.inputTensor!) as tf.Tensor;
    // Force sync to ensure computation completes
    await result.data();
    const elapsed = performance.now() - start;
    result.dispose();
    return elapsed;
  }

  async dispose(): Promise<void> {
    this.inputTensor?.dispose();
    this.model?.dispose();
    this.model = null;
    this.inputTensor = null;
  }
}
