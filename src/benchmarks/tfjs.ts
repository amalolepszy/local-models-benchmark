import type { BackendId, ClassificationResult, FrameworkBenchmark } from './types';
import type { PreprocessedImage } from '../utils/image-input';
import { IMAGENET_LABELS } from '../utils/imagenet-labels';
import * as tf from '@tensorflow/tfjs';

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
    const MODEL_URL = '/mobilenet_v2_tfjs/model.json';
    this.model = await tf.loadGraphModel(MODEL_URL);
    // Default to random input
    this.inputTensor = tf.randomUniform([1, 224, 224, 3]);
  }

  setImage(image: PreprocessedImage): void {
    this.inputTensor?.dispose();
    // TF.js MobileNet v2 from TFHub expects NHWC [0, 1]
    this.inputTensor = tf.tensor(image.nhwcZeroOne, [1, 224, 224, 3]);
  }

  async runInference(): Promise<number> {
    const start = performance.now();
    const result = this.model!.predict(this.inputTensor!) as tf.Tensor;
    await result.data();
    const elapsed = performance.now() - start;
    result.dispose();
    return elapsed;
  }

  async classify(topK = 5): Promise<ClassificationResult[]> {
    const result = this.model!.predict(this.inputTensor!) as tf.Tensor;
    const probabilities = await result.data();
    result.dispose();
    return extractTopK(probabilities as Float32Array, topK);
  }

  async dispose(): Promise<void> {
    this.inputTensor?.dispose();
    this.model?.dispose();
    this.model = null;
    this.inputTensor = null;
  }
}

function extractTopK(logits: Float32Array, topK: number): ClassificationResult[] {
  // Apply softmax
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((s, e) => s + e, 0);
  const probs = exps.map(e => e / sumExp);

  const indexed = Array.from(probs).map((score, i) => ({ score, i }));
  indexed.sort((a, b) => b.score - a.score);

  return indexed.slice(0, topK).map(({ score, i }) => ({
    label: IMAGENET_LABELS[i] ?? `class_${i}`,
    labelIndex: i,
    score,
  }));
}
