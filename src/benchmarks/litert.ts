import type { BackendId, BenchmarkInput, ClassificationResult, FrameworkBenchmark } from './types';
import { IMAGENET_LABELS } from '../utils/imagenet-labels';
import { loadLiteRt, loadAndCompile, Tensor, type CompiledModel } from '@litertjs/core';
import { measureNewResources } from '../utils/metrics';

let liteRtLoaded = false;
const cachedModels: Record<string, CompiledModel> = {};

export class LiteRTBenchmark implements FrameworkBenchmark {
  name = 'LiteRT.js';
  frameworkBytes = 0;
  supportedBackends: BackendId[] = ['wasm-simd-threads', 'webgpu'];
  private model: CompiledModel | null = null;
  private inputData: Float32Array = new Float32Array(0);
  private backend: BackendId = 'wasm';
  private readonly modelUrl = '/mobilenet_v2-1-0-224.tflite';

  async initFramework(backend: BackendId): Promise<void> {
    const before = performance.getEntriesByType('resource').length;

    this.backend = backend;
    if (!liteRtLoaded) {
      await loadLiteRt('/litert-wasm/');
      liteRtLoaded = true;
    }

    this.frameworkBytes = measureNewResources(before);
  }

  async prefetchModel(): Promise<void> {
    await fetch(this.modelUrl);
  }

  async loadModel(): Promise<void> {
    const accelerator = this.backend === 'webgpu' ? 'webgpu' : 'wasm';
    const cacheKey = `${this.modelUrl}:${accelerator}`;

    if (cachedModels[cacheKey]) {
      this.model = cachedModels[cacheKey];
    } else {
      this.model = await loadAndCompile(this.modelUrl, { accelerator });
      cachedModels[cacheKey] = this.model;
    }

    // Default to random input — NHWC [1, 224, 224, 3]
    this.inputData = new Float32Array(1 * 224 * 224 * 3);
    for (let i = 0; i < this.inputData.length; i++) {
      this.inputData[i] = Math.random();
    }
  }

  setInput(input: BenchmarkInput): void {
    if (input.type !== 'image') throw new Error('LiteRTBenchmark only supports image input');
    // TFLite MobileNet v2 expects NHWC [-1, 1]
    this.inputData = input.image.nhwcNegOneOne;
  }

  async runInference(): Promise<number> {
    const inputTensor = new Tensor(this.inputData, [1, 224, 224, 3]);

    const tensor = this.backend === 'webgpu'
      ? await inputTensor.moveTo('webgpu')
      : inputTensor;

    const start = performance.now();
    const results = await this.model!.run(tensor);
    await results[0]!.data();
    const elapsed = performance.now() - start;

    if (this.backend === 'webgpu' && tensor !== inputTensor) {
      tensor.delete();
    } else {
      inputTensor.delete();
    }
    results[0]!.delete();

    return elapsed;
  }

  async classify(topK = 5): Promise<ClassificationResult[]> {
    const inputTensor = new Tensor(this.inputData, [1, 224, 224, 3]);

    const tensor = this.backend === 'webgpu'
      ? await inputTensor.moveTo('webgpu')
      : inputTensor;

    const results = await this.model!.run(tensor);
    const outputData = await results[0]!.data();
    const logits = new Float32Array(outputData.buffer, outputData.byteOffset, outputData.length);

    if (this.backend === 'webgpu' && tensor !== inputTensor) {
      tensor.delete();
    } else {
      inputTensor.delete();
    }
    results[0]!.delete();

    return extractTopK(logits, topK);
  }

  async dispose(): Promise<void> {
    // Don't delete the compiled model — it's cached for reuse across sessions
    this.model = null;
    this.inputData = new Float32Array(0);
  }
}

function extractTopK(probs: Float32Array, topK: number): ClassificationResult[] {
  // TFLite model already includes softmax — output is probabilities, not logits
  const indexed = Array.from(probs).map((score, i) => ({ score, i }));
  indexed.sort((a, b) => b.score - a.score);

  return indexed.slice(0, topK).map(({ score, i }) => ({
    label: IMAGENET_LABELS[i] ?? `class_${i}`,
    labelIndex: i,
    score,
  }));
}
