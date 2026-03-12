import type { BackendId, FrameworkBenchmark } from './types';
import { pipeline, env, type ImageClassificationPipeline } from '@huggingface/transformers';

/**
 * Transformers.js benchmark adapter.
 * Uses MobileNet v2 for image classification (via ONNX under the hood).
 */
export class TransformersJsBenchmark implements FrameworkBenchmark {
  name = 'Transformers.js';
  supportedBackends: BackendId[] = ['wasm', 'webgpu', 'webnn'];
  private classifier: ImageClassificationPipeline | null = null;
  private dummyImageUrl: string = '';
  private backend: BackendId = 'wasm';

  async initFramework(backend: BackendId): Promise<void> {
    this.backend = backend;
    // Transformers.js uses ONNX Runtime under the hood
    // Configure the backend / device
    env.allowLocalModels = false;

    // Create a small dummy image as data URL (3x3 red pixel)
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#888888';
    ctx.fillRect(0, 0, 224, 224);
    this.dummyImageUrl = canvas.toDataURL('image/png');
  }

  async loadModel(): Promise<void> {
    let device: 'wasm' | 'webgpu' | 'webnn' = 'wasm';
    if (this.backend === 'webgpu') device = 'webgpu';
    if (this.backend === 'webnn') device = 'webnn';

    // @ts-expect-error — pipeline() union type too complex for TS
    this.classifier = await pipeline(
      'image-classification',
      'onnx-community/mobilenet_v2_1.0_224',
      { device }
    );
  }

  async runInference(): Promise<number> {
    const start = performance.now();
    const result = await this.classifier!(this.dummyImageUrl);
    const elapsed = performance.now() - start;
    void result;
    return elapsed;
  }

  async dispose(): Promise<void> {
    await this.classifier?.dispose();
    this.classifier = null;
  }
}
