import type { BackendId, ClassificationResult, FrameworkBenchmark } from './types';
import type { PreprocessedImage } from '../utils/image-input';
import { pipeline, env, type ImageClassificationPipeline } from '@huggingface/transformers';

export class TransformersJsBenchmark implements FrameworkBenchmark {
  name = 'Transformers.js';
  supportedBackends: BackendId[] = ['wasm', 'webgpu', 'webnn'];
  private classifier: ImageClassificationPipeline | null = null;
  private imageUrl: string = '';
  private backend: BackendId = 'wasm';

  async initFramework(backend: BackendId): Promise<void> {
    this.backend = backend;
    env.allowLocalModels = false;

    // Default dummy image
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#888888';
    ctx.fillRect(0, 0, 224, 224);
    this.imageUrl = canvas.toDataURL('image/png');
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

  setImage(image: PreprocessedImage): void {
    // Transformers.js pipeline handles its own preprocessing from the image
    this.imageUrl = image.dataUrl;
  }

  async runInference(): Promise<number> {
    const start = performance.now();
    const result = await this.classifier!(this.imageUrl);
    const elapsed = performance.now() - start;
    void result;
    return elapsed;
  }

  async classify(topK = 5): Promise<ClassificationResult[]> {
    const result = await this.classifier!(this.imageUrl, { top_k: topK });
    const output = Array.isArray(result) ? result : [result];
    return output.map((r: any) => ({
      label: r.label as string,
      labelIndex: -1, // Pipeline doesn't expose raw index
      score: r.score as number,
    }));
  }

  async dispose(): Promise<void> {
    await this.classifier?.dispose();
    this.classifier = null;
  }
}
