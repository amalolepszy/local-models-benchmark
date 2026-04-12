import { isWasmBackend, getWasmFlags, type BackendId, type BenchmarkInput, type ClassificationResult, type FrameworkBenchmark } from './types';
import { pipeline, env, type ImageClassificationPipeline } from '@huggingface/transformers';
import { measureNewResources } from '../utils/metrics';

export class TransformersJsBenchmark implements FrameworkBenchmark {
  name = 'Transformers.js';
  frameworkBytes = 0;
  supportedBackends: BackendId[] = ['wasm-simd-threads', 'webgpu', 'webnn'];
  private classifier: ImageClassificationPipeline | null = null;
  private imageUrl: string = '';
  private backend: BackendId = 'wasm';
  private readonly modelId = 'onnx-community/mobilenet_v2_1.0_224';

  async initFramework(backend: BackendId): Promise<void> {
    const before = performance.getEntriesByType('resource').length;

    this.backend = backend;
    env.allowLocalModels = false;

    // Configure WASM variant for the underlying ONNX Runtime
    if (isWasmBackend(backend)) {
      const { simd, threads } = getWasmFlags(backend);
      (env as any).backends = (env as any).backends ?? {};
      (env as any).backends.onnx = (env as any).backends.onnx ?? {};
      (env as any).backends.onnx.wasm = (env as any).backends.onnx.wasm ?? {};
      (env as any).backends.onnx.wasm.simd = simd;
      (env as any).backends.onnx.wasm.numThreads = threads ? 4 : 1;
    }

    // Default dummy image
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#888888';
    ctx.fillRect(0, 0, 224, 224);
    this.imageUrl = canvas.toDataURL('image/png');

    this.frameworkBytes = measureNewResources(before);
  }

  async prefetchModel(): Promise<void> {
    // Pre-download model files into browser cache so loadModel() only measures compilation
    // @ts-expect-error — pipeline() union type too complex for TS
    const tempPipeline = await pipeline(
      'image-classification',
      this.modelId,
      { device: 'wasm', dtype: 'fp32' }
    );
    await tempPipeline.dispose();
  }

  async loadModel(): Promise<void> {
    let device: 'wasm' | 'webgpu' | 'webnn' = 'wasm';
    if (this.backend === 'webgpu') device = 'webgpu';
    else if (this.backend === 'webnn') device = 'webnn';
    else if (isWasmBackend(this.backend)) device = 'wasm';

    // @ts-expect-error — pipeline() union type too complex for TS
    // Model files already cached from prefetch — this only measures compilation
    this.classifier = await pipeline(
      'image-classification',
      this.modelId,
      { device, dtype: 'fp32' }
    );
  }

  setInput(input: BenchmarkInput): void {
    if (input.type !== 'image') throw new Error('TransformersJsBenchmark only supports image input');
    // Transformers.js pipeline handles its own preprocessing from the image
    this.imageUrl = input.image.dataUrl;
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
