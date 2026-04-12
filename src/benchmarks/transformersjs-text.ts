import { isWasmBackend, getWasmFlags, type BackendId, type BenchmarkInput, type ClassificationResult, type FrameworkBenchmark } from './types';
import { pipeline, env, type TextClassificationPipeline } from '@huggingface/transformers';
import { measureNewResources } from '../utils/metrics';

export class TransformersJsTextBenchmark implements FrameworkBenchmark {
  name = 'Transformers.js';
  frameworkBytes = 0;
  supportedBackends: BackendId[] = ['wasm', 'wasm-simd', 'wasm-threads', 'wasm-simd-threads', 'webgpu', 'webnn'];
  private classifier: TextClassificationPipeline | null = null;
  private rawText: string = 'This is a test.';
  private backend: BackendId = 'wasm';
  private readonly modelId = 'distilbert-base-uncased-finetuned-sst-2-english';

  async initFramework(backend: BackendId): Promise<void> {
    const before = performance.getEntriesByType('resource').length;

    this.backend = backend;
    env.allowLocalModels = false;

    if (isWasmBackend(backend)) {
      const { simd, threads } = getWasmFlags(backend);
      (env as any).backends = (env as any).backends ?? {};
      (env as any).backends.onnx = (env as any).backends.onnx ?? {};
      (env as any).backends.onnx.wasm = (env as any).backends.onnx.wasm ?? {};
      (env as any).backends.onnx.wasm.simd = simd;
      (env as any).backends.onnx.wasm.numThreads = threads ? 4 : 1;
    }

    this.frameworkBytes = measureNewResources(before);
  }

  async prefetchModel(): Promise<void> {
    // Pre-download model files into browser cache
    const tempPipeline = await pipeline(
      'text-classification',
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

    this.classifier = await pipeline(
      'text-classification',
      this.modelId,
      { device, dtype: 'fp32' }
    );
  }

  setInput(input: BenchmarkInput): void {
    if (input.type !== 'text') throw new Error('TransformersJsTextBenchmark only supports text input');
    // Transformers.js pipeline handles its own tokenization
    this.rawText = input.rawText;
  }

  async runInference(): Promise<number> {
    const start = performance.now();
    const result = await this.classifier!(this.rawText);
    const elapsed = performance.now() - start;
    void result;
    return elapsed;
  }

  async classify(_topK = 5): Promise<ClassificationResult[]> {
    const result = await this.classifier!(this.rawText, { top_k: 2 });
    const output = Array.isArray(result) ? result : [result];
    return output.map((r: any) => ({
      label: r.label as string,
      labelIndex: r.label === 'POSITIVE' ? 1 : 0,
      score: r.score as number,
    }));
  }

  async dispose(): Promise<void> {
    await this.classifier?.dispose();
    this.classifier = null;
  }
}
