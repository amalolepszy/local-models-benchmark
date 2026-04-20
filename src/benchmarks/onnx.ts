import { isWasmBackend, getWasmFlags, type BackendId, type BenchmarkInput, type ClassificationResult, type FrameworkBenchmark } from './types';
import { IMAGENET_LABELS } from '../utils/imagenet-labels';
import { measureNewResources, getContentLength } from '../utils/metrics';

let cachedSession: import('onnxruntime-web').InferenceSession | null = null;

export class OnnxBenchmark implements FrameworkBenchmark {
  name = 'ONNX Runtime Web';
  frameworkBytes = 0;
  supportedBackends: BackendId[] = ['wasm-simd-threads', 'webgl', 'webgpu', 'webnn'];
  private ort: typeof import('onnxruntime-web') | null = null;
  private session: import('onnxruntime-web').InferenceSession | null = null;
  private inputData: Float32Array = new Float32Array(0);
  private backend: BackendId = 'wasm';
  private modelBuffer: ArrayBuffer | null = null;
  private readonly modelUrl = '/mobilenet_v2_1.0_224.onnx';

  async initFramework(backend: BackendId): Promise<void> {
    const before = performance.getEntriesByType('resource').length;

    this.backend = backend;
    if (backend === 'webgl') {
      this.ort = await import('onnxruntime-web/webgl');
    } else {
      this.ort = await import('onnxruntime-web');
    }
    this.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';

    // Configure WASM variant
    if (isWasmBackend(backend)) {
      const { simd, threads } = getWasmFlags(backend);
      this.ort.env.wasm.simd = simd;
      this.ort.env.wasm.numThreads = threads ? 4 : 1;
    }

    this.frameworkBytes = measureNewResources(before);

    // WASM is loaded from CDN during InferenceSession.create(), not during init.
    if (isWasmBackend(backend) || backend === 'webgpu' || backend === 'webnn') {
      const wasmBase = this.ort.env.wasm.wasmPaths as string;
      const wasmFile = (backend === 'webgpu' || backend === 'webnn')
        ? 'ort-wasm-simd-threaded.jsep.wasm'
        : 'ort-wasm-simd-threaded.wasm';
      this.frameworkBytes += await getContentLength(wasmBase + wasmFile);
    }
  }

  async prefetchModel(): Promise<void> {
    const resp = await fetch(this.modelUrl);
    this.modelBuffer = await resp.arrayBuffer();
  }

  async loadModel(): Promise<void> {
    const ort = this.ort!;

    let executionProviders: import('onnxruntime-web').InferenceSession.ExecutionProviderConfig[];
    if (isWasmBackend(this.backend))       executionProviders = ['wasm'];
    else if (this.backend === 'webgl')     executionProviders = ['webgl'];
    else if (this.backend === 'webgpu')    executionProviders = ['webgpu'];
    else if (this.backend === 'webnn')     executionProviders = ['webnn'];
    else                                   executionProviders = ['wasm'];

    if (cachedSession) {
      this.session = cachedSession;
    } else {
      const source = this.modelBuffer ?? this.modelUrl;
      this.session = await ort.InferenceSession.create(source, { executionProviders });
      cachedSession = this.session;
    }
    // Default to random input — NCHW [1, 3, 224, 224]
    this.inputData = new Float32Array(1 * 3 * 224 * 224);
    for (let i = 0; i < this.inputData.length; i++) {
      this.inputData[i] = Math.random();
    }
  }

  setInput(input: BenchmarkInput): void {
    if (input.type !== 'image') throw new Error('OnnxBenchmark only supports image input');
    // ONNX model (google/mobilenet_v2) expects NCHW [-1, 1]
    this.inputData = input.image.nchwNegOneOne;
  }

  async runInference(): Promise<number> {
    const ort = this.ort!;
    const tensor = new ort.Tensor('float32', this.inputData, [1, 3, 224, 224]);
    const inputNames = this.session!.inputNames;
    const feeds: Record<string, import('onnxruntime-web').Tensor> = { [inputNames[0]!]: tensor };

    const start = performance.now();
    const results = await this.session!.run(feeds);
    const elapsed = performance.now() - start;

    const _output = Object.values(results)[0];
    return elapsed;
  }

  async classify(topK = 5): Promise<ClassificationResult[]> {
    const ort = this.ort!;
    const tensor = new ort.Tensor('float32', this.inputData, [1, 3, 224, 224]);
    const inputNames = this.session!.inputNames;
    const feeds: Record<string, import('onnxruntime-web').Tensor> = { [inputNames[0]!]: tensor };

    const results = await this.session!.run(feeds);
    const output = Object.values(results)[0]!;
    const logits = output.data as Float32Array;
    return extractTopK(logits, topK);
  }

  async dispose(): Promise<void> {
    // Don't release the session — it's cached for reuse across sessions
    this.session = null;
    this.inputData = new Float32Array(0);
    this.ort = null;
  }
}

function extractTopK(logits: Float32Array, topK: number): ClassificationResult[] {
  const maxLogit = Math.max(...logits);
  const exps = Array.from(logits).map(l => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((s, e) => s + e, 0);
  const probs = exps.map(e => e / sumExp);

  const indexed = probs.map((score, i) => ({ score, i }));
  indexed.sort((a, b) => b.score - a.score);

  return indexed.slice(0, topK).map(({ score, i }) => ({
    label: IMAGENET_LABELS[i] ?? `class_${i}`,
    labelIndex: i,
    score,
  }));
}
