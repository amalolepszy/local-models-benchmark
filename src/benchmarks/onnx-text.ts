import { isWasmBackend, getWasmFlags, type BackendId, type BenchmarkInput, type ClassificationResult, type FrameworkBenchmark } from './types';
import type { TokenizedText } from '../utils/tokenizer';
import { measureNewResources, getContentLength } from '../utils/metrics';

const LABELS = ['NEGATIVE', 'POSITIVE'];

let cachedSession: import('onnxruntime-web').InferenceSession | null = null;

export class OnnxTextBenchmark implements FrameworkBenchmark {
  name = 'ONNX Runtime Web';
  frameworkBytes = 0;
  supportedBackends: BackendId[] = ['wasm-simd-threads', 'webgl', 'webgpu', 'webnn'];
  private ort: typeof import('onnxruntime-web') | null = null;
  private session: import('onnxruntime-web').InferenceSession | null = null;
  private tokenized: TokenizedText | null = null;
  private backend: BackendId = 'wasm';
  private modelBuffer: ArrayBuffer | null = null;
  private readonly modelUrl = '/distilbert-base-uncased-finetuned-sst-2-english/onnx/model.onnx';

  async initFramework(backend: BackendId): Promise<void> {
    const before = performance.getEntriesByType('resource').length;

    this.backend = backend;
    if (backend === 'webgl') {
      this.ort = await import('onnxruntime-web/webgl');
    } else {
      this.ort = await import('onnxruntime-web');
    }
    this.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';

    if (isWasmBackend(backend)) {
      const { simd, threads } = getWasmFlags(backend);
      this.ort.env.wasm.simd = simd;
      this.ort.env.wasm.numThreads = threads ? 4 : 1;
    }

    this.frameworkBytes = measureNewResources(before);

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
      const source: string | Uint8Array = this.modelBuffer ? new Uint8Array(this.modelBuffer) : this.modelUrl;
      this.session = await ort.InferenceSession.create(source as string, { executionProviders });
      cachedSession = this.session;
    }

    // Default dummy tokenized input
    this.tokenized = {
      inputIds: new Int32Array(128),
      attentionMask: new Int32Array(128),
      seqLength: 0,
    };
  }

  setInput(input: BenchmarkInput): void {
    if (input.type !== 'text') throw new Error('OnnxTextBenchmark only supports text input');
    this.tokenized = input.text;
  }

  async runInference(): Promise<number> {
    const ort = this.ort!;
    const t = this.tokenized!;
    const len = t.inputIds.length;

    // ONNX DistilBERT expects int64 tensors
    const inputIds = new ort.Tensor('int64', BigInt64Array.from(t.inputIds, v => BigInt(v)), [1, len]);
    const attentionMask = new ort.Tensor('int64', BigInt64Array.from(t.attentionMask, v => BigInt(v)), [1, len]);

    const feeds: Record<string, import('onnxruntime-web').Tensor> = {
      input_ids: inputIds,
      attention_mask: attentionMask,
    };

    const start = performance.now();
    const results = await this.session!.run(feeds);
    const elapsed = performance.now() - start;

    const _output = Object.values(results)[0];
    return elapsed;
  }

  async classify(_topK = 5): Promise<ClassificationResult[]> {
    const ort = this.ort!;
    const t = this.tokenized!;
    const len = t.inputIds.length;

    const inputIds = new ort.Tensor('int64', BigInt64Array.from(t.inputIds, v => BigInt(v)), [1, len]);
    const attentionMask = new ort.Tensor('int64', BigInt64Array.from(t.attentionMask, v => BigInt(v)), [1, len]);

    const feeds: Record<string, import('onnxruntime-web').Tensor> = {
      input_ids: inputIds,
      attention_mask: attentionMask,
    };

    const results = await this.session!.run(feeds);
    const output = Object.values(results)[0]!;
    const logits = new Float32Array(output.data as Float32Array);
    return logitsToResults(logits);
  }

  async dispose(): Promise<void> {
    // Don't release the session — it's cached for reuse across sessions
    this.session = null;
    this.tokenized = null;
    this.ort = null;
  }
}

function logitsToResults(logits: Float32Array): ClassificationResult[] {
  const maxLogit = Math.max(...logits);
  const exps = Array.from(logits).map(l => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((s, e) => s + e, 0);
  const probs = exps.map(e => e / sumExp);

  return probs
    .map((score, i) => ({ label: LABELS[i] ?? `class_${i}`, labelIndex: i, score }))
    .sort((a, b) => b.score - a.score);
}
