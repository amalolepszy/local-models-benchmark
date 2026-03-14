import type { BackendId, ClassificationResult, FrameworkBenchmark } from './types';
import type { PreprocessedImage } from '../utils/image-input';
import { IMAGENET_LABELS } from '../utils/imagenet-labels';

export class OnnxBenchmark implements FrameworkBenchmark {
  name = 'ONNX Runtime Web';
  supportedBackends: BackendId[] = ['wasm', 'webgl', 'webgpu', 'webnn'];
  private ort: typeof import('onnxruntime-web') | null = null;
  private session: import('onnxruntime-web').InferenceSession | null = null;
  private inputData: Float32Array = new Float32Array(0);
  private backend: BackendId = 'wasm';

  async initFramework(backend: BackendId): Promise<void> {
    this.backend = backend;
    if (backend === 'webgl') {
      this.ort = await import('onnxruntime-web/webgl');
    } else {
      this.ort = await import('onnxruntime-web');
    }
    this.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';
  }

  async loadModel(): Promise<void> {
    const ort = this.ort!;
    const MODEL_URL = '/mobilenet_v2_1.0_224.onnx';

    let executionProviders: import('onnxruntime-web').InferenceSession.ExecutionProviderConfig[];
    switch (this.backend) {
      case 'webgl':
        executionProviders = ['webgl'];
        break;
      case 'webgpu':
        executionProviders = ['webgpu'];
        break;
      case 'webnn':
        executionProviders = ['webnn'];
        break;
      default:
        executionProviders = ['wasm'];
    }

    this.session = await ort.InferenceSession.create(MODEL_URL, { executionProviders });
    // Default to random input — NCHW [1, 3, 224, 224]
    this.inputData = new Float32Array(1 * 3 * 224 * 224);
    for (let i = 0; i < this.inputData.length; i++) {
      this.inputData[i] = Math.random();
    }
  }

  setImage(image: PreprocessedImage): void {
    // ONNX model (google/mobilenet_v2) expects NCHW [-1, 1]
    this.inputData = image.nchwNegOneOne;
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
    await this.session?.release();
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
