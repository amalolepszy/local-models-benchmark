import type { BackendId, FrameworkBenchmark } from './types';

/**
 * ONNX Runtime Web benchmark adapter.
 * Uses MobileNet v2 ONNX model for inference.
 *
 * The WebGL backend ships as a separate bundle (onnxruntime-web/webgl)
 * that must be imported instead of the default entry point.
 */
export class OnnxBenchmark implements FrameworkBenchmark {
  name = 'ONNX Runtime Web';
  supportedBackends: BackendId[] = ['wasm', 'webgl', 'webgpu', 'webnn'];
  private ort: typeof import('onnxruntime-web') | null = null;
  private session: import('onnxruntime-web').InferenceSession | null = null;
  private inputData: Float32Array | null = null;
  private backend: BackendId = 'wasm';

  async initFramework(backend: BackendId): Promise<void> {
    this.backend = backend;
    // WebGL has its own standalone bundle; other backends use the default entry
    if (backend === 'webgl') {
      this.ort = await import('onnxruntime-web/webgl');
    } else {
      this.ort = await import('onnxruntime-web');
    }
    // Configure WASM paths (used by wasm/webgpu/webnn backends)
    this.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';
  }

  async loadModel(): Promise<void> {
    const ort = this.ort!;
    // MobileNet v2 1.0 224x224 classification (served locally, from onnx-community/mobilenet_v2_1.0_224)
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
    // MobileNet v2 expects (1, 3, 224, 224)
    this.inputData = new Float32Array(1 * 3 * 224 * 224);
    for (let i = 0; i < this.inputData.length; i++) {
      this.inputData[i] = Math.random();
    }
  }

  async runInference(): Promise<number> {
    const ort = this.ort!;
    const tensor = new ort.Tensor('float32', this.inputData!, [1, 3, 224, 224]);
    const inputNames = this.session!.inputNames;
    const feeds: Record<string, import('onnxruntime-web').Tensor> = { [inputNames[0]!]: tensor };

    const start = performance.now();
    const results = await this.session!.run(feeds);
    const elapsed = performance.now() - start;

    // Access output to ensure completion
    const _output = Object.values(results)[0];
    return elapsed;
  }

  async dispose(): Promise<void> {
    await this.session?.release();
    this.session = null;
    this.inputData = null;
    this.ort = null;
  }
}
