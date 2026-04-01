import type { BackendId, ClassificationResult, FrameworkBenchmark } from './types';
import type { PreprocessedImage } from '../utils/image-input';
import { IMAGENET_LABELS } from '../utils/imagenet-labels';

declare global {
  interface Window {
    tfliteNative?: {
      createSession(options?: { delegate?: 'cpu' | 'gpu' | 'nnapi' }): Promise<TFLiteNativeSession>;
    };
  }
}

interface TFLiteNativeSession {
  loadModel(modelName: string): Promise<void>;
  run(inputData: Float32Array): Promise<Float32Array>;
  dispose(): void;
}

export class TFLiteNativeBenchmark implements FrameworkBenchmark {
  name = 'TFLite Native';
  supportedBackends: BackendId[] = ['cpu', 'gpu'];
  private session: TFLiteNativeSession | null = null;
  private inputData: Float32Array = new Float32Array(224 * 224 * 3);

  async initFramework(backend: BackendId): Promise<void> {
    if (!window.tfliteNative) {
      throw new Error(
        'window.tfliteNative not available. Launch Chrome with --enable-blink-features=TFLiteNativeInference',
      );
    }
    const delegate = backend as 'cpu' | 'gpu';
    this.session = await window.tfliteNative.createSession({ delegate });
  }

  async loadModel(): Promise<void> {
    await this.session!.loadModel('mobilenet_v2-1-0-224');
  }

  setImage(image: PreprocessedImage): void {
    // Native API expects HWC Float32Array of length 224*224*3 (no batch dim)
    // Use nhwcNegOneOne which is [1, 224, 224, 3] and strip the batch dimension
    const withBatch = image.nhwcNegOneOne;
    this.inputData = withBatch.length === 224 * 224 * 3
      ? withBatch
      : withBatch.slice(0, 224 * 224 * 3);
  }

  async runInference(): Promise<number> {
    const start = performance.now();
    await this.session!.run(this.inputData);
    return performance.now() - start;
  }

  async classify(topK = 5): Promise<ClassificationResult[]> {
    const output = await this.session!.run(this.inputData);
    return extractTopK(output, topK);
  }

  async dispose(): Promise<void> {
    this.session?.dispose();
    this.session = null;
    this.inputData = new Float32Array(0);
  }
}

function extractTopK(probs: Float32Array, topK: number): ClassificationResult[] {
  // Same TFLite model — output is probabilities (softmax baked in)
  const indexed = Array.from(probs).map((score, i) => ({ score, i }));
  indexed.sort((a, b) => b.score - a.score);

  return indexed.slice(0, topK).map(({ score, i }) => ({
    label: IMAGENET_LABELS[i] ?? `class_${i}`,
    labelIndex: i,
    score,
  }));
}
