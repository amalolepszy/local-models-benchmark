import type { BackendId, BenchmarkInput, ClassificationResult, FrameworkBenchmark } from './types';
import type { TokenizedText } from '../utils/tokenizer';

declare global {
  interface Window {
    tfliteNative?: {
      createSession(options?: { delegate?: 'cpu' | 'gpu' }): Promise<TFLiteNativeSession>;
    };
  }
}

interface TFLiteNativeSession {
  loadModel(modelName: string): Promise<void>;
  run(inputData: Float32Array): Promise<Float32Array>;
  runInt32(inputData: Int32Array): Promise<Float32Array>;
  dispose(): void;
}

const LABELS = ['NEGATIVE', 'POSITIVE'];

export class TFLiteNativeTextBenchmark implements FrameworkBenchmark {
  name = 'TFLite Native';
  frameworkBytes = 0;
  supportedBackends: BackendId[] = ['cpu', 'gpu'];
  private session: TFLiteNativeSession | null = null;
  private tokenized: TokenizedText | null = null;

  async initFramework(backend: BackendId): Promise<void> {
    if (!window.tfliteNative) {
      throw new Error(
        'window.tfliteNative not available. Launch Chrome with --enable-blink-features=TFLiteNativeInference',
      );
    }
    const delegate = backend as 'cpu' | 'gpu';
    this.session = await window.tfliteNative.createSession({ delegate });
  }

  async prefetchModel(): Promise<void> {
    // Model is bundled on disk — no network fetch needed
  }

  async loadModel(): Promise<void> {
    await this.session!.loadModel('distilbert-base-uncased-finetuned-sst-2-english');

    this.tokenized = {
      inputIds: new Int32Array(128),
      attentionMask: new Int32Array(128),
      seqLength: 0,
    };
  }

  setInput(input: BenchmarkInput): void {
    if (input.type !== 'text') throw new Error('TFLiteNativeTextBenchmark only supports text input');
    this.tokenized = input.text;
  }

  async runInference(): Promise<number> {
    // Native API expects just input_ids as Int32Array(128)
    const start = performance.now();
    await this.session!.runInt32(this.tokenized!.inputIds);
    return performance.now() - start;
  }

  async classify(_topK = 5): Promise<ClassificationResult[]> {
    const output = await this.session!.runInt32(this.tokenized!.inputIds);
    return logitsToResults(output);
  }

  async dispose(): Promise<void> {
    this.session?.dispose();
    this.session = null;
    this.tokenized = null;
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
