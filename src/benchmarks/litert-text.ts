import type { BackendId, BenchmarkInput, ClassificationResult, FrameworkBenchmark } from './types';
import type { TokenizedText } from '../utils/tokenizer';
import { loadLiteRt, loadAndCompile, Tensor, type CompiledModel } from '@litertjs/core';
import { measureNewResources } from '../utils/metrics';

const LABELS = ['NEGATIVE', 'POSITIVE'];

let liteRtLoaded = false;

export class LiteRTTextBenchmark implements FrameworkBenchmark {
  name = 'LiteRT.js';
  frameworkBytes = 0;
  supportedBackends: BackendId[] = ['wasm-simd-threads'];
  private model: CompiledModel | null = null;
  private tokenized: TokenizedText | null = null;
  private backend: BackendId = 'wasm';
  private readonly modelUrl = '/distilbert-base-uncased-finetuned-sst-2-english/tflite/model.tflite';

  async initFramework(backend: BackendId): Promise<void> {
    const before = performance.getEntriesByType('resource').length;

    this.backend = backend;
    if (!liteRtLoaded) {
      await loadLiteRt('/litert-wasm/');
      liteRtLoaded = true;
    }

    this.frameworkBytes = measureNewResources(before);
  }

  async prefetchModel(): Promise<void> {
    await fetch(this.modelUrl);
  }

  async loadModel(): Promise<void> {
    // TFLite DistilBERT — no WebGPU support (text models typically WASM only)
    this.model = await loadAndCompile(this.modelUrl, { accelerator: 'wasm' });

    // Default dummy tokenized input
    this.tokenized = {
      inputIds: new Int32Array(128),
      attentionMask: new Int32Array(128),
      seqLength: 0,
    };
  }

  setInput(input: BenchmarkInput): void {
    if (input.type !== 'text') throw new Error('LiteRTTextBenchmark only supports text input');
    this.tokenized = input.text;
  }

  async runInference(): Promise<number> {
    const t = this.tokenized!;
    // TFLite sorts inputs alphabetically: attention_mask=0, input_ids=1
    const attentionMaskTensor = new Tensor(t.attentionMask, [1, t.attentionMask.length]);
    const inputIdsTensor = new Tensor(t.inputIds, [1, t.inputIds.length]);

    const start = performance.now();
    const results = await this.model!.run([attentionMaskTensor, inputIdsTensor]);
    await results[0]!.data();
    const elapsed = performance.now() - start;

    attentionMaskTensor.delete();
    inputIdsTensor.delete();
    results[0]!.delete();

    return elapsed;
  }

  async classify(_topK = 5): Promise<ClassificationResult[]> {
    const t = this.tokenized!;
    // TFLite sorts inputs alphabetically: attention_mask=0, input_ids=1
    const attentionMaskTensor = new Tensor(t.attentionMask, [1, t.attentionMask.length]);
    const inputIdsTensor = new Tensor(t.inputIds, [1, t.inputIds.length]);

    const results = await this.model!.run([attentionMaskTensor, inputIdsTensor]);
    const outputData = await results[0]!.data();
    const logits = new Float32Array(outputData.buffer, outputData.byteOffset, outputData.length);

    attentionMaskTensor.delete();
    inputIdsTensor.delete();
    results[0]!.delete();

    return logitsToResults(logits);
  }

  async dispose(): Promise<void> {
    this.model?.delete();
    this.model = null;
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
