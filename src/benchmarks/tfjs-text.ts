import { isWasmBackend, getWasmFlags, type BackendId, type BenchmarkInput, type ClassificationResult, type FrameworkBenchmark } from './types';
import type { TokenizedText } from '../utils/tokenizer';
import * as tf from '@tensorflow/tfjs';
import { measureNewResources, getContentLength } from '../utils/metrics';

const LABELS = ['NEGATIVE', 'POSITIVE'];

export class TfjsTextBenchmark implements FrameworkBenchmark {
  name = 'TensorFlow.js';
  supportedBackends: BackendId[] = ['wasm-simd-threads', 'webgl', 'webgpu'];
  frameworkBytes = 0;
  private model: tf.GraphModel | null = null;
  private tokenized: TokenizedText | null = null;
  private modelUrl = '/distilbert-base-uncased-finetuned-sst-2-english/model_tfjs/model.json';

  async initFramework(backend: BackendId): Promise<void> {
    const before = performance.getEntriesByType('resource').length;

    if (isWasmBackend(backend)) {
      const { simd, threads } = getWasmFlags(backend);
      const wasmModule = await import('@tensorflow/tfjs-backend-wasm');
      wasmModule.setWasmPaths('/');
      tf.env().set('WASM_HAS_SIMD_SUPPORT', simd);
      tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', threads);
      await tf.setBackend('wasm');
    } else if (backend === 'webgl') {
      await import('@tensorflow/tfjs-backend-webgl');
      await tf.setBackend('webgl');
    } else if (backend === 'webgpu') {
      await import('@tensorflow/tfjs-backend-webgpu');
      await tf.setBackend('webgpu');
    }
    await tf.ready();

    this.frameworkBytes = measureNewResources(before);

    if (isWasmBackend(backend)) {
      const { simd, threads } = getWasmFlags(backend);
      let wasmFile: string;
      if (simd && threads) wasmFile = '/tfjs-backend-wasm-threaded-simd.wasm';
      else if (simd)       wasmFile = '/tfjs-backend-wasm-simd.wasm';
      else                 wasmFile = '/tfjs-backend-wasm.wasm';
      this.frameworkBytes += await getContentLength(wasmFile);
    }
  }

  async prefetchModel(): Promise<void> {
    const resp = await fetch(this.modelUrl);
    const modelJson = await resp.json();
    const shards: string[] = (modelJson.weightsManifest ?? [])
      .flatMap((m: any) => m.paths ?? []);
    const base = this.modelUrl.replace(/\/[^/]+$/, '/');
    await Promise.all(shards.map(s => fetch(base + s)));
  }

  async loadModel(): Promise<void> {
    this.model = await tf.loadGraphModel(this.modelUrl);

    // Default dummy tokenized input (sequence of zeros)
    this.tokenized = {
      inputIds: new Int32Array(128),
      attentionMask: new Int32Array(128),
      seqLength: 0,
    };
  }

  setInput(input: BenchmarkInput): void {
    if (input.type !== 'text') throw new Error('TfjsTextBenchmark only supports text input');
    this.tokenized = input.text;
  }

  async runInference(): Promise<number> {
    const t = this.tokenized!;
    const inputIds = tf.tensor(Array.from(t.inputIds), [1, t.inputIds.length], 'int32');
    const attentionMask = tf.tensor(Array.from(t.attentionMask), [1, t.attentionMask.length], 'int32');

    const feeds: Record<string, tf.Tensor> = {
      input_ids: inputIds,
      attention_mask: attentionMask,
    };

    const start = performance.now();
    const result = this.model!.predict(feeds);
    const output = Array.isArray(result) ? result[0]! : result;
    await (output as tf.Tensor).data();
    const elapsed = performance.now() - start;

    (output as tf.Tensor).dispose();
    if (Array.isArray(result)) result.forEach(r => r.dispose());
    inputIds.dispose();
    attentionMask.dispose();

    return elapsed;
  }

  async classify(_topK = 5): Promise<ClassificationResult[]> {
    const t = this.tokenized!;
    const inputIds = tf.tensor(Array.from(t.inputIds), [1, t.inputIds.length], 'int32');
    const attentionMask = tf.tensor(Array.from(t.attentionMask), [1, t.attentionMask.length], 'int32');

    const feeds: Record<string, tf.Tensor> = {
      input_ids: inputIds,
      attention_mask: attentionMask,
    };

    const result = this.model!.predict(feeds);
    const output = Array.isArray(result) ? result[0]! : result;
    const logits = await (output as tf.Tensor).data();

    (output as tf.Tensor).dispose();
    if (Array.isArray(result)) result.forEach(r => r.dispose());
    inputIds.dispose();
    attentionMask.dispose();

    return logitsToResults(logits as Float32Array);
  }

  async dispose(): Promise<void> {
    this.model?.dispose();
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
