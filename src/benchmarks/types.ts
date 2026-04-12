import type { PreprocessedImage } from '../utils/image-input';
import type { TokenizedText } from '../utils/tokenizer';

export type TaskId = 'image-classification' | 'text-classification';
export type FrameworkId = 'tfjs' | 'onnx' | 'litert' | 'transformersjs' | 'tflite-native';
export type BackendId = 'wasm' | 'wasm-simd' | 'wasm-threads' | 'wasm-simd-threads' | 'webgl' | 'webgpu' | 'webnn' | 'cpu' | 'gpu';

export interface ClassificationResult {
  label: string;
  labelIndex: number;
  score: number;
}

export type BenchmarkInput =
  | { type: 'image'; image: PreprocessedImage }
  | { type: 'text'; text: TokenizedText; rawText: string };

export interface ProgressCallback {
  (phase: string, detail: string): void;
}

/**
 * Each framework adapter must implement this interface.
 */
export interface FrameworkBenchmark {
  /** Human-readable name */
  name: string;
  /** Supported backends for this framework */
  supportedBackends: BackendId[];
  /** Framework runtime size in bytes (JS + WASM, excluding model). Populated after initFramework(). */
  frameworkBytes: number;
  /** Initialize the framework runtime with given backend */
  initFramework(backend: BackendId): Promise<void>;
  /** Pre-fetch model data (download only, not timed) */
  prefetchModel(): Promise<void>;
  /** Load / compile the model (from prefetched data) */
  loadModel(): Promise<void>;
  /** Set input data for inference (image or tokenized text) */
  setInput(input: BenchmarkInput): void;
  /** Run a single inference and return the elapsed time in ms */
  runInference(): Promise<number>;
  /** Run inference and return top-K classification results */
  classify(topK?: number): Promise<ClassificationResult[]>;
  /** Cleanup resources */
  dispose(): Promise<void>;
}

/** Returns whether a backend ID is a WASM variant */
export function isWasmBackend(backend: BackendId): boolean {
  return backend === 'wasm' || backend.startsWith('wasm-');
}

/** Parse WASM variant flags from a backend ID */
export function getWasmFlags(backend: BackendId): { simd: boolean; threads: boolean } {
  switch (backend) {
    case 'wasm':           return { simd: false, threads: false };
    case 'wasm-simd':      return { simd: true,  threads: false };
    case 'wasm-threads':   return { simd: false, threads: true };
    case 'wasm-simd-threads': return { simd: true, threads: true };
    default:               return { simd: true,  threads: true }; // non-wasm, doesn't matter
  }
}

export const FRAMEWORK_BACKENDS: Record<FrameworkId, BackendId[]> = {
  tfjs: ['wasm', 'wasm-simd', 'wasm-threads', 'wasm-simd-threads', 'webgl', 'webgpu'],
  onnx: ['wasm', 'wasm-simd', 'wasm-threads', 'wasm-simd-threads', 'webgl', 'webgpu', 'webnn'],
  litert: ['wasm', 'wasm-simd-threads', 'webgpu'],
  transformersjs: ['wasm', 'wasm-simd', 'wasm-threads', 'wasm-simd-threads', 'webgpu', 'webnn'],
  'tflite-native': ['cpu', 'gpu'],
};
