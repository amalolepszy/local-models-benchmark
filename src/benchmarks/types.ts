import type { PreprocessedImage } from '../utils/image-input';

export type FrameworkId = 'tfjs' | 'onnx' | 'litert' | 'transformersjs' | 'tflite-native';
export type BackendId = 'wasm' | 'webgl' | 'webgpu' | 'webnn' | 'cpu' | 'gpu';

export interface ClassificationResult {
  label: string;
  labelIndex: number;
  score: number;
}

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
  /** Set preprocessed image input for inference */
  setImage(image: PreprocessedImage): void;
  /** Run a single inference and return the elapsed time in ms */
  runInference(): Promise<number>;
  /** Run inference and return top-K classification results */
  classify(topK?: number): Promise<ClassificationResult[]>;
  /** Cleanup resources */
  dispose(): Promise<void>;
}

export const FRAMEWORK_BACKENDS: Record<FrameworkId, BackendId[]> = {
  tfjs: ['wasm', 'webgl', 'webgpu'],
  onnx: ['wasm', 'webgl', 'webgpu', 'webnn'],
  litert: ['wasm', 'webgpu'],
  transformersjs: ['wasm', 'webgpu', 'webnn'],
  'tflite-native': ['cpu', 'gpu'],
};
