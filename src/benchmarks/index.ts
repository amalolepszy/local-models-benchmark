import type { FrameworkBenchmark, FrameworkId } from './types';
import { TfjsBenchmark } from './tfjs';
import { OnnxBenchmark } from './onnx';
import { LiteRTBenchmark } from './litert';
import { TransformersJsBenchmark } from './transformersjs';

export function createBenchmark(id: FrameworkId): FrameworkBenchmark {
  switch (id) {
    case 'tfjs': return new TfjsBenchmark();
    case 'onnx': return new OnnxBenchmark();
    case 'litert': return new LiteRTBenchmark();
    case 'transformersjs': return new TransformersJsBenchmark();
  }
}

export { FRAMEWORK_BACKENDS } from './types';
export type { FrameworkId, BackendId } from './types';
