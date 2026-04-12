import type { FrameworkBenchmark, FrameworkId, TaskId } from './types';
import { TfjsBenchmark } from './tfjs';
import { OnnxBenchmark } from './onnx';
import { LiteRTBenchmark } from './litert';
import { TransformersJsBenchmark } from './transformersjs';
import { TFLiteNativeBenchmark } from './tflite-native';
import { TfjsTextBenchmark } from './tfjs-text';
import { OnnxTextBenchmark } from './onnx-text';
import { LiteRTTextBenchmark } from './litert-text';
import { TransformersJsTextBenchmark } from './transformersjs-text';
import { TFLiteNativeTextBenchmark } from './tflite-native-text';

export function createBenchmark(id: FrameworkId, task: TaskId = 'image-classification'): FrameworkBenchmark {
  if (task === 'text-classification') {
    switch (id) {
      case 'tfjs': return new TfjsTextBenchmark();
      case 'onnx': return new OnnxTextBenchmark();
      case 'litert': return new LiteRTTextBenchmark();
      case 'transformersjs': return new TransformersJsTextBenchmark();
      case 'tflite-native': return new TFLiteNativeTextBenchmark();
    }
  }

  switch (id) {
    case 'tfjs': return new TfjsBenchmark();
    case 'onnx': return new OnnxBenchmark();
    case 'litert': return new LiteRTBenchmark();
    case 'transformersjs': return new TransformersJsBenchmark();
    case 'tflite-native': return new TFLiteNativeBenchmark();
  }
}

export { FRAMEWORK_BACKENDS } from './types';
export type { FrameworkId, BackendId, TaskId } from './types';
