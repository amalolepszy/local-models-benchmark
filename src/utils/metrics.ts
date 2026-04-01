import type { ClassificationResult } from '../benchmarks/types';

export interface BenchmarkMetrics {
  framework: string;
  backend: string;
  frameworkInitMs: number;
  modelLoadMs: number;
  inferenceTimes: number[];
  avgInferenceMs: number;
  minInferenceMs: number;
  maxInferenceMs: number;
  p95InferenceMs: number;
  memoryBeforeMB: number | null;
  memoryAfterMB: number | null;
  memoryDeltaMB: number | null;
  predictions: ClassificationResult[];
  expectedClass: string | null;
  top1Correct: boolean | null;
  top5Correct: boolean | null;
  top1Confidence: number | null;
}

export function computeStats(times: number[]): {
  avg: number;
  min: number;
  max: number;
  p95: number;
} {
  const sorted = [...times].sort((a, b) => a - b);
  const avg = times.reduce((s, t) => s + t, 0) / times.length;
  const p95Index = Math.ceil(sorted.length * 0.95) - 1;
  return {
    avg: round(avg),
    min: round(sorted[0]!),
    max: round(sorted[sorted.length - 1]!),
    p95: round(sorted[Math.max(0, p95Index)]!),
  };
}

export function round(n: number, decimals = 2): number {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}

export async function getMemoryUsageMB(): Promise<number | null> {
  const perf = performance as any;
  if (perf.memory) {
    return round(perf.memory.usedJSHeapSize / (1024 * 1024));
  }
  if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated && perf.measureUserAgentSpecificMemory) {
    try {
      const result = await perf.measureUserAgentSpecificMemory();
      return round(result.bytes / (1024 * 1024));
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Sum decodedBodySize of resource entries added since the given index.
 * Used to measure bytes loaded during a specific phase (e.g., framework init).
 */
export function measureNewResources(snapshotIndex: number): number {
  const entries = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
  let total = 0;
  for (let i = snapshotIndex; i < entries.length; i++) {
    total += entries[i]!.decodedBodySize || entries[i]!.transferSize || 0;
  }
  return total;
}

/**
 * Measure file size via HEAD request (Content-Length header).
 * Returns 0 if the request fails or header is missing.
 */
export async function getContentLength(url: string): Promise<number> {
  try {
    const r = await fetch(url, { method: 'HEAD' });
    if (r.ok) return parseInt(r.headers.get('content-length') || '0', 10);
  } catch { /* ignore */ }
  return 0;
}

/**
 * Check if a prediction matches the expected class (case-insensitive substring match).
 */
export function matchesExpectedClass(prediction: ClassificationResult, expected: string): boolean {
  const lowerLabel = prediction.label.toLowerCase();
  const lowerExpected = expected.toLowerCase().trim();
  return lowerLabel.includes(lowerExpected) || lowerExpected.includes(lowerLabel.split(',')[0]!);
}

export function evaluateAccuracy(
  predictions: ClassificationResult[],
  expectedClass: string | null,
): { top1Correct: boolean | null; top5Correct: boolean | null; top1Confidence: number | null } {
  if (!expectedClass || predictions.length === 0) {
    return { top1Correct: null, top5Correct: null, top1Confidence: null };
  }
  const top1Correct = matchesExpectedClass(predictions[0]!, expectedClass);
  const top5Correct = predictions.slice(0, 5).some(p => matchesExpectedClass(p, expectedClass));
  const top1Confidence = round(predictions[0]!.score * 100);
  return { top1Correct, top5Correct, top1Confidence };
}

export function metricsToCSVRow(m: BenchmarkMetrics): string {
  return [
    m.framework,
    m.backend,
    m.frameworkInitMs,
    m.modelLoadMs,
    m.avgInferenceMs,
    m.minInferenceMs,
    m.maxInferenceMs,
    m.p95InferenceMs,
    m.memoryBeforeMB ?? 'N/A',
    m.memoryAfterMB ?? 'N/A',
    m.memoryDeltaMB ?? 'N/A',
    m.expectedClass ?? '',
    m.top1Correct ?? 'N/A',
    m.top5Correct ?? 'N/A',
    m.top1Confidence != null ? `${m.top1Confidence}%` : 'N/A',
    `"${m.predictions.slice(0, 3).map(p => `${p.label.split(',')[0]} (${round(p.score * 100)}%)`).join('; ')}"`,
  ].join(',');
}

export const CSV_HEADER = 'Framework,Backend,FrameworkInit(ms),ModelLoad(ms),AvgInference(ms),MinInference(ms),MaxInference(ms),P95Inference(ms),MemoryBefore(MB),MemoryAfter(MB),MemoryDelta(MB),ExpectedClass,Top1Correct,Top5Correct,Top1Confidence,TopPredictions';
