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
    min: round(sorted[0]),
    max: round(sorted[sorted.length - 1]),
    p95: round(sorted[Math.max(0, p95Index)]),
  };
}

export function round(n: number, decimals = 2): number {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}

export async function getMemoryUsageMB(): Promise<number | null> {
  // Chrome-only: performance.memory (requires --enable-precise-memory-info flag)
  const perf = performance as any;
  if (perf.memory) {
    return round(perf.memory.usedJSHeapSize / (1024 * 1024));
  }
  // Fallback: crossOriginIsolated API (requires COOP/COEP headers)
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
  ].join(',');
}

export const CSV_HEADER = 'Framework,Backend,FrameworkInit(ms),ModelLoad(ms),AvgInference(ms),MinInference(ms),MaxInference(ms),P95Inference(ms),MemoryBefore(MB),MemoryAfter(MB),MemoryDelta(MB)';
