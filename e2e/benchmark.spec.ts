import { test, chromium } from '@playwright/test';
import type { CDPSession, Page, BrowserContext } from '@playwright/test';
import { writeFileSync } from 'fs';
import { join } from 'path';

/**
 * Automated benchmark runner with CDP memory measurements.
 *
 * Each framework × backend combo runs in a fresh browser context
 * to avoid cross-contamination and COOP/COEP navigation issues.
 *
 * Measures ProcessPrivateMemoryFootprint (total process memory)
 * and JSHeapUsedSize before and after each benchmark run.
 */

const MATRIX: [string, string][] = [
  ['tfjs', 'wasm'],
  ['tfjs', 'webgl'],
  ['tfjs', 'webgpu'],
  ['onnx', 'wasm'],
  ['onnx', 'webgl'],
  ['onnx', 'webgpu'],
  ['onnx', 'webnn'],
  ['litert', 'wasm'],
  ['litert', 'webgpu'],
  ['transformersjs', 'wasm'],
  ['transformersjs', 'webgpu'],
  ['transformersjs', 'webnn'],
  ['tflite-native', 'cpu'],
  ['tflite-native', 'gpu'],
];

interface CDPMemorySnapshot {
  processMemoryMB: number;
  jsHeapUsedMB: number;
}

interface BenchmarkResult {
  framework: string;
  backend: string;
  memoryBefore: CDPMemorySnapshot;
  memoryAfter: CDPMemorySnapshot;
  memoryDeltaMB: number;
  frameworkInitMs: number;
  modelLoadMs: number;
  avgInferenceMs: number;
  minInferenceMs: number;
  maxInferenceMs: number;
  p95InferenceMs: number;
  predictions: { label: string; score: number }[];
  error?: string;
}

async function getCDPMemory(cdp: CDPSession, page: Page): Promise<CDPMemorySnapshot> {
  try {
    await cdp.send('HeapProfiler.collectGarbage');
  } catch { /* GC may not be available */ }

  await new Promise(r => setTimeout(r, 300));

  // Get JS heap from performance.memory (Chrome-only, enabled via --enable-precise-memory-info)
  const jsMemory = await page.evaluate(() => {
    const perf = performance as any;
    if (perf.memory) {
      return {
        usedJSHeapSize: perf.memory.usedJSHeapSize as number,
        totalJSHeapSize: perf.memory.totalJSHeapSize as number,
      };
    }
    return null;
  });

  // Get process memory via CDP Performance.getMetrics
  const { metrics } = await cdp.send('Performance.getMetrics');
  const processMemory = metrics.find(m => m.name === 'ProcessPrivateMemoryFootprint');
  const jsHeapCDP = metrics.find(m => m.name === 'JSHeapUsedSize');

  // Also try to get detailed memory breakdown via Memory.getBrowserSamplingProfile
  let totalFromSampling = 0;
  try {
    const sampling = await cdp.send('Memory.getDOMCounters' as any);
    totalFromSampling = (sampling as any)?.jsHeapSizeUsed ?? 0;
  } catch { /* may not be available */ }

  const jsHeapMB = jsMemory
    ? jsMemory.usedJSHeapSize / 1024 / 1024
    : (jsHeapCDP?.value ?? 0) / 1024 / 1024;

  const processMemMB = (processMemory?.value ?? 0) / 1024 / 1024;

  // If ProcessPrivateMemoryFootprint is 0, use totalJSHeapSize as a rough proxy
  const effectiveProcessMB = processMemMB > 0
    ? processMemMB
    : jsMemory
      ? jsMemory.totalJSHeapSize / 1024 / 1024
      : 0;

  return {
    processMemoryMB: Math.round(effectiveProcessMB * 100) / 100,
    jsHeapUsedMB: Math.round(jsHeapMB * 100) / 100,
  };
}

async function runSingleBenchmark(
  framework: string,
  backend: string,
  baseURL: string,
): Promise<BenchmarkResult> {
  // Fresh browser context per combo — avoids COOP context destruction
  const browser = await chromium.launch({
    executablePath: 'D:/chr-build/chromium/src/out/Release/chrome.exe',
    headless: false,
    args: [
      '--enable-precise-memory-info',
      '--js-flags=--expose-gc',
      '--enable-features=WebMachineLearningNeuralNetwork,ExperimentalWebMachineLearningNeuralNetwork,WebNNDirectML,WebNNOnnxRuntime,ExperimentalWebAssemblyFeatures,ExperimentalWebAssemblySharedEverything,ExperimentalWebAssemblyStackSwitching,WebAssemblyBaseline,WebAssemblyLazyCompilation,WebAssemblyTiering',
      '--enable-unsafe-webgpu',
      '--enable-blink-features=TFLiteNativeInference',
    ],
  });

  const context = await browser.newContext();
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send('Performance.enable');

  try {
    await page.goto(baseURL, { waitUntil: 'load', timeout: 30000 });

    // Wait for the __benchmark API to be ready
    await page.waitForFunction(
      () => (window as any).__benchmark !== undefined,
      { timeout: 15000 },
    );

    // Generate noise and configure
    await page.evaluate(
      ([fw, be]) => {
        const b = (window as any).__benchmark;
        b.generateNoise();
        b.configure(fw, be, 10, 3);
      },
      [framework, backend],
    );

    // Measure memory before
    const memoryBefore = await getCDPMemory(cdp, page);

    // Run the benchmark (this blocks until done)
    const metrics: any = await page.evaluate(
      () => (window as any).__benchmark.run(),
      { timeout: 4 * 60_000 },
    );

    // Measure memory after
    const memoryAfter = await getCDPMemory(cdp, page);

    return {
      framework: metrics.framework,
      backend: metrics.backend,
      memoryBefore,
      memoryAfter,
      memoryDeltaMB: Math.round((memoryAfter.processMemoryMB - memoryBefore.processMemoryMB) * 100) / 100,
      frameworkInitMs: metrics.frameworkInitMs,
      modelLoadMs: metrics.modelLoadMs,
      avgInferenceMs: metrics.avgInferenceMs,
      minInferenceMs: metrics.minInferenceMs,
      maxInferenceMs: metrics.maxInferenceMs,
      p95InferenceMs: metrics.p95InferenceMs,
      predictions: metrics.predictions.map((p: any) => ({
        label: p.label,
        score: Math.round(p.score * 10000) / 100,
      })),
    };
  } finally {
    await browser.close();
  }
}

function emptyResult(framework: string, backend: string, error: string): BenchmarkResult {
  const zero: CDPMemorySnapshot = { processMemoryMB: 0, jsHeapUsedMB: 0 };
  return {
    framework, backend: backend.toUpperCase(),
    memoryBefore: zero, memoryAfter: zero, memoryDeltaMB: 0,
    frameworkInitMs: 0, modelLoadMs: 0,
    avgInferenceMs: 0, minInferenceMs: 0, maxInferenceMs: 0, p95InferenceMs: 0,
    predictions: [], error,
  };
}

test('run all benchmarks with CDP memory profiling', async () => {
  const baseURL = 'http://localhost:5173';
  const results: BenchmarkResult[] = [];

  for (const [framework, backend] of MATRIX) {
    const combo = `${framework}:${backend}`;
    console.log(`  RUN  ${combo}`);

    try {
      const result = await runSingleBenchmark(framework, backend, baseURL);
      results.push(result);

      console.log(
        `  DONE ${combo} — ` +
        `init: ${result.frameworkInitMs}ms, ` +
        `load: ${result.modelLoadMs}ms, ` +
        `avg: ${result.avgInferenceMs}ms, ` +
        `mem: ${result.memoryBefore.processMemoryMB} → ${result.memoryAfter.processMemoryMB} MB ` +
        `(+${result.memoryDeltaMB} MB)`,
      );
    } catch (err: any) {
      const msg = err.message || String(err);
      console.log(`  FAIL ${combo} — ${msg}`);
      results.push(emptyResult(framework, backend, msg));
    }
  }

  // Save JSON
  const outputPath = join(process.cwd(), 'benchmark_results.json');
  const report = {
    timestamp: new Date().toISOString(),
    results,
    errors: results.filter(r => r.error),
  };
  writeFileSync(outputPath, JSON.stringify(report, null, 2));
  console.log(`\nResults saved to ${outputPath}`);

  // Save CSV
  const csvPath = join(process.cwd(), 'benchmark_results.csv');
  const csvHeader = [
    'Framework', 'Backend',
    'FrameworkInit(ms)', 'ModelLoad(ms)',
    'AvgInference(ms)', 'MinInference(ms)', 'MaxInference(ms)', 'P95Inference(ms)',
    'MemBefore_Process(MB)', 'MemAfter_Process(MB)', 'MemDelta_Process(MB)',
    'MemBefore_JSHeap(MB)', 'MemAfter_JSHeap(MB)',
    'Top1Prediction', 'Top1Score(%)',
    'Error',
  ].join(',');
  const csvRows = results.map(r => [
    r.framework, r.backend,
    r.frameworkInitMs, r.modelLoadMs,
    r.avgInferenceMs, r.minInferenceMs, r.maxInferenceMs, r.p95InferenceMs,
    r.memoryBefore.processMemoryMB, r.memoryAfter.processMemoryMB, r.memoryDeltaMB,
    r.memoryBefore.jsHeapUsedMB, r.memoryAfter.jsHeapUsedMB,
    `"${r.predictions[0]?.label ?? 'N/A'}"`, r.predictions[0]?.score ?? 'N/A',
    `"${r.error ?? ''}"`,
  ].join(','));
  writeFileSync(csvPath, [csvHeader, ...csvRows].join('\n'));
  console.log(`CSV saved to ${csvPath}`);
});
