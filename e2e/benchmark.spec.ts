import { test, chromium } from '@playwright/test';
import type { CDPSession, Page } from '@playwright/test';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

/**
 * Automated session-mode benchmark runner with CDP memory measurements.
 *
 * For each framework × backend combo, launches a single browser and runs
 * SESSIONS iterations.  Each iteration creates a fresh benchmark instance
 * (init → prefetch → load → single inference → dispose) inside the SAME
 * browser so we can observe caching effects (WASM compilation cache,
 * model data in Cache API, etc.) that make init/load faster over time.
 *
 * 0 warmup iterations — every inference counts.
 */

// Set via env: BENCHMARK_TASK=text npx playwright test e2e/benchmark.spec.ts
const TASK = process.env.BENCHMARK_TASK === 'text' ? 'text-classification' : 'image-classification';
const TEXT_INPUT = 'This movie was absolutely wonderful and I loved every moment of it.';
const SESSIONS = parseInt(process.env.BENCHMARK_SESSIONS ?? '10', 10);

const IMAGE_MATRIX: [string, string][] = [
  ['tfjs', 'wasm-simd-threads'],
  ['tfjs', 'webgl'],
  ['tfjs', 'webgpu'],
  ['onnx', 'wasm-simd-threads'],
  ['onnx', 'webgl'],
  ['onnx', 'webgpu'],
  ['onnx', 'webnn'],
  ['litert', 'wasm-simd-threads'],
  ['litert', 'webgpu'],
  ['transformersjs', 'wasm-simd-threads'],
  ['transformersjs', 'webgpu'],
  ['transformersjs', 'webnn'],
  ['tflite-native', 'cpu'],
  ['tflite-native', 'gpu'],
];

const TEXT_MATRIX: [string, string][] = [
  ['tfjs', 'wasm-simd-threads'],
  ['tfjs', 'webgl'],
  ['tfjs', 'webgpu'],
  ['onnx', 'wasm-simd-threads'],
  ['onnx', 'webgl'],
  ['onnx', 'webgpu'],
  ['onnx', 'webnn'],
  ['litert', 'wasm-simd-threads'],
  ['transformersjs', 'wasm-simd-threads'],
  ['transformersjs', 'webgpu'],
  ['transformersjs', 'webnn'],
  ['tflite-native', 'cpu'],
  ['tflite-native', 'gpu'],
];

const MATRIX = TASK === 'image-classification' ? IMAGE_MATRIX : TEXT_MATRIX;

interface CDPMemorySnapshot {
  processMemoryMB: number;
  jsHeapUsedMB: number;
}

interface SessionIterationResult {
  framework: string;
  backend: string;
  session: number;
  memoryBefore: CDPMemorySnapshot;
  memoryAfter: CDPMemorySnapshot;
  memoryDeltaMB: number;
  frameworkInitMs: number;
  modelLoadMs: number;
  inferenceMs: number;
  totalMs: number;
  error?: string;
}

async function getCDPMemory(cdp: CDPSession, page: Page): Promise<CDPMemorySnapshot> {
  try {
    await cdp.send('HeapProfiler.collectGarbage');
  } catch { /* GC may not be available */ }

  await new Promise(r => setTimeout(r, 300));

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

  const { metrics } = await cdp.send('Performance.getMetrics');
  const processMemory = metrics.find(m => m.name === 'ProcessPrivateMemoryFootprint');
  const jsHeapCDP = metrics.find(m => m.name === 'JSHeapUsedSize');

  const jsHeapMB = jsMemory
    ? jsMemory.usedJSHeapSize / 1024 / 1024
    : (jsHeapCDP?.value ?? 0) / 1024 / 1024;

  const processMemMB = (processMemory?.value ?? 0) / 1024 / 1024;

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

/**
 * Run N session iterations for a single framework × backend combo
 * inside the same browser instance.
 */
async function runSessionBenchmark(
  framework: string,
  backend: string,
  baseURL: string,
  sessions: number,
): Promise<SessionIterationResult[]> {
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

  const results: SessionIterationResult[] = [];

  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    const cdp = await context.newCDPSession(page);
    await cdp.send('Performance.enable');

    page.setDefaultTimeout(4 * 60_000);
    await page.goto(baseURL, { waitUntil: 'load', timeout: 30000 });
    await page.waitForFunction(
      () => (window as any).__benchmark !== undefined,
      { timeout: 15000 },
    );

    // Configure task and input once
    await page.evaluate(
      async ([fw, be, task, textInput]) => {
        const b = (window as any).__benchmark;
        b.setTask(task);
        b.configure(fw, be, 1, 0);
        if (task === 'image-classification') {
          await b.loadImage('/rocky.jpg');
        } else {
          b.setTextInput(textInput);
        }
      },
      [framework, backend, TASK, TEXT_INPUT],
    );

    for (let i = 1; i <= sessions; i++) {
      try {
        const memoryBefore = await getCDPMemory(cdp, page);

        // Each iteration: create adapter → init → prefetch → load → inference → dispose
        const metrics: any = await page.evaluate(
          async ([fw, be, task]) => {
            const { createBenchmark } = await import('/src/benchmarks/index.ts');

            const benchmark = createBenchmark(fw, task);

            const initStart = performance.now();
            await benchmark.initFramework(be);
            const frameworkInitMs = performance.now() - initStart;

            await benchmark.prefetchModel();

            const loadStart = performance.now();
            await benchmark.loadModel();
            const modelLoadMs = performance.now() - loadStart;

            // Prepare input
            const b = (window as any).__benchmark;
            if (task === 'image-classification') {
              const img = b.__getCurrentImage();
              benchmark.setInput({ type: 'image', image: img });
            } else {
              const { getTokenizer } = await import('/src/utils/tokenizer.ts');
              const tokenizer = await getTokenizer();
              const text = (document.getElementById('text-input') as HTMLInputElement).value;
              const tokenized = tokenizer.tokenize(text);
              benchmark.setInput({ type: 'text', text: tokenized, rawText: text });
            }

            const infStart = performance.now();
            await benchmark.runInference();
            const inferenceMs = performance.now() - infStart;

            await benchmark.dispose();

            return {
              framework: benchmark.name,
              backend: be.toUpperCase(),
              frameworkInitMs: Math.round(frameworkInitMs * 100) / 100,
              modelLoadMs: Math.round(modelLoadMs * 100) / 100,
              inferenceMs: Math.round(inferenceMs * 100) / 100,
            };
          },
          [framework, backend, TASK],
        );

        const memoryAfter = await getCDPMemory(cdp, page);

        results.push({
          framework: metrics.framework,
          backend: metrics.backend,
          session: i,
          memoryBefore,
          memoryAfter,
          memoryDeltaMB: Math.round((memoryAfter.processMemoryMB - memoryBefore.processMemoryMB) * 100) / 100,
          frameworkInitMs: metrics.frameworkInitMs,
          modelLoadMs: metrics.modelLoadMs,
          inferenceMs: metrics.inferenceMs,
          totalMs: Math.round((metrics.frameworkInitMs + metrics.modelLoadMs + metrics.inferenceMs) * 100) / 100,
        });
      } catch (err: any) {
        const zero: CDPMemorySnapshot = { processMemoryMB: 0, jsHeapUsedMB: 0 };
        results.push({
          framework, backend: backend.toUpperCase(),
          session: i,
          memoryBefore: zero, memoryAfter: zero, memoryDeltaMB: 0,
          frameworkInitMs: 0, modelLoadMs: 0, inferenceMs: 0, totalMs: 0,
          error: err.message || String(err),
        });
      }
    }
  } finally {
    await browser.close();
  }

  return results;
}

test('run all benchmarks — session mode with CDP memory', async () => {
  const baseURL = 'http://localhost:5173';
  const allResults: SessionIterationResult[] = [];

  for (const [framework, backend] of MATRIX) {
    const combo = `${framework}:${backend}`;
    console.log(`\n  === ${combo} (${SESSIONS} sessions) ===`);

    try {
      const iterResults = await runSessionBenchmark(framework, backend, baseURL, SESSIONS);

      for (const r of iterResults) {
        allResults.push(r);
        if (r.error) {
          console.log(`    #${r.session} FAIL — ${r.error}`);
        } else {
          console.log(
            `    #${r.session}  init:${r.frameworkInitMs}ms  load:${r.modelLoadMs}ms  ` +
            `inf:${r.inferenceMs}ms  total:${r.totalMs}ms  ` +
            `mem:${r.memoryBefore.processMemoryMB}→${r.memoryAfter.processMemoryMB}MB`,
          );
        }
      }
    } catch (err: any) {
      const msg = err.message || String(err);
      console.log(`  FAIL ${combo} — ${msg}`);
      const zero: CDPMemorySnapshot = { processMemoryMB: 0, jsHeapUsedMB: 0 };
      allResults.push({
        framework, backend: backend.toUpperCase(),
        session: 0,
        memoryBefore: zero, memoryAfter: zero, memoryDeltaMB: 0,
        frameworkInitMs: 0, modelLoadMs: 0, inferenceMs: 0, totalMs: 0,
        error: msg,
      });
    }
  }

  // Save JSON
  const taskSuffix = TASK === 'text-classification' ? 'text' : 'image';
  const outputDir = join(process.cwd(), 'benchmark_results');
  mkdirSync(outputDir, { recursive: true });

  const jsonPath = join(outputDir, `session_results_${taskSuffix}.json`);
  const report = {
    timestamp: new Date().toISOString(),
    task: TASK,
    sessionsPerCombo: SESSIONS,
    results: allResults,
    errors: allResults.filter(r => r.error),
  };
  writeFileSync(jsonPath, JSON.stringify(report, null, 2));
  console.log(`\nJSON saved to ${jsonPath}`);

  // Save CSV
  const csvPath = join(outputDir, `session_results_${taskSuffix}.csv`);
  const csvHeader = [
    'Framework', 'Backend', 'Session#',
    'FrameworkInit(ms)', 'ModelLoad(ms)', 'Inference(ms)', 'Total(ms)',
    'MemBefore_Process(MB)', 'MemAfter_Process(MB)', 'MemDelta_Process(MB)',
    'MemBefore_JSHeap(MB)', 'MemAfter_JSHeap(MB)',
    'Error',
  ].join(',');
  const csvRows = allResults.map(r => [
    r.framework, r.backend, r.session,
    r.frameworkInitMs, r.modelLoadMs, r.inferenceMs, r.totalMs,
    r.memoryBefore.processMemoryMB, r.memoryAfter.processMemoryMB, r.memoryDeltaMB,
    r.memoryBefore.jsHeapUsedMB, r.memoryAfter.jsHeapUsedMB,
    `"${r.error ?? ''}"`,
  ].join(','));
  writeFileSync(csvPath, [csvHeader, ...csvRows].join('\n'));
  console.log(`CSV saved to ${csvPath}`);
});
