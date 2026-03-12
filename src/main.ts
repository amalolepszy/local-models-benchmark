import { createBenchmark, FRAMEWORK_BACKENDS, type FrameworkId, type BackendId } from './benchmarks';
import { computeStats, getMemoryUsageMB, round, metricsToCSVRow, CSV_HEADER, type BenchmarkMetrics } from './utils/metrics';

// --- DOM elements ---
const frameworkSelect = document.getElementById('framework') as HTMLSelectElement;
const backendSelect = document.getElementById('backend') as HTMLSelectElement;
const iterationsInput = document.getElementById('iterations') as HTMLInputElement;
const warmupInput = document.getElementById('warmup') as HTMLInputElement;
const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const exportBtn = document.getElementById('export-btn') as HTMLButtonElement;
const progressSection = document.getElementById('progress-section') as HTMLElement;
const progressBar = document.getElementById('progress-bar') as HTMLElement;
const progressText = document.getElementById('progress-text') as HTMLElement;
const resultsBody = document.getElementById('results-body') as HTMLTableSectionElement;

// --- State ---
const allResults: BenchmarkMetrics[] = [];

// --- Backend dropdown population ---
function updateBackendOptions() {
  const fw = frameworkSelect.value as FrameworkId;
  const backends = FRAMEWORK_BACKENDS[fw];
  backendSelect.innerHTML = backends
    .map(b => `<option value="${b}">${b.toUpperCase()}</option>`)
    .join('');
}

frameworkSelect.addEventListener('change', updateBackendOptions);
updateBackendOptions();

// --- Progress helpers ---
function showProgress(phase: string, detail: string, pct: number) {
  progressSection.hidden = false;
  progressBar.style.width = `${Math.min(100, pct)}%`;
  progressText.textContent = `${phase}: ${detail}`;
}

function hideProgress() {
  progressSection.hidden = true;
  progressBar.style.width = '0%';
}

// --- Run benchmark ---
runBtn.addEventListener('click', async () => {
  const frameworkId = frameworkSelect.value as FrameworkId;
  const backendId = backendSelect.value as BackendId;
  const iterations = parseInt(iterationsInput.value, 10) || 10;
  const warmup = parseInt(warmupInput.value, 10) || 3;

  runBtn.disabled = true;

  try {
    const benchmark = createBenchmark(frameworkId);

    // Phase 1: Framework init
    showProgress('Framework Init', `Initializing ${benchmark.name} with ${backendId.toUpperCase()}...`, 5);
    const memBefore = await getMemoryUsageMB();

    const initStart = performance.now();
    await benchmark.initFramework(backendId);
    const frameworkInitMs = round(performance.now() - initStart);

    showProgress('Framework Init', `Done in ${frameworkInitMs}ms`, 20);

    // Phase 2: Model load
    showProgress('Model Load', 'Downloading and compiling model...', 25);
    const loadStart = performance.now();
    await benchmark.loadModel();
    const modelLoadMs = round(performance.now() - loadStart);

    showProgress('Model Load', `Done in ${modelLoadMs}ms`, 50);

    // Phase 3: Warmup
    if (warmup > 0) {
      showProgress('Warmup', `Running ${warmup} warmup iterations...`, 55);
      for (let i = 0; i < warmup; i++) {
        await benchmark.runInference();
      }
    }

    // Phase 4: Inference
    const inferenceTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const pct = 60 + (i / iterations) * 35;
      showProgress('Inference', `Iteration ${i + 1}/${iterations}`, pct);
      const elapsed = await benchmark.runInference();
      inferenceTimes.push(round(elapsed));
    }

    const memAfter = await getMemoryUsageMB();
    const stats = computeStats(inferenceTimes);

    const metrics: BenchmarkMetrics = {
      framework: benchmark.name,
      backend: backendId.toUpperCase(),
      frameworkInitMs,
      modelLoadMs,
      inferenceTimes,
      avgInferenceMs: stats.avg,
      minInferenceMs: stats.min,
      maxInferenceMs: stats.max,
      p95InferenceMs: stats.p95,
      memoryBeforeMB: memBefore,
      memoryAfterMB: memAfter,
      memoryDeltaMB: memBefore != null && memAfter != null ? round(memAfter - memBefore) : null,
    };

    allResults.push(metrics);
    addResultRow(metrics);

    // Cleanup
    await benchmark.dispose();

    showProgress('Done', 'Benchmark complete!', 100);
    setTimeout(hideProgress, 2000);

  } catch (err: any) {
    showProgress('Error', err.message || String(err), 100);
    progressText.classList.add('status-error');
    console.error('Benchmark error:', err);
  } finally {
    runBtn.disabled = false;
    exportBtn.disabled = allResults.length === 0;
  }
});

// --- Results table ---
function addResultRow(m: BenchmarkMetrics) {
  // Remove placeholder
  const placeholder = resultsBody.querySelector('.placeholder-row');
  if (placeholder) placeholder.remove();

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${m.framework}</td>
    <td>${m.backend}</td>
    <td>${m.frameworkInitMs}</td>
    <td>${m.modelLoadMs}</td>
    <td>${m.avgInferenceMs}</td>
    <td>${m.minInferenceMs}</td>
    <td>${m.maxInferenceMs}</td>
    <td>${m.p95InferenceMs}</td>
    <td>${m.memoryBeforeMB ?? 'N/A'}</td>
    <td>${m.memoryAfterMB ?? 'N/A'}</td>
    <td>${m.memoryDeltaMB ?? 'N/A'}</td>
  `;
  resultsBody.appendChild(tr);
}

// --- Export CSV ---
exportBtn.addEventListener('click', () => {
  const csv = [CSV_HEADER, ...allResults.map(metricsToCSVRow)].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `benchmark_results_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});
