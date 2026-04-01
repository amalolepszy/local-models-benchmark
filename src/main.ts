import { createBenchmark, FRAMEWORK_BACKENDS, type FrameworkId, type BackendId } from './benchmarks';
import { computeStats, getMemoryUsageMB, round, metricsToCSVRow, CSV_HEADER, evaluateAccuracy, type BenchmarkMetrics } from './utils/metrics';
import { preprocessImage, generateDummyImage, type PreprocessedImage } from './utils/image-input';

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
const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
const expectedClassInput = document.getElementById('expected-class') as HTMLInputElement;
const imagePreview = document.getElementById('image-preview') as HTMLElement;
const predictionResults = document.getElementById('prediction-results') as HTMLElement;
const predictionList = document.getElementById('prediction-list') as HTMLOListElement;
const noiseBtn = document.getElementById('noise-btn') as HTMLButtonElement;
const modeSelect = document.getElementById('mode') as HTMLSelectElement;
const aggregateSection = document.getElementById('aggregate-results-section') as HTMLElement;
const sessionSection = document.getElementById('session-results-section') as HTMLElement;
const sessionResultsBody = document.getElementById('session-results-body') as HTMLTableSectionElement;

// --- State ---
const allResults: BenchmarkMetrics[] = [];
let currentImage: PreprocessedImage | null = null;

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

// --- Mode toggle ---
modeSelect.addEventListener('change', () => {
  const isSession = modeSelect.value === 'session';
  aggregateSection.hidden = isSession;
  sessionSection.hidden = !isSession;
});


// --- Image upload handling ---
imageUpload.addEventListener('change', async () => {
  const file = imageUpload.files?.[0];
  if (!file) return;

  currentImage = await preprocessImage(file);
  showImagePreview(currentImage.dataUrl, 'Uploaded image');
});

// --- Noise generation ---
noiseBtn.addEventListener('click', () => {
  currentImage = generateDummyImage();
  showImagePreview(currentImage.dataUrl, 'Random noise');
});

function showImagePreview(dataUrl: string, alt: string) {
  imagePreview.innerHTML = '';
  const img = document.createElement('img');
  img.src = dataUrl;
  img.alt = alt;
  imagePreview.appendChild(img);
}

// --- Progress helpers ---
function showProgress(phase: string, detail: string, pct: number) {
  progressSection.hidden = false;
  progressBar.style.width = `${Math.min(100, pct)}%`;
  progressText.textContent = `${phase}: ${detail}`;
  progressText.className = '';
}

function hideProgress() {
  progressSection.hidden = true;
  progressBar.style.width = '0%';
}

// --- Display predictions ---
function showPredictions(
  predictions: import('./benchmarks/types').ClassificationResult[],
  expectedClass: string | null,
) {
  predictionResults.hidden = false;
  predictionList.innerHTML = '';

  for (const pred of predictions) {
    const li = document.createElement('li');

    const labelSpan = document.createElement('span');
    labelSpan.className = 'pred-label';
    labelSpan.textContent = pred.label;

    const scoreSpan = document.createElement('span');
    scoreSpan.className = 'pred-score';
    scoreSpan.textContent = `${round(pred.score * 100)}%`;

    li.appendChild(labelSpan);
    li.appendChild(scoreSpan);

    if (expectedClass) {
      const { matchesExpectedClass } = await_import_metrics();
      const matches = matchesExpectedClass(pred, expectedClass);
      const matchSpan = document.createElement('span');
      matchSpan.className = matches ? 'pred-match' : 'pred-no-match';
      matchSpan.textContent = matches ? 'MATCH' : '';
      li.appendChild(matchSpan);
    }

    predictionList.appendChild(li);
  }
}

// Inline the match check to avoid async import in sync context
function checkMatch(label: string, expected: string): boolean {
  const lowerLabel = label.toLowerCase();
  const lowerExpected = expected.toLowerCase().trim();
  return lowerLabel.includes(lowerExpected) || lowerExpected.includes(lowerLabel.split(',')[0]!);
}

function await_import_metrics() {
  return { matchesExpectedClass: (pred: { label: string }, expected: string) => checkMatch(pred.label, expected) };
}

// --- Run benchmark ---
runBtn.addEventListener('click', async () => {
  if (modeSelect.value === 'session') {
    await runSessionMode();
  } else {
    await runAggregateMode();
  }
});

async function runAggregateMode() {
  const frameworkId = frameworkSelect.value as FrameworkId;
  const backendId = backendSelect.value as BackendId;
  const iterations = parseInt(iterationsInput.value, 10) || 10;
  const warmup = parseInt(warmupInput.value, 10) || 3;
  const expectedClass = expectedClassInput.value.trim() || null;

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

    // Pre-fetch model (not timed — download time varies too much)
    showProgress('Model Prefetch', 'Downloading model...', 25);
    await benchmark.prefetchModel();

    // Phase 2: Model load (compilation/initialization only)
    showProgress('Model Load', 'Compiling model...', 35);
    const loadStart = performance.now();
    await benchmark.loadModel();
    const modelLoadMs = round(performance.now() - loadStart);

    showProgress('Model Load', `Done in ${modelLoadMs}ms`, 50);

    if (!currentImage) {
      throw new Error('No image set. Upload an image or click "Generate noise" first.');
    }
    benchmark.setImage(currentImage);

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
      const pct = 60 + (i / iterations) * 30;
      showProgress('Inference', `Iteration ${i + 1}/${iterations}`, pct);
      const elapsed = await benchmark.runInference();
      inferenceTimes.push(round(elapsed));
    }

    // Phase 5: Classification
    showProgress('Classification', 'Getting predictions...', 95);
    const predictions = await benchmark.classify(5);
    showPredictions(predictions, expectedClass);

    const memAfter = await getMemoryUsageMB();
    const stats = computeStats(inferenceTimes);
    const accuracy = evaluateAccuracy(predictions, expectedClass);
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
      predictions,
      expectedClass,
      ...accuracy,
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
}

async function runSessionMode() {
  const frameworkId = frameworkSelect.value as FrameworkId;
  const backendId = backendSelect.value as BackendId;
  const sessions = parseInt(iterationsInput.value, 10) || 10;

  runBtn.disabled = true;

  // Clear previous session results
  const placeholder = sessionResultsBody.querySelector('.placeholder-row');
  if (placeholder) placeholder.remove();

  if (!currentImage) {
    showProgress('Error', 'No image set. Upload an image or click "Generate noise" first.', 100);
    progressText.classList.add('status-error');
    runBtn.disabled = false;
    return;
  }

  let prevMemory = await getMemoryUsageMB();

  for (let i = 0; i < sessions; i++) {
    const pct = ((i + 1) / sessions) * 100;
    showProgress('Session', `${i + 1}/${sessions}`, pct);

    try {
      const benchmark = createBenchmark(frameworkId);

      // Init
      const initStart = performance.now();
      await benchmark.initFramework(backendId);
      const initMs = round(performance.now() - initStart);

      // Prefetch (not timed)
      await benchmark.prefetchModel();

      // Model load
      const loadStart = performance.now();
      await benchmark.loadModel();
      const loadMs = round(performance.now() - loadStart);

      // Set image and run single inference
      benchmark.setImage(currentImage);
      const inferenceMs = round(await benchmark.runInference());

      // Memory
      const memNow = await getMemoryUsageMB();
      const memDelta = prevMemory != null && memNow != null ? round(memNow - prevMemory) : null;

      // Add row
      addSessionRow(i + 1, benchmark.name, backendId.toUpperCase(), initMs, loadMs, inferenceMs, memNow, memDelta);

      prevMemory = memNow;

      await benchmark.dispose();
    } catch (err: any) {
      addSessionRow(i + 1, frameworkId, backendId.toUpperCase(), 0, 0, 0, null, null, err.message);
    }
  }

  showProgress('Done', `${sessions} sessions complete!`, 100);
  setTimeout(hideProgress, 2000);

  runBtn.disabled = false;
  exportBtn.disabled = false;
}

function addSessionRow(
  num: number, framework: string, backend: string,
  initMs: number, loadMs: number, inferenceMs: number,
  memMB: number | null, memDeltaMB: number | null,
  error?: string,
) {
  const tr = document.createElement('tr');
  if (error) {
    tr.innerHTML = `<td>${num}</td><td>${framework}</td><td>${backend}</td><td colspan="6" class="status-error">${error}</td>`;
  } else {
    tr.innerHTML = `
      <td>${num}</td>
      <td>${framework}</td>
      <td>${backend}</td>
      <td>${initMs}</td>
      <td>${loadMs}</td>
      <td>${inferenceMs}</td>
      <td>${round(initMs + loadMs + inferenceMs)}</td>
      <td>${memMB ?? 'N/A'}</td>
      <td>${memDeltaMB ?? 'N/A'}</td>
    `;
  }
  sessionResultsBody.appendChild(tr);
}

// --- Results table ---
function addResultRow(m: BenchmarkMetrics) {
  const placeholder = resultsBody.querySelector('.placeholder-row');
  if (placeholder) placeholder.remove();

  const topPred = m.predictions[0];
  const topLabel = topPred ? `${topPred.label.split(',')[0]} (${round(topPred.score * 100)}%)` : 'N/A';

  const accuracyBadge = (value: boolean | null) => {
    if (value === null) return '<span class="accuracy-badge na">N/A</span>';
    return value
      ? '<span class="accuracy-badge correct">YES</span>'
      : '<span class="accuracy-badge incorrect">NO</span>';
  };

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
    <td>${round(m.frameworkInitMs + m.modelLoadMs + m.avgInferenceMs)}</td>
    <td>${m.memoryBeforeMB ?? 'N/A'}</td>
    <td>${m.memoryAfterMB ?? 'N/A'}</td>
    <td>${m.memoryDeltaMB ?? 'N/A'}</td>
    <td>${topLabel}</td>
    <td>${accuracyBadge(m.top1Correct)}</td>
    <td>${accuracyBadge(m.top5Correct)}</td>
  `;
  resultsBody.appendChild(tr);
}

// --- Programmatic API for Playwright automation ---
// Exposes benchmark operations so Playwright can orchestrate runs
// and take CDP memory measurements between phases.
(window as any).__benchmark = {
  /** Set framework and backend dropdowns */
  configure(frameworkId: string, backendId: string, iterations = 10, warmup = 3) {
    frameworkSelect.value = frameworkId;
    updateBackendOptions();
    backendSelect.value = backendId;
    iterationsInput.value = String(iterations);
    warmupInput.value = String(warmup);
  },

  /** Generate noise image (same as clicking the button) */
  generateNoise() {
    currentImage = generateDummyImage();
    showImagePreview(currentImage.dataUrl, 'Random noise');
  },

  /** Set expected class */
  setExpectedClass(cls: string) {
    expectedClassInput.value = cls;
  },

  /** Run benchmark and return full metrics (resolves when done) */
  async run(): Promise<BenchmarkMetrics> {
    const frameworkId = frameworkSelect.value as FrameworkId;
    const backendId = backendSelect.value as BackendId;
    const iterations = parseInt(iterationsInput.value, 10) || 10;
    const warmup = parseInt(warmupInput.value, 10) || 3;
    const expectedClass = expectedClassInput.value.trim() || null;

    const benchmark = createBenchmark(frameworkId);

    // Emit phase events for CDP measurement timing
    window.dispatchEvent(new CustomEvent('benchmark:phase', { detail: 'framework-init:start' }));
    const initStart = performance.now();
    await benchmark.initFramework(backendId);
    const frameworkInitMs = round(performance.now() - initStart);
    window.dispatchEvent(new CustomEvent('benchmark:phase', { detail: 'framework-init:end' }));

    // Pre-fetch model (not timed)
    await benchmark.prefetchModel();

    window.dispatchEvent(new CustomEvent('benchmark:phase', { detail: 'model-load:start' }));
    const loadStart = performance.now();
    await benchmark.loadModel();
    const modelLoadMs = round(performance.now() - loadStart);
    window.dispatchEvent(new CustomEvent('benchmark:phase', { detail: 'model-load:end' }));

    if (!currentImage) {
      throw new Error('No image set. Call __benchmark.generateNoise() first.');
    }
    benchmark.setImage(currentImage);

    // Warmup
    for (let i = 0; i < warmup; i++) {
      await benchmark.runInference();
    }

    // Inference
    window.dispatchEvent(new CustomEvent('benchmark:phase', { detail: 'inference:start' }));
    const inferenceTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const elapsed = await benchmark.runInference();
      inferenceTimes.push(round(elapsed));
    }
    window.dispatchEvent(new CustomEvent('benchmark:phase', { detail: 'inference:end' }));

    const predictions = await benchmark.classify(5);
    const stats = computeStats(inferenceTimes);
    const accuracy = evaluateAccuracy(predictions, expectedClass);
    await benchmark.dispose();

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
      memoryBeforeMB: null,
      memoryAfterMB: null,
      memoryDeltaMB: null,
      predictions,
      expectedClass,
      ...accuracy,
    };

    allResults.push(metrics);
    addResultRow(metrics);

    return metrics;
  },

  /** Get all results collected so far */
  getResults(): BenchmarkMetrics[] {
    return allResults;
  },

  /** Get the current preprocessed image (for phased Playwright automation) */
  __getCurrentImage() {
    return currentImage;
  },

  /** Get the framework-backend matrix */
  getMatrix() {
    return FRAMEWORK_BACKENDS;
  },
};

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
