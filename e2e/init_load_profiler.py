"""
Init + Model Load cold-start profiler.

For each (framework, backend) combo in the matrix, runs N fully-cold
iterations. Every iteration launches a *fresh* Chromium process, runs
exactly one inference (no warmup), and closes the browser before the
next iteration. This isolates init / model-load latency from any
cross-iteration caching the engine might do in the same process.

Per iteration we record:
    - framework_init_ms      (timed inside the page)
    - model_load_ms          (timed inside the page)
    - inference_ms           (one inference, kept just so the pipeline
                              is exercised end-to-end)
    - phase-resource metrics for init and load (CPU/GPU/RAM/VRAM)

Per (framework, backend) we summarise framework_init_ms and
model_load_ms with mean / min / max / p95 / stdev, plus averages of the
resource metrics over the successful runs.

Usage:
    npm run dev
    python e2e/init_load_profiler.py [--task image|text]
                                     [--iterations 30]
                                     [--combos fw:be,fw:be]

Requires:
    pip install playwright psutil gputil
"""

import argparse
import csv
import json
import math
import statistics
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import psutil
from playwright.sync_api import sync_playwright

try:
    import GPUtil
    HAS_GPU_MONITOR = True
except ImportError:
    HAS_GPU_MONITOR = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:5173"
SAMPLE_INTERVAL_MS = 50
DEFAULT_ITERATIONS = 30
TEXT_INPUT = "This movie was absolutely wonderful and I loved every moment of it."
CHROME_PATH = r"D:\chr-build\chromium\src\out\Release\chrome.exe"

CHROME_ARGS = [
    "--enable-precise-memory-info",
    "--js-flags=--expose-gc",
    "--enable-features="
    "WebMachineLearningNeuralNetwork,"
    "ExperimentalWebMachineLearningNeuralNetwork,"
    "WebNNDirectML,"
    "WebNNOnnxRuntime,"
    "ExperimentalWebAssemblyFeatures,"
    "ExperimentalWebAssemblySharedEverything,"
    "ExperimentalWebAssemblyStackSwitching,"
    "WebAssemblyBaseline,"
    "WebAssemblyLazyCompilation,"
    "WebAssemblyTiering",
    "--enable-unsafe-webgpu",
    "--enable-blink-features=TFLiteNativeInference",
]

IMAGE_MATRIX = [
    ("tfjs", "wasm-simd-threads"),
    ("tfjs", "webgl"),
    ("tfjs", "webgpu"),
    ("onnx", "wasm-simd-threads"),
    ("onnx", "webgl"),
    ("onnx", "webgpu"),
    ("onnx", "webnn"),
    ("litert", "wasm-simd-threads"),
    ("litert", "webgpu"),
    ("transformersjs", "wasm-simd-threads"),
    ("transformersjs", "webgpu"),
    ("transformersjs", "webnn"),
    ("tflite-native", "cpu"),
    ("tflite-native", "gpu"),
]

TEXT_MATRIX = [
    ("tfjs", "wasm-simd-threads"),
    ("tfjs", "webgl"),
    ("tfjs", "webgpu"),
    ("onnx", "wasm-simd-threads"),
    ("onnx", "webgl"),
    ("onnx", "webgpu"),
    ("onnx", "webnn"),
    ("litert", "wasm-simd-threads"),
    ("litert", "webgpu"),
    ("transformersjs", "wasm-simd-threads"),
    ("transformersjs", "webgpu"),
    ("transformersjs", "webnn"),
    ("tflite-native", "cpu"),
    ("tflite-native", "gpu"),
]


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    timestamp: float
    phase: str
    cpu_percent: float
    memory_rss_mb: float
    gpu_util_percent: float
    gpu_memory_used_mb: float


@dataclass
class PhaseMetrics:
    name: str
    wall_time_ms: float = 0
    avg_cpu_percent: float = 0
    peak_cpu_percent: float = 0
    avg_gpu_percent: float = 0
    peak_gpu_percent: float = 0
    memory_rss_start_mb: float = 0
    memory_rss_end_mb: float = 0
    memory_rss_delta_mb: float = 0
    gpu_memory_start_mb: float = 0
    gpu_memory_end_mb: float = 0
    gpu_memory_delta_mb: float = 0
    sample_count: int = 0


class ResourceSampler:
    def __init__(self, pids: list[int]):
        self.processes: list[psutil.Process] = []
        for pid in pids:
            try:
                self.processes.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                pass
        self.samples: list[Sample] = []
        self.current_phase = "idle"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        for p in self.processes:
            try:
                p.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def set_phase(self, phase: str):
        self.current_phase = phase
        s = self._take_sample()
        if s:
            self.samples.append(s)

    def _loop(self):
        while not self._stop.is_set():
            s = self._take_sample()
            if s:
                self.samples.append(s)
            self._stop.wait(SAMPLE_INTERVAL_MS / 1000)

    def _take_sample(self) -> Optional[Sample]:
        total_cpu = 0.0
        total_rss = 0.0
        for p in self.processes:
            try:
                total_cpu += p.cpu_percent()
                total_rss += p.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        gpu_util = 0.0
        gpu_mem = 0.0
        if HAS_GPU_MONITOR:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util = gpus[0].load * 100
                    gpu_mem = gpus[0].memoryUsed
            except Exception:
                pass

        return Sample(
            timestamp=time.time(),
            phase=self.current_phase,
            cpu_percent=round(total_cpu, 1),
            memory_rss_mb=round(total_rss, 2),
            gpu_util_percent=round(gpu_util, 1),
            gpu_memory_used_mb=round(gpu_mem, 2),
        )

    def get_phase_metrics(self, phase: str) -> PhaseMetrics:
        phase_samples = [s for s in self.samples if s.phase == phase]
        if not phase_samples:
            return PhaseMetrics(name=phase)
        cpu = [s.cpu_percent for s in phase_samples]
        gpu = [s.gpu_util_percent for s in phase_samples]
        return PhaseMetrics(
            name=phase,
            wall_time_ms=round(
                (phase_samples[-1].timestamp - phase_samples[0].timestamp) * 1000, 2
            ),
            avg_cpu_percent=round(sum(cpu) / len(cpu), 1),
            peak_cpu_percent=round(max(cpu), 1),
            avg_gpu_percent=round(sum(gpu) / len(gpu), 1),
            peak_gpu_percent=round(max(gpu), 1),
            memory_rss_start_mb=phase_samples[0].memory_rss_mb,
            memory_rss_end_mb=phase_samples[-1].memory_rss_mb,
            memory_rss_delta_mb=round(
                phase_samples[-1].memory_rss_mb - phase_samples[0].memory_rss_mb, 2
            ),
            gpu_memory_start_mb=phase_samples[0].gpu_memory_used_mb,
            gpu_memory_end_mb=phase_samples[-1].gpu_memory_used_mb,
            gpu_memory_delta_mb=round(
                phase_samples[-1].gpu_memory_used_mb
                - phase_samples[0].gpu_memory_used_mb,
                2,
            ),
            sample_count=len(phase_samples),
        )


def get_browser_pids(browser) -> list[int]:
    pids: list[int] = []
    try:
        cdp = browser.new_browser_cdp_session()
        info = cdp.send("SystemInfo.getProcessInfo")
        for proc in info.get("processInfo", []):
            pids.append(proc["id"])
        cdp.detach()
        if pids:
            return pids
    except Exception:
        pass

    chrome_names = {"chrome.exe", "chromium.exe", "chrome-headless-shell.exe"}
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info["name"] or "").lower()
            if any(cn in name for cn in chrome_names):
                pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    framework: str
    backend: str
    task: str
    iteration: int
    framework_init_ms: float = 0
    model_load_ms: float = 0
    inference_ms: float = 0
    total_ms: float = 0  # framework_init + model_load + inference for this iteration
    init_phase: dict = field(default_factory=dict)
    load_phase: dict = field(default_factory=dict)
    inf_phase: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ComboSummary:
    framework: str
    backend: str
    task: str
    successful_runs: int
    failed_runs: int

    # Latency stats per phase (across the N iterations)
    init_mean_ms: float = 0
    init_min_ms: float = 0
    init_max_ms: float = 0
    init_p95_ms: float = 0
    init_stdev_ms: float = 0
    load_mean_ms: float = 0
    load_min_ms: float = 0
    load_max_ms: float = 0
    load_p95_ms: float = 0
    load_stdev_ms: float = 0
    inf_mean_ms: float = 0
    inf_min_ms: float = 0
    inf_max_ms: float = 0
    inf_p95_ms: float = 0
    inf_stdev_ms: float = 0
    total_mean_ms: float = 0
    total_min_ms: float = 0
    total_max_ms: float = 0
    total_p95_ms: float = 0
    total_stdev_ms: float = 0

    # Resource metrics per phase: mean & stdev across iterations.
    # Naming: <phase>_<metric>_<mean|stdev>
    # phase ∈ {init, load, inf}; metric ∈ {avg_cpu, peak_cpu, avg_gpu, peak_gpu, mem_delta, gpu_mem_delta}
    init_avg_cpu_mean: float = 0
    init_avg_cpu_stdev: float = 0
    init_peak_cpu_mean: float = 0
    init_peak_cpu_stdev: float = 0
    init_avg_gpu_mean: float = 0
    init_avg_gpu_stdev: float = 0
    init_peak_gpu_mean: float = 0
    init_peak_gpu_stdev: float = 0
    init_mem_delta_mean: float = 0
    init_mem_delta_stdev: float = 0
    init_gpu_mem_delta_mean: float = 0
    init_gpu_mem_delta_stdev: float = 0

    load_avg_cpu_mean: float = 0
    load_avg_cpu_stdev: float = 0
    load_peak_cpu_mean: float = 0
    load_peak_cpu_stdev: float = 0
    load_avg_gpu_mean: float = 0
    load_avg_gpu_stdev: float = 0
    load_peak_gpu_mean: float = 0
    load_peak_gpu_stdev: float = 0
    load_mem_delta_mean: float = 0
    load_mem_delta_stdev: float = 0
    load_gpu_mem_delta_mean: float = 0
    load_gpu_mem_delta_stdev: float = 0

    inf_avg_cpu_mean: float = 0
    inf_avg_cpu_stdev: float = 0
    inf_peak_cpu_mean: float = 0
    inf_peak_cpu_stdev: float = 0
    inf_avg_gpu_mean: float = 0
    inf_avg_gpu_stdev: float = 0
    inf_peak_gpu_mean: float = 0
    inf_peak_gpu_stdev: float = 0
    inf_mem_delta_mean: float = 0
    inf_mem_delta_stdev: float = 0
    inf_gpu_mem_delta_mean: float = 0
    inf_gpu_mem_delta_stdev: float = 0

    error_samples: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "p95": 0, "stdev": 0}
    sorted_v = sorted(values)
    p95_idx = max(0, math.ceil(len(sorted_v) * 0.95) - 1)
    return {
        "mean": round(sum(sorted_v) / len(sorted_v), 2),
        "min": round(sorted_v[0], 2),
        "max": round(sorted_v[-1], 2),
        "p95": round(sorted_v[p95_idx], 2),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def _phase_stats(
    runs: list[IterationResult], phase_attr: str, key: str
) -> tuple[float, float]:
    """Return (mean, sample_stdev) of `key` across iterations for the given phase."""
    vals = []
    for r in runs:
        phase = getattr(r, phase_attr) or {}
        v = phase.get(key)
        if v is not None:
            vals.append(v)
    if not vals:
        return 0.0, 0.0
    mean = round(sum(vals) / len(vals), 2)
    stdev = round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0
    return mean, stdev


# ---------------------------------------------------------------------------
# Single cold-start iteration
# ---------------------------------------------------------------------------

def run_iteration(framework: str, backend: str, task: str, iteration_num: int) -> IterationResult:
    """Launch a fresh browser, time init + model load, run one inference, close."""
    result = IterationResult(
        framework=framework, backend=backend, task=task, iteration=iteration_num
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=False,
            args=CHROME_ARGS,
        )
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(60000)

        pids = get_browser_pids(browser)
        sampler = ResourceSampler(pids)

        try:
            page.goto(BASE_URL, wait_until="load", timeout=30000)
            page.wait_for_function(
                "() => window.__benchmark !== undefined", timeout=15000
            )

            # Configure UI (1 iteration, 0 warmup — but we drive phases manually
            # below, so configure() is only used to set the selects).
            page.evaluate(
                """async ([fw, be, task, textInput]) => {
                    const b = window.__benchmark;
                    b.setTask(task);
                    b.configure(fw, be, 1, 0);
                    if (task === 'image-classification') {
                        await b.loadImage('/rocky.jpg');
                    } else {
                        b.setTextInput(textInput);
                    }
                }""",
                [framework, backend, task, TEXT_INPUT],
            )

            sampler.start()

            # --- framework_init ---
            sampler.set_phase("framework_init")
            init_ms = page.evaluate(
                """async ([task]) => {
                    const { createBenchmark } = await import('/src/benchmarks/index.ts');
                    const fw = document.getElementById('framework').value;
                    const be = document.getElementById('backend').value;
                    window.__currentBenchmark = createBenchmark(fw, task);
                    const t0 = performance.now();
                    await window.__currentBenchmark.initFramework(be);
                    return performance.now() - t0;
                }""",
                [task],
            )
            result.framework_init_ms = round(init_ms, 2)

            # Prefetch is not part of either timed phase (download is
            # network-bound and noisy). Tag samples taken during the
            # prefetch as a separate "prefetch" phase so neither
            # framework_init nor model_load metrics include it.
            sampler.set_phase("prefetch")
            page.evaluate(
                "async () => await window.__currentBenchmark.prefetchModel()"
            )

            # --- model_load ---
            sampler.set_phase("model_load")
            load_ms = page.evaluate(
                """async () => {
                    const t0 = performance.now();
                    await window.__currentBenchmark.loadModel();
                    return performance.now() - t0;
                }"""
            )
            result.model_load_ms = round(load_ms, 2)

            # --- single inference (untimed in stats, but recorded) ---
            sampler.set_phase("inference")
            page.evaluate(
                """async ([task]) => {
                    if (task === 'image-classification') {
                        const img = window.__benchmark.__getCurrentImage();
                        window.__currentBenchmark.setInput({ type: 'image', image: img });
                    } else {
                        const { getTokenizer } = await import('/src/utils/tokenizer.ts');
                        const tokenizer = await getTokenizer();
                        const text = document.getElementById('text-input').value;
                        const tokenized = tokenizer.tokenize(text);
                        window.__currentBenchmark.setInput({ type: 'text', text: tokenized, rawText: text });
                    }
                }""",
                [task],
            )
            inf_ms = page.evaluate(
                "async () => await window.__currentBenchmark.runInference()"
            )
            result.inference_ms = round(inf_ms, 2)
            result.total_ms = round(
                result.framework_init_ms + result.model_load_ms + result.inference_ms, 2
            )
            # Settling sleep before we flip the phase tag: gives the sampler
            # ~3 extra samples while the page is in post-inference state, so
            # short inferences (~10 ms image-cls) still get a meaningful
            # memory_rss_delta_mb instead of "start == end == 0".
            time.sleep(0.15)

            # --- cleanup ---
            sampler.set_phase("cleanup")
            page.evaluate("async () => await window.__currentBenchmark.dispose()")
            time.sleep(0.2)
            sampler.stop()

            result.init_phase = asdict(sampler.get_phase_metrics("framework_init"))
            result.load_phase = asdict(sampler.get_phase_metrics("model_load"))
            result.inf_phase = asdict(sampler.get_phase_metrics("inference"))

        except Exception as e:
            try:
                sampler.stop()
            except Exception:
                pass
            result.error = str(e)
            traceback.print_exc()

        finally:
            browser.close()

    return result


# ---------------------------------------------------------------------------
# Summarise a combo
# ---------------------------------------------------------------------------

_RESOURCE_KEYS = [
    ("avg_cpu", "avg_cpu_percent"),
    ("peak_cpu", "peak_cpu_percent"),
    ("avg_gpu", "avg_gpu_percent"),
    ("peak_gpu", "peak_gpu_percent"),
    ("mem_delta", "memory_rss_delta_mb"),
    ("gpu_mem_delta", "gpu_memory_delta_mb"),
]


def summarise(framework: str, backend: str, task: str,
              iterations: list[IterationResult]) -> ComboSummary:
    ok = [r for r in iterations if r.error is None]
    bad = [r for r in iterations if r.error is not None]

    init_s = _stats([r.framework_init_ms for r in ok])
    load_s = _stats([r.model_load_ms for r in ok])
    inf_s = _stats([r.inference_ms for r in ok])
    total_s = _stats([r.total_ms for r in ok])

    summary = ComboSummary(
        framework=framework, backend=backend, task=task,
        successful_runs=len(ok), failed_runs=len(bad),
        init_mean_ms=init_s["mean"], init_min_ms=init_s["min"],
        init_max_ms=init_s["max"], init_p95_ms=init_s["p95"],
        init_stdev_ms=init_s["stdev"],
        load_mean_ms=load_s["mean"], load_min_ms=load_s["min"],
        load_max_ms=load_s["max"], load_p95_ms=load_s["p95"],
        load_stdev_ms=load_s["stdev"],
        inf_mean_ms=inf_s["mean"], inf_min_ms=inf_s["min"],
        inf_max_ms=inf_s["max"], inf_p95_ms=inf_s["p95"],
        inf_stdev_ms=inf_s["stdev"],
        total_mean_ms=total_s["mean"], total_min_ms=total_s["min"],
        total_max_ms=total_s["max"], total_p95_ms=total_s["p95"],
        total_stdev_ms=total_s["stdev"],
        error_samples=[r.error for r in bad[:3]],
    )

    # Populate resource metrics (mean + stdev) for each phase.
    for phase_short, phase_attr in [
        ("init", "init_phase"),
        ("load", "load_phase"),
        ("inf", "inf_phase"),
    ]:
        for metric_short, metric_key in _RESOURCE_KEYS:
            mean, stdev = _phase_stats(ok, phase_attr, metric_key)
            setattr(summary, f"{phase_short}_{metric_short}_mean", mean)
            setattr(summary, f"{phase_short}_{metric_short}_stdev", stdev)

    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(per_iter: list[IterationResult],
                 summaries: list[ComboSummary],
                 task: str) -> None:
    out_dir = Path(__file__).parent.parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    suffix = "image" if task == "image-classification" else "text"

    iter_csv = out_dir / f"init_load_iterations_{suffix}.csv"
    with iter_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Framework", "Backend", "Task", "Iteration",
            "FrameworkInit(ms)", "ModelLoad(ms)", "Inference(ms)", "Total(ms)",
            "Init_AvgCPU%", "Init_PeakCPU%",
            "Init_AvgGPU%", "Init_PeakGPU%",
            "Init_MemRSSDelta(MB)", "Init_GPUMemDelta(MB)",
            "Init_WallTime(ms)",
            "Load_AvgCPU%", "Load_PeakCPU%",
            "Load_AvgGPU%", "Load_PeakGPU%",
            "Load_MemRSSDelta(MB)", "Load_GPUMemDelta(MB)",
            "Load_WallTime(ms)",
            "Inf_AvgCPU%", "Inf_PeakCPU%",
            "Inf_AvgGPU%", "Inf_PeakGPU%",
            "Inf_MemRSSDelta(MB)", "Inf_GPUMemDelta(MB)",
            "Inf_WallTime(ms)",
            "Error",
        ])
        for r in per_iter:
            ip = r.init_phase or {}
            lp = r.load_phase or {}
            fp = r.inf_phase or {}
            w.writerow([
                r.framework, r.backend, r.task, r.iteration,
                r.framework_init_ms, r.model_load_ms, r.inference_ms, r.total_ms,
                ip.get("avg_cpu_percent", ""), ip.get("peak_cpu_percent", ""),
                ip.get("avg_gpu_percent", ""), ip.get("peak_gpu_percent", ""),
                ip.get("memory_rss_delta_mb", ""), ip.get("gpu_memory_delta_mb", ""),
                ip.get("wall_time_ms", ""),
                lp.get("avg_cpu_percent", ""), lp.get("peak_cpu_percent", ""),
                lp.get("avg_gpu_percent", ""), lp.get("peak_gpu_percent", ""),
                lp.get("memory_rss_delta_mb", ""), lp.get("gpu_memory_delta_mb", ""),
                lp.get("wall_time_ms", ""),
                fp.get("avg_cpu_percent", ""), fp.get("peak_cpu_percent", ""),
                fp.get("avg_gpu_percent", ""), fp.get("peak_gpu_percent", ""),
                fp.get("memory_rss_delta_mb", ""), fp.get("gpu_memory_delta_mb", ""),
                fp.get("wall_time_ms", ""),
                r.error or "",
            ])

    sum_csv = out_dir / f"init_load_summary_{suffix}.csv"
    # Build resource-metric columns programmatically (3 phases × 6 metrics × 2 stats = 36 cols)
    res_headers: list[str] = []
    for phase_short, phase_label in [("init", "Init"), ("load", "Load"), ("inf", "Inf")]:
        for metric_short, label in [
            ("avg_cpu", "AvgCPU%"), ("peak_cpu", "PeakCPU%"),
            ("avg_gpu", "AvgGPU%"), ("peak_gpu", "PeakGPU%"),
            ("mem_delta", "MemDelta(MB)"), ("gpu_mem_delta", "GPUMemDelta(MB)"),
        ]:
            res_headers.append(f"{phase_label}_{label}_Mean")
            res_headers.append(f"{phase_label}_{label}_Stdev")

    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Framework", "Backend", "Task", "Successful", "Failed",
            "InitMean(ms)", "InitMin(ms)", "InitMax(ms)",
            "InitP95(ms)", "InitStdev(ms)",
            "LoadMean(ms)", "LoadMin(ms)", "LoadMax(ms)",
            "LoadP95(ms)", "LoadStdev(ms)",
            "InfMean(ms)", "InfMin(ms)", "InfMax(ms)",
            "InfP95(ms)", "InfStdev(ms)",
            "TotalMean(ms)", "TotalMin(ms)", "TotalMax(ms)",
            "TotalP95(ms)", "TotalStdev(ms)",
            *res_headers,
        ])
        for s in summaries:
            row = [
                s.framework, s.backend, s.task,
                s.successful_runs, s.failed_runs,
                s.init_mean_ms, s.init_min_ms, s.init_max_ms,
                s.init_p95_ms, s.init_stdev_ms,
                s.load_mean_ms, s.load_min_ms, s.load_max_ms,
                s.load_p95_ms, s.load_stdev_ms,
                s.inf_mean_ms, s.inf_min_ms, s.inf_max_ms,
                s.inf_p95_ms, s.inf_stdev_ms,
                s.total_mean_ms, s.total_min_ms, s.total_max_ms,
                s.total_p95_ms, s.total_stdev_ms,
            ]
            for phase_short in ("init", "load", "inf"):
                for metric_short, _ in _RESOURCE_KEYS:
                    row.append(getattr(s, f"{phase_short}_{metric_short}_mean"))
                    row.append(getattr(s, f"{phase_short}_{metric_short}_stdev"))
            w.writerow(row)

    json_path = out_dir / f"init_load_results_{suffix}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "iterations": [asdict(r) for r in per_iter],
                "summaries": [asdict(s) for s in summaries],
            },
            f,
            indent=2,
        )

    print(f"\nSaved:\n  {iter_csv}\n  {sum_csv}\n  {json_path}")


def print_summary_table(summaries: list[ComboSummary]) -> None:
    print(
        f"\n{'Framework':<16} {'Backend':<22} "
        f"{'InitMean':>9} {'InitStd':>8}  "
        f"{'LoadMean':>10} {'LoadStd':>8}  "
        f"{'InfMean':>9} {'InfStd':>8}  "
        f"{'TotalMean':>10} {'TotalStd':>9}  "
        f"{'InitMemΔ':>9} {'LoadMemΔ':>9} {'InfMemΔ':>9}  "
        f"{'OK':>4} {'Err':>4}"
    )
    print("-" * 165)
    for s in summaries:
        print(
            f"{s.framework:<16} {s.backend:<22} "
            f"{s.init_mean_ms:>9.1f} {s.init_stdev_ms:>8.1f}  "
            f"{s.load_mean_ms:>10.1f} {s.load_stdev_ms:>8.1f}  "
            f"{s.inf_mean_ms:>9.1f} {s.inf_stdev_ms:>8.1f}  "
            f"{s.total_mean_ms:>10.1f} {s.total_stdev_ms:>9.1f}  "
            f"{s.init_mem_delta_mean:>9.1f} {s.load_mem_delta_mean:>9.1f} {s.inf_mem_delta_mean:>9.1f}  "
            f"{s.successful_runs:>4} {s.failed_runs:>4}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cold-start profiler for framework init + model load."
    )
    parser.add_argument(
        "--task", choices=["image", "text"], default="image",
        help="image (classification) or text (sentiment)",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help="Cold-start iterations per (framework, backend) (default: 30)",
    )
    parser.add_argument(
        "--combos", default=None,
        help="Comma-separated fw:be pairs to restrict the run, e.g. 'tfjs:webgpu,onnx:wasm-simd-threads'",
    )
    args = parser.parse_args()

    task = "image-classification" if args.task == "image" else "text-classification"
    matrix = IMAGE_MATRIX if task == "image-classification" else TEXT_MATRIX

    if args.combos:
        wanted = set()
        for c in args.combos.split(","):
            fw, _, be = c.strip().partition(":")
            if fw and be:
                wanted.add((fw, be))
        matrix = [c for c in matrix if c in wanted]
        if not matrix:
            print(f"No combos in matrix matched --combos={args.combos!r}")
            return

    print(
        f"task={task}  iterations={args.iterations}  "
        f"combos={len(matrix)}  total_browser_launches={len(matrix) * args.iterations}"
    )

    per_iter: list[IterationResult] = []
    summaries: list[ComboSummary] = []

    for fw, be in matrix:
        print(f"\n=== {fw}:{be} ===")
        combo_runs: list[IterationResult] = []
        for i in range(1, args.iterations + 1):
            print(f"  [{i:>2}/{args.iterations}] launching browser...", end=" ", flush=True)
            r = run_iteration(fw, be, task, i)
            combo_runs.append(r)
            per_iter.append(r)
            if r.error:
                print(f"ERROR: {r.error[:100]}")
            else:
                print(
                    f"init={r.framework_init_ms:>8.1f}ms  "
                    f"load={r.model_load_ms:>9.1f}ms  "
                    f"inf={r.inference_ms:>7.1f}ms  "
                    f"total={r.total_ms:>9.1f}ms"
                )
        s = summarise(fw, be, task, combo_runs)
        summaries.append(s)
        print(
            f"  --> init mean={s.init_mean_ms} stdev={s.init_stdev_ms} | "
            f"load mean={s.load_mean_ms} stdev={s.load_stdev_ms} | "
            f"inf mean={s.inf_mean_ms} stdev={s.inf_stdev_ms} | "
            f"total mean={s.total_mean_ms} p95={s.total_p95_ms} stdev={s.total_stdev_ms} | "
            f"ΔRSS init/load/inf={s.init_mem_delta_mean}/{s.load_mem_delta_mean}/{s.inf_mem_delta_mean} MB | "
            f"ok={s.successful_runs} err={s.failed_runs}"
        )

    save_results(per_iter, summaries, task)
    print_summary_table(summaries)


if __name__ == "__main__":
    main()
