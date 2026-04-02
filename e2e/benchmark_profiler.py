"""
Automated benchmark runner with CPU and GPU profiling.

Uses Python Playwright to drive the browser, psutil for per-process
CPU/memory sampling, and GPUtil/nvidia-smi for GPU utilization.

Samples are taken in a background thread at ~100ms intervals and
correlated with benchmark phases.

Usage:
    python e2e/benchmark_profiler.py

Requires:
    pip install playwright psutil gputil
    python -m playwright install chromium
    npm run dev  (Vite dev server must be running on port 5173)
"""

import json
import csv
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import psutil
from playwright.sync_api import sync_playwright, Page, CDPSession

# Try GPU monitoring — falls back gracefully if not available
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
ITERATIONS = 30
WARMUP = 5

MATRIX = [
    ("tfjs", "wasm"),
    ("tfjs", "wasm-simd"),
    ("tfjs", "wasm-threads"),
    ("tfjs", "wasm-simd-threads"),
    ("tfjs", "webgl"),
    ("tfjs", "webgpu"),
    ("onnx", "wasm"),
    ("onnx", "wasm-simd"),
    ("onnx", "wasm-threads"),
    ("onnx", "wasm-simd-threads"),
    ("onnx", "webgl"),
    ("onnx", "webgpu"),
    ("onnx", "webnn"),
    ("litert", "wasm"),
    ("litert", "wasm-simd-threads"),
    ("litert", "webgpu"),
    ("transformersjs", "wasm"),
    ("transformersjs", "wasm-simd"),
    ("transformersjs", "wasm-threads"),
    ("transformersjs", "wasm-simd-threads"),
    ("transformersjs", "webgpu"),
    ("transformersjs", "webnn"),
    ("tflite-native", "cpu"),
    ("tflite-native", "gpu"),
]

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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    timestamp: float
    phase: str
    cpu_percent: float          # per-process CPU% (can exceed 100 on multi-core)
    memory_rss_mb: float        # resident set size
    memory_private_mb: float    # private working set
    gpu_util_percent: float     # GPU compute utilization (0-100)
    gpu_memory_used_mb: float   # GPU VRAM used


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


@dataclass
class BenchmarkResult:
    framework: str
    backend: str
    phases: dict = field(default_factory=dict)
    framework_init_ms: float = 0
    model_load_ms: float = 0
    avg_inference_ms: float = 0
    min_inference_ms: float = 0
    max_inference_ms: float = 0
    p95_inference_ms: float = 0
    predictions: list = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Resource sampler — runs in a background thread
# ---------------------------------------------------------------------------

class ResourceSampler:
    def __init__(self, pids: list[int]):
        self.pids = pids
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
        # Prime cpu_percent (first call always returns 0)
        for p in self.processes:
            try:
                p.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def set_phase(self, phase: str):
        self.current_phase = phase
        # Take an immediate sample so short phases get at least one data point
        sample = self._take_sample()
        if sample:
            self.samples.append(sample)

    def _sample_loop(self):
        while not self._stop.is_set():
            sample = self._take_sample()
            if sample:
                self.samples.append(sample)
            self._stop.wait(SAMPLE_INTERVAL_MS / 1000)

    def _take_sample(self) -> Optional[Sample]:
        total_cpu = 0.0
        total_rss = 0.0
        total_private = 0.0

        for p in self.processes:
            try:
                cpu = p.cpu_percent()
                mem = p.memory_info()
                total_cpu += cpu
                total_rss += mem.rss / (1024 * 1024)
                # On Windows, 'private' is available via memory_full_info
                try:
                    full_mem = p.memory_full_info()
                    total_private += full_mem.private / (1024 * 1024)
                except (psutil.AccessDenied, AttributeError):
                    total_private += mem.rss / (1024 * 1024)
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
            memory_private_mb=round(total_private, 2),
            gpu_util_percent=round(gpu_util, 1),
            gpu_memory_used_mb=round(gpu_mem, 2),
        )

    def get_phase_metrics(self, phase_name: str) -> PhaseMetrics:
        phase_samples = [s for s in self.samples if s.phase == phase_name]
        if not phase_samples:
            return PhaseMetrics(name=phase_name)

        cpu_values = [s.cpu_percent for s in phase_samples]
        gpu_values = [s.gpu_util_percent for s in phase_samples]

        return PhaseMetrics(
            name=phase_name,
            wall_time_ms=round(
                (phase_samples[-1].timestamp - phase_samples[0].timestamp) * 1000, 2
            ),
            avg_cpu_percent=round(sum(cpu_values) / len(cpu_values), 1),
            peak_cpu_percent=round(max(cpu_values), 1),
            avg_gpu_percent=round(sum(gpu_values) / len(gpu_values), 1),
            peak_gpu_percent=round(max(gpu_values), 1),
            memory_rss_start_mb=phase_samples[0].memory_rss_mb,
            memory_rss_end_mb=phase_samples[-1].memory_rss_mb,
            memory_rss_delta_mb=round(
                phase_samples[-1].memory_rss_mb - phase_samples[0].memory_rss_mb, 2
            ),
            gpu_memory_start_mb=phase_samples[0].gpu_memory_used_mb,
            gpu_memory_end_mb=phase_samples[-1].gpu_memory_used_mb,
            gpu_memory_delta_mb=round(
                phase_samples[-1].gpu_memory_used_mb - phase_samples[0].gpu_memory_used_mb, 2
            ),
            sample_count=len(phase_samples),
        )


# ---------------------------------------------------------------------------
# Get Chromium child PIDs (renderer, GPU process, etc.)
# ---------------------------------------------------------------------------

def get_browser_pids(browser) -> list[int]:
    """
    Find all Chromium process PIDs associated with a Playwright browser.
    Uses browser-level CDP to get process IDs, falls back to psutil scan.
    """
    pids: list[int] = []

    # Approach 1: Use browser-level CDP to get process info
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

    # Approach 2: Find chromium processes by name via psutil
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
# Run a single benchmark combo
# ---------------------------------------------------------------------------

def run_benchmark(framework: str, backend: str) -> BenchmarkResult:
    result = BenchmarkResult(framework=framework, backend=backend)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"D:\chr-build\chromium\src\out\Release\chrome.exe",
            headless=False,
            args=CHROME_ARGS,
        )
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(30000)

        # Get all browser PIDs for monitoring
        pids = get_browser_pids(browser)
        print(f"    Monitoring PIDs: {pids}")

        sampler = ResourceSampler(pids)

        try:
            page.goto(BASE_URL, wait_until="load", timeout=30000)
            page.wait_for_function(
                "() => window.__benchmark !== undefined",
                timeout=15000,
            )

            # Generate noise and configure
            page.evaluate(
                """([fw, be, iter, warm]) => {
                    const b = window.__benchmark;
                    b.generateNoise();
                    b.configure(fw, be, iter, warm);
                }""",
                [framework, backend, ITERATIONS, WARMUP],
            )

            # Start sampling
            sampler.start()

            # Phase 1: Framework init
            sampler.set_phase("framework_init")
            page.evaluate(
                """() => {
                    window.__benchPhaseResult = null;
                    const b = window.__benchmark;
                    const fw = document.getElementById('framework').value;
                    const be = document.getElementById('backend').value;
                    const { createBenchmark } = window.__benchmarkInternals || {};

                    // We run via the full run() which handles all phases
                    // but we need phase-level timing, so we rely on the
                    // benchmark's own timing plus our external sampling.
                }"""
            )

            # Run the full benchmark — phases are timed internally
            metrics = page.evaluate("() => window.__benchmark.run()")

            # Mark final phase
            sampler.set_phase("done")
            time.sleep(0.5)  # Let a few more samples collect
            sampler.stop()

            # Extract timing from benchmark
            result.framework_init_ms = metrics["frameworkInitMs"]
            result.model_load_ms = metrics["modelLoadMs"]
            result.avg_inference_ms = metrics["avgInferenceMs"]
            result.min_inference_ms = metrics["minInferenceMs"]
            result.max_inference_ms = metrics["maxInferenceMs"]
            result.p95_inference_ms = metrics["p95InferenceMs"]
            result.predictions = [
                {"label": pred["label"], "score": round(pred["score"] * 100, 2)}
                for pred in metrics.get("predictions", [])
            ]

            # Since __benchmark.run() runs all phases sequentially,
            # we use the sampling timestamps to approximate phase boundaries.
            # The sampler was running during the entire run() call with
            # phase="framework_init". We'll report the overall resource usage.
            overall = sampler.get_phase_metrics("framework_init")
            result.phases = {
                "overall": asdict(overall),
            }

        except Exception as e:
            sampler.stop()
            result.error = str(e)

        finally:
            browser.close()

    return result


# ---------------------------------------------------------------------------
# Run with per-phase sampling
# ---------------------------------------------------------------------------

def run_benchmark_phased(framework: str, backend: str) -> BenchmarkResult:
    """
    Runs the benchmark with separate phase sampling by calling each
    phase individually via page.evaluate.
    """
    result = BenchmarkResult(framework=framework, backend=backend)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"D:\chr-build\chromium\src\out\Release\chrome.exe",
            headless=False,
            args=CHROME_ARGS,
        )
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(30000)

        pids = get_browser_pids(browser)
        print(f"    Monitoring PIDs: {pids}")

        sampler = ResourceSampler(pids)

        try:
            page.goto(BASE_URL, wait_until="load", timeout=30000)
            page.wait_for_function(
                "() => window.__benchmark !== undefined",
                timeout=15000,
            )

            # Configure
            page.evaluate(
                """([fw, be, iter, warm]) => {
                    const b = window.__benchmark;
                    b.generateNoise();
                    b.configure(fw, be, iter, warm);
                }""",
                [framework, backend, ITERATIONS, WARMUP],
            )

            sampler.start()

            # --- Phase 1: Framework Init ---
            sampler.set_phase("framework_init")
            init_ms = page.evaluate(
                """async () => {
                    const { createBenchmark } = await import('/src/benchmarks/index.ts');
                    const fw = document.getElementById('framework').value;
                    const be = document.getElementById('backend').value;
                    window.__currentBenchmark = createBenchmark(fw);
                    const start = performance.now();
                    await window.__currentBenchmark.initFramework(be);
                    return performance.now() - start;
                }""",
            )
            result.framework_init_ms = round(init_ms, 2)

            # Prefetch model (not timed — download time varies)
            page.evaluate(
                "async () => await window.__currentBenchmark.prefetchModel()",
            )

            # --- Phase 2: Model Load (compilation only) ---
            sampler.set_phase("model_load")
            load_ms = page.evaluate(
                """async () => {
                    const start = performance.now();
                    await window.__currentBenchmark.loadModel();
                    return performance.now() - start;
                }""",
            )
            result.model_load_ms = round(load_ms, 2)

            # Set image
            page.evaluate(
                """() => {
                    const img = window.__benchmark.__getCurrentImage();
                    window.__currentBenchmark.setImage(img);
                }"""
            )

            # --- Phase 3: Warmup ---
            sampler.set_phase("warmup")
            page.evaluate(
                f"""async () => {{
                    for (let i = 0; i < {WARMUP}; i++) {{
                        await window.__currentBenchmark.runInference();
                    }}
                }}""",
            )

            # --- Phase 4: Inference ---
            sampler.set_phase("inference")
            inference_result = page.evaluate(
                f"""async () => {{
                    const times = [];
                    for (let i = 0; i < {ITERATIONS}; i++) {{
                        const t = await window.__currentBenchmark.runInference();
                        times.push(t);
                    }}
                    return times;
                }}""",
            )
            times = sorted(inference_result)
            result.avg_inference_ms = round(sum(times) / len(times), 2)
            result.min_inference_ms = round(times[0], 2)
            result.max_inference_ms = round(times[-1], 2)
            p95_idx = max(0, int(len(times) * 0.95) - 1)
            result.p95_inference_ms = round(times[p95_idx], 2)

            # --- Phase 5: Classification ---
            sampler.set_phase("classification")
            preds = page.evaluate(
                """async () => {
                    const results = await window.__currentBenchmark.classify(5);
                    return results;
                }""",
            )
            result.predictions = [
                {"label": pred["label"], "score": round(pred["score"] * 100, 2)}
                for pred in preds
            ]

            # Cleanup
            sampler.set_phase("cleanup")
            page.evaluate("async () => await window.__currentBenchmark.dispose()")

            sampler.stop()

            # Collect per-phase metrics
            result.phases = {
                phase: asdict(sampler.get_phase_metrics(phase))
                for phase in [
                    "framework_init", "model_load", "warmup",
                    "inference", "classification", "cleanup",
                ]
            }

        except Exception as e:
            sampler.stop()
            result.error = str(e)
            import traceback
            traceback.print_exc()

        finally:
            browser.close()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"GPU monitoring available: {HAS_GPU_MONITOR}")
    if HAS_GPU_MONITOR:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"  GPU: {gpu.name}, {gpu.memoryTotal}MB VRAM")
    print()

    results: list[BenchmarkResult] = []

    for framework, backend in MATRIX:
        combo = f"{framework}:{backend}"
        print(f"  RUN  {combo}")

        try:
            result = run_benchmark_phased(framework, backend)
            results.append(result)

            init_phase = result.phases.get("framework_init", {})
            inference_phase = result.phases.get("inference", {})
            print(
                f"  DONE {combo} — "
                f"init: {result.framework_init_ms}ms, "
                f"load: {result.model_load_ms}ms, "
                f"avg_inf: {result.avg_inference_ms}ms, "
                f"cpu_avg: {inference_phase.get('avg_cpu_percent', 0)}%, "
                f"gpu_avg: {inference_phase.get('avg_gpu_percent', 0)}%"
            )
        except Exception as e:
            print(f"  FAIL {combo} — {e}")
            results.append(BenchmarkResult(
                framework=framework, backend=backend, error=str(e)
            ))

    # --- Save results ---
    output_dir = Path(__file__).parent.parent

    # JSON
    json_path = output_dir / "benchmark_results.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu_available": HAS_GPU_MONITOR,
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\nJSON saved to {json_path}")

    # CSV
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Framework", "Backend",
            "FrameworkInit(ms)", "ModelLoad(ms)",
            "AvgInference(ms)", "MinInference(ms)", "MaxInference(ms)", "P95Inference(ms)",
            # Framework init phase
            "Init_AvgCPU(%)", "Init_PeakCPU(%)",
            "Init_AvgGPU(%)", "Init_PeakGPU(%)",
            # Model load phase
            "Load_AvgCPU(%)", "Load_PeakCPU(%)",
            "Load_MemRSS_Start(MB)", "Load_MemRSS_End(MB)", "Load_MemDelta(MB)",
            # Inference phase
            "Inf_AvgCPU(%)", "Inf_PeakCPU(%)",
            "Inf_AvgGPU(%)", "Inf_PeakGPU(%)",
            "Inf_MemRSS_Start(MB)", "Inf_MemRSS_End(MB)", "Inf_MemDelta(MB)",
            "Inf_GPU_Mem_Start(MB)", "Inf_GPU_Mem_End(MB)", "Inf_GPU_Mem_Delta(MB)",
            # Predictions
            "Top1Prediction", "Top1Score(%)",
            "Error",
        ])

        for r in results:
            init_p = r.phases.get("framework_init", {})
            load_p = r.phases.get("model_load", {})
            inf_p = r.phases.get("inference", {})
            top_pred = r.predictions[0] if r.predictions else {}

            writer.writerow([
                r.framework, r.backend,
                r.framework_init_ms, r.model_load_ms,
                r.avg_inference_ms, r.min_inference_ms, r.max_inference_ms, r.p95_inference_ms,
                init_p.get("avg_cpu_percent", ""),
                init_p.get("peak_cpu_percent", ""),
                init_p.get("avg_gpu_percent", ""),
                init_p.get("peak_gpu_percent", ""),
                load_p.get("avg_cpu_percent", ""),
                load_p.get("peak_cpu_percent", ""),
                load_p.get("memory_rss_start_mb", ""),
                load_p.get("memory_rss_end_mb", ""),
                load_p.get("memory_rss_delta_mb", ""),
                inf_p.get("avg_cpu_percent", ""),
                inf_p.get("peak_cpu_percent", ""),
                inf_p.get("avg_gpu_percent", ""),
                inf_p.get("peak_gpu_percent", ""),
                inf_p.get("memory_rss_start_mb", ""),
                inf_p.get("memory_rss_end_mb", ""),
                inf_p.get("memory_rss_delta_mb", ""),
                inf_p.get("gpu_memory_start_mb", ""),
                inf_p.get("gpu_memory_end_mb", ""),
                inf_p.get("gpu_memory_delta_mb", ""),
                top_pred.get("label", "N/A"),
                top_pred.get("score", "N/A"),
                r.error or "",
            ])

    print(f"CSV saved to {csv_path}")

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Framework':<18} {'Backend':<8} {'Init':>8} {'Load':>8} {'Avg Inf':>8} "
          f"{'CPU%':>6} {'GPU%':>6} {'RSS(MB)':>10} {'GPU Mem':>10} {'Top-1'}")
    print("-" * 120)
    for r in results:
        if r.error:
            print(f"{r.framework:<18} {r.backend:<8} ERROR: {r.error[:60]}")
            continue
        inf = r.phases.get("inference", {})
        pred = r.predictions[0]["label"].split(",")[0] if r.predictions else "N/A"
        score = f"{r.predictions[0]['score']}%" if r.predictions else ""
        print(
            f"{r.framework:<18} {r.backend:<8} "
            f"{r.framework_init_ms:>7.1f} {r.model_load_ms:>7.1f} {r.avg_inference_ms:>7.1f} "
            f"{inf.get('avg_cpu_percent', 0):>5.1f} {inf.get('avg_gpu_percent', 0):>5.1f} "
            f"{inf.get('memory_rss_end_mb', 0):>9.1f} "
            f"{inf.get('gpu_memory_end_mb', 0):>9.1f} "
            f"{pred} ({score})"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()
