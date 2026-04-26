"""
Speedometer-under-inference-load runner (single-tab injection mode).

Speedometer 4 is served locally by Vite from `vendor/Speedometer/` at
`http://localhost:5173/speedometer/` (see vite.config.ts's
`serveSpeedometer()` plugin). Because Speedometer and the rest of the app
share an origin, we can inject an inference loop *into the Speedometer
page itself* using the same `src/benchmarks/*` modules as the regular
benchmark UI — no CORS, no CDN, no version drift, full multi-threaded
WASM (crossOriginIsolated holds because Vite sets COOP/COEP).

For each (framework, backend) in the matrix:
  1. Open http://localhost:5173/speedometer/
  2. Inject + boot the inference loop via page.evaluate(); wait until it
     is actually running.
  3. Click Speedometer's `.start-tests-button`.
  4. Wait for `#result-number` to render a numeric score.
  5. Stop the inference loop, read inferences-completed.
  6. Write a CSV row.

A no-inference baseline run is included by default for comparison.

Usage:
    npm run dev      (Vite must be running on :5173)
    python e2e/speedometer_under_load.py
        [--combos fw1:be1,fw2:be2,...]
        [--task image|text]
        [--no-baseline]
        [--speedometer-timeout 600]
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, Page

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPEEDOMETER_URL = "http://localhost:5173/speedometer/"
TEXT_INPUT = "This movie was absolutely wonderful and I loved every moment of it."
IMAGE_URL = "/rocky.jpg"

# Mirrors the matrices in benchmark_profiler.py — keep these in sync if a
# framework/backend pairing changes there.
IMAGE_MATRIX: list[tuple[str, str]] = [
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

TEXT_MATRIX: list[tuple[str, str]] = [
    ("tfjs", "wasm-simd-threads"),
    ("tfjs", "webgl"),
    ("tfjs", "webgpu"),
    ("onnx", "wasm-simd-threads"),
    ("onnx", "webgl"),
    ("onnx", "webgpu"),
    ("onnx", "webnn"),
    ("litert", "wasm-simd-threads"),
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

CHROME_EXECUTABLE = r"D:\chr-build\chromium\src\out\Release\chrome.exe"


# ---------------------------------------------------------------------------
# Injected JS — runs inside the Speedometer page
# ---------------------------------------------------------------------------

# Boots the inference loop, returns once it's loaded + the loop has at least
# kicked off its first iteration. After this resolves, window.__inferenceLoad
# carries the controller. The loop yields via setTimeout(0) every iteration so
# it never starves the macrotask queue (Speedometer's measurements depend on
# the renderer being responsive).
INFERENCE_BOOT_JS = """
async ([fwId, beId, taskId, imageUrl, textInput]) => {
    if (window.__inferenceLoad) {
        throw new Error('inferenceLoad already exists; previous run did not clean up');
    }
    const { createBenchmark } = await import('/src/benchmarks/index.ts');
    const benchmark = createBenchmark(fwId, taskId);
    await benchmark.initFramework(beId);
    await benchmark.prefetchModel();
    await benchmark.loadModel();

    let input;
    if (taskId === 'image-classification') {
        const { preprocessImage } = await import('/src/utils/image-input.ts');
        const image = await preprocessImage(imageUrl);
        input = { type: 'image', image };
    } else {
        const { getTokenizer } = await import('/src/utils/tokenizer.ts');
        const tokenizer = await getTokenizer();
        const tokenized = tokenizer.tokenize(textInput);
        input = { type: 'text', text: tokenized, rawText: textInput };
    }
    benchmark.setInput(input);

    const state = {
        benchmark,
        running: true,
        count: 0,
        startTime: performance.now(),
        error: null,
        loop: null,
    };
    state.loop = (async () => {
        while (state.running) {
            try {
                await benchmark.runInference();
                state.count++;
                // Yield to macrotasks so Speedometer's UI work runs.
                await new Promise(r => setTimeout(r, 0));
            } catch (err) {
                state.error = err && err.message ? err.message : String(err);
                state.running = false;
                break;
            }
        }
    })();
    window.__inferenceLoad = state;
}
"""

INFERENCE_STATUS_JS = """
() => {
    const s = window.__inferenceLoad;
    if (!s) return { running: false, count: 0, elapsedMs: 0, error: null };
    return {
        running: s.running,
        count: s.count,
        elapsedMs: performance.now() - s.startTime,
        error: s.error,
    };
}
"""

INFERENCE_STOP_JS = """
async () => {
    const s = window.__inferenceLoad;
    if (!s) return { count: 0, elapsedMs: 0, error: null };
    s.running = false;
    await s.loop;
    const elapsedMs = performance.now() - s.startTime;
    try { await s.benchmark.dispose(); } catch (e) { /* ignore */ }
    const result = { count: s.count, elapsedMs, error: s.error };
    delete window.__inferenceLoad;
    return result;
}
"""

# Speedometer 4 renders the final geomean score into #result-number.
SPEEDOMETER_SCORE_PREDICATE = """
() => {
    const el = document.getElementById('result-number');
    if (!el) return null;
    const text = (el.textContent || '').trim();
    const m = text.match(/-?\\d+(?:\\.\\d+)?/);
    return m ? parseFloat(m[0]) : null;
}
"""


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------

@dataclass
class SpeedometerRow:
    timestamp: str
    scenario: str          # "baseline" | "under_load"
    framework: str
    backend: str
    task: str
    speedometer_score: Optional[float]
    inferences_completed: int
    inference_loop_seconds: float
    inference_error: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------

def wait_for_speedometer_score(page: Page, timeout_seconds: int) -> float:
    page.wait_for_function(
        f"() => ({SPEEDOMETER_SCORE_PREDICATE.strip()})() !== null",
        timeout=timeout_seconds * 1000,
    )
    score = page.evaluate(SPEEDOMETER_SCORE_PREDICATE)
    if score is None:
        raise RuntimeError("Speedometer finished but #result-number had no number")
    return float(score)


def run_scenario(
    framework: Optional[str],
    backend: Optional[str],
    task: str,
    speedometer_timeout: int,
) -> SpeedometerRow:
    is_baseline = framework is None or backend is None
    scenario = "baseline" if is_baseline else "under_load"
    label = "baseline" if is_baseline else f"{framework}:{backend}"
    print(f"\n  === {scenario} ({label}) ===")

    row = SpeedometerRow(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        scenario=scenario,
        framework=framework or "",
        backend=backend or "",
        task=task,
        speedometer_score=None,
        inferences_completed=0,
        inference_loop_seconds=0.0,
    )

    playwright_task = (
        "text-classification" if task == "text" else "image-classification"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_EXECUTABLE,
            headless=False,
            args=CHROME_ARGS,
        )
        # Speedometer measures responsiveness — give it a real window size.
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        # Speedometer model load + N iterations can take several minutes.
        page.set_default_timeout(speedometer_timeout * 1000)

        try:
            print(f"    Opening {SPEEDOMETER_URL}")
            page.goto(SPEEDOMETER_URL, wait_until="load")

            # Wait for Speedometer's start button to render before doing
            # anything — that tells us the page bootstrapped.
            page.wait_for_selector(".start-tests-button", state="visible", timeout=30_000)

            # --- Inject + boot inference (skip for baseline) ---
            if not is_baseline:
                print("    Booting inference loop (init → prefetch → load → setInput)...")
                page.evaluate(
                    INFERENCE_BOOT_JS,
                    [framework, backend, playwright_task, IMAGE_URL, TEXT_INPUT],
                )

                # Sanity check: confirm the loop is advancing before we click Start.
                time.sleep(1.5)
                status = page.evaluate(INFERENCE_STATUS_JS)
                print(f"    Loop status after 1.5s: {status}")
                if not status["running"]:
                    raise RuntimeError(
                        f"Inference loop is not running: {status.get('error')}"
                    )

            # --- Start Speedometer ---
            print("    Clicking Speedometer Start Test...")
            page.click(".start-tests-button")

            # --- Wait for score ---
            print(f"    Waiting for #result-number (timeout {speedometer_timeout}s)...")
            score = wait_for_speedometer_score(page, speedometer_timeout)
            row.speedometer_score = round(score, 2)
            print(f"    Speedometer score: {row.speedometer_score}")

            # --- Stop inference + collect counts ---
            if not is_baseline:
                stop_result = page.evaluate(INFERENCE_STOP_JS)
                row.inferences_completed = int(stop_result["count"])
                row.inference_loop_seconds = round(stop_result["elapsedMs"] / 1000, 2)
                row.inference_error = stop_result.get("error")
                throughput = (
                    row.inferences_completed / row.inference_loop_seconds
                    if row.inference_loop_seconds > 0 else 0
                )
                print(
                    f"    Inferences during run: {row.inferences_completed} "
                    f"in {row.inference_loop_seconds}s ({throughput:.2f}/s)"
                )

        except Exception as e:
            row.error = str(e)
            import traceback
            traceback.print_exc()

        finally:
            browser.close()

    return row


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _throughput(r: SpeedometerRow) -> Optional[float]:
    if r.inference_loop_seconds and r.inference_loop_seconds > 0 and r.inferences_completed:
        return round(r.inferences_completed / r.inference_loop_seconds, 2)
    return None


def _delta_pct(r: SpeedometerRow, baseline_score: Optional[float]) -> Optional[float]:
    if baseline_score is None or r.speedometer_score is None or r.scenario == "baseline":
        return None
    return round(((r.speedometer_score - baseline_score) / baseline_score) * 100, 2)


def save_results(rows: list[SpeedometerRow]):
    output_dir = Path(__file__).parent.parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)

    baseline_score = next(
        (r.speedometer_score for r in rows
         if r.scenario == "baseline" and r.speedometer_score is not None),
        None,
    )

    # Augment each row with derived fields for JSON consumers.
    json_rows = []
    for r in rows:
        d = asdict(r)
        d["throughput_inf_per_s"] = _throughput(r)
        d["speedometer_delta_pct"] = _delta_pct(r, baseline_score)
        json_rows.append(d)

    json_path = output_dir / "speedometer_under_load.json"
    json_path.write_text(json.dumps(json_rows, indent=2))
    print(f"\nJSON saved to {json_path}")

    csv_path = output_dir / "speedometer_under_load.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Scenario", "Framework", "Backend", "Task",
            "SpeedometerScore", "InferencesCompleted", "InferenceLoopSeconds",
            "Throughput(inf/s)", "SpeedometerDeltaPct",
            "InferenceError", "Error",
        ])
        for r in rows:
            tput = _throughput(r)
            delta = _delta_pct(r, baseline_score)
            writer.writerow([
                r.timestamp, r.scenario, r.framework, r.backend, r.task,
                r.speedometer_score if r.speedometer_score is not None else "",
                r.inferences_completed, r.inference_loop_seconds,
                tput if tput is not None else "",
                delta if delta is not None else "",
                r.inference_error or "", r.error or "",
            ])
    print(f"CSV saved to {csv_path}")
    print("\n" + "=" * 115)
    header = (
        f"{'Scenario':<12} {'Framework':<16} {'Backend':<22} "
        f"{'Score':>10} {'Infers':>8} {'Loop(s)':>8} {'Tput/s':>8}"
    )
    if baseline_score is not None:
        header += f" {'Δ vs base':>10}"
    print(header)
    print("-" * 115)
    for r in rows:
        if r.error:
            print(f"{r.scenario:<12} {r.framework:<16} {r.backend:<22} ERROR: {r.error[:50]}")
            continue
        tput = _throughput(r)
        line = (
            f"{r.scenario:<12} {r.framework:<16} {r.backend:<22} "
            f"{r.speedometer_score:>10.2f} {r.inferences_completed:>8} "
            f"{r.inference_loop_seconds:>8.2f} "
            f"{(f'{tput:>7.2f}' if tput is not None else '       -')}"
        )
        delta = _delta_pct(r, baseline_score)
        if delta is not None:
            line += f" {delta:>+9.2f}%"
        print(line)
    print("=" * 115)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_combos(s: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"--combos entry '{part}' must be framework:backend")
        fw, be = part.split(":", 1)
        out.append((fw.strip(), be.strip()))
    return out


def main():
    parser = argparse.ArgumentParser(description="Speedometer 4 under inference load (local + injection)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--combos",
        help="Comma-separated framework:backend pairs (default: onnx:wasm-simd-threads)",
    )
    selection.add_argument(
        "--all", action="store_true",
        help="Run the full IMAGE_MATRIX or TEXT_MATRIX (matches benchmark_profiler.py).",
    )
    parser.add_argument(
        "--task", choices=["image", "text"], default="image",
        help="Inference task to inject (default: image)",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Skip the no-inference baseline measurement",
    )
    parser.add_argument(
        "--speedometer-timeout", type=int, default=600,
        help="Max seconds to wait for Speedometer to finish (default: 600)",
    )
    args = parser.parse_args()

    if args.all:
        combos = IMAGE_MATRIX if args.task == "image" else TEXT_MATRIX
    elif args.combos:
        combos = parse_combos(args.combos)
    else:
        combos = parse_combos("onnx:wasm-simd-threads")

    print(f"Task: {args.task}")
    print(f"Combos: {len(combos)} — {combos}")
    print(f"Baseline: {'skipped' if args.no_baseline else 'included'}")

    rows: list[SpeedometerRow] = []

    if not args.no_baseline:
        rows.append(run_scenario(None, None, args.task, args.speedometer_timeout))
        # Persist after baseline so a mid-matrix crash doesn't lose it.
        save_results(rows)

    for fw, be in combos:
        rows.append(run_scenario(fw, be, args.task, args.speedometer_timeout))
        # Save incrementally so a crash on combo N doesn't wipe combos 0..N-1.
        save_results(rows)


if __name__ == "__main__":
    main()
