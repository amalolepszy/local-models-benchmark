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
    ("litert", "webgpu"),
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
    repeat: int            # 1..repeats — which run within the per-combo set
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
    repeat: int = 1,
    total_repeats: int = 1,
) -> SpeedometerRow:
    is_baseline = framework is None or backend is None
    scenario = "baseline" if is_baseline else "under_load"
    label = "baseline" if is_baseline else f"{framework}:{backend}"
    rep_label = f" [{repeat}/{total_repeats}]" if total_repeats > 1 else ""
    print(f"\n  === {scenario} ({label}){rep_label} ===")

    row = SpeedometerRow(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        scenario=scenario,
        framework=framework or "",
        backend=backend or "",
        task=task,
        repeat=repeat,
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


def _delta_pct(score: Optional[float], baseline_score: Optional[float]) -> Optional[float]:
    if baseline_score is None or score is None or baseline_score == 0:
        return None
    return round(((score - baseline_score) / baseline_score) * 100, 2)


def _stats(values: list[float]) -> dict:
    """avg/min/max/p95 over a list of values. p95 index matches benchmark_profiler.py."""
    if not values:
        return {"n": 0, "avg": None, "min": None, "max": None, "p95": None, "stdev": None}
    s = sorted(values)
    p95_idx = max(0, int(len(s) * 0.95) - 1)
    avg = sum(values) / len(values)
    if len(values) > 1:
        var = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
        stdev = var ** 0.5
    else:
        stdev = 0.0
    return {
        "n": len(values),
        "avg": round(avg, 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "p95": round(s[p95_idx], 2),
        "stdev": round(stdev, 2),
    }


def _aggregate(rows: list[SpeedometerRow]) -> list[dict]:
    """
    Group rows by (scenario, framework, backend, task) and compute summary
    stats over the SpeedometerScore + Throughput + Inferences across repeats.
    Skips rows that errored out (no score) when computing stats but reports
    how many succeeded vs were attempted.
    """
    groups: dict[tuple, list[SpeedometerRow]] = {}
    for r in rows:
        key = (r.scenario, r.framework, r.backend, r.task)
        groups.setdefault(key, []).append(r)

    # Mean baseline (per-task) for delta calculations.
    baseline_means: dict[str, float] = {}
    for (scenario, _fw, _be, t), grp in groups.items():
        if scenario != "baseline":
            continue
        scores = [r.speedometer_score for r in grp if r.speedometer_score is not None]
        if scores:
            baseline_means[t] = sum(scores) / len(scores)

    out: list[dict] = []
    for (scenario, fw, be, t), grp in groups.items():
        scores = [r.speedometer_score for r in grp if r.speedometer_score is not None]
        tputs = [tp for tp in (_throughput(r) for r in grp) if tp is not None]
        infers = [r.inferences_completed for r in grp if r.speedometer_score is not None]

        score_stats = _stats(scores)
        tput_stats = _stats(tputs)
        infer_stats = _stats([float(x) for x in infers])

        baseline_mean = baseline_means.get(t)
        out.append({
            "scenario": scenario,
            "framework": fw,
            "backend": be,
            "task": t,
            "attempts": len(grp),
            "successful": len(scores),
            "errors": [r.error or r.inference_error for r in grp if r.error or r.inference_error],
            "score_avg": score_stats["avg"],
            "score_min": score_stats["min"],
            "score_max": score_stats["max"],
            "score_p95": score_stats["p95"],
            "score_stdev": score_stats["stdev"],
            "throughput_avg": tput_stats["avg"],
            "throughput_min": tput_stats["min"],
            "throughput_max": tput_stats["max"],
            "throughput_p95": tput_stats["p95"],
            "inferences_avg": infer_stats["avg"],
            "inferences_min": infer_stats["min"],
            "inferences_max": infer_stats["max"],
            "speedometer_delta_pct_avg": _delta_pct(score_stats["avg"], baseline_mean),
        })
    return out


def save_results(rows: list[SpeedometerRow], task: str):
    output_dir = Path(__file__).parent.parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    suffix = task  # "image" or "text"

    summary = _aggregate(rows)
    # Per-task mean baseline used for per-row delta below.
    baseline_means: dict[str, float] = {
        s["task"]: s["score_avg"] for s in summary
        if s["scenario"] == "baseline" and s["score_avg"] is not None
    }

    # --- Per-run JSON (every individual repeat, with derived fields). ---
    json_rows = []
    for r in rows:
        d = asdict(r)
        d["throughput_inf_per_s"] = _throughput(r)
        d["speedometer_delta_pct"] = _delta_pct(
            r.speedometer_score, baseline_means.get(r.task)
        )
        json_rows.append(d)

    json_path = output_dir / f"speedometer_under_load_{suffix}.json"
    json_path.write_text(json.dumps(
        {"runs": json_rows, "summary": summary}, indent=2,
    ))
    print(f"\nJSON saved to {json_path}")

    # --- Per-run CSV. ---
    csv_path = output_dir / f"speedometer_under_load_{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Scenario", "Framework", "Backend", "Task", "Repeat",
            "SpeedometerScore", "InferencesCompleted", "InferenceLoopSeconds",
            "Throughput(inf/s)", "SpeedometerDeltaPct",
            "InferenceError", "Error",
        ])
        for r in rows:
            tput = _throughput(r)
            delta = _delta_pct(r.speedometer_score, baseline_means.get(r.task))
            writer.writerow([
                r.timestamp, r.scenario, r.framework, r.backend, r.task, r.repeat,
                r.speedometer_score if r.speedometer_score is not None else "",
                r.inferences_completed, r.inference_loop_seconds,
                tput if tput is not None else "",
                delta if delta is not None else "",
                r.inference_error or "", r.error or "",
            ])
    print(f"CSV saved to {csv_path}")

    # --- Aggregate CSV (one row per fw:be:task). ---
    summary_csv = output_dir / f"speedometer_under_load_summary_{suffix}.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario", "Framework", "Backend", "Task", "Attempts", "Successful",
            "Score_Avg", "Score_Min", "Score_Max", "Score_P95", "Score_Stdev",
            "Throughput_Avg", "Throughput_Min", "Throughput_Max", "Throughput_P95",
            "Inferences_Avg", "Inferences_Min", "Inferences_Max",
            "SpeedometerDeltaPct_Avg",
        ])
        for s in summary:
            writer.writerow([
                s["scenario"], s["framework"], s["backend"], s["task"],
                s["attempts"], s["successful"],
                s["score_avg"] or "", s["score_min"] or "", s["score_max"] or "",
                s["score_p95"] or "", s["score_stdev"] or "",
                s["throughput_avg"] or "", s["throughput_min"] or "",
                s["throughput_max"] or "", s["throughput_p95"] or "",
                s["inferences_avg"] or "", s["inferences_min"] or "",
                s["inferences_max"] or "",
                s["speedometer_delta_pct_avg"] if s["speedometer_delta_pct_avg"] is not None else "",
            ])
    print(f"Summary CSV saved to {summary_csv}")

    # --- Console: per-combo summary table (only worth printing if N>1, but always shown). ---
    print("\n" + "=" * 130)
    header = (
        f"{'Scenario':<12} {'Framework':<16} {'Backend':<22} {'N':>3} "
        f"{'AvgScore':>10} {'MinScore':>10} {'MaxScore':>10} {'P95':>10} "
        f"{'AvgTput':>9} {'AvgInfers':>10}"
    )
    has_baseline = any(s["scenario"] == "baseline" for s in summary)
    if has_baseline:
        header += f" {'Δ vs base':>10}"
    print(header)
    print("-" * 130)
    for s in summary:
        if s["successful"] == 0:
            err = (s["errors"][0] if s["errors"] else "(no score)")[:50]
            print(f"{s['scenario']:<12} {s['framework']:<16} {s['backend']:<22} {s['attempts']:>3} ERROR: {err}")
            continue
        line = (
            f"{s['scenario']:<12} {s['framework']:<16} {s['backend']:<22} "
            f"{s['successful']:>3} "
            f"{s['score_avg']:>10.2f} {s['score_min']:>10.2f} "
            f"{s['score_max']:>10.2f} {s['score_p95']:>10.2f} "
            f"{(s['throughput_avg'] if s['throughput_avg'] is not None else 0):>9.2f} "
            f"{(s['inferences_avg'] if s['inferences_avg'] is not None else 0):>10.1f}"
        )
        delta = s["speedometer_delta_pct_avg"]
        if has_baseline and delta is not None and s["scenario"] != "baseline":
            line += f" {delta:>+9.2f}%"
        print(line)
    print("=" * 130)


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
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of Speedometer runs per scenario (default: 1). Each run uses a fresh Chromium. Aggregates (avg/min/max/p95) are written to speedometer_under_load_summary_<task>.csv.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    if args.all:
        combos = IMAGE_MATRIX if args.task == "image" else TEXT_MATRIX
    elif args.combos:
        combos = parse_combos(args.combos)
    else:
        combos = parse_combos("onnx:wasm-simd-threads")

    print(f"Task: {args.task}")
    print(f"Combos: {len(combos)} — {combos}")
    print(f"Baseline: {'skipped' if args.no_baseline else 'included'}")
    print(f"Repeats per scenario: {args.repeats}")

    rows: list[SpeedometerRow] = []

    if not args.no_baseline:
        for rep in range(1, args.repeats + 1):
            rows.append(run_scenario(
                None, None, args.task, args.speedometer_timeout,
                repeat=rep, total_repeats=args.repeats,
            ))
            # Save incrementally so a crash mid-run doesn't lose data.
            save_results(rows, args.task)

    for fw, be in combos:
        for rep in range(1, args.repeats + 1):
            rows.append(run_scenario(
                fw, be, args.task, args.speedometer_timeout,
                repeat=rep, total_repeats=args.repeats,
            ))
            save_results(rows, args.task)


if __name__ == "__main__":
    main()
