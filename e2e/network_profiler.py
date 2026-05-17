"""
Network footprint profiler.

For each (framework, backend) combo, launches a fresh Chromium, hooks
page.on("response"), drives the page through framework_init → prefetchModel
→ model_load → one inference, and records how many bytes were transferred
in each phase, broken down by category (JS, WASM, model/weights, other).

The intent is to answer "how much does the browser actually download to
run this combo cold" — separately from on-disk node_modules sizes (which
include source maps, type defs, examples) and separately from production
bundle sizes (which depend on tree-shaking).

Reads Content-Length headers — that's the encoded transfer size (post-gzip
if compressed). When testing against the dev server (npm run dev) most
responses are uncompressed, so numbers approximate raw asset sizes. When
testing against a preview server (npm run preview, serving dist/) responses
may be gzip-compressed.

Usage:
    # Against dev server (per-phase split available):
    npm run dev
    python e2e/network_profiler.py [--task image|text]

    # Against production bundle (totals only — no per-phase split):
    npx vite build
    npx vite preview --port 4173
    python e2e/network_profiler.py --bundled --base-url http://localhost:4173

    Other options:
        [--iterations 1] [--combos fw:be,fw:be]

The --bundled flag is required when running against a built bundle
(npm run preview) because the per-phase flow relies on dynamic-importing
TS source files at /src/, which only the Vite dev server serves. In
bundled mode the script drives the page via window.__benchmark.run(),
which exists in both dev and prod, but cannot split network bytes per
phase — all responses are tagged "running".

Outputs:
    benchmark_results/network_summary_{image,text}.csv     — one row per combo
    benchmark_results/network_per_phase_{image,text}.csv   — combo × phase rows
    benchmark_results/network_requests_{image,text}.csv    — every request
    benchmark_results/network_results_{image,text}.json    — full structured dump
"""

import argparse
import csv
import json
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:5173"
DEFAULT_ITERATIONS = 1
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

TEXT_MATRIX = list(IMAGE_MATRIX)  # same backends; only the model differs

PHASES = ("startup", "framework_init", "prefetch", "model_load", "inference", "running")
CATEGORIES = ("js", "wasm", "model", "other")


# ---------------------------------------------------------------------------
# URL categorization & size extraction
# ---------------------------------------------------------------------------

def _categorize_url(url: str) -> str:
    """Bucket a response URL into js / wasm / model / other."""
    lower = url.lower()
    path = urlparse(lower).path
    if path.endswith(".wasm"):
        return "wasm"
    if path.endswith((".onnx", ".tflite")):
        return "model"
    if path.endswith(".bin") and ("shard" in path or "model" in path or "weight" in path):
        return "model"
    if path.endswith("model.json") or path.endswith("tokenizer.json"):
        return "model"
    if "huggingface.co" in lower or "/resolve/" in lower or "cdn.jsdelivr.net" in lower:
        # Transformers.js fetches models from HF; some packages from jsdelivr
        if path.endswith((".js", ".mjs")):
            return "js"
        return "model"
    if path.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx")):
        return "js"
    return "other"


def _response_size(response) -> int:
    """Best-effort byte count. Prefers Content-Length header, falls back to body()."""
    try:
        headers = response.headers
    except Exception:
        headers = {}
    cl = headers.get("content-length") if isinstance(headers, dict) else None
    if cl:
        try:
            n = int(cl)
            if n > 0:
                return n
        except ValueError:
            pass
    try:
        return len(response.body())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Capture state per iteration
# ---------------------------------------------------------------------------

@dataclass
class RequestRecord:
    phase: str
    category: str
    url: str
    bytes_: int  # trailing underscore avoids 'bytes' shadowing builtin


class NetworkCapture:
    """Records every response into per-phase × per-category buckets."""

    def __init__(self) -> None:
        self.current_phase = "startup"
        self.by_phase: dict[str, dict[str, int]] = {
            p: {c: 0 for c in CATEGORIES} | {"count": 0} for p in PHASES
        }
        self.requests: list[RequestRecord] = []

    def set_phase(self, phase: str) -> None:
        if phase not in self.by_phase:
            self.by_phase[phase] = {c: 0 for c in CATEGORIES} | {"count": 0}
        self.current_phase = phase

    def on_response(self, response) -> None:
        try:
            cat = _categorize_url(response.url)
            size = _response_size(response)
            bucket = self.by_phase[self.current_phase]
            bucket[cat] += size
            bucket["count"] += 1
            self.requests.append(
                RequestRecord(
                    phase=self.current_phase,
                    category=cat,
                    url=response.url,
                    bytes_=size,
                )
            )
        except Exception:
            # Never crash the iteration because of bookkeeping.
            pass

    def totals(self) -> dict[str, int]:
        out = {c: 0 for c in CATEGORIES} | {"count": 0, "total": 0}
        for phase_bucket in self.by_phase.values():
            for c in CATEGORIES:
                out[c] += phase_bucket[c]
            out["count"] += phase_bucket["count"]
        out["total"] = sum(out[c] for c in CATEGORIES)
        return out


# ---------------------------------------------------------------------------
# One cold-start iteration
# ---------------------------------------------------------------------------

@dataclass
class IterationCapture:
    framework: str
    backend: str
    task: str
    iteration: int
    by_phase: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)
    requests: list = field(default_factory=list)
    error: Optional[str] = None


def _drive_phased(
    page, framework: str, backend: str, task: str, net: NetworkCapture,
    init_only: bool = False,
) -> None:
    """Dev-server flow: dynamic-import /src/ and drive each phase explicitly.

    When init_only=True, stops right after framework_init and reports just
    framework runtime + WASM downloads (no model fetch).
    """
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

    net.set_phase("framework_init")
    page.evaluate(
        """async ([task]) => {
            const { createBenchmark } = await import('/src/benchmarks/index.ts');
            const fw = document.getElementById('framework').value;
            const be = document.getElementById('backend').value;
            window.__currentBenchmark = createBenchmark(fw, task);
            await window.__currentBenchmark.initFramework(be);
        }""",
        [task],
    )

    if init_only:
        # Stop here. Any straggler responses (e.g. late-arriving WASM tiers)
        # will still fire on page.on("response") during the drain sleep
        # added in run_iteration.
        return

    net.set_phase("prefetch")
    page.evaluate("async () => await window.__currentBenchmark.prefetchModel()")

    net.set_phase("model_load")
    page.evaluate("async () => await window.__currentBenchmark.loadModel()")

    net.set_phase("inference")
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
    page.evaluate("async () => await window.__currentBenchmark.runInference()")
    page.evaluate("async () => await window.__currentBenchmark.dispose()")


def _drive_bundled(page, framework: str, backend: str, task: str, net: NetworkCapture) -> None:
    """Production-bundle flow: drive via window.__benchmark.run().

    Cannot split per-phase since the bundle encapsulates init/load/inference
    in one method call; everything goes under the "running" phase tag.
    """
    net.set_phase("running")
    page.evaluate(
        """async ([fw, be, task, textInput]) => {
            const b = window.__benchmark;
            b.setTask(task);
            b.configure(fw, be, 1, 0);  // 1 iteration, 0 warmup
            if (task === 'image-classification') {
                await b.loadImage('/rocky.jpg');
            } else {
                b.setTextInput(textInput);
            }
        }""",
        [framework, backend, task, TEXT_INPUT],
    )
    page.evaluate("() => window.__benchmark.run()")


def run_iteration(
    framework: str,
    backend: str,
    task: str,
    iteration_num: int,
    base_url: str,
    bundled: bool,
    init_only: bool = False,
) -> IterationCapture:
    cap = IterationCapture(
        framework=framework, backend=backend, task=task, iteration=iteration_num
    )
    net = NetworkCapture()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=False,
            args=CHROME_ARGS,
        )
        # Fresh context, no shared cache between iterations.
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(120_000)

        # Bypass HTTP cache so every iteration re-downloads everything.
        # Without this, iteration #2+ would show zero bytes for cached assets.
        try:
            cdp = context.new_cdp_session(page)
            cdp.send("Network.enable")
            cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
        except Exception:
            pass

        # Hook BEFORE goto so the very first HTML/JS response is counted.
        page.on("response", net.on_response)

        try:
            net.set_phase("startup")
            page.goto(base_url, wait_until="load", timeout=60_000)
            page.wait_for_function(
                "() => window.__benchmark !== undefined", timeout=15_000
            )

            if bundled:
                if init_only:
                    # In bundled mode `run()` is a single black-box call —
                    # we can't slice it. Still useful to capture startup
                    # bytes (HTML + main JS chunks) by doing only the goto
                    # and skipping run().
                    net.set_phase("running")  # tag for consistency
                else:
                    _drive_bundled(page, framework, backend, task, net)
            else:
                _drive_phased(page, framework, backend, task, net, init_only=init_only)

            # Drain any pending response events so late-arriving bytes are
            # counted rather than lost.
            time.sleep(0.25)

            cap.by_phase = {k: dict(v) for k, v in net.by_phase.items()}
            cap.totals = net.totals()
            cap.requests = [asdict(r) for r in net.requests]

        except Exception as e:
            cap.error = str(e)
            cap.by_phase = {k: dict(v) for k, v in net.by_phase.items()}
            cap.totals = net.totals()
            cap.requests = [asdict(r) for r in net.requests]
            traceback.print_exc()

        finally:
            try:
                browser.close()
            except Exception:
                pass

    return cap


# ---------------------------------------------------------------------------
# Summarise across iterations of one combo
# ---------------------------------------------------------------------------

@dataclass
class ComboSummary:
    framework: str
    backend: str
    task: str
    successful: int
    failed: int
    # Per-category mean+stdev across iterations (total over all phases)
    js_mean_mb: float = 0
    js_stdev_mb: float = 0
    wasm_mean_mb: float = 0
    wasm_stdev_mb: float = 0
    model_mean_mb: float = 0
    model_stdev_mb: float = 0
    other_mean_mb: float = 0
    other_stdev_mb: float = 0
    total_mean_mb: float = 0
    total_stdev_mb: float = 0
    requests_mean: float = 0
    # Per-phase × per-category mean (MB)
    by_phase_mean_mb: dict = field(default_factory=dict)
    error_samples: list = field(default_factory=list)


def _b_to_mb(n: float) -> float:
    return round(n / (1024 * 1024), 3)


def _mean_stdev_mb(values_bytes: list[int]) -> tuple[float, float]:
    if not values_bytes:
        return 0.0, 0.0
    mb = [v / (1024 * 1024) for v in values_bytes]
    mean = round(sum(mb) / len(mb), 3)
    stdev = round(statistics.stdev(mb), 3) if len(mb) > 1 else 0.0
    return mean, stdev


def summarise(
    framework: str, backend: str, task: str, runs: list[IterationCapture]
) -> ComboSummary:
    ok = [r for r in runs if r.error is None]
    bad = [r for r in runs if r.error is not None]

    summary = ComboSummary(
        framework=framework, backend=backend, task=task,
        successful=len(ok), failed=len(bad),
        error_samples=[r.error for r in bad[:3]],
    )

    if not ok:
        return summary

    for cat in CATEGORIES:
        vals = [r.totals.get(cat, 0) for r in ok]
        mean, stdev = _mean_stdev_mb(vals)
        setattr(summary, f"{cat}_mean_mb", mean)
        setattr(summary, f"{cat}_stdev_mb", stdev)

    totals = [r.totals.get("total", 0) for r in ok]
    summary.total_mean_mb, summary.total_stdev_mb = _mean_stdev_mb(totals)

    counts = [r.totals.get("count", 0) for r in ok]
    summary.requests_mean = round(sum(counts) / len(counts), 1)

    # Per-phase × per-category mean across iterations (MB)
    phases_seen: set[str] = set()
    for r in ok:
        phases_seen.update(r.by_phase.keys())
    by_phase: dict[str, dict[str, float]] = {}
    for phase in sorted(phases_seen, key=lambda p: PHASES.index(p) if p in PHASES else 999):
        by_phase[phase] = {}
        for cat in CATEGORIES:
            vals = [r.by_phase.get(phase, {}).get(cat, 0) for r in ok]
            by_phase[phase][cat] = _b_to_mb(sum(vals) / len(vals))
        counts = [r.by_phase.get(phase, {}).get("count", 0) for r in ok]
        by_phase[phase]["count"] = round(sum(counts) / len(counts), 1)
    summary.by_phase_mean_mb = by_phase
    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(
    per_iter: list[IterationCapture], summaries: list[ComboSummary], task: str,
    init_only: bool = False,
) -> None:
    out_dir = Path(__file__).parent.parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    base = "image" if task == "image-classification" else "text"
    suffix = f"{base}_initonly" if init_only else base

    # ---- per-combo summary CSV ----
    sum_csv = out_dir / f"network_summary_{suffix}.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Framework", "Backend", "Task", "Successful", "Failed",
            "JS_Mean(MB)", "JS_Stdev(MB)",
            "WASM_Mean(MB)", "WASM_Stdev(MB)",
            "Model_Mean(MB)", "Model_Stdev(MB)",
            "Other_Mean(MB)", "Other_Stdev(MB)",
            "Total_Mean(MB)", "Total_Stdev(MB)",
            "Requests_Mean",
        ])
        for s in summaries:
            w.writerow([
                s.framework, s.backend, s.task, s.successful, s.failed,
                s.js_mean_mb, s.js_stdev_mb,
                s.wasm_mean_mb, s.wasm_stdev_mb,
                s.model_mean_mb, s.model_stdev_mb,
                s.other_mean_mb, s.other_stdev_mb,
                s.total_mean_mb, s.total_stdev_mb,
                s.requests_mean,
            ])

    # ---- per-phase CSV (long format: one row per combo × phase) ----
    phase_csv = out_dir / f"network_per_phase_{suffix}.csv"
    with phase_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Framework", "Backend", "Task", "Phase",
            "JS(MB)", "WASM(MB)", "Model(MB)", "Other(MB)", "Total(MB)",
            "Requests",
        ])
        for s in summaries:
            for phase, vals in s.by_phase_mean_mb.items():
                total = round(sum(vals[c] for c in CATEGORIES), 3)
                w.writerow([
                    s.framework, s.backend, s.task, phase,
                    vals.get("js", 0), vals.get("wasm", 0),
                    vals.get("model", 0), vals.get("other", 0),
                    total, vals.get("count", 0),
                ])

    # ---- raw per-request log ----
    req_csv = out_dir / f"network_requests_{suffix}.csv"
    with req_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Framework", "Backend", "Task", "Iteration", "Phase",
            "Category", "Bytes", "URL",
        ])
        for r in per_iter:
            for req in r.requests:
                w.writerow([
                    r.framework, r.backend, r.task, r.iteration,
                    req["phase"], req["category"], req["bytes_"], req["url"],
                ])

    json_path = out_dir / f"network_results_{suffix}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "iterations": [asdict(r) for r in per_iter],
                "summaries": [asdict(s) for s in summaries],
            },
            f, indent=2,
        )

    print(f"\nSaved:\n  {sum_csv}\n  {phase_csv}\n  {req_csv}\n  {json_path}")


def print_summary_table(summaries: list[ComboSummary]) -> None:
    print(
        f"\n{'Framework':<16} {'Backend':<22} "
        f"{'JS(MB)':>8} {'WASM(MB)':>10} {'Model(MB)':>10} "
        f"{'Other(MB)':>10} {'Total(MB)':>10} {'Reqs':>5}  {'OK':>3} {'Err':>3}"
    )
    print("-" * 110)
    for s in summaries:
        print(
            f"{s.framework:<16} {s.backend:<22} "
            f"{s.js_mean_mb:>8.2f} {s.wasm_mean_mb:>10.2f} "
            f"{s.model_mean_mb:>10.2f} {s.other_mean_mb:>10.2f} "
            f"{s.total_mean_mb:>10.2f} {s.requests_mean:>5.0f}  "
            f"{s.successful:>3} {s.failed:>3}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cold-start network footprint per (framework, backend)."
    )
    parser.add_argument(
        "--task", choices=["image", "text"], default="image",
        help="image (classification) or text (sentiment)",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help="Cold-start iterations per combo. Downloads are deterministic, "
             "so 1 is usually enough; >1 gives stdev for CDN-fetched assets.",
    )
    parser.add_argument(
        "--combos", default=None,
        help="Comma-separated fw:be pairs to restrict the run, "
             "e.g. 'tfjs:webgpu,onnx:wasm-simd-threads'",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=f"Server to test against (default: {DEFAULT_BASE_URL}). "
             "Use http://localhost:4173 to test a built bundle via "
             "'npm run preview' (requires --bundled).",
    )
    parser.add_argument(
        "--bundled", action="store_true",
        help="Drive the page via window.__benchmark.run() instead of "
             "dynamic-importing /src/ modules. REQUIRED when running "
             "against a production bundle (vite preview / dist). Loses "
             "per-phase splitting — all responses tagged 'running'.",
    )
    parser.add_argument(
        "--init-only", action="store_true",
        help="Stop after framework_init phase: measures ONLY the framework "
             "runtime + WASM downloads, with no model fetch. Useful for "
             "answering 'how big is the framework + WASM by itself'. "
             "In phased mode this gives a clean per-combo breakdown; in "
             "bundled mode it can only capture startup chunks (no init "
             "phase splitting available).",
    )
    args = parser.parse_args()

    # Sanity hint: if base_url is the typical preview port, suggest --bundled.
    if not args.bundled and "4173" in args.base_url:
        print(
            "WARNING: --base-url points at the preview port (4173) but "
            "--bundled is not set. The phased flow will fail because "
            "/src/ paths are not served from the built bundle. Re-run "
            "with --bundled.",
        )

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
        f"task={task}  iterations={args.iterations}  combos={len(matrix)}  "
        f"base_url={args.base_url}  mode={'bundled' if args.bundled else 'phased'}  "
        f"init_only={args.init_only}  "
        f"total_launches={len(matrix) * args.iterations}"
    )

    if args.bundled and args.init_only:
        print(
            "NOTE: --init-only in --bundled mode only captures startup "
            "responses (HTML + main JS chunks). For per-combo framework+WASM "
            "isolation, use phased mode (omit --bundled).",
        )

    per_iter: list[IterationCapture] = []
    summaries: list[ComboSummary] = []

    for fw, be in matrix:
        print(f"\n=== {fw}:{be} ===")
        runs: list[IterationCapture] = []
        for i in range(1, args.iterations + 1):
            print(f"  [{i:>2}/{args.iterations}] launching browser...", end=" ", flush=True)
            r = run_iteration(
                fw, be, task, i, args.base_url, args.bundled, args.init_only,
            )
            runs.append(r)
            per_iter.append(r)
            if r.error:
                print(f"ERROR: {r.error[:120]}")
            else:
                t = r.totals
                print(
                    f"js={_b_to_mb(t.get('js', 0)):>6.2f}MB  "
                    f"wasm={_b_to_mb(t.get('wasm', 0)):>6.2f}MB  "
                    f"model={_b_to_mb(t.get('model', 0)):>7.2f}MB  "
                    f"total={_b_to_mb(t.get('total', 0)):>7.2f}MB  "
                    f"reqs={t.get('count', 0)}"
                )
        s = summarise(fw, be, task, runs)
        summaries.append(s)

    save_outputs(per_iter, summaries, task, init_only=args.init_only)
    print_summary_table(summaries)


if __name__ == "__main__":
    main()
