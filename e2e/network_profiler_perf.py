"""
Framework + WASM transfer size, measured via Performance Resource Timing API.

By default, this profiler measures **production** values — runs against the
Vite preview server (port 4173), which serves the built bundle with gzip
compression. That's what real users would download in production.

For each (framework, backend) combo:
  1. Launch a fresh Chromium with cache disabled.
  2. Navigate to the page; wait until __benchmark is registered.
  3. Snapshot performance.getEntriesByType('resource') and clear it
     — this is the "startup" bucket (page HTML + main JS chunks loaded
     before any framework is touched).
  4. Call window.__benchmark.initOnly(framework, backend, task), which
     creates the adapter and calls initFramework(backend). Nothing else.
  5. Snapshot performance.getEntriesByType('resource') again — this
     is the "framework_init" bucket (the framework runtime JS + its
     WASM files, post-gzip).
  6. Stop. No prefetch, no model load, no inference.

For every resource entry we read three sizes (same fields DevTools uses):
  - transferSize    — bytes over the wire incl. headers (gzip/brotli respected)
  - encodedBodySize — body size as compressed by server
  - decodedBodySize — body size after the browser decompressed it

`transferSize` is what to report as "downloaded bytes". `decodedBodySize`
is what to compare against on-disk file sizes.

Usage:
    # Production measurement (default — vite preview, gzipped):
    npx vite build
    npx vite preview --port 4173
    python e2e/network_profiler_perf.py [--task image|text]
                                        [--iterations 1]
                                        [--combos fw:be,fw:be]

    # Dev-server measurement (raw, uncompressed):
    npm run dev
    python e2e/network_profiler_perf.py --base-url http://localhost:5173

Outputs in benchmark_results/:
    network_perf_summary_{task}.csv   — one row per (combo × iteration)
    network_perf_details_{task}.csv   — one row per resource entry
    network_perf_results_{task}.json  — full structured dump
"""

import argparse
import csv
import json
import statistics
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

DEFAULT_BASE_URL = "http://localhost:4173"   # vite preview (production bundle, gzipped)
CHROME_PATH = r"D:\chr-build\chromium\src\out\Release\chrome.exe"
TEXT_INPUT = "This movie was absolutely wonderful and I loved every moment of it."

CHROME_ARGS = [
    "--enable-precise-memory-info",
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

MATRIX = [
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
# Helpers
# ---------------------------------------------------------------------------

def _categorize(url: str) -> str:
    path = urlparse(url.lower()).path
    if path.endswith(".wasm"):
        return "wasm"
    if path.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx")):
        return "js"
    if path.endswith(".css"):
        return "css"
    if path.endswith((".html", ".htm")):
        return "html"
    return "other"


def _snapshot_and_clear(page) -> list[dict]:
    """Read all resource timing entries since the last clear, then clear."""
    return page.evaluate(
        """() => {
            const out = performance.getEntriesByType('resource').map(r => ({
                name: r.name,
                transferSize: r.transferSize | 0,
                encodedBodySize: r.encodedBodySize | 0,
                decodedBodySize: r.decodedBodySize | 0,
                initiatorType: r.initiatorType,
                duration: Math.round(r.duration),
            }));
            performance.clearResourceTimings();
            return out;
        }"""
    )


def _bucket(entries: list[dict]) -> dict:
    """Aggregate a list of entries into per-category sums."""
    out = {c: {"transfer": 0, "encoded": 0, "decoded": 0, "count": 0}
           for c in ("js", "wasm", "css", "html", "other")}
    out["total"] = {"transfer": 0, "encoded": 0, "decoded": 0, "count": 0}
    for e in entries:
        cat = _categorize(e["name"])
        for key, field_name in (("transfer", "transferSize"),
                                ("encoded", "encodedBodySize"),
                                ("decoded", "decodedBodySize")):
            out[cat][key] += e[field_name]
            out["total"][key] += e[field_name]
        out[cat]["count"] += 1
        out["total"]["count"] += 1
    return out


# ---------------------------------------------------------------------------
# Single iteration
# ---------------------------------------------------------------------------

@dataclass
class IterCapture:
    framework: str
    backend: str
    task: str
    iteration: int
    startup_entries: list = field(default_factory=list)
    init_entries: list = field(default_factory=list)
    startup_bucket: dict = field(default_factory=dict)
    init_bucket: dict = field(default_factory=dict)
    error: Optional[str] = None


def run_iteration(
    framework: str, backend: str, task: str, iter_num: int, base_url: str,
) -> IterCapture:
    cap = IterCapture(framework=framework, backend=backend, task=task, iteration=iter_num)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH, headless=False, args=CHROME_ARGS,
        )
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(60_000)

        # Disable HTTP cache so each iteration sees real network transfer.
        try:
            cdp = context.new_cdp_session(page)
            cdp.send("Network.enable")
            cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
        except Exception:
            pass

        try:
            page.goto(base_url, wait_until="load", timeout=60_000)
            page.wait_for_function(
                "() => window.__benchmark !== undefined", timeout=15_000
            )

            # Sanity check: initOnly() must exist. Older bundles (built
            # before that method was added to src/main.ts) won't have it.
            has_init_only = page.evaluate(
                "() => typeof window.__benchmark.initOnly === 'function'"
            )
            if not has_init_only:
                raise RuntimeError(
                    "window.__benchmark.initOnly() is not exposed. "
                    "Rebuild the bundle (npx vite build) — initOnly was "
                    "added to src/main.ts and your preview bundle is stale."
                )

            # Configure task+input but DON'T initialize the framework yet —
            # configure() only sets dropdowns; loadImage / setTextInput only
            # prepare the input. No framework runtime code runs here.
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

            # ---- Snapshot #1: everything loaded so far is "startup" ----
            cap.startup_entries = _snapshot_and_clear(page)

            # ---- Drive framework_init only ----
            page.evaluate(
                """async ([fw, be, task]) => {
                    await window.__benchmark.initOnly(fw, be, task);
                }""",
                [framework, backend, task],
            )

            # Settling — let any late-arriving WASM tier responses register.
            time.sleep(0.3)

            # ---- Snapshot #2: new entries since clear = init bucket ----
            cap.init_entries = _snapshot_and_clear(page)

            cap.startup_bucket = _bucket(cap.startup_entries)
            cap.init_bucket = _bucket(cap.init_entries)

        except Exception as e:
            cap.error = str(e)
            traceback.print_exc()

        finally:
            try:
                browser.close()
            except Exception:
                pass

    return cap


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _kb(n: int) -> float:
    return round(n / 1024, 1)


def save(results: list[IterCapture], task: str) -> None:
    out_dir = Path(__file__).parent.parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    suffix = "image" if task == "image-classification" else "text"

    # ---- Summary CSV (one row per iteration) ----
    sum_csv = out_dir / f"network_perf_summary_{suffix}.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Framework", "Backend", "Task", "Iteration",
            "Startup_JS_kB", "Startup_WASM_kB", "Startup_Other_kB", "Startup_Total_kB",
            "Init_JS_kB", "Init_WASM_kB", "Init_Other_kB", "Init_Total_kB",
            "Init_Requests",
            # Same as Init_*_kB but in decoded (uncompressed) bytes
            "Init_JS_kB_decoded", "Init_WASM_kB_decoded", "Init_Total_kB_decoded",
            "Error",
        ])
        for r in results:
            sb = r.startup_bucket
            ib = r.init_bucket
            s_other = (sb.get("css", {}).get("transfer", 0)
                       + sb.get("html", {}).get("transfer", 0)
                       + sb.get("other", {}).get("transfer", 0))
            i_other = (ib.get("css", {}).get("transfer", 0)
                       + ib.get("html", {}).get("transfer", 0)
                       + ib.get("other", {}).get("transfer", 0))
            w.writerow([
                r.framework, r.backend, r.task, r.iteration,
                _kb(sb.get("js", {}).get("transfer", 0)),
                _kb(sb.get("wasm", {}).get("transfer", 0)),
                _kb(s_other),
                _kb(sb.get("total", {}).get("transfer", 0)),
                _kb(ib.get("js", {}).get("transfer", 0)),
                _kb(ib.get("wasm", {}).get("transfer", 0)),
                _kb(i_other),
                _kb(ib.get("total", {}).get("transfer", 0)),
                ib.get("total", {}).get("count", 0),
                _kb(ib.get("js", {}).get("decoded", 0)),
                _kb(ib.get("wasm", {}).get("decoded", 0)),
                _kb(ib.get("total", {}).get("decoded", 0)),
                r.error or "",
            ])

    # ---- Detail CSV (one row per resource entry) ----
    det_csv = out_dir / f"network_perf_details_{suffix}.csv"
    with det_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Framework", "Backend", "Task", "Iteration", "Phase",
            "Category", "Transfer_kB", "Encoded_kB", "Decoded_kB",
            "InitiatorType", "Duration_ms", "URL",
        ])
        for r in results:
            for phase, entries in (("startup", r.startup_entries),
                                   ("framework_init", r.init_entries)):
                for e in entries:
                    w.writerow([
                        r.framework, r.backend, r.task, r.iteration, phase,
                        _categorize(e["name"]),
                        _kb(e["transferSize"]), _kb(e["encodedBodySize"]),
                        _kb(e["decodedBodySize"]),
                        e["initiatorType"], e["duration"], e["name"],
                    ])

    json_path = out_dir / f"network_perf_results_{suffix}.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved:\n  {sum_csv}\n  {det_csv}\n  {json_path}")


def print_table(results: list[IterCapture]) -> None:
    print(
        f"\n{'Framework':<16} {'Backend':<22} "
        f"{'Init JS':>10} {'Init WASM':>11} {'Init Total':>12} "
        f"{'Reqs':>5} {'(decoded WASM)':>16}"
    )
    print("-" * 100)
    for r in results:
        if r.error:
            print(f"{r.framework:<16} {r.backend:<22}  ERROR: {r.error[:60]}")
            continue
        ib = r.init_bucket
        print(
            f"{r.framework:<16} {r.backend:<22} "
            f"{_kb(ib['js']['transfer']):>9.1f}kB "
            f"{_kb(ib['wasm']['transfer']):>10.1f}kB "
            f"{_kb(ib['total']['transfer']):>11.1f}kB "
            f"{ib['total']['count']:>5} "
            f"{_kb(ib['wasm']['decoded']):>14.1f}kB"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Framework + WASM transfer size via Performance Resource Timing API."
    )
    parser.add_argument("--task", choices=["image", "text"], default="image")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Iterations per combo (default 1; downloads are deterministic)")
    parser.add_argument("--combos", default=None,
                        help="Comma-separated fw:be pairs to restrict the run")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=(
            f"Server to test against. Default: {DEFAULT_BASE_URL} (vite preview, "
            "production bundle WITH gzip — recommended for thesis-grade numbers). "
            "Use http://localhost:5173 to test against `npm run dev` (raw, uncompressed)."
        ),
    )
    args = parser.parse_args()

    task = "image-classification" if args.task == "image" else "text-classification"
    matrix = list(MATRIX)
    if args.combos:
        wanted = set()
        for c in args.combos.split(","):
            fw, _, be = c.strip().partition(":")
            if fw and be:
                wanted.add((fw, be))
        matrix = [c for c in matrix if c in wanted]
        if not matrix:
            print(f"No combos matched --combos={args.combos!r}")
            return

    is_preview = "4173" in args.base_url
    mode_label = "production (vite preview, gzipped)" if is_preview else "dev (raw, uncompressed)"
    print(
        f"task={task}  iterations={args.iterations}  combos={len(matrix)}  "
        f"base_url={args.base_url}  mode={mode_label}  "
        f"total_launches={len(matrix) * args.iterations}"
    )
    if is_preview:
        print(
            "Reminder: ensure you have run `npx vite build && npx vite preview --port 4173` "
            "before this script. If you see 'initOnly is not exposed', rebuild the bundle."
        )

    results: list[IterCapture] = []
    for fw, be in matrix:
        print(f"\n=== {fw}:{be} ===")
        for i in range(1, args.iterations + 1):
            print(f"  [{i:>2}/{args.iterations}] launching browser...", end=" ", flush=True)
            r = run_iteration(fw, be, task, i, args.base_url)
            results.append(r)
            if r.error:
                print(f"ERROR: {r.error[:120]}")
            else:
                ib = r.init_bucket
                print(
                    f"JS={_kb(ib['js']['transfer']):>7.1f}kB "
                    f"WASM={_kb(ib['wasm']['transfer']):>9.1f}kB "
                    f"total={_kb(ib['total']['transfer']):>9.1f}kB "
                    f"reqs={ib['total']['count']}"
                )

    save(results, task)
    print_table(results)


if __name__ == "__main__":
    main()
