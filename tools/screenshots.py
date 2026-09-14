"""Drive the built site with a real browser, verify it renders, and capture the tour.

Run:
    python3 tools/screenshots.py

This is a verification step first and a screenshot generator second. It asserts that each
page actually rendered its artifacts — not merely that it returned HTTP 200 — by waiting
for a known element and failing loudly if the page shows a loading or error state. Console
errors are collected and reported, so a chart that silently throws is caught here rather
than by a reader.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "web" / "dist"
OUT = ROOT / "docs" / "screenshots"
PORT = 8899

# route, output name, selector that only appears once real artifact data has rendered
PAGES = [
    ("/", "00_home", ".project-card"),
    ("/#/methodology", "00_methodology", ".card"),
    ("/#/p/nyc-mobility", "01_nyc_mobility", ".stat-value"),
    ("/#/p/segmentation", "02_segmentation", ".stat-value"),
    ("/#/p/market-basket", "03_market_basket", ".stat-value"),
    ("/#/p/fraud", "04_fraud", ".stat-value"),
    ("/#/p/forecasting", "05_forecasting", ".stat-value"),
    ("/#/p/automl", "06_automl", ".stat-value"),
    ("/#/p/transformer", "07_transformer", ".stat-value"),
    ("/#/p/academy", "08_academy", ".card"),
    ("/#/p/similarity-search", "09_similarity_search", ".stat-value"),
    ("/#/p/fairness", "10_fairness", ".stat-value"),
    ("/#/p/dag-engine", "11_dag_engine", ".stat-value"),
    ("/#/p/backtest", "12_backtest", ".stat-value"),
]

# Tabs worth capturing separately: the CRISP-DM record and the provenance table are two of
# the portfolio's central claims and are invisible on the default tab.
TAB_SHOTS = [
    ("/#/p/fraud", "04_fraud_method", "Method (CRISP-DM)"),
    ("/#/p/nyc-mobility", "01_nyc_mobility_data", "Data provenance"),
    ("/#/p/fairness", "10_fairness_method", "Method (CRISP-DM)"),
]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def log_message(self, *args):  # noqa: A003 - silence per-request logging
        pass


def chromium_path() -> str | None:
    """Locate a usable Chromium, preferring one already present on the machine.

    The Python Playwright package pins a browser revision and refuses to launch when the
    installed browser is a different build. Rather than download several hundred megabytes
    of a second Chromium, point the launcher at whichever one is already here. Returns
    ``None`` to fall back to Playwright's own resolution when nothing is found.
    """
    candidates = sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))
    candidates += sorted(Path("/opt/pw-browsers").glob("chromium/chrome-linux/chrome"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    for fallback in ("/usr/bin/chromium", "/usr/bin/chromium-browser",
                     "/usr/bin/google-chrome"):
        if Path(fallback).is_file():
            return fallback
    return None


def serve() -> socketserver.TCPServer:
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), QuietHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def ensure_dist_served_at_root() -> None:
    """Build the site with a root base path if the current build has a different one.

    The committed Vite config targets the GitHub Pages sub-path (``/Projects/``), while this
    script serves ``web/dist`` at ``/``. A build made for Pages therefore requests
    ``/Projects/assets/...`` here, every asset 404s, and each page renders an empty body —
    which previously showed up as fourteen identical "selector never appeared" failures
    rather than as the one-line cause. Detect the mismatch and rebuild rather than making
    the caller remember an environment variable.
    """
    index = DIST / "index.html"
    if index.exists() and "/Projects/assets/" not in index.read_text():
        return

    reason = "web/dist not found" if not index.exists() else "web/dist is built for the Pages sub-path"
    print(f"{reason} — rebuilding with VITE_BASE=/ for local serving …")
    subprocess.run(
        ["npm", "run", "build"],
        cwd=ROOT / "web",
        check=True,
        env={**os.environ, "VITE_BASE": "/"},
    )


def main() -> None:
    ensure_dist_served_at_root()

    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    httpd = serve()
    failures: list[str] = []
    captured: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_path())
        page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                device_scale_factor=2)

        console_errors: list[str] = []
        page.on("console", lambda m: console_errors.append(m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

        def capture(route: str, name: str, selector: str, tab: str | None = None) -> None:
            before = len(console_errors)
            page.goto(f"http://127.0.0.1:{PORT}{route}", wait_until="networkidle")

            if tab:
                page.get_by_role("tab", name=tab).click()
                page.wait_for_timeout(600)

            try:
                page.wait_for_selector(selector, timeout=15000)
            except Exception:
                failures.append(f"{name}: selector {selector!r} never appeared")
                return

            # A page showing the loading or error placeholder has not really rendered.
            body = page.inner_text("body")
            if "Could not load artifacts" in body:
                failures.append(f"{name}: page rendered the artifact load-error state")
                return
            if body.strip().count("Loading") > 2:
                failures.append(f"{name}: page still showing loading placeholders")
                return

            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            captured.append(name)

            new_errors = console_errors[before:]
            if new_errors:
                failures.append(f"{name}: {len(new_errors)} console error(s): {new_errors[:2]}")
            print(f"  ✓ {name}")

        print(f"Capturing {len(PAGES) + len(TAB_SHOTS)} views …")
        for route, name, selector in PAGES:
            capture(route, name, selector)
        for route, name, tab in TAB_SHOTS:
            capture(route, name, ".card", tab=tab)

        browser.close()

    httpd.shutdown()

    print(f"\n{len(captured)} screenshots written to {OUT.relative_to(ROOT)}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("All pages rendered real artifact data with no console errors.")


if __name__ == "__main__":
    main()
