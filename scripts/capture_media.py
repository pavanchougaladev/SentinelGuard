"""Capture README screenshots and a short walkthrough video of the dashboard.

The script drives headless Chrome through the DevTools protocol (no extra Python
packages beyond ``websockets``), scores demo traffic through the API, clicks the
dashboard controls that need interaction (score form, drift check), and stitches
the screenshots into an MP4 with ffmpeg.

Usage (from the project root):
    python scripts/capture_media.py

It reuses an already running API/dashboard, or starts both and shuts them down
again. Requires Chrome and ffmpeg on PATH (override with CHROME_PATH / FFMPEG).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ("overview", "/", [], "Overview"),
    ("score", "/score", ["Account takeover", "Score transaction"], "Score console"),
    ("activity", "/activity", [], "Live activity"),
    ("batch", "/batch", ["Score 50 sample transactions"], "Batch scoring"),
    ("performance", "/performance", [], "Model performance"),
    ("drift", "/drift", ["Run drift check"], "Drift monitor"),
]

CHROME_CANDIDATES = [
    os.getenv("CHROME_PATH", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    found = shutil.which("chrome") or shutil.which("google-chrome") or shutil.which("chromium")
    if not found:
        raise SystemExit("Chrome not found. Set CHROME_PATH to the Chrome executable.")
    return found


def http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def wait_for(url: str, seconds: int = 90) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if http_ok(url):
            return True
        time.sleep(1)
    return False


def start_stack(api_url: str, app_url: str) -> list[subprocess.Popen]:
    started: list[subprocess.Popen] = []
    env = dict(os.environ)

    if not http_ok(f"{api_url}/health"):
        print("starting scoring API ...")
        started.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    if not http_ok(f"{app_url}/_stcore/health"):
        print("starting Streamlit dashboard ...")
        started.append(
            subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                    "--server.port", "8501",
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false",
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    if not wait_for(f"{api_url}/health"):
        raise SystemExit("The scoring API did not become ready on port 8000.")
    if not api_has_policy(api_url):
        raise SystemExit(
            "The API on port 8000 is running an old build without /policy. "
            "Stop it and rerun so the current code is captured."
        )
    if not wait_for(f"{app_url}/_stcore/health"):
        raise SystemExit("The dashboard did not become ready on port 8501.")
    return started


def api_has_policy(api_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{api_url}/policy?mode=kaggle", timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def stop_stack(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


# ---------------------------------------------------------------------------
# Demo traffic
# ---------------------------------------------------------------------------


def generate_traffic(api_url: str, count: int) -> None:
    """Score real PCA rows so the decision feed and drift window are populated."""
    import pandas as pd
    import requests

    dataset = ROOT / "data" / "creditcard.csv"
    if not dataset.exists():
        print("data/creditcard.csv missing; skipping demo traffic")
        return

    frame = pd.read_csv(dataset, nrows=max(count * 4, 400))
    frame = frame.sample(n=min(count, len(frame)), random_state=7)
    scored = 0
    for _, row in frame.iterrows():
        payload = {"Amount": float(row["Amount"])}
        payload.update({f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)})
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=20)
            scored += int(response.ok)
        except requests.RequestException:
            break
    print(f"scored {scored} demo transactions")


# ---------------------------------------------------------------------------
# Chrome DevTools Protocol driver
# ---------------------------------------------------------------------------


class Chrome:
    def __init__(self, websocket_url: str) -> None:
        self.websocket_url = websocket_url
        self.ws = None
        self.next_id = 0

    async def __aenter__(self) -> "Chrome":
        self.ws = await websockets.connect(self.websocket_url, max_size=None)
        await self.cmd("Page.enable")
        await self.cmd("Runtime.enable")
        await self.cmd(
            "Emulation.setDeviceMetricsOverride",
            width=1600,
            height=1000,
            deviceScaleFactor=1,
            mobile=False,
        )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self.ws is not None:
            await self.ws.close()

    async def cmd(self, method: str, **params) -> dict:
        self.next_id += 1
        message_id = self.next_id
        await self.ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("id") != message_id:
                continue
            if "error" in message:
                raise RuntimeError(f"{method}: {message['error']}")
            return message.get("result", {})

    async def go(self, url: str, settle: float = 5.0) -> None:
        await self.cmd("Page.navigate", url=url)
        await asyncio.sleep(1.0)
        for _ in range(40):
            state = await self.cmd(
                "Runtime.evaluate",
                expression="document.readyState",
                returnByValue=True,
            )
            if state.get("result", {}).get("value") == "complete":
                break
            await asyncio.sleep(0.25)
        await asyncio.sleep(settle)

    async def wait_js(self, expression: str, timeout: float = 20.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = await self.cmd(
                "Runtime.evaluate", expression=expression, returnByValue=True
            )
            if result.get("result", {}).get("value"):
                return True
            await asyncio.sleep(0.5)
        return False

    async def click(self, label: str, timeout: float = 20.0) -> bool:
        expression = """
        (() => {
          const buttons = [...document.querySelectorAll('button')];
          const target = buttons.find(
            (node) => node.textContent.replace(/\\s+/g, ' ').includes(%s)
          );
          if (!target) return 'missing';
          target.click();
          return 'clicked';
        })()
        """ % json.dumps(label)
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = await self.cmd(
                "Runtime.evaluate", expression=expression, returnByValue=True
            )
            if result.get("result", {}).get("value") == "clicked":
                return True
            await asyncio.sleep(0.5)
        return False

    async def screenshot(self, path: Path, clip: dict | None = None) -> None:
        params: dict = {"format": "png"}
        if clip:
            params.update(clip=clip, captureBeyondViewport=True)
        result = await self.cmd("Page.captureScreenshot", **params)
        path.write_bytes(base64.b64decode(result["data"]))


async def capture_pages(base_url: str, out_dir: Path, chrome_path: str) -> list[Path]:
    profile = tempfile.mkdtemp(prefix="sentinelguard-chrome-")
    browser = subprocess.Popen(
        [
            chrome_path,
            "--headless=new",
            "--remote-debugging-port=9222",
            f"--user-data-dir={profile}",
            "--window-size=1600,1000",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            "--force-device-scale-factor=1",
            "--disable-background-timer-throttling",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for("http://127.0.0.1:9222/json/version", seconds=30):
            raise SystemExit("Chrome DevTools endpoint did not start.")
        with urllib.request.urlopen("http://127.0.0.1:9222/json/list", timeout=10) as response:
            targets = json.load(response)
        page_target = next(item for item in targets if item["type"] == "page")

        saved: list[Path] = []
        async with Chrome(page_target["webSocketDebuggerUrl"]) as chrome:
            for slug, path, actions, label in PAGES:
                print(f"capturing {label} ...")
                await chrome.go(f"{base_url}{path}", settle=4.0)
                for action in actions:
                    clicked = await chrome.click(action)
                    print(f"  {action!r}: {'clicked' if clicked else 'button not found'}")
                    await asyncio.sleep(2.0)
                # Charts render after Streamlit finishes the run; wait for an
                # ECharts canvas before taking the frame.
                await chrome.wait_js(
                    "document.querySelectorAll('canvas').length > 0", timeout=25
                )
                await asyncio.sleep(2.0)
                target = out_dir / f"{slug}.png"
                await chrome.screenshot(target)
                saved.append(target)

            # Second frame of Model performance showing the calibration,
            # baseline and threshold-policy cards below the fold. Chrome does
            # not paint content outside the viewport, so render a tall viewport
            # and crop the lower half.
            print("capturing Model performance (policy section) ...")
            await chrome.cmd(
                "Emulation.setDeviceMetricsOverride",
                width=1600,
                height=2400,
                deviceScaleFactor=1,
                mobile=False,
            )
            await chrome.go(f"{base_url}/performance", settle=9.0)
            await chrome.wait_js(
                "document.body.innerText.includes('Threshold policy simulation')",
                timeout=30,
            )
            await asyncio.sleep(2.0)
            full = out_dir / "_performance_full.png"
            await chrome.screenshot(full)

            from PIL import Image

            with Image.open(full) as image:
                image.crop((0, 1120, 1600, 2000)).save(out_dir / "performance-policy.png")
            full.unlink(missing_ok=True)
            saved.append(out_dir / "performance-policy.png")
        return saved
    finally:
        browser.terminate()
        try:
            browser.wait(timeout=10)
        except subprocess.TimeoutExpired:
            browser.kill()
        shutil.rmtree(profile, ignore_errors=True)


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------


def build_video(frames: list[Path], output: Path, seconds: float = 3.5, fade: float = 0.8) -> bool:
    ffmpeg = os.getenv("FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found; skipping video assembly")
        return False
    if len(frames) < 2:
        return False

    command = [ffmpeg, "-y", "-loglevel", "error"]
    for frame in frames:
        command += ["-loop", "1", "-framerate", "30", "-t", str(seconds), "-i", str(frame)]

    step = seconds - fade
    filters = []
    for index in range(len(frames)):
        # Frames can differ slightly in height (viewport vs cropped region);
        # normalise every input to the same canvas before crossfading.
        filters.append(
            f"[{index}:v]scale=1600:1000:force_original_aspect_ratio=decrease,"
            f"pad=1600:1000:(ow-iw)/2:(oh-ih)/2:color=black,settb=AVTB[v{index}]"
        )
    previous = "v0"
    for index in range(1, len(frames)):
        label = f"x{index}"
        filters.append(
            f"[{previous}][v{index}]xfade=transition=fade:duration={fade}:"
            f"offset={round(index * step, 2)}[{label}]"
        )
        previous = label
    filters.append(f"[{previous}]format=yuv420p[v]")

    command += [
        "-filter_complex", ";".join(filters),
        "-map", "[v]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-movflags", "+faststart",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1200:])
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--app-url", default="http://localhost:8501")
    parser.add_argument("--out-dir", default="docs/screenshots")
    parser.add_argument("--video", default="docs/demo.mp4")
    parser.add_argument("--traffic", type=int, default=160)
    parser.add_argument("--skip-traffic", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    chrome_path = find_chrome()
    started = start_stack(args.api_url, args.app_url)
    try:
        if not args.skip_traffic:
            generate_traffic(args.api_url, args.traffic)
        frames = asyncio.run(capture_pages(args.app_url, out_dir, chrome_path))
    finally:
        stop_stack(started)

    for frame in frames:
        print(f"wrote {frame.relative_to(ROOT)} ({frame.stat().st_size // 1024} KB)")
    if build_video(frames, ROOT / args.video):
        print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
