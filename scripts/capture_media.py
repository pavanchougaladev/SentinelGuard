"""Capture README screenshots and a short walkthrough video of the dashboard.

The script drives headless Chrome through the DevTools protocol (no extra Python
packages beyond ``websockets``), scores demo traffic through the API, clicks the
dashboard controls that need interaction (score form, agent cases, cheque upload,
drift check), and stitches the screenshots into an MP4 with ffmpeg.

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
from typing import Any

import websockets

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ("overview", "/", [], "Overview"),
    ("activity", "/activity", [], "Live activity"),
    ("score", "/score", ["Account takeover", "Score transaction"], "Score console"),
    ("batch", "/batch", ["Score 50 sample transactions"], "Batch scoring"),
    ("agents", "/agents", ["High-risk mule", "Run agents"], "Agent console"),
    ("cheque", "/cheque", [], "Cheque vision"),
    ("performance", "/performance", [], "Model performance"),
    ("performance-policy", "/performance", [], "Threshold policy"),
    ("drift", "/drift", ["Run drift check"], "Drift monitor"),
    ("adaptive", "/adaptive", [], "Adaptive learning"),
    ("federated", "/adaptive", ["Federated training"], "Federated training"),
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


def start_stack(api_url: str, app_url: str, reuse: bool = False) -> list[subprocess.Popen]:
    started: list[subprocess.Popen] = []
    env = dict(os.environ)

    api_running = http_ok(f"{api_url}/health")
    app_running = http_ok(f"{app_url}/_stcore/health")
    if not reuse and (api_running or app_running):
        busy = []
        if api_running:
            busy.append("8000 (API)")
        if app_running:
            busy.append("8501 (dashboard)")
        raise SystemExit(
            "A stack is already running on port "
            + " and ".join(busy)
            + ". Stop it so fresh code and the current virtual environment are "
            "captured, or pass --reuse-existing."
        )

    if not api_running:
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

    if not app_running:
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
    if not api_has_features(api_url):
        raise SystemExit(
            "The API on port 8000 is running an old build without the agent, "
            "cheque, or online-learning endpoints. Stop it and rerun so the "
            "current code is captured."
        )
    if not wait_for(f"{app_url}/_stcore/health"):
        raise SystemExit("The dashboard did not become ready on port 8501.")
    return started


def api_has_features(api_url: str) -> bool:
    for path in ("/agents/catalog", "/online/status", "/policy?mode=kaggle"):
        try:
            with urllib.request.urlopen(f"{api_url}{path}", timeout=3) as response:
                if response.status != 200:
                    return False
        except (urllib.error.URLError, OSError):
            return False
    return True


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


def warm_online_learner(api_url: str, rows: int = 3000, labels: int = 40) -> None:
    """Bootstrap the River learner and feed labelled outcomes for a live view."""
    import requests

    try:
        response = requests.post(
            f"{api_url}/online/bootstrap",
            json={"rows": rows, "reset": True},
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"online bootstrap failed: {exc}")
        return

    fraud = {
        "amount": 9_400.0, "tx_type": "transfer", "merchant_category": "crypto",
        "country": "RU", "device_risk": 0.92, "ip_risk": 0.88,
        "account_age_days": 4, "tx_count_24h": 34,
    }
    legitimate = {
        "amount": 85.0, "tx_type": "purchase", "merchant_category": "grocery",
        "country": "IN", "device_risk": 0.10, "ip_risk": 0.05,
        "account_age_days": 900, "tx_count_24h": 1,
    }
    scored = 0
    for index in range(labels):
        label = 1 if index % 5 == 0 else 0
        payload = {
            "transaction": fraud if label else legitimate,
            "label": label,
        }
        try:
            response = requests.post(
                f"{api_url}/online/learn", json=payload, timeout=30
            )
            scored += int(response.ok)
        except requests.RequestException:
            break
    print(f"bootstrapped {rows} rows and fed {scored} labelled online updates")


def make_demo_cheque(path: Path) -> Path:
    """Render a synthetic cheque with editing-software EXIF for the capture."""
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int) -> Any:
        for candidate in (
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default()

    width, height = 1600, 720
    image = Image.new("RGB", (width, height), "#F6F3E8")
    draw = ImageDraw.Draw(image)
    ink = "#1F2937"

    draw.rectangle((28, 28, width - 28, height - 28), outline="#B9C2CF", width=3)
    for offset in range(90, height - 90, 46):
        draw.line((40, offset, width - 40, offset), fill="#EDE8D6", width=1)

    draw.text((70, 66), "NORTHWIND TRUST BANK", font=font(42), fill="#0F4C81")
    draw.text((70, 122), "221B Harbour Street · Singapore 018989", font=font(20), fill="#475569")
    draw.text((width - 420, 70), "CHEQUE NO.  004821", font=font(24), fill=ink)
    draw.text((width - 420, 108), "DATE  20 Nov 2026", font=font(24), fill=ink)

    draw.text((70, 230), "PAY TO THE ORDER OF", font=font(20), fill="#64748B")
    draw.text((70, 258), "Acme Holdings Pte Ltd", font=font(38), fill=ink)
    draw.line((70, 310, 1010, 310), fill="#94A3B8", width=2)

    draw.text((70, 350), "AMOUNT IN WORDS", font=font(20), fill="#64748B")
    draw.text((70, 378), "Nine thousand four hundred and 00/100", font=font(28), fill=ink)
    draw.line((70, 424, 1010, 424), fill="#94A3B8", width=2)

    draw.rectangle((1080, 220, 1510, 300), outline="#94A3B8", width=2)
    draw.text((1100, 240), "$ 9,400.00", font=font(40), fill=ink)

    draw.text((70, 470), "MEMO", font=font(18), fill="#64748B")
    draw.text((70, 496), "Invoice INV-8842", font=font(24), fill=ink)
    draw.text((1080, 470), "AUTHORISED SIGNATURE", font=font(18), fill="#64748B")
    draw.line((1080, 540, 1510, 540), fill="#94A3B8", width=2)
    draw.text((1180, 505), "A. Tan", font=font(30), fill="#0F4C81")

    draw.text((70, 620), "MICR  021000021  4821  1042", font=font(30), fill=ink)

    exif = Image.Exif()
    exif[0x0131] = "Adobe Photoshop 25.0 (Windows)"
    exif[0x0132] = "2026:11:20 14:32:11"
    image.save(path, "JPEG", quality=90, exif=exif)
    return path


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
          const buttons = [...document.querySelectorAll('button, [role="tab"]')];
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

    async def scroll_to(
        self, text: str, offset: int = -24, timeout: float = 20.0
    ) -> bool:
        expression = """
        (() => {
          const all = [...document.querySelectorAll('body *')];
          const exact = all.find(
            (node) => node.children.length === 0 && node.textContent.trim() === %s
          );
          const matches = all.filter(
            (node) => node.textContent && node.textContent.includes(%s)
          );
          const target = exact || matches[matches.length - 1];
          if (!target) return false;
          target.scrollIntoView({block: 'start'});
          let scroller = target.parentElement;
          while (scroller) {
            const overflow = getComputedStyle(scroller).overflowY;
            if ((overflow === 'auto' || overflow === 'scroll')
                && scroller.scrollHeight > scroller.clientHeight) {
              break;
            }
            scroller = scroller.parentElement;
          }
          if (scroller) scroller.scrollTop += %d;
          else window.scrollBy(0, %d);
          return true;
        })()
        """ % (json.dumps(text), json.dumps(text), offset, offset)
        return bool(await self.wait_js(expression, timeout=timeout))

    async def wait_for_number(
        self, label_text: str, expected: float, timeout: float = 30.0
    ) -> bool:
        """Wait until a labelled number field shows a preset value."""
        expression = """
        (() => {
          const labels = [...document.querySelectorAll('label')];
          const label = labels.find(
            (node) => node.textContent.trim().startsWith(%s)
          );
          const field = label && label.closest('[data-testid="stNumberInput"]');
          const input = field && field.querySelector('input');
          return !!input && Math.abs(parseFloat(input.value) - %s) < 0.01;
        })()
        """ % (json.dumps(label_text), expected)
        return bool(await self.wait_js(expression, timeout=timeout))

    async def wait_for_amount(self, expected: float, timeout: float = 30.0) -> bool:
        """Wait until the score form shows the amount from the selected preset."""
        return await self.wait_for_number("Amount (USD)", expected, timeout=timeout)

    async def click_submit(self, label: str, timeout: float = 20.0) -> bool:
        """Click a form submit button, ignoring matching navigation labels."""
        expression = """
        (() => {
          const form = document.querySelector('[data-testid="stForm"]');
          const buttons = [...(form || document).querySelectorAll('button')];
          const target = buttons.find(
            (node) => node.textContent.replace(/\\s+/g, ' ').includes(%s)
          );
          if (!target || target.disabled) return false;
          target.click();
          return true;
        })()
        """ % json.dumps(label)
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = await self.cmd(
                "Runtime.evaluate", expression=expression, returnByValue=True
            )
            if result.get("result", {}).get("value"):
                return True
            await asyncio.sleep(0.5)
        return False

    async def button_enabled(self, label: str, timeout: float = 30.0) -> bool:
        expression = """
        (() => {
          const buttons = [...document.querySelectorAll('button')];
          const target = buttons.find(
            (node) => node.textContent.replace(/\\s+/g, ' ').includes(%s)
          );
          return !!target && !target.disabled;
        })()
        """ % json.dumps(label)
        return bool(
            await self.wait_js(expression, timeout=timeout)
        )

    async def upload_file(
        self, path: Path, selector: str = "input[type=file]", timeout: float = 30.0
    ) -> bool:
        await self.cmd("DOM.enable")
        deadline = time.time() + timeout
        while time.time() < deadline:
            document = await self.cmd("DOM.getDocument", depth=1)
            node = await self.cmd(
                "DOM.querySelector",
                nodeId=document["root"]["nodeId"],
                selector=selector,
            )
            if node.get("nodeId"):
                await self.cmd(
                    "DOM.setFileInputFiles", files=[str(path)], nodeId=node["nodeId"]
                )
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
        demo_cheque = make_demo_cheque(out_dir / "_cheque_sample.jpg")
        try:
            async with Chrome(page_target["webSocketDebuggerUrl"]) as chrome:
                for slug, path, actions, label in PAGES:
                    print(f"capturing {label} ...")

                    # Second view of Model performance focused on the
                    # threshold-policy cards below the fold.
                    if slug == "performance-policy":
                        await chrome.go(f"{base_url}/performance", settle=9.0)
                        await chrome.wait_js(
                            "document.body.innerText.includes('Threshold policy simulation')",
                            timeout=30,
                        )
                        await chrome.scroll_to(
                            "Threshold policy simulation", offset=-100
                        )
                        await asyncio.sleep(2.0)
                        target = out_dir / "performance-policy.png"
                        await chrome.screenshot(target)
                        saved.append(target)
                        continue

                    await chrome.go(f"{base_url}{path}", settle=4.0)
                    if slug == "cheque":
                        uploaded = await chrome.upload_file(demo_cheque)
                        print(f"  upload cheque: {'ok' if uploaded else 'input not found'}")
                        if uploaded:
                            enabled = await chrome.button_enabled("Analyse cheque")
                            if enabled:
                                await chrome.click("Analyse cheque")
                            await chrome.wait_js(
                                "document.body.innerText.includes('Reason codes')",
                                timeout=45,
                            )
                    else:
                        for action in actions:
                            # Presets rerun the app; submit only after the widgets
                            # show the preset values so the frames capture the
                            # high-risk cases instead of the initial defaults.
                            if slug in {"score", "agents"} and action in {
                                "Score transaction",
                                "Run agents",
                            }:
                                continue
                            clicked = await chrome.click(action)
                            print(
                                f"  {action!r}: "
                                f"{'clicked' if clicked else 'button not found'}"
                            )
                            await asyncio.sleep(2.0)
                        if slug == "score":
                            applied = await chrome.wait_for_amount(950.0)
                            # Let the preset rerun finish before submitting the form.
                            await asyncio.sleep(1.5)
                            submitted = await chrome.click_submit("Score transaction")
                            print(
                                f"  preset applied: {'ok' if applied else 'timeout'}; "
                                f"submit: {'clicked' if submitted else 'button not found'}"
                            )
                            await chrome.wait_js(
                                "document.body.innerText.includes('Analyst reason codes')",
                                timeout=45,
                            )
                        if slug == "agents":
                            applied = await chrome.wait_for_number("Inbound / 24h", 40.0)
                            await asyncio.sleep(1.0)
                            submitted = await chrome.click("Run agents")
                            print(
                                f"  preset applied: {'ok' if applied else 'timeout'}; "
                                f"submit: {'clicked' if submitted else 'button not found'}"
                            )

                    if slug == "agents":
                        await chrome.wait_js(
                            "document.body.innerText.includes('Fused risk')",
                            timeout=45,
                        )
                        await chrome.scroll_to("Fused risk", offset=-130)
                        await asyncio.sleep(1.5)
                    if slug == "federated":
                        await chrome.wait_js(
                            "document.body.innerText.includes('Privacy controls')",
                            timeout=30,
                        )
                        await chrome.scroll_to("Privacy controls")
                        await asyncio.sleep(1.5)

                    # Charts render after Streamlit finishes the run; wait for an
                    # ECharts canvas before taking the frame where one is expected.
                    if slug in {"overview", "activity", "score", "batch", "performance", "drift"}:
                        await chrome.wait_js(
                            "document.querySelectorAll('canvas').length > 0", timeout=25
                        )
                    await asyncio.sleep(2.0)
                    target = out_dir / f"{slug}.png"
                    await chrome.screenshot(target)
                    saved.append(target)
        finally:
            demo_cheque.unlink(missing_ok=True)
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
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Capture against an already running API/dashboard instead of failing",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    chrome_path = find_chrome()
    started = start_stack(args.api_url, args.app_url, reuse=args.reuse_existing)
    try:
        if not args.skip_traffic:
            generate_traffic(args.api_url, args.traffic)
            warm_online_learner(args.api_url)
        frames = asyncio.run(capture_pages(args.app_url, out_dir, chrome_path))
    finally:
        stop_stack(started)

    for frame in frames:
        print(f"wrote {frame.relative_to(ROOT)} ({frame.stat().st_size // 1024} KB)")
    if build_video(frames, ROOT / args.video):
        print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
