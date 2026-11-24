"""Add a narration track to the dashboard demo video.

Synthesises one clip per dashboard scene with the edge-tts neural voice, then
rebuilds ``docs/demo.mp4`` from the screenshots in ``docs/screenshots`` so each
scene lasts long enough for its line. Safe to re-run after ``capture_media.py``.

Usage::

    python scripts/voiceover.py                  # build docs/demo.mp4
    python scripts/voiceover.py --voice en-GB-SoniaNeural --rate -5
    python scripts/voiceover.py --keep-audio     # keep docs/audio/*.mp3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
AUDIO = ROOT / "docs" / "audio"

# Order matches the capture order in capture_media.py: the six pages, then the
# cropped policy section of Model performance.
SCENES: list[tuple[str, str]] = [
    (
        "overview",
        "SentinelGuard flags fraudulent card transactions in real time, "
        "from a scoring API to a live dashboard.",
    ),
    (
        "score",
        "The score console turns one transaction into a decision, "
        "with SHAP drivers and an analyst-ready reason code.",
    ),
    (
        "activity",
        "Live activity streams every decision as it lands, "
        "so blocks, reviews and alerts stay visible.",
    ),
    (
        "batch",
        "Batch scoring tags a whole file in one call, "
        "useful for backfills, portfolio sweeps and investigations.",
    ),
    (
        "performance",
        "Model performance reports precision, recall, Brier score, "
        "and calibration against simple baselines.",
    ),
    (
        "drift",
        "The drift monitor compares live traffic with the training window, "
        "using population stability per feature to raise alerts.",
    ),
    (
        "performance-policy",
        "And a threshold policy simulation turns analyst review capacity "
        "into a cost-optimal operating point.",
    ),
]

LEAD = 0.45      # silence before a line starts inside its scene
TAIL = 0.75      # silence after the line before the next crossfade
MIN_SCENE = 4.0  # never hold a frame shorter than this
FADE = 0.6       # crossfade between scenes
WIDTH, HEIGHT = 1600, 1000


def tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found on PATH")
    return found


def duration(path: Path) -> float:
    result = subprocess.run(
        [
            tool("ffprobe"), "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def synthesize(text: str, target: Path, voice: str, rate: str) -> Path:
    """Render ``text`` to a 48 kHz wav with the edge-tts neural voice."""
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.with_suffix(".mp3")
    command = [
        sys.executable, "-m", "edge_tts",
        "--voice", voice,
        "--rate", rate,
        "--text", text,
        "--write-media", str(raw),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not raw.exists() or raw.stat().st_size == 0:
        print(result.stderr[-800:] or "edge-tts produced no audio")
        raise SystemExit(
            "voice synthesis failed; check the network or pass --voice/--rate"
        )

    # edge-tts pads each clip with leading and trailing silence; strip it so the
    # scene timing is driven by speech, not padding.
    trim = subprocess.run(
        [
            tool("ffmpeg"), "-y", "-loglevel", "error", "-i", str(raw),
            "-af",
            "silenceremove=start_periods=1:start_threshold=-45dB:"
            "stop_periods=-1:stop_threshold=-45dB:stop_silence=0.12",
            "-ar", "48000", "-ac", "1", str(target),
        ],
        capture_output=True,
        text=True,
    )
    raw.unlink(missing_ok=True)
    if trim.returncode != 0:
        print(trim.stderr[-800:])
        raise SystemExit("ffmpeg failed to trim the narration clip")
    return target


def build(frames: list[Path], clips: list[Path], output: Path) -> None:
    ffmpeg = tool("ffmpeg")
    scene = [max(MIN_SCENE, LEAD + duration(c) + TAIL) for c in clips]

    starts: list[float] = []
    cursor = 0.0
    for length in scene:
        starts.append(cursor)
        cursor += length - FADE
    total = cursor + FADE

    command = [ffmpeg, "-y", "-loglevel", "error"]
    for frame, length in zip(frames, scene):
        command += ["-loop", "1", "-framerate", "30", "-t", f"{length:.3f}", "-i", str(frame)]
    for clip in clips:
        command += ["-i", str(clip)]

    filters: list[str] = []
    for index in range(len(frames)):
        filters.append(
            f"[{index}:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"settb=AVTB,fps=30[v{index}]"
        )

    previous = "v0"
    for index in range(1, len(frames)):
        label = f"x{index}"
        filters.append(
            f"[{previous}][v{index}]xfade=transition=fade:duration={FADE}:"
            f"offset={starts[index]:.3f}[{label}]"
        )
        previous = label
    filters.append(f"[{previous}]format=yuv420p[v]")

    audio_labels = []
    for index in range(len(clips)):
        delay = int(round((starts[index] + LEAD) * 1000))
        label = f"a{index}"
        filters.append(
            f"[{len(frames) + index}:a]aresample=48000,"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"adelay={delay}:all=1[{label}]"
        )
        audio_labels.append(f"[{label}]")
    filters.append(
        "".join(audio_labels)
        + f"amix=inputs={len(clips)}:duration=longest:normalize=0,"
        # Pad to the full timeline: the mix stops at the last word, and
        # -shortest would otherwise cut the closing scene short.
        f"apad=whole_dur={total:.3f},"
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        f"afade=t=out:st={max(0.0, total - 1.2):.3f}:d=1.2,"
        "atrim=0:%s[a]" % f"{total:.3f}"
    )

    command += [
        "-filter_complex", ";".join(filters),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-shortest", "-movflags", "+faststart",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise SystemExit("ffmpeg failed to build the narrated video")
    print(f"wrote {output.relative_to(ROOT)} ({total:.1f}s, {output.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="en-US-AriaNeural")
    parser.add_argument("--rate", default="+6%", help="edge-tts rate, e.g. -5%%")
    parser.add_argument("--video", default="docs/demo.mp4")
    parser.add_argument("--keep-audio", action="store_true")
    args = parser.parse_args()

    frames = [SHOTS / f"{slug}.png" for slug, _ in SCENES]
    missing = [str(p.relative_to(ROOT)) for p in frames if not p.exists()]
    if missing:
        raise SystemExit("missing screenshots: " + ", ".join(missing))

    clips: list[Path] = []
    for slug, text in SCENES:
        clip = synthesize(text, AUDIO / f"{slug}.wav", args.voice, args.rate)
        print(f"{slug}: {duration(clip):.2f}s")
        clips.append(clip)

    build(frames, clips, ROOT / args.video)
    if not args.keep_audio:
        shutil.rmtree(AUDIO, ignore_errors=True)


if __name__ == "__main__":
    main()
