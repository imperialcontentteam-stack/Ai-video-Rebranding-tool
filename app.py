"""
Video Rebranding Tool v17.1 Cloud Safe
Uses the EXACT uploaded Intro.mp4 — only changes the course name, unit number and chapter name.
All animations, logo, 3D shapes and audio are preserved perfectly.

v17.1 cloud-safety changes:
 0. Streamlit Community Cloud-safe FFmpeg execution: one encode at a time,
    one encoder/filter thread, deferred downloads and streamed upload copies.

v17 changes:
 1. Full course name always displayed — text wraps up to 4 lines and auto-shrinks;
    never truncated with "…".
 2. Processing queue — add multiple videos, see pending / processing / completed
    status with a live progress bar per job.
 3. Direct MP4 download only (no ZIP).
 4. Redesigned modern UI.
 5. Faster processing — cached outro & intro clips, parallel intro generation,
    real ffmpeg progress reporting.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
#  Video constants
# ─────────────────────────────────────────────
TARGET_W   = 1920
TARGET_H   = 1080
TARGET_FPS = 30
AUDIO_RATE = 48000

# Community Cloud has limited shared CPU/RAM. Using all available FFmpeg
# threads and running two 1080p encodes at once can cause the app process to
# be killed, which appears in the logs as a /healthz connection reset.
# Set FFMPEG_THREADS=2 or higher only on a larger private server.
try:
    FFMPEG_THREADS = max(1, int(os.environ.get("FFMPEG_THREADS", "1")))
except ValueError:
    FFMPEG_THREADS = 1

# ─────────────────────────────────────────────
#  Pixel-measured regions in Intro.mp4
#  (measured directly from Intro.mp4 at t=8.5s:
#   line1 rows 380-436, line2 rows 478-565,
#   pill fill rows 622-708 / cols 642-1276)
# ─────────────────────────────────────────────
# The erase band is enlarged upward vs v16 so long course names can wrap to
# 3-4 lines without ever colliding with the pill below (pill erase starts 595).
TITLE_ERASE_Y = 295
TITLE_ERASE_H = 295   # covers Y 295-590

# Text must stay inside this safe zone (10px padding inside the erase band).
TITLE_SAFE_TOP    = 305
TITLE_SAFE_BOTTOM = 582
TITLE_CENTER_Y    = 460   # preferred vertical midpoint of the text block
TITLE_MAX_LINES   = 4     # never truncate — wrap up to 4 lines, shrinking font

# Original pill: Y=622-708, X=642-1276, centre X≈959
PILL_ERASE_Y  = 595
PILL_ERASE_H  = 145   # covers Y 595-740
PILL_CENTER_Y = 665
PILL_MIN_W    = 580

# ─────────────────────────────────────────────
#  Entrance-animation timing (measured from Intro.mp4)
# ─────────────────────────────────────────────
TITLE_FADE_START = 0.90
TITLE_FADE_DUR   = 0.30
PILL_FADE_START  = 1.90
PILL_FADE_DUR    = 0.10

TITLE_BG_HEX  = "9B5EE1"
PILL_BG_HEX   = "945BE1"

SLC_LOGO_BOX  = (1722, 966, 106, 60)

# ─────────────────────────────────────────────
#  Brand config
# ─────────────────────────────────────────────
BRANDS = {
    "Aspirex": {
        "prefix":         "aspirex",
        "logo":           "aspirex_logo.png",
        "bg_ffmpeg":      "0x9051D9",
        "title_color":    (255, 255, 255),
        "pill_bg":        (255, 255, 255),
        "pill_text":      (109,  50, 181),
    },
    "GEL": {
        "prefix":         "gel",
        "logo":           "gel_logo.png",
        "bg_ffmpeg":      "0xF7FBFF",
        "title_color":    ( 26,  46,  74),
        "pill_bg":        (255, 255, 255),
        "pill_text":      ( 26,  46,  74),
    },
}

SPEED_PROFILES = {
    "High quality":       {"preset": "fast",      "crf": 18},
    "Fast (recommended)": {"preset": "veryfast",  "crf": 24},
    "Very fast":          {"preset": "ultrafast", "crf": 26},
}
DEFAULT_SPEED = "Fast (recommended)"


# ─────────────────────────────────────────────
#  FFmpeg helpers
# ─────────────────────────────────────────────
_FFMPEG: Optional[str] = None

def get_ffmpeg() -> str:
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    for candidate in [
        os.environ.get("FFMPEG_BINARY", ""),
        shutil.which("ffmpeg") or "",
    ]:
        if candidate and _test_exe(candidate):
            _FFMPEG = candidate
            return _FFMPEG
    try:
        import imageio_ffmpeg
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if _test_exe(candidate):
            _FFMPEG = candidate
            return _FFMPEG
    except Exception:
        pass
    raise RuntimeError("FFmpeg not found. Run: pip install -r requirements.txt")

def _test_exe(path: str) -> bool:
    try:
        return subprocess.run([path, "-version"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False

def ff(cmd: list[str]) -> list[str]:
    if cmd and cmd[0] == "ffmpeg":
        return [
            get_ffmpeg(),
            "-filter_threads", str(FFMPEG_THREADS),
            "-filter_complex_threads", str(FFMPEG_THREADS),
            *cmd[1:],
        ]
    return cmd

def run(cmd: list[str], label: str) -> None:
    r = subprocess.run(ff(cmd), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{label} failed.\n\n{(r.stderr or '')[-4000:]}")

def run_progress(cmd: list[str], label: str, total_dur: float,
                 cb: Optional[Callable[[float], None]] = None) -> None:
    """Run an ffmpeg command, streaming real encode progress to `cb` (0..1)."""
    if not cb or total_dur <= 0:
        run(cmd, label)
        return
    full = ff(cmd)
    full = [full[0], "-progress", "pipe:1", "-nostats", *full[1:]]

    # Do not leave stderr unread in a PIPE while FFmpeg runs. A full pipe can
    # block FFmpeg indefinitely. A temporary disk-backed log also keeps long
    # encodes from accumulating log output in RAM.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as err_file:
        p = subprocess.Popen(
            full,
            stdout=subprocess.PIPE,
            stderr=err_file,
            text=True,
            bufsize=1,
        )
        try:
            assert p.stdout is not None
            for line in p.stdout:
                line = line.strip()
                if line.startswith("out_time_ms=") or line.startswith("out_time_us="):
                    val = line.split("=", 1)[1]
                    if val.isdigit():
                        try:
                            cb(min(int(val) / 1_000_000.0 / total_dur, 1.0))
                        except Exception:
                            pass
        finally:
            if p.stdout is not None:
                p.stdout.close()
            p.wait()
            err_file.seek(0)
            err = err_file.read()

    if p.returncode != 0:
        raise RuntimeError(f"{label} failed.\n\n{(err or '')[-4000:]}")

def probe_text(path: Path) -> str:
    r = subprocess.run(ff(["ffmpeg", "-hide_banner", "-i", str(path)]),
                       capture_output=True, text=True)
    return (r.stderr or "") + "\n" + (r.stdout or "")

def get_duration(path: Path) -> float:
    txt = probe_text(path)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if not m:
        return 0.0
    return int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))

def has_audio(path: Path) -> bool:
    return bool(re.search(r"Stream #.*Audio:", probe_text(path)))

def get_media_info(path: Path) -> tuple[float, int, int, bool]:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    fc  = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = fc/fps if fps > 0 and fc > 0 else 0.0
    cap.release()
    if dur <= 0:
        dur = get_duration(path)
    return dur, w, h, has_audio(path)

def enc_args(speed: str, crf: Optional[int] = None) -> list[str]:
    p = SPEED_PROFILES.get(speed, SPEED_PROFILES[DEFAULT_SPEED])
    return ["-c:v", "libx264", "-preset", p["preset"], "-crf", str(crf or p["crf"]),
            "-threads", str(FFMPEG_THREADS)]

def scale_filter() -> str:
    return (f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={TARGET_FPS}")


# ─────────────────────────────────────────────
#  Font helpers
# ─────────────────────────────────────────────
def find_font(bold: bool = True) -> Optional[str]:
    base = Path(__file__).parent
    candidates = [
        str(base / ("Poppins-Bold.ttf" if bold else "Poppins-Regular.ttf")),
        os.environ.get("POPPINS_BOLD_FONT", "") if bold else "",
        "/usr/share/fonts/truetype/poppins/Poppins-Bold.ttf" if bold else "/usr/share/fonts/truetype/poppins/Poppins-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf" if bold else "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/Poppins-Bold.ttf" if bold else "C:/Windows/Fonts/Poppins-Regular.ttf",
        "/Library/Fonts/Poppins-Bold.ttf" if bold else "/Library/Fonts/Poppins-Regular.ttf",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None

_FONT_FALLBACK_WARNED = False

def make_font(size: int, bold: bool = True):
    global _FONT_FALLBACK_WARNED
    fp = find_font(bold)
    if fp:
        return ImageFont.truetype(fp, size)
    if not _FONT_FALLBACK_WARNED:
        _FONT_FALLBACK_WARNED = True
        warnings.warn(
            "Poppins-Bold.ttf was not found next to app.py and no system font "
            "was found either. Falling back to a small placeholder font; title "
            "and pill text may look wrong.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ImageFont.load_default()

def tsz(draw, text: str, font) -> Tuple[int, int]:
    bb = draw.textbbox((0, 0), text or " ", font=font)
    return bb[2]-bb[0], bb[3]-bb[1]


# ─────────────────────────────────────────────
#  Title wrapping — NEVER truncates.
#  Words wrap onto new lines; over-long single words are hard-broken.
# ─────────────────────────────────────────────
def wrap_full(draw, text: str, font, max_w: int) -> list[str]:
    """Greedy word-wrap that keeps every character of the text.
    A single word wider than max_w is split across lines (no ellipsis)."""
    text = " ".join((text or "").strip().upper().split())
    if not text:
        return [" "]
    lines: list[str] = []
    cur = ""
    for word in text.split(" "):
        trial = word if not cur else f"{cur} {word}"
        if tsz(draw, trial, font)[0] <= max_w:
            cur = trial
            continue
        if cur:
            lines.append(cur)
            cur = ""
        # word alone still too wide → hard-break it, keeping every character
        while tsz(draw, word, font)[0] > max_w and len(word) > 1:
            cut = len(word)
            while cut > 1 and tsz(draw, word[:cut], font)[0] > max_w:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        cur = word
    if cur:
        lines.append(cur)
    return lines or [" "]


def fit_title(draw, text: str, max_w: int):
    """Pick the largest font size at which the FULL text fits within max_w and
    within the vertical safe zone using at most TITLE_MAX_LINES lines.
    The text is never shortened — only wrapped and scaled."""
    avail_h = TITLE_SAFE_BOTTOM - TITLE_SAFE_TOP
    best = None
    for size in range(82, 23, -2):
        font = make_font(size)
        lines = wrap_full(draw, text, font, max_w)
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        if (len(lines) <= TITLE_MAX_LINES
                and len(lines) * line_h <= avail_h
                and all(tsz(draw, l, font)[0] <= max_w for l in lines)):
            return font, lines
        if best is None:
            best = (font, lines)
    # Fallback: smallest size, full wrap, still no truncation.
    font = make_font(24)
    return font, wrap_full(draw, text, font, max_w)


# ─────────────────────────────────────────────
#  Extract clean background rows from intro video
# ─────────────────────────────────────────────
_CLEAN_BG_CACHE: dict[str, Optional[np.ndarray]] = {}

def _extract_clean_bg(intro: Path) -> Optional[np.ndarray]:
    """Sample the intro at t=280ms (before any text animates in) to get clean
    background pixels for the erase zones. Cached in-process for speed."""
    key = str(intro)
    if key in _CLEAN_BG_CACHE:
        return _CLEAN_BG_CACHE[key]
    cap = cv2.VideoCapture(str(intro))
    cap.set(cv2.CAP_PROP_POS_MSEC, 280)
    ok, frame = cap.read()
    cap.release()
    arr = None
    if ok:
        arr = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).astype(np.uint8)
    _CLEAN_BG_CACHE[key] = arr
    return arr


# ─────────────────────────────────────────────
#  Core: build overlay layers
# ─────────────────────────────────────────────
def _clean_bg_layer(intro_path: Optional[Path], erase_y: int, erase_h: int) -> Image.Image:
    arr = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
    clean_bg = _extract_clean_bg(intro_path) if intro_path else None
    if clean_bg is not None:
        arr[erase_y:erase_y+erase_h, :, :3] = clean_bg[erase_y:erase_y+erase_h, :, :3]
        arr[erase_y:erase_y+erase_h, :, 3]  = 255
    else:
        bg_hex = TITLE_BG_HEX if erase_y == TITLE_ERASE_Y else PILL_BG_HEX
        bg = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
        arr[erase_y:erase_y+erase_h, :, :3] = bg
        arr[erase_y:erase_y+erase_h, :, 3]  = 255
    return Image.fromarray(arr, "RGBA")


def build_title_layers(brand: str, course: str,
                        intro_path: Optional[Path] = None) -> Tuple[Image.Image, Image.Image]:
    tc = BRANDS[brand]["title_color"]
    course_text = (course or "COURSE NAME").strip().upper()

    erase = _clean_bg_layer(intro_path, TITLE_ERASE_Y, TITLE_ERASE_H)
    text_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer, "RGBA")

    max_title_w = 1580
    t_font, t_lines = fit_title(draw, course_text, max_title_w)
    ascent, descent = t_font.getmetrics()
    line_h = ascent + descent
    total_h = len(t_lines) * line_h
    # Centre on the preferred midpoint but clamp inside the safe zone so long
    # titles never spill into the pill area or above the erase band.
    title_top = TITLE_CENTER_Y - total_h // 2
    title_top = max(TITLE_SAFE_TOP, min(title_top, TITLE_SAFE_BOTTOM - total_h))

    for i, line in enumerate(t_lines):
        w, _ = tsz(draw, line, t_font)
        x = TARGET_W // 2 - w // 2
        y = title_top + i * line_h
        draw.text((x + 3, y + 5), line, font=t_font, fill=(0, 0, 0, 60))
        draw.text((x, y),         line, font=t_font, fill=(*tc, 255))

    return erase, text_layer


def build_pill_layers(brand: str, unit_no: str, chapter: str,
                       intro_path: Optional[Path] = None) -> Tuple[Image.Image, Image.Image]:
    pb = BRANDS[brand]["pill_bg"]
    pt = BRANDS[brand]["pill_text"]
    unit_text    = (unit_no or "UNIT 01").strip().upper()
    chapter_text = (chapter or "CHAPTER 01").strip().upper()
    pill_line    = f"{unit_text} -  {chapter_text}"

    erase = _clean_bg_layer(intro_path, PILL_ERASE_Y, PILL_ERASE_H)
    text_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer, "RGBA")

    # Shrink pill font if the unit+chapter line is extremely long, so it also
    # never truncates or overflows the frame.
    p_size = 46
    while p_size > 24:
        p_font = make_font(p_size)
        bbox = draw.textbbox((0, 0), pill_line, font=p_font)
        if bbox[2] - bbox[0] + 160 <= 1700:
            break
        p_size -= 2
    p_font = make_font(p_size)
    bbox = draw.textbbox((0, 0), pill_line, font=p_font)
    top_offset = bbox[1]
    uw, uh = bbox[2]-bbox[0], bbox[3]-bbox[1]
    pw = min(max(uw + 160, PILL_MIN_W), 1700)
    ph = 88
    px = TARGET_W // 2 - pw // 2
    py = PILL_CENTER_Y - ph // 2

    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=44, fill=(*pb, 255))
    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=44, outline=(*pt, 80), width=5)
    y_drawn = py + (ph - uh) // 2 - top_offset
    draw.text((TARGET_W // 2 - uw // 2, y_drawn),
              pill_line, font=p_font, fill=(*pt, 255))

    return erase, text_layer


def build_overlay_png(out_png: Path, brand: str, course: str, unit_no: str,
                      chapter: str, intro_path: Optional[Path] = None) -> None:
    """Static combined overlay — used for the still preview image."""
    te, tt = build_title_layers(brand, course, intro_path)
    pe, pt_ = build_pill_layers(brand, unit_no, chapter, intro_path)
    img = Image.alpha_composite(Image.alpha_composite(te, tt),
                                Image.alpha_composite(pe, pt_))
    img.save(out_png)


# ─────────────────────────────────────────────
#  Preview (static still from the video)
# ─────────────────────────────────────────────
def make_preview(brand: str, course: str, unit_no: str, chapter: str) -> Optional[Image.Image]:
    intro = _intro_path(brand)
    if not intro:
        return None
    cap = cv2.VideoCapture(str(intro))
    cap.set(cv2.CAP_PROP_POS_MSEC, 280)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
        (TARGET_W, TARGET_H), Image.Resampling.LANCZOS
    ).convert("RGBA")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = Path(f.name)
    try:
        build_overlay_png(tmp, brand, course, unit_no, chapter, intro_path=intro)
        overlay = Image.open(tmp).convert("RGBA")
        return Image.alpha_composite(base, overlay).convert("RGB")
    finally:
        tmp.unlink(missing_ok=True)


# ─────────────────────────────────────────────
#  Generate the intro clip with new text
# ─────────────────────────────────────────────
ERASE_SNAP_DUR = 0.08
ERASE_LEAD     = 0.05

def generate_intro_clip(out_path: Path, brand: str, course: str, unit_no: str,
                        chapter: str, speed: str = DEFAULT_SPEED) -> None:
    intro = _intro_path(brand)
    if not intro:
        raise FileNotFoundError(
            f"Intro video not found: expected '{BRANDS[brand]['prefix']}_intro.mp4' "
            "beside app.py"
        )

    dur = get_duration(intro)
    audio = has_audio(intro)

    paths = {
        "title_erase": out_path.with_suffix(".terase.png"),
        "title_text":  out_path.with_suffix(".ttext.png"),
        "pill_erase":  out_path.with_suffix(".perase.png"),
        "pill_text":   out_path.with_suffix(".ptext.png"),
    }
    try:
        te, tt = build_title_layers(brand, course, intro_path=intro)
        pe, pt_ = build_pill_layers(brand, unit_no, chapter, intro_path=intro)
        te.save(paths["title_erase"]);  tt.save(paths["title_text"])
        pe.save(paths["pill_erase"]);   pt_.save(paths["pill_text"])

        title_erase_start = max(TITLE_FADE_START - ERASE_LEAD, 0.0)
        pill_erase_start  = max(PILL_FADE_START - ERASE_LEAD, 0.0)

        vf = (
            f"[0:v]{scale_filter()}[base];"
            f"[1:v]format=rgba,fade=t=in:st={title_erase_start}:d={ERASE_SNAP_DUR}:alpha=1[te];"
            f"[2:v]format=rgba,fade=t=in:st={TITLE_FADE_START}:d={TITLE_FADE_DUR}:alpha=1[tt];"
            f"[3:v]format=rgba,fade=t=in:st={pill_erase_start}:d={ERASE_SNAP_DUR}:alpha=1[pe];"
            f"[4:v]format=rgba,fade=t=in:st={PILL_FADE_START}:d={PILL_FADE_DUR}:alpha=1[pt];"
            f"[base][te]overlay=0:0:format=auto[s1];"
            f"[s1][tt]overlay=0:0:format=auto[s2];"
            f"[s2][pe]overlay=0:0:format=auto[s3];"
            f"[s3][pt]overlay=0:0:format=auto,format=yuv420p[v]"
        )

        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", str(intro),
            "-loop", "1", "-i", str(paths["title_erase"]),
            "-loop", "1", "-i", str(paths["title_text"]),
            "-loop", "1", "-i", str(paths["pill_erase"]),
            "-loop", "1", "-i", str(paths["pill_text"]),
            "-filter_complex", vf,
            "-map", "[v]",
        ]
        if audio:
            cmd += ["-map", "0:a:0"]
        else:
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}",
                    "-i", f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
                    "-map", "5:a:0"]

        cmd += [
            *enc_args(speed),
            "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-ac", "2",
            "-movflags", "+faststart",
            "-t", f"{dur:.3f}",
            str(out_path),
        ]
        run(cmd, "Generate intro with text overlay")
    finally:
        for p in paths.values():
            p.unlink(missing_ok=True)


# ─────────────────────────────────────────────
#  Content segment: trim + replace SLC logo
# ─────────────────────────────────────────────
def process_content(src: Path, out: Path, media_info, trim_start: float,
                    trim_end: float, logo: Path, speed: str,
                    cb: Optional[Callable[[float], None]] = None) -> None:
    dur, _, _, audio = media_info
    seg = trim_end - trim_start
    if seg <= 0:
        raise ValueError("Trim settings leave no content.")
    x, y, w, h = SLC_LOGO_BOX
    cover = "0xFFFFFF@1.0"
    fc = (
        f"[0:v]{scale_filter()},"
        f"drawbox=x={x}:y={y}:w={w}:h={h}:color={cover}:t=fill,format=rgba[base];"
        f"[1:v]format=rgba,scale={w}:{h}:flags=lanczos[logo];"
        f"[base][logo]overlay=x={x}:y={y}:format=auto,format=yuv420p[v]"
    )
    cmd = ["ffmpeg", "-y", "-hide_banner",
           "-accurate_seek", "-ss", f"{trim_start:.3f}", "-t", f"{seg:.3f}",
           "-i", str(src), "-loop", "1", "-i", str(logo)]
    if audio:
        cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "0:a:0",
                *enc_args(speed), "-c:a", "aac", "-b:a", "192k",
                "-ar", str(AUDIO_RATE), "-ac", "2",
                "-movflags", "+faststart", "-shortest", str(out)]
    else:
        cmd += ["-f", "lavfi", "-t", f"{seg:.3f}",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
                "-filter_complex", fc, "-map", "[v]", "-map", "2:a:0",
                *enc_args(speed), "-c:a", "aac", "-b:a", "192k",
                "-ar", str(AUDIO_RATE), "-ac", "2",
                "-movflags", "+faststart", "-shortest", str(out)]
    run_progress(cmd, "Process content segment", seg, cb)


def normalize_clip(src: Path, out: Path, speed: str) -> None:
    _, _, _, audio = get_media_info(src)
    dur = get_duration(src)
    vf = f"[0:v]{scale_filter()},format=yuv420p[v]"
    if audio:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-i", str(src),
               "-filter_complex", vf, "-map", "[v]", "-map", "0:a:0",
               *enc_args(speed), "-c:a", "aac", "-b:a", "192k",
               "-ar", str(AUDIO_RATE), "-ac", "2",
               "-movflags", "+faststart", "-shortest", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-i", str(src),
               "-f", "lavfi", "-t", f"{dur:.3f}",
               "-i", f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
               "-filter_complex", vf, "-map", "[v]", "-map", "1:a:0",
               *enc_args(speed), "-c:a", "aac", "-b:a", "192k",
               "-ar", str(AUDIO_RATE), "-ac", "2",
               "-movflags", "+faststart", "-shortest", str(out)]
    run(cmd, "Normalize clip")


def logo_slate(logo: Path, out: Path, brand: str, dur: float, speed: str) -> None:
    bg = BRANDS[brand]["bg_ffmpeg"]
    cmd = ["ffmpeg", "-y", "-hide_banner",
           "-f", "lavfi", "-t", f"{dur:.3f}",
           "-i", f"color=c={bg}:s={TARGET_W}x{TARGET_H}:r={TARGET_FPS}",
           "-loop", "1", "-t", f"{dur:.3f}", "-i", str(logo),
           "-f", "lavfi", "-t", f"{dur:.3f}",
           "-i", f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
           "-filter_complex",
           "[1:v]format=rgba,scale=620:-1[l];[0:v][l]overlay=x=(W-w)/2:y=(H-h)/2:format=auto,format=yuv420p[v]",
           "-map", "[v]", "-map", "2:a:0",
           *enc_args(speed), "-c:a", "aac", "-b:a", "192k",
           "-ar", str(AUDIO_RATE), "-ac", "2",
           "-movflags", "+faststart", "-shortest", str(out)]
    run(cmd, "Logo slate")


def concat_clips(a: Path, b: Path, c: Path, out: Path, speed: str) -> None:
    txt = out.with_suffix(".concat.txt")
    txt.write_text(f"file '{a.as_posix()}'\nfile '{b.as_posix()}'\nfile '{c.as_posix()}'\n")
    try:
        r = subprocess.run(ff(["ffmpeg", "-y", "-hide_banner",
                                "-f", "concat", "-safe", "0", "-i", str(txt),
                                "-c", "copy", "-movflags", "+faststart", str(out)]),
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
        run(["ffmpeg", "-y", "-hide_banner",
             "-i", str(a), "-i", str(b), "-i", str(c),
             "-filter_complex", "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]",
             "-map", "[v]", "-map", "[a]",
             *enc_args(speed), "-c:a", "aac", "-b:a", "192k",
             "-ar", str(AUDIO_RATE), "-ac", "2",
             "-movflags", "+faststart", str(out)], "Final concat")
    finally:
        txt.unlink(missing_ok=True)


# ─────────────────────────────────────────────
#  Path helpers
# ─────────────────────────────────────────────
def _base() -> Path:
    return Path(__file__).resolve().parent

def _intro_path(brand: str) -> Optional[Path]:
    generic = _base() / "Intro.mp4"
    if generic.exists():
        return generic
    p = _base() / f"{BRANDS[brand]['prefix']}_intro.mp4"
    return p if p.exists() else None

def _outro_path(brand: str) -> Optional[Path]:
    p = _base() / f"{BRANDS[brand]['prefix']}_outro.mp4"
    return p if p.exists() else None

def _logo_path(brand: str) -> Optional[Path]:
    p = _base() / BRANDS[brand]["logo"]
    return p if p.exists() else None

def safe_name(v: str, stem: str, brand: str) -> str:
    raw = (v or "").strip() or f"{stem}_{brand.lower()}_rebranded.mp4"
    raw = re.sub(r"[^A-Za-z0-9._ -]+", "_", raw.replace("\\","_").replace("/","_")).strip(" ._")
    return (raw or f"{stem}_{brand.lower()}_rebranded") + ("" if raw.lower().endswith(".mp4") else ".mp4")

def fmt_time(s: float) -> str:
    s = max(float(s), 0)
    h, m = int(s)//3600, (int(s)%3600)//60
    sec, ms = int(s)%60, int(round((s-int(s))*1000))
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}" if h else f"{m:02d}:{sec:02d}.{ms:03d}"


# ─────────────────────────────────────────────
#  Caching (speed optimization)
# ─────────────────────────────────────────────
def _cache_dir(out_root: Path) -> Path:
    """Return the cache folder without reading Streamlit session state.

    Passing ordinary Path objects keeps this helper safe when it is called by
    background worker threads, which do not own Streamlit's script context.
    """
    d = Path(out_root) / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def upload_cache_path(uploaded, cache_dir: Path) -> Path:
    key = hashlib.sha1(f"{uploaded.name}|{uploaded.size}".encode()).hexdigest()[:16]
    p = Path(cache_dir) / f"{key}.mp4"
    if not p.exists():
        uploaded.seek(0)
        with p.open("wb") as dst:
            shutil.copyfileobj(uploaded, dst, length=1024 * 1024)
        uploaded.seek(0)
    return p

def cached_outro(brand: str, speed: str, logo: Path, cache_dir: Path) -> Path:
    """Build the normalized outro once per (brand, speed) and reuse it across
    every job in the queue — previously it was rebuilt for every video."""
    slug = re.sub(r"[^a-z0-9]+", "", speed.lower())
    p = Path(cache_dir) / f"outro_{BRANDS[brand]['prefix']}_{slug}.mp4"
    if p.exists() and p.stat().st_size > 0:
        return p
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "outro_raw.mp4"
        op = _outro_path(brand)
        if op:
            shutil.copy(op, raw)
        else:
            logo_slate(logo, raw, brand, 3.0, speed)
        normalize_clip(raw, p, speed)
    return p

def cached_intro(brand: str, course: str, unit: str, chapter: str, speed: str,
                 cache_dir: Path) -> Path:
    """Generate the branded intro once per unique text/brand/speed combination.
    Repeated runs (or many chapters of the same course) reuse it instantly."""
    key = hashlib.sha1(f"{brand}|{course.strip().upper()}|{unit.strip().upper()}|"
                       f"{chapter.strip().upper()}|{speed}".encode()).hexdigest()[:16]
    p = Path(cache_dir) / f"intro_{key}.mp4"
    if p.exists() and p.stat().st_size > 0:
        return p
    tmp = p.with_suffix(".building.mp4")
    generate_intro_clip(tmp, brand, course, unit, chapter, speed)
    tmp.replace(p)
    return p


def extract_frame_at(path: Path, t: float) -> Optional[Image.Image]:
    t = max(t, 0.0)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def trim_preview(src: Path, trim_start: float, outro_remove: float):
    dur, _, _, _ = get_media_info(src)
    t0 = float(trim_start)
    t1 = max(dur - float(outro_remove), 0.0)
    first = extract_frame_at(src, t0)
    last  = extract_frame_at(src, max(t1 - 0.05, 0.0))
    return first, last, dur


# ─────────────────────────────────────────────
#  Queue engine
# ─────────────────────────────────────────────
STATUS_PENDING    = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE       = "done"
STATUS_ERROR      = "error"

def new_job(src_path: Path, src_name: str, brand: str, course: str, unit: str,
            chapter: str, trim_start: float, outro_remove: float,
            out_name: str, speed: str) -> dict:
    return {
        "id": uuid.uuid4().hex[:10],
        "src": str(src_path),
        "src_name": src_name,
        "brand": brand,
        "course": course,
        "unit": unit,
        "chapter": chapter,
        "trim_start": trim_start,
        "outro_remove": outro_remove,
        "out_name": out_name,
        "speed": speed,
        "status": STATUS_PENDING,
        "progress": 0.0,
        "stage": "Waiting in queue",
        "result": None,
        "error": None,
        "added": time.time(),
        "elapsed": None,
    }


def run_job(job: dict, on_update: Callable[[], None], out_root: Path) -> None:
    """Process one queue job with live progress.

    Intro, lesson and outro preparation is deliberately sequential. Running
    the intro and lesson encoders in parallel can exceed Streamlit Community
    Cloud's shared CPU/RAM limits and terminate the entire app process.
    """
    t_start = time.time()
    out_root = Path(out_root)
    cache_dir = _cache_dir(out_root)

    logo = _logo_path(job["brand"])
    if not logo:
        raise FileNotFoundError(f"Missing logo: {BRANDS[job['brand']]['logo']}")

    def setp(frac: float, stage: str):
        job["progress"] = max(job["progress"], min(frac, 1.0))
        job["stage"] = stage
        on_update()

    setp(0.02, "Preparing outro")
    outro = cached_outro(job["brand"], job["speed"], logo, cache_dir)

    src = Path(job["src"])
    info = get_media_info(src)
    dur, w, h, _ = info
    if w == 0:
        raise ValueError(f"{job['src_name']}: no video stream found.")
    t0 = float(job["trim_start"])
    t1 = dur - float(job["outro_remove"])
    if t1 <= t0:
        raise ValueError(f"Trim removes entire video (duration {fmt_time(dur)}).")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        setp(0.05, "Preparing intro")
        intro_clip = cached_intro(
            job["brand"], job["course"], job["unit"],
            job["chapter"], job["speed"], cache_dir
        )

        setp(0.12, "Rebranding lesson video")
        content_out = tmpdir / "content.mp4"
        process_content(
            src, content_out, info, t0, t1, logo, job["speed"],
            cb=lambda f: setp(0.12 + f * 0.75, "Rebranding lesson video"),
        )

        setp(0.90, "Stitching intro + lesson + outro")
        final = tmpdir / job["out_name"]
        concat_clips(intro_clip, content_out, outro, final, job["speed"])
        if not final.exists() or final.stat().st_size == 0:
            raise RuntimeError("Output file not created.")

        out_dir = out_root / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{job['id']}_{job['out_name']}"
        shutil.copy(final, dest)
        job["result"] = str(dest)

    job["elapsed"] = time.time() - t_start
    setp(1.0, "Completed")


# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.block-container { max-width: 900px; padding-top: 1.0rem; }

/* ── Hero ── */
.hero{
  position:relative; overflow:hidden;
  padding:30px 34px; border-radius:22px;
  background:linear-gradient(120deg,#6D28D9 0%,#8B5CF6 45%,#C084FC 100%);
  color:#fff; margin-bottom:20px;
  box-shadow:0 16px 40px -14px rgba(124,58,237,.55);
}
.hero::after{
  content:""; position:absolute; right:-70px; top:-70px; width:240px; height:240px;
  border-radius:50%; background:rgba(255,255,255,.14);
}
.hero::before{
  content:""; position:absolute; right:60px; bottom:-90px; width:180px; height:180px;
  border-radius:50%; background:rgba(255,255,255,.10);
}
.hero h1{margin:0;font-size:1.95rem;font-weight:800;color:#fff;letter-spacing:-.5px}
.hero p{margin:8px 0 0;color:#F3E8FF;font-size:.95rem;max-width:600px}
.hero .badge{
  display:inline-block;margin-bottom:10px;padding:4px 12px;border-radius:999px;
  background:rgba(255,255,255,.18);font-size:.72rem;font-weight:600;
  letter-spacing:.8px;text-transform:uppercase;
}

/* ── Section headers ── */
.sec{
  display:flex;align-items:center;gap:10px;margin:4px 0 2px;
}
.sec .num{
  width:30px;height:30px;border-radius:10px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#7C3AED,#A855F7);color:#fff;font-weight:700;font-size:.9rem;
  box-shadow:0 5px 12px -4px rgba(124,58,237,.6);
}
.sec h3{margin:0;font-size:1.06rem;font-weight:700}

/* ── Info box ── */
.info-box{
  background:linear-gradient(120deg,#F5F3FF,#FAF5FF);
  border:1px solid #DDD6FE;border-left:4px solid #7C3AED;
  padding:11px 14px;border-radius:12px;color:#4C1D95;font-size:.88rem;
}

/* ── Status chips ── */
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 12px;border-radius:999px;
      font-size:.75rem;font-weight:700;letter-spacing:.3px}
.chip.pending    {background:#F1F5F9;color:#475569;border:1px solid #E2E8F0}
.chip.processing {background:#EDE9FE;color:#6D28D9;border:1px solid #DDD6FE}
.chip.done       {background:#DCFCE7;color:#15803D;border:1px solid #BBF7D0}
.chip.error      {background:#FEE2E2;color:#B91C1C;border:1px solid #FECACA}

.jobmeta{color:#64748B;font-size:.8rem;margin-top:2px}
.jobtitle{font-weight:700;font-size:.95rem;line-height:1.3}

/* ── Buttons ── */
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{
  background:linear-gradient(120deg,#7C3AED,#A855F7);
  border:none;border-radius:12px;font-weight:700;
  box-shadow:0 8px 20px -8px rgba(124,58,237,.65);
  transition:transform .12s ease, box-shadow .12s ease;
}
.stButton>button[kind="primary"]:hover, .stDownloadButton>button[kind="primary"]:hover{
  transform:translateY(-1px);
  box-shadow:0 12px 26px -8px rgba(124,58,237,.75);
}
.stButton>button{border-radius:12px;font-weight:600}
.stDownloadButton>button{border-radius:12px;font-weight:600}

/* progress bar tint */
.stProgress > div > div > div > div{
  background:linear-gradient(90deg,#7C3AED,#C084FC);
}

[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#FAF5FF 0%,#F5F3FF 100%);
}
</style>
"""

def sec(num: int, title: str):
    st.markdown(f'<div class="sec"><div class="num">{num}</div><h3>{title}</h3></div>',
                unsafe_allow_html=True)


def setup_page():
    st.set_page_config(page_title="Video Rebranding Studio", page_icon="🎬", layout="centered")
    st.markdown(APP_CSS, unsafe_allow_html=True)


def init_state():
    if "out_dir" not in st.session_state:
        st.session_state["out_dir"] = tempfile.mkdtemp(prefix="rebrand_")
    if "queue" not in st.session_state:
        st.session_state.queue = []          # list[dict] job records
    if "queue_running" not in st.session_state:
        st.session_state.queue_running = False


CHIP = {
    STATUS_PENDING:    ('<span class="chip pending">⏳ Pending</span>'),
    STATUS_PROCESSING: ('<span class="chip processing">⚙️ Processing</span>'),
    STATUS_DONE:       ('<span class="chip done">✅ Completed</span>'),
    STATUS_ERROR:      ('<span class="chip error">❌ Failed</span>'),
}


def render_job_card(job: dict, live: bool = False):
    """Render one queue entry. When `live` is True the card returns
    placeholders so the processing loop can update them in place."""
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f'<div class="jobtitle">🎞 {job["out_name"]}</div>'
                f'<div class="jobmeta">{job["course"][:70]}{"…" if len(job["course"])>70 else ""}'
                f' · {job["unit"]} · {job["chapter"]} · source: {job["src_name"]}</div>',
                unsafe_allow_html=True)
        with c2:
            chip_ph = st.empty()
            chip_ph.markdown(CHIP[job["status"]], unsafe_allow_html=True)

        prog_ph  = st.empty()
        stage_ph = st.empty()

        if job["status"] == STATUS_PROCESSING or live:
            prog_ph.progress(min(job["progress"], 1.0))
            stage_ph.caption(job["stage"])
        elif job["status"] == STATUS_PENDING:
            stage_ph.caption("Waiting in queue…")
        elif job["status"] == STATUS_ERROR:
            st.error(job["error"] or "Unknown error")
        elif job["status"] == STATUS_DONE and job["result"] and Path(job["result"]).exists():
            p = Path(job["result"])
            dur, _, _, _ = get_media_info(p)
            took = f" · finished in {job['elapsed']:.0f}s" if job.get("elapsed") else ""
            stage_ph.caption(f"{p.stat().st_size/1_048_576:.1f} MB · {fmt_time(dur)}{took}")
            def open_result(path=p):
                return path.open("rb")

            st.download_button(
                "⬇ Download video",
                data=open_result,
                file_name=job["out_name"],
                mime="video/mp4",
                type="primary",
                width="stretch",
                on_click="ignore",
                key=f"dl_{job['id']}",
            )

        # Remove button (not while processing)
        if job["status"] in (STATUS_PENDING, STATUS_DONE, STATUS_ERROR) and not live:
            if st.button("Remove", key=f"rm_{job['id']}", width="stretch"):
                if job.get("result"):
                    Path(job["result"]).unlink(missing_ok=True)
                st.session_state.queue = [j for j in st.session_state.queue
                                          if j["id"] != job["id"]]
                st.rerun()

        return chip_ph, prog_ph, stage_ph


def process_queue():
    """Run all pending jobs sequentially with live in-place progress updates."""
    # Read session state only in Streamlit's main script thread. The resulting
    # normal Path is then passed through the processing pipeline explicitly.
    out_root = Path(st.session_state["out_dir"])
    pending = [j for j in st.session_state.queue if j["status"] == STATUS_PENDING]
    if not pending:
        return
    st.session_state.queue_running = True
    holder = st.container()
    with holder:
        st.markdown("#### ⚙️ Processing queue…")
        for job in pending:
            job["status"] = STATUS_PROCESSING
            job["progress"] = 0.0
            chip_ph, prog_ph, stage_ph = render_job_card(job, live=True)
            chip_ph.markdown(CHIP[STATUS_PROCESSING], unsafe_allow_html=True)

            last = {"t": 0.0}
            def on_update(j=job, pp=prog_ph, sp=stage_ph):
                now = time.time()
                if now - last["t"] < 0.15:   # throttle UI updates
                    return
                last["t"] = now
                pp.progress(min(j["progress"], 1.0))
                sp.caption(f"{j['stage']} — {int(j['progress']*100)}%")

            try:
                run_job(job, on_update, out_root)
                job["status"] = STATUS_DONE
                chip_ph.markdown(CHIP[STATUS_DONE], unsafe_allow_html=True)
                prog_ph.progress(1.0)
                stage_ph.caption("Completed — 100%")
            except Exception as e:
                job["status"] = STATUS_ERROR
                job["error"] = str(e)
                chip_ph.markdown(CHIP[STATUS_ERROR], unsafe_allow_html=True)
                stage_ph.caption("Failed")
    st.session_state.queue_running = False
    st.rerun()


def main():
    setup_page()
    init_state()
    try:
        get_ffmpeg()
    except RuntimeError as e:
        st.error(str(e)); st.stop()

    session_out_dir = Path(st.session_state["out_dir"])
    cache_dir = _cache_dir(session_out_dir)

    st.markdown("""
<div class="hero">
  <span class="badge">v17.1 · Cloud-Safe Queue</span>
  <h1>🎬 Video Rebranding Studio</h1>
  <p>Replace the course name, unit and chapter in your exact Intro.mp4 — every
  animation, logo and note of music kept intact. Queue up multiple videos and
  download each finished MP4 directly. Cloud-safe mode processes only one FFmpeg encode at a time.</p>
</div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        brand = st.radio("Brand", list(BRANDS.keys()))
        logo  = _logo_path(brand)
        intro = _intro_path(brand)

        if logo:
            try:
                li = Image.open(logo).convert("RGBA")
                bg = Image.new("RGB", li.size, (255,255,255))
                bg.paste(li, mask=li.split()[3] if len(li.split())==4 else None)
                st.image(bg, width=160)
            except Exception: pass
        else:
            st.error(f"Missing: {BRANDS[brand]['logo']}")

        if intro:
            st.success(f"✓ {intro.name}")
        else:
            st.warning("Missing: Intro.mp4 (place it beside app.py)")

        st.divider()
        speed = st.selectbox("Processing speed", list(SPEED_PROFILES.keys()),
                             index=list(SPEED_PROFILES.keys()).index(DEFAULT_SPEED))
        st.divider()
        fp = find_font()
        st.caption(f"Font: {Path(fp).name if fp else 'PIL default'}")
        st.caption(f"FFmpeg: {Path(get_ffmpeg()).name}")
        st.caption(f"FFmpeg threads: {FFMPEG_THREADS} (cloud-safe)")

    # ── 1 · Video details ─────────────────────
    with st.container(border=True):
        sec(1, "Video details")
        course = st.text_input("Course name (full name is always shown — long names wrap automatically)",
                               "LEVEL 4 DIPLOMA IN EDUCATION STUDIES (RQF)")
        c1, c2 = st.columns(2)
        with c1:
            unit = st.text_input("Unit number", "UNIT 01")
        with c2:
            chapter = st.text_input("Chapter name", "CHAPTER 01")

    # ── 2 · Intro preview ─────────────────────
    with st.container(border=True):
        sec(2, "Intro preview")
        st.caption("Real intro frame with your text applied — exactly as it will render. "
                   "Long course names wrap onto multiple lines, never cut off.")
        prev = make_preview(brand, course, unit, chapter)
        if prev:
            st.image(prev, width="stretch")
        else:
            st.info("Intro.mp4 not found.")

    # ── 3 · Upload & trim ─────────────────────
    with st.container(border=True):
        sec(3, "Upload & trim")
        uploads = st.file_uploader("Choose one or more SLC videos",
                                   ["mp4", "mov", "avi", "mkv"],
                                   accept_multiple_files=True)

        t1, t2 = st.columns(2)
        with t1:
            trim_start = st.number_input("Remove from start (s)", 0.0, 300.0, 9.0, 0.5, "%.1f",
                                          help="Seconds of the old SLC intro to cut.")
        with t2:
            outro_remove = st.number_input("Remove from end (s)", 0.0, 300.0, 10.0, 0.5, "%.1f",
                                            help="Seconds of the old SLC outro to cut.")

        if uploads and st.button("👁 Preview trim points (first video)", width="stretch"):
            with st.spinner("Grabbing frames…"):
                src0 = upload_cache_path(uploads[0], cache_dir)
                first, last, src_dur = trim_preview(src0, trim_start, outro_remove)
            if trim_start + outro_remove >= src_dur:
                st.error(f"Trim removes the entire video (source is only {fmt_time(src_dur)}).")
            else:
                p1, p2 = st.columns(2)
                with p1:
                    if first is not None:
                        st.image(first, width="stretch",
                                 caption=f"First kept frame @ {fmt_time(trim_start)}")
                    else:
                        st.warning("Couldn't read that frame.")
                with p2:
                    if last is not None:
                        st.image(last, width="stretch",
                                 caption=f"Last kept frame @ {fmt_time(src_dur - outro_remove)}")
                    else:
                        st.warning("Couldn't read that frame.")
                st.caption(f"Source duration: {fmt_time(src_dur)}. If the \"last kept frame\" "
                           "still shows real lesson content (not the old outro), lower "
                           "\"Remove from end\".")

    # ── 4 · Add to queue ──────────────────────
    with st.container(border=True):
        sec(4, "Add to queue")
        st.markdown(
            f'<div class="info-box">The exact <b>Intro.mp4</b> is used — only the course '
            f'name, unit and chapter are replaced. All animations, music and the {brand} '
            f'logo stay intact. Each video is added as a job with the details above.</div>',
            unsafe_allow_html=True)
        st.write("")

        custom_name = ""
        if uploads and len(uploads) == 1:
            default_stem = Path(uploads[0].name).stem
            custom_name = st.text_input("Output filename",
                                        f"{default_stem}_{brand.lower()}_rebranded.mp4")

        n = len(uploads) if uploads else 0
        add = st.button(f"➕ Add {n or ''} video{'s' if n != 1 else ''} to queue".replace("  ", " "),
                        type="primary", width="stretch",
                        disabled=not (uploads and logo and intro))
        if add and uploads:
            for up in uploads:
                src = upload_cache_path(up, cache_dir)
                stem = Path(up.name).stem
                if len(uploads) == 1 and custom_name:
                    out_name = safe_name(custom_name, stem, brand)
                else:
                    out_name = safe_name("", stem, brand)
                st.session_state.queue.append(
                    new_job(src, up.name, brand, course, unit, chapter,
                            trim_start, outro_remove, out_name, speed))
            st.toast(f"Added {len(uploads)} job(s) to the queue", icon="✅")
            st.rerun()

    # ── 5 · Queue ─────────────────────────────
    with st.container(border=True):
        sec(5, "Processing queue")
        q = st.session_state.queue
        n_pend = sum(1 for j in q if j["status"] == STATUS_PENDING)
        n_done = sum(1 for j in q if j["status"] == STATUS_DONE)
        n_err  = sum(1 for j in q if j["status"] == STATUS_ERROR)

        if not q:
            st.caption("Queue is empty — add videos above.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", len(q))
            m2.metric("Pending", n_pend)
            m3.metric("Completed", n_done)
            m4.metric("Failed", n_err)

            b1, b2 = st.columns([2, 1])
            with b1:
                start = st.button(f"▶ Start queue ({n_pend} pending)", type="primary",
                                  width="stretch", disabled=n_pend == 0)
            with b2:
                if st.button("🧹 Clear finished", width="stretch",
                             disabled=(n_done + n_err) == 0):
                    for j in q:
                        if j["status"] in (STATUS_DONE, STATUS_ERROR) and j.get("result"):
                            Path(j["result"]).unlink(missing_ok=True)
                    st.session_state.queue = [j for j in q if j["status"]
                                              in (STATUS_PENDING, STATUS_PROCESSING)]
                    st.rerun()

            if start:
                process_queue()
            else:
                for job in q:
                    render_job_card(job)


if __name__ == "__main__":
    main()
