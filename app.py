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
import html
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

# The source wording is removed with softly feathered, localized clean-background
# patches. This avoids the hard horizontal edges created by full-width erase bands.
CLEAN_PATCH_TIME_MS = 750
MASK_REFERENCE_TIME_MS = 8500
TITLE_MASK_ROI = (230, 320, 1690, 590)  # x1, y1, x2, y2

INTRO_RENDER_VERSION = "feathered-clean-patches-with-pill-v3"

SLC_COVER_BOX = (1688, 972, 160, 82)   # cleanup stays inside the existing rounded white pill
SLC_LOGO_BOX  = (1722, 983, 106, 60)   # unchanged maximum logo size; centered in the pill

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
def _read_intro_frame(intro_path: Path, position_ms: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(intro_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(position_ms, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_clean_patch_layers(intro_path: Path) -> Tuple[Image.Image, Image.Image]:
    """Build two localized, feathered patches that hide the wording baked into
    the intro without creating visible rectangular background bands."""
    clean_rgb = _read_intro_frame(intro_path, CLEAN_PATCH_TIME_MS)
    reference_rgb = _read_intro_frame(intro_path, MASK_REFERENCE_TIME_MS)
    if clean_rgb is None:
        transparent = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
        return transparent.copy(), transparent
    if reference_rgb is None:
        reference_rgb = clean_rgb

    reference_bgr = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV)

    # Detect the bright, low-saturation title lettering, then expand and feather
    # the mask so the original drop shadow is also covered without hard edges.
    title_alpha = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    x1, y1, x2, y2 = TITLE_MASK_ROI
    title_raw = (
        (gray[y1:y2, x1:x2] > 175)
        & (hsv[y1:y2, x1:x2, 1] < 125)
    ).astype(np.uint8) * 255
    title_raw = cv2.morphologyEx(
        title_raw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    title_core = cv2.dilate(
        title_raw,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    title_soft = cv2.GaussianBlur(title_core, (0, 0), sigmaX=12, sigmaY=12)
    title_alpha[y1:y2, x1:x2] = np.maximum(title_core, title_soft)

    # The original unit/chapter element includes a large white pill and shadow.
    # Cover it with a wide, softly feathered patch so no pill-shaped or rectangular
    # outline remains when only the replacement wording is drawn.
    pill_outer = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    cv2.rectangle(pill_outer, (500, 555), (1420, 765), 255, thickness=-1)
    pill_outer = cv2.GaussianBlur(pill_outer, (0, 0), sigmaX=80, sigmaY=55)
    pill_core = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    cv2.rectangle(pill_core, (550, 585), (1370, 745), 255, thickness=-1)
    pill_alpha = np.maximum(pill_outer, pill_core)

    def make_patch(alpha: np.ndarray) -> Image.Image:
        rgba = np.dstack([clean_rgb, alpha]).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")

    return make_patch(title_alpha), make_patch(pill_alpha)


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

    # The erase layer intentionally stays transparent. The source wording is
    # removed dynamically in FFmpeg, so no full-width replacement panel is
    # composited over the animated background.
    erase = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    text_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer, "RGBA")

    max_title_w = 1580
    t_font, t_lines = fit_title(draw, course_text, max_title_w)
    ascent, descent = t_font.getmetrics()
    line_h = ascent + descent
    total_h = len(t_lines) * line_h
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
    # Keep the white rounded unit/chapter pill while the large full-width
    # translucent background bands remain removed.
    pb = BRANDS[brand]["pill_bg"]
    pt = BRANDS[brand]["pill_text"]
    unit_text    = (unit_no or "UNIT 01").strip().upper()
    chapter_text = (chapter or "CHAPTER 01").strip().upper()
    pill_line    = f"{unit_text} -  {chapter_text}"

    # The original pill is removed by the localized feathered clean patch in
    # build_clean_patch_layers(); this layer only draws the replacement pill.
    erase = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    text_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer, "RGBA")

    # Shrink the font for unusually long unit/chapter wording so the pill stays
    # inside the frame without truncating the text.
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

    draw.rounded_rectangle(
        [px, py, px + pw, py + ph],
        radius=44,
        fill=(*pb, 255),
    )
    draw.rounded_rectangle(
        [px, py, px + pw, py + ph],
        radius=44,
        outline=(*pt, 80),
        width=5,
    )
    y_drawn = py + (ph - uh) // 2 - top_offset
    draw.text(
        (TARGET_W // 2 - uw // 2, y_drawn),
        pill_line,
        font=p_font,
        fill=(*pt, 255),
    )

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
        "title_clean": out_path.with_suffix(".tclean.png"),
        "pill_clean":  out_path.with_suffix(".pclean.png"),
        "title_text":  out_path.with_suffix(".ttext.png"),
        "unit_text":   out_path.with_suffix(".utext.png"),
    }
    try:
        title_clean, pill_clean = build_clean_patch_layers(intro)
        _, title_text = build_title_layers(brand, course, intro_path=intro)
        _, unit_text = build_pill_layers(brand, unit_no, chapter, intro_path=intro)
        title_clean.save(paths["title_clean"])
        pill_clean.save(paths["pill_clean"])
        title_text.save(paths["title_text"])
        unit_text.save(paths["unit_text"])

        title_clean_start = max(TITLE_FADE_START - ERASE_LEAD, 0.0)
        pill_clean_start = max(PILL_FADE_START - ERASE_LEAD, 0.0)
        vf = (
            f"[0:v]{scale_filter()}[base];"
            f"[1:v]format=rgba,fade=t=in:st={title_clean_start}:d={ERASE_SNAP_DUR}:alpha=1[tc];"
            f"[2:v]format=rgba,fade=t=in:st={pill_clean_start}:d={ERASE_SNAP_DUR}:alpha=1[pc];"
            f"[3:v]format=rgba,fade=t=in:st={TITLE_FADE_START}:d={TITLE_FADE_DUR}:alpha=1[tt];"
            f"[4:v]format=rgba,fade=t=in:st={PILL_FADE_START}:d={PILL_FADE_DUR}:alpha=1[ut];"
            f"[base][tc]overlay=0:0:format=auto[s1];"
            f"[s1][tt]overlay=0:0:format=auto[s2];"
            f"[s2][pc]overlay=0:0:format=auto[s3];"
            f"[s3][ut]overlay=0:0:format=auto,format=yuv420p[v]"
        )

        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", str(intro),
            "-loop", "1", "-i", str(paths["title_clean"]),
            "-loop", "1", "-i", str(paths["pill_clean"]),
            "-loop", "1", "-i", str(paths["title_text"]),
            "-loop", "1", "-i", str(paths["unit_text"]),
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
        run(cmd, "Generate intro with restored unit/chapter pill")
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)


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
    cx, cy, cw, ch = SLC_COVER_BOX
    x, y, w, h = SLC_LOGO_BOX
    cover = "0xFAFAFA@1.0"
    fc = (
        f"[0:v]{scale_filter()},"
        f"drawbox=x={cx}:y={cy}:w={cw}:h={ch}:color={cover}:t=fill,format=rgba[base];"
        f"[1:v]format=rgba,scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos[logo];"
        f"[base][logo]overlay=x={x}+({w}-overlay_w)/2:y={y}+({h}-overlay_h)/2:"
        f"format=auto,format=yuv420p[v]"
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
    key = hashlib.sha1(
        f"{INTRO_RENDER_VERSION}|{brand}|{course.strip().upper()}|"
        f"{unit.strip().upper()}|{chapter.strip().upper()}|{speed}".encode()
    ).hexdigest()[:16]
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
# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────
APP_CSS = """
<style>
:root {
  --app-bg: #070B14;
  --surface: #0D1321;
  --card: #121A2A;
  --card-elevated: #182235;
  --purple: #7C3AED;
  --violet: #8B5CF6;
  --blue: #3B82F6;
  --cyan: #22D3EE;
  --text: #F8FAFC;
  --text-muted: #94A3B8;
  --text-subtle: #64748B;
  --border: rgba(148, 163, 184, 0.18);
  --border-strong: rgba(139, 92, 246, 0.45);
  --success: #22C55E;
  --warning: #F59E0B;
  --error: #EF4444;
  --radius-sm: 10px;
  --radius-md: 14px;
  --radius-lg: 18px;
  --space-1: 0.35rem;
  --space-2: 0.65rem;
  --space-3: 1rem;
  --space-4: 1.35rem;
  --space-5: 1.8rem;
  --shadow-card: 0 18px 50px rgba(0, 0, 0, 0.22);
  --shadow-glow: 0 14px 36px rgba(124, 58, 237, 0.28);
  --gradient-primary: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 45%, #3B82F6 100%);
  --gradient-soft: linear-gradient(135deg, rgba(124, 58, 237, 0.20), rgba(59, 130, 246, 0.12));
}

html, body, [data-testid="stAppViewContainer"], button, input, textarea, select {
  font-family: Inter, Manrope, "Plus Jakarta Sans", ui-sans-serif, system-ui,
               -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

html, body {
  background: var(--app-bg);
  color: var(--text);
}

[data-testid="stAppViewContainer"] {
  color: var(--text);
  background:
    radial-gradient(circle at 8% 2%, rgba(124, 58, 237, 0.14), transparent 30rem),
    radial-gradient(circle at 92% 12%, rgba(59, 130, 246, 0.12), transparent 28rem),
    linear-gradient(180deg, #070B14 0%, #090E19 52%, #070B14 100%);
}

[data-testid="stHeader"] {
  background: rgba(7, 11, 20, 0.72);
  border-bottom: 1px solid rgba(148, 163, 184, 0.08);
  backdrop-filter: blur(14px);
}

[data-testid="stToolbar"] {
  color: var(--text-muted);
}

.block-container {
  max-width: 1480px;
  padding-top: 1.35rem;
  padding-bottom: 4rem;
}

h1, h2, h3, h4, h5, h6, p, label, span {
  color: inherit;
}

/* Page header */
.dashboard-header {
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.5rem;
  margin-bottom: 1rem;
  padding: 1.75rem 1.9rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background:
    linear-gradient(125deg, rgba(24, 34, 53, 0.96), rgba(13, 19, 33, 0.94)),
    var(--surface);
  box-shadow: var(--shadow-card);
}

.dashboard-header::before {
  content: "";
  position: absolute;
  width: 22rem;
  height: 22rem;
  right: -10rem;
  top: -14rem;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(139, 92, 246, 0.28), transparent 68%);
  pointer-events: none;
}

.dashboard-header::after {
  content: "";
  position: absolute;
  width: 17rem;
  height: 17rem;
  right: 8rem;
  bottom: -14rem;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(34, 211, 238, 0.14), transparent 68%);
  pointer-events: none;
}

.header-copy, .header-meta {
  position: relative;
  z-index: 1;
}

.header-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.65rem;
  color: #C4B5FD;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

.header-eyebrow::before {
  content: "";
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 999px;
  background: var(--cyan);
  box-shadow: 0 0 18px rgba(34, 211, 238, 0.55);
}

.dashboard-header h1 {
  margin: 0;
  color: var(--text);
  font-size: clamp(1.8rem, 3vw, 2.65rem);
  line-height: 1.06;
  letter-spacing: -0.045em;
  font-weight: 800;
}

.dashboard-header p {
  max-width: 47rem;
  margin: 0.75rem 0 0;
  color: var(--text-muted);
  font-size: 0.96rem;
  line-height: 1.65;
}

.header-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.65rem;
  min-width: 10rem;
}

.brand-chip, .version-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  width: fit-content;
  padding: 0.48rem 0.78rem;
  border-radius: 999px;
  border: 1px solid rgba(139, 92, 246, 0.30);
  background: rgba(124, 58, 237, 0.12);
  color: #DDD6FE;
  font-size: 0.76rem;
  font-weight: 750;
  white-space: nowrap;
}

.version-chip {
  border-color: var(--border);
  background: rgba(15, 23, 42, 0.55);
  color: var(--text-muted);
  font-weight: 650;
}

.brand-chip::before {
  content: "";
  width: 0.48rem;
  height: 0.48rem;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--violet), var(--blue));
  box-shadow: 0 0 12px rgba(139, 92, 246, 0.65);
}

/* Workflow rail */
.workflow-rail {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.65rem;
  margin: 0 0 1.35rem;
}

.workflow-item {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  min-height: 3.45rem;
  padding: 0.72rem 0.8rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: rgba(13, 19, 33, 0.82);
  color: var(--text-muted);
  transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease;
}

.workflow-item.active {
  border-color: rgba(139, 92, 246, 0.52);
  background: var(--gradient-soft);
  box-shadow: inset 0 0 0 1px rgba(139, 92, 246, 0.08),
              0 10px 28px rgba(20, 12, 45, 0.22);
  color: var(--text);
}

.workflow-item.done {
  border-color: rgba(34, 197, 94, 0.28);
  background: rgba(34, 197, 94, 0.07);
  color: #D1FAE5;
}

.workflow-number {
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  width: 1.8rem;
  height: 1.8rem;
  border-radius: 0.62rem;
  background: rgba(148, 163, 184, 0.10);
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 0.75rem;
  font-weight: 800;
}

.workflow-item.active .workflow-number {
  border-color: transparent;
  background: var(--gradient-primary);
  color: #FFFFFF;
  box-shadow: 0 8px 18px rgba(124, 58, 237, 0.30);
}

.workflow-item.done .workflow-number {
  border-color: rgba(34, 197, 94, 0.32);
  background: rgba(34, 197, 94, 0.15);
  color: #86EFAC;
}

.workflow-label {
  min-width: 0;
  font-size: 0.75rem;
  font-weight: 700;
  line-height: 1.25;
}

/* Streamlit bordered containers become cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background: linear-gradient(180deg, rgba(18, 26, 42, 0.96), rgba(13, 19, 33, 0.94));
  box-shadow: var(--shadow-card);
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
  padding: 0.2rem;
}

/* Step and panel headings */
.section-heading {
  display: flex;
  align-items: flex-start;
  gap: 0.85rem;
  margin-bottom: 0.35rem;
}

.section-number {
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  width: 2.35rem;
  height: 2.35rem;
  border-radius: 0.82rem;
  border: 1px solid var(--border);
  background: rgba(148, 163, 184, 0.08);
  color: var(--text-muted);
  font-size: 0.83rem;
  font-weight: 800;
}

.section-heading.active .section-number {
  border-color: transparent;
  color: #FFFFFF;
  background: var(--gradient-primary);
  box-shadow: 0 10px 24px rgba(124, 58, 237, 0.28);
}

.section-heading.done .section-number {
  border-color: rgba(34, 197, 94, 0.35);
  color: #86EFAC;
  background: rgba(34, 197, 94, 0.12);
}

.section-title {
  margin: 0;
  color: var(--text);
  font-size: 1.06rem;
  line-height: 1.3;
  font-weight: 780;
  letter-spacing: -0.015em;
}

.section-subtitle {
  margin-top: 0.2rem;
  color: var(--text-muted);
  font-size: 0.8rem;
  line-height: 1.45;
}

.panel-heading {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  margin-bottom: 0.25rem;
}

.panel-icon {
  display: grid;
  place-items: center;
  width: 2rem;
  height: 2rem;
  border-radius: 0.7rem;
  background: var(--gradient-soft);
  border: 1px solid rgba(139, 92, 246, 0.24);
  color: #C4B5FD;
  font-size: 0.95rem;
}

.panel-title {
  color: var(--text);
  font-weight: 760;
  font-size: 0.96rem;
}

.panel-subtitle {
  color: var(--text-muted);
  font-size: 0.76rem;
  margin-top: 0.1rem;
}

/* Inputs and labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] label,
label[data-testid="stWidgetLabel"] {
  color: #CBD5E1 !important;
  font-size: 0.82rem !important;
  font-weight: 650 !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  min-height: 2.75rem;
  color: var(--text) !important;
  background: rgba(7, 11, 20, 0.72) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-sm) !important;
  caret-color: var(--cyan);
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stNumberInput"] input::placeholder {
  color: var(--text-subtle) !important;
}

[data-testid="stTextInput"] > div:focus-within,
[data-testid="stNumberInput"] > div:focus-within,
[data-testid="stSelectbox"] > div:focus-within {
  border-color: var(--violet) !important;
  box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.16) !important;
  border-radius: var(--radius-sm);
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  min-height: 2.75rem;
  color: var(--text) !important;
  background: rgba(7, 11, 20, 0.72) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-sm) !important;
}

[data-testid="stSelectbox"] svg,
[data-testid="stNumberInput"] button svg {
  fill: var(--text-muted);
}

[data-baseweb="popover"], [role="listbox"] {
  background: var(--card-elevated) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

[role="option"] {
  color: var(--text) !important;
}

[role="option"]:hover,
[aria-selected="true"][role="option"] {
  background: rgba(124, 58, 237, 0.18) !important;
}

[data-testid="stRadio"] > div {
  gap: 0.55rem;
}

[data-testid="stRadio"] label {
  padding: 0.52rem 0.65rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: rgba(7, 11, 20, 0.50);
  transition: border-color 0.16s ease, background 0.16s ease;
}

[data-testid="stRadio"] label:hover {
  border-color: rgba(139, 92, 246, 0.48);
  background: rgba(124, 58, 237, 0.10);
}

[data-testid="stRadio"] label:has(input:checked) {
  border-color: rgba(139, 92, 246, 0.60);
  background: rgba(124, 58, 237, 0.16);
  box-shadow: inset 0 0 0 1px rgba(139, 92, 246, 0.12);
}

/* Upload area */
.upload-intro {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin: 0.2rem 0 0.75rem;
  color: var(--text-muted);
  font-size: 0.8rem;
}

.upload-intro-icon {
  display: grid;
  place-items: center;
  width: 2.35rem;
  height: 2.35rem;
  border-radius: 0.8rem;
  background: var(--gradient-soft);
  color: #C4B5FD;
  border: 1px solid rgba(139, 92, 246, 0.28);
  font-size: 1.05rem;
  font-weight: 800;
}

[data-testid="stFileUploaderDropzone"] {
  min-height: 9.5rem;
  padding: 1.3rem !important;
  border: 1.5px dashed rgba(139, 92, 246, 0.52) !important;
  border-radius: var(--radius-md) !important;
  background:
    linear-gradient(135deg, rgba(124, 58, 237, 0.09), rgba(59, 130, 246, 0.06)),
    rgba(7, 11, 20, 0.48) !important;
  transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(34, 211, 238, 0.72) !important;
  background:
    linear-gradient(135deg, rgba(124, 58, 237, 0.13), rgba(34, 211, 238, 0.08)),
    rgba(7, 11, 20, 0.56) !important;
}

[data-testid="stFileUploaderDropzone"] svg {
  fill: #A78BFA !important;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p {
  color: var(--text-muted) !important;
}

[data-testid="stFileUploaderFile"] {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: rgba(24, 34, 53, 0.60);
}

/* Buttons */
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
  min-height: 2.75rem;
  border-radius: var(--radius-sm) !important;
  font-weight: 740 !important;
  letter-spacing: -0.005em;
  transition: transform 0.15s ease, border-color 0.15s ease,
              box-shadow 0.15s ease, background 0.15s ease;
}

[data-testid="stButton"] button[kind="primary"],
[data-testid="stDownloadButton"] button[kind="primary"] {
  color: #FFFFFF !important;
  border: 1px solid rgba(255, 255, 255, 0.10) !important;
  background: var(--gradient-primary) !important;
  box-shadow: var(--shadow-glow);
}

[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stDownloadButton"] button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 17px 42px rgba(124, 58, 237, 0.38);
  filter: brightness(1.06);
}

[data-testid="stButton"] button[kind="secondary"],
[data-testid="stDownloadButton"] button[kind="secondary"] {
  color: #E2E8F0 !important;
  border: 1px solid var(--border) !important;
  background: rgba(7, 11, 20, 0.54) !important;
}

[data-testid="stButton"] button[kind="secondary"]:hover,
[data-testid="stDownloadButton"] button[kind="secondary"]:hover {
  transform: translateY(-1px);
  border-color: rgba(139, 92, 246, 0.55) !important;
  background: rgba(124, 58, 237, 0.10) !important;
}

[data-testid="stButton"] button[kind="tertiary"] {
  color: #FCA5A5 !important;
  border: 1px solid rgba(239, 68, 68, 0.28) !important;
  background: rgba(239, 68, 68, 0.07) !important;
}

[data-testid="stButton"] button[kind="tertiary"]:hover {
  color: #FECACA !important;
  border-color: rgba(239, 68, 68, 0.56) !important;
  background: rgba(239, 68, 68, 0.12) !important;
}

[data-testid="stButton"] button:focus-visible,
[data-testid="stDownloadButton"] button:focus-visible {
  outline: 3px solid rgba(34, 211, 238, 0.35) !important;
  outline-offset: 2px;
}

[data-testid="stButton"] button:disabled,
[data-testid="stDownloadButton"] button:disabled {
  color: #64748B !important;
  border-color: rgba(148, 163, 184, 0.10) !important;
  background: rgba(30, 41, 59, 0.38) !important;
  box-shadow: none !important;
  transform: none !important;
  cursor: not-allowed;
  opacity: 0.72;
}

/* Information, alerts and helper elements */
.info-box {
  padding: 0.9rem 1rem;
  border: 1px solid rgba(139, 92, 246, 0.27);
  border-left: 3px solid var(--violet);
  border-radius: var(--radius-sm);
  background: linear-gradient(135deg, rgba(124, 58, 237, 0.11), rgba(59, 130, 246, 0.06));
  color: #CBD5E1;
  font-size: 0.82rem;
  line-height: 1.55;
}

.system-status {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.68rem 0.75rem;
  margin: 0.25rem 0;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: rgba(7, 11, 20, 0.46);
  color: var(--text-muted);
  font-size: 0.77rem;
}

.system-status .status-light {
  width: 0.52rem;
  height: 0.52rem;
  border-radius: 999px;
  background: var(--success);
  box-shadow: 0 0 12px rgba(34, 197, 94, 0.44);
}

.system-status.warning .status-light {
  background: var(--warning);
  box-shadow: 0 0 12px rgba(245, 158, 11, 0.42);
}

[data-testid="stAlert"] {
  color: #E2E8F0 !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  background: rgba(15, 23, 42, 0.88) !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
  color: inherit !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption {
  color: var(--text-muted) !important;
}

hr {
  border-color: var(--border) !important;
}

/* Metrics and summary */
[data-testid="stMetric"] {
  min-height: 6.1rem;
  padding: 0.85rem 0.9rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: rgba(7, 11, 20, 0.48);
}

[data-testid="stMetricLabel"] {
  color: var(--text-muted) !important;
  font-size: 0.72rem !important;
  font-weight: 650;
}

[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-size: 1.25rem !important;
  font-weight: 790 !important;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.55rem;
  margin-top: 0.6rem;
}

.summary-metric {
  min-width: 0;
  padding: 0.72rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: rgba(7, 11, 20, 0.48);
}

.summary-label {
  color: var(--text-subtle);
  font-size: 0.66rem;
  font-weight: 750;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.summary-value {
  margin-top: 0.22rem;
  overflow: hidden;
  color: var(--text);
  font-size: 0.92rem;
  font-weight: 780;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Queue cards and status badges */
.status-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.45rem;
  padding: 0.4rem 0.68rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 0.69rem;
  font-weight: 800;
  letter-spacing: 0.025em;
  white-space: nowrap;
}

.status-dot {
  width: 0.46rem;
  height: 0.46rem;
  border-radius: 999px;
  background: currentColor;
}

.status-pending {
  color: #FCD34D;
  border-color: rgba(245, 158, 11, 0.28);
  background: rgba(245, 158, 11, 0.09);
}

.status-processing {
  color: #C4B5FD;
  border-color: rgba(139, 92, 246, 0.34);
  background: rgba(124, 58, 237, 0.12);
  box-shadow: 0 0 22px rgba(124, 58, 237, 0.10);
}

.status-done {
  color: #86EFAC;
  border-color: rgba(34, 197, 94, 0.28);
  background: rgba(34, 197, 94, 0.09);
}

.status-error {
  color: #FCA5A5;
  border-color: rgba(239, 68, 68, 0.30);
  background: rgba(239, 68, 68, 0.09);
}

.job-title {
  color: var(--text);
  font-size: 0.96rem;
  line-height: 1.35;
  font-weight: 780;
  overflow-wrap: anywhere;
}

.job-meta {
  margin-top: 0.3rem;
  color: var(--text-muted);
  font-size: 0.76rem;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.job-details {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.65rem;
}

.job-detail-pill {
  padding: 0.3rem 0.5rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: rgba(7, 11, 20, 0.42);
  color: var(--text-muted);
  font-size: 0.68rem;
  font-weight: 650;
}

.download-callout {
  margin: 0.4rem 0 0.7rem;
  padding: 0.72rem 0.82rem;
  border: 1px solid rgba(34, 197, 94, 0.24);
  border-radius: var(--radius-sm);
  background: rgba(34, 197, 94, 0.06);
  color: #BBF7D0;
  font-size: 0.76rem;
  line-height: 1.45;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 13rem;
  padding: 1.4rem;
  border: 1px dashed rgba(148, 163, 184, 0.24);
  border-radius: var(--radius-md);
  background: rgba(7, 11, 20, 0.30);
  text-align: center;
}

.empty-icon {
  display: grid;
  place-items: center;
  width: 3.2rem;
  height: 3.2rem;
  margin-bottom: 0.75rem;
  border-radius: 1rem;
  border: 1px solid rgba(139, 92, 246, 0.26);
  background: var(--gradient-soft);
  color: #C4B5FD;
  font-size: 1.35rem;
  font-weight: 800;
}

.empty-title {
  color: var(--text);
  font-size: 0.95rem;
  font-weight: 760;
}

.empty-copy {
  max-width: 26rem;
  margin-top: 0.35rem;
  color: var(--text-muted);
  font-size: 0.78rem;
  line-height: 1.5;
}

/* Progress */
[data-testid="stProgress"] > div,
[data-testid="stProgressBar"] > div {
  background: rgba(148, 163, 184, 0.12) !important;
  border-radius: 999px !important;
  overflow: hidden;
}

[data-testid="stProgress"] > div > div,
[data-testid="stProgressBar"] > div > div {
  background: var(--gradient-primary) !important;
  border-radius: 999px !important;
  box-shadow: 0 0 18px rgba(124, 58, 237, 0.34);
}

/* Expanders and images */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  background: rgba(7, 11, 20, 0.34);
}

[data-testid="stExpander"] summary {
  color: #CBD5E1 !important;
  font-size: 0.78rem;
  font-weight: 650;
}

[data-testid="stImage"] img {
  border-radius: var(--radius-md);
}

.brand-preview {
  margin: 0.4rem 0 0.7rem;
  padding: 0.7rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: #FFFFFF;
}

/* Toast */
[data-testid="stToast"] {
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--card-elevated) !important;
}

/* Responsive */
@media (max-width: 1050px) {
  .workflow-rail {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .block-container {
    padding-top: 0.85rem;
    padding-left: 0.85rem;
    padding-right: 0.85rem;
  }

  .dashboard-header {
    flex-direction: column;
    padding: 1.3rem;
  }

  .header-meta {
    flex-direction: row;
    align-items: center;
    flex-wrap: wrap;
  }

  .workflow-rail {
    grid-template-columns: 1fr;
    gap: 0.45rem;
  }

  .workflow-item {
    min-height: 2.8rem;
  }

  .summary-grid {
    grid-template-columns: 1fr 1fr;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition: none !important;
    animation: none !important;
  }
}
</style>
"""


def setup_page():
    st.set_page_config(
        page_title="AI Video Rebranding Studio",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)


def init_state():
    if "out_dir" not in st.session_state:
        st.session_state["out_dir"] = tempfile.mkdtemp(prefix="rebrand_")
    if "queue" not in st.session_state:
        st.session_state.queue = []          # list[dict] job records
    if "queue_running" not in st.session_state:
        st.session_state.queue_running = False


def render_header(brand: str):
    safe_brand = html.escape(brand)
    st.markdown(
        f"""
<div class="dashboard-header">
  <div class="header-copy">
    <div class="header-eyebrow">AI video processing workspace</div>
    <h1>Video Rebranding Studio</h1>
    <p>Prepare polished, correctly branded lesson videos with a guided five step
    workflow. Preview the replacement intro, trim the original video, manage a
    processing queue, and download each completed MP4.</p>
  </div>
  <div class="header-meta">
    <div class="brand-chip">Current brand: {safe_brand}</div>
    <div class="version-chip">CDD Department</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _workflow_states(brand: str) -> list[str]:
    details_done = all(
        str(st.session_state.get(key, default)).strip()
        for key, default in (
            ("course_name", "LEVEL 4 DIPLOMA IN EDUCATION STUDIES (RQF)"),
            ("unit_number", "UNIT 01"),
            ("chapter_name", "CHAPTER 01"),
        )
    )
    preview_done = details_done and _intro_path(brand) is not None
    uploads_done = bool(st.session_state.get("video_uploads"))
    queue = st.session_state.get("queue", [])
    queue_added = bool(queue)
    queue_complete = bool(queue) and all(j["status"] == STATUS_DONE for j in queue)

    complete_flags = [details_done, preview_done, uploads_done, queue_added, queue_complete]
    states = ["done" if flag else "upcoming" for flag in complete_flags]
    try:
        active_index = complete_flags.index(False)
    except ValueError:
        active_index = len(complete_flags) - 1
    states[active_index] = "active"
    return states


def render_workflow_tracker(brand: str):
    labels = (
        "Video details",
        "Intro preview",
        "Upload and trim",
        "Add to queue",
        "Processing queue",
    )
    states = _workflow_states(brand)
    items = []
    for index, (label, state) in enumerate(zip(labels, states), start=1):
        number = "✓" if state == "done" else str(index)
        items.append(
            f'<div class="workflow-item {state}">'
            f'<div class="workflow-number">{number}</div>'
            f'<div class="workflow-label">{html.escape(label)}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="workflow-rail">{"".join(items)}</div>', unsafe_allow_html=True)


def sec(num: int, title: str, subtitle: str = "", state: str = "active"):
    safe_state = state if state in {"active", "done", "upcoming"} else "upcoming"
    number = "✓" if safe_state == "done" else str(num)
    subtitle_html = (
        f'<div class="section-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    )
    st.markdown(
        f'<div class="section-heading {safe_state}">'
        f'<div class="section-number">{number}</div>'
        f'<div><div class="section-title">{html.escape(title)}</div>{subtitle_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def panel_heading(icon: str, title: str, subtitle: str = ""):
    subtitle_html = (
        f'<div class="panel-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    )
    st.markdown(
        f'<div class="panel-heading">'
        f'<div class="panel-icon">{html.escape(icon)}</div>'
        f'<div><div class="panel-title">{html.escape(title)}</div>{subtitle_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _status_html(ok: bool, text: str) -> str:
    state_class = "" if ok else " warning"
    return (
        f'<div class="system-status{state_class}">'
        f'<span class="status-light"></span>{html.escape(text)}</div>'
    )


def render_settings_panel():
    with st.container(border=True):
        panel_heading("⚙", "Processing settings", "Brand and encoding profile")
        brand = st.radio(
            "Brand",
            list(BRANDS.keys()),
            horizontal=True,
            key="brand_selector",
        )
        logo = _logo_path(brand)
        intro = _intro_path(brand)

        if logo:
            try:
                li = Image.open(logo).convert("RGBA")
                bg = Image.new("RGB", li.size, (255, 255, 255))
                alpha = li.split()[3] if len(li.split()) == 4 else None
                bg.paste(li, mask=alpha)
                st.image(bg, width="stretch")
            except Exception:
                st.warning("The selected logo could not be previewed.")
        else:
            st.error(f"Missing: {BRANDS[brand]['logo']}")

        st.markdown(
            _status_html(bool(intro), f"Intro source: {intro.name}" if intro else "Intro source is missing"),
            unsafe_allow_html=True,
        )
        st.markdown(
            _status_html(bool(logo), f"Brand asset: {logo.name}" if logo else "Brand logo is missing"),
            unsafe_allow_html=True,
        )

        speed = st.selectbox(
            "Processing speed",
            list(SPEED_PROFILES.keys()),
            index=list(SPEED_PROFILES.keys()).index(DEFAULT_SPEED),
            key="processing_speed",
            help="Fast is recommended for Streamlit Community Cloud.",
        )

       

    return brand, speed, logo, intro


def _format_size(total_bytes: int) -> str:
    if total_bytes <= 0:
        return "0 MB"
    return f"{total_bytes / 1_048_576:.1f} MB"


def render_processing_summary():
    uploads = st.session_state.get("video_uploads") or []
    total_size = sum(int(getattr(upload, "size", 0) or 0) for upload in uploads)
    trim_start = float(st.session_state.get("trim_start_seconds", 9.0))
    trim_end = float(st.session_state.get("trim_end_seconds", 10.0))
    queue = st.session_state.get("queue", [])
    processing = sum(1 for job in queue if job["status"] == STATUS_PROCESSING)
    pending = sum(1 for job in queue if job["status"] == STATUS_PENDING)
    completed = sum(1 for job in queue if job["status"] == STATUS_DONE)

    if processing:
        queue_status = "Processing"
    elif pending:
        queue_status = "Ready"
    elif queue and completed == len(queue):
        queue_status = "Completed"
    elif queue:
        queue_status = "Needs review"
    else:
        queue_status = "Idle"

    metrics = (
        ("Uploads", str(len(uploads))),
        ("File size", _format_size(total_size)),
        ("Total trim", f"{trim_start + trim_end:.1f}s"),
        ("Queue", str(len(queue))),
        ("Status", queue_status),
        ("Completed", str(completed)),
    )
    cards = "".join(
        f'<div class="summary-metric"><div class="summary-label">{html.escape(label)}</div>'
        f'<div class="summary-value">{html.escape(value)}</div></div>'
        for label, value in metrics
    )

    with st.container(border=True):
        panel_heading("◫", "Processing summary", "Current workspace activity")
        st.markdown(f'<div class="summary-grid">{cards}</div>', unsafe_allow_html=True)
        st.caption("Source duration and kept duration are shown after trim-point preview.")


def render_queue_metrics(queue: list[dict]):
    n_pending = sum(1 for job in queue if job["status"] == STATUS_PENDING)
    n_processing = sum(1 for job in queue if job["status"] == STATUS_PROCESSING)
    n_done = sum(1 for job in queue if job["status"] == STATUS_DONE)
    n_error = sum(1 for job in queue if job["status"] == STATUS_ERROR)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Queued", n_pending)
    m2.metric("Processing", n_processing)
    m3.metric("Completed", n_done)
    m4.metric("Failed", n_error)
    return n_pending, n_done, n_error


def render_empty_queue():
    st.markdown(
        """
<div class="empty-state">
  <div class="empty-icon">＋</div>
  <div class="empty-title">Your processing queue is empty</div>
  <div class="empty-copy">Upload one or more source videos, confirm the trim settings,
  and add them to the queue. Jobs will appear here with live status and progress.</div>
</div>
""",
        unsafe_allow_html=True,
    )


CHIP = {
    STATUS_PENDING: (
        '<span class="status-badge status-pending"><span class="status-dot"></span>Queued</span>'
    ),
    STATUS_PROCESSING: (
        '<span class="status-badge status-processing"><span class="status-dot"></span>Processing</span>'
    ),
    STATUS_DONE: (
        '<span class="status-badge status-done"><span class="status-dot"></span>Completed</span>'
    ),
    STATUS_ERROR: (
        '<span class="status-badge status-error"><span class="status-dot"></span>Failed</span>'
    ),
}


def render_job_card(job: dict, live: bool = False):
    """Render one queue entry. When `live` is True the card returns
    placeholders so the processing loop can update them in place."""
    with st.container(border=True):
        title_col, badge_col = st.columns([4, 1])
        with title_col:
            st.markdown(
                f'<div class="job-title">{html.escape(job["out_name"])}</div>'
                f'<div class="job-meta">{html.escape(job["course"])}<br>'
                f'{html.escape(job["unit"])} · {html.escape(job["chapter"])} · '
                f'Source: {html.escape(job["src_name"])}</div>'
                f'<div class="job-details">'
                f'<span class="job-detail-pill">{html.escape(job["brand"])}</span>'
                f'<span class="job-detail-pill">Start trim {job["trim_start"]:.1f}s</span>'
                f'<span class="job-detail-pill">End trim {job["outro_remove"]:.1f}s</span>'
                f'<span class="job-detail-pill">{html.escape(job["speed"])}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with badge_col:
            chip_ph = st.empty()
            chip_ph.markdown(CHIP[job["status"]], unsafe_allow_html=True)

        prog_ph = st.empty()
        stage_ph = st.empty()

        if job["status"] == STATUS_PROCESSING or live:
            prog_ph.progress(min(job["progress"], 1.0))
            stage_ph.caption(job["stage"])
        elif job["status"] == STATUS_PENDING:
            stage_ph.caption("Waiting for processing to begin.")
        elif job["status"] == STATUS_ERROR:
            st.error(job["error"] or "Unknown error")
        elif job["status"] == STATUS_DONE and job["result"] and Path(job["result"]).exists():
            result_path = Path(job["result"])
            duration, _, _, _ = get_media_info(result_path)
            elapsed = f" · completed in {job['elapsed']:.0f}s" if job.get("elapsed") else ""
            stage_ph.caption(
                f"{result_path.stat().st_size / 1_048_576:.1f} MB · {fmt_time(duration)}{elapsed}"
            )
            st.markdown(
                '<div class="download-callout">Your rebranded MP4 is ready. '
                'Download it directly to your device.</div>',
                unsafe_allow_html=True,
            )

            def open_result(path=result_path):
                return path.open("rb")

            st.download_button(
                "Download completed video",
                data=open_result,
                file_name=job["out_name"],
                mime="video/mp4",
                type="primary",
                width="stretch",
                on_click="ignore",
                key=f"dl_{job['id']}",
            )

        if job["status"] in (STATUS_PENDING, STATUS_DONE, STATUS_ERROR) and not live:
            if st.button(
                "Remove job",
                key=f"rm_{job['id']}",
                width="stretch",
                type="tertiary",
            ):
                if job.get("result"):
                    Path(job["result"]).unlink(missing_ok=True)
                st.session_state.queue = [
                    queued_job for queued_job in st.session_state.queue
                    if queued_job["id"] != job["id"]
                ]
                st.rerun()

        return chip_ph, prog_ph, stage_ph


def process_queue():
    """Run all pending jobs sequentially with live in-place progress updates."""
    # Read session state only in Streamlit's main script thread. The resulting
    # normal Path is then passed through the processing pipeline explicitly.
    out_root = Path(st.session_state["out_dir"])
    pending = [job for job in st.session_state.queue if job["status"] == STATUS_PENDING]
    if not pending:
        return
    st.session_state.queue_running = True
    holder = st.container()
    with holder:
        with st.container(border=True):
            panel_heading("◉", "Processing queue", "Jobs run sequentially in cloud-safe mode")
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
                    sp.caption(f"{j['stage']} — {int(j['progress'] * 100)}%")

                try:
                    run_job(job, on_update, out_root)
                    job["status"] = STATUS_DONE
                    chip_ph.markdown(CHIP[STATUS_DONE], unsafe_allow_html=True)
                    prog_ph.progress(1.0)
                    stage_ph.caption("Completed — 100%")
                except Exception as exc:
                    job["status"] = STATUS_ERROR
                    job["error"] = str(exc)
                    chip_ph.markdown(CHIP[STATUS_ERROR], unsafe_allow_html=True)
                    stage_ph.caption("Failed")
    st.session_state.queue_running = False
    st.rerun()


def main():
    setup_page()
    init_state()
    try:
        get_ffmpeg()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    session_out_dir = Path(st.session_state["out_dir"])
    cache_dir = _cache_dir(session_out_dir)
    current_brand = st.session_state.get("brand_selector", list(BRANDS.keys())[0])

    render_header(current_brand)
    render_workflow_tracker(current_brand)

    workflow_col, utility_col = st.columns([2.25, 0.85], gap="large")

    with utility_col:
        brand, speed, logo, intro = render_settings_panel()
        render_processing_summary()

    with workflow_col:
        # 1 · Video details
        with st.container(border=True):
            sec(
                1,
                "Video details",
                "Enter the exact titles that should appear in the branded intro.",
                _workflow_states(brand)[0],
            )
            course = st.text_input(
                "Course name",
                "LEVEL 4 DIPLOMA IN EDUCATION STUDIES (RQF)",
                key="course_name",
                help="The full course name is always shown. Long names wrap automatically.",
            )
            details_left, details_right = st.columns(2)
            with details_left:
                unit = st.text_input("Unit number", "UNIT 01", key="unit_number")
            with details_right:
                chapter = st.text_input("Chapter name", "CHAPTER 01", key="chapter_name")

        # 2 · Intro preview
        with st.container(border=True):
            sec(
                2,
                "Intro preview",
                "Review a real frame from the selected brand intro before processing.",
                _workflow_states(brand)[1],
            )
            preview = make_preview(brand, course, unit, chapter)
            if preview:
                st.image(
                    preview,
                    width="stretch",
                    caption="Preview of the final branded intro frame",
                )
                st.caption(
                    "Long course names wrap across multiple lines and are not truncated."
                )
            else:
                st.info("The intro preview is unavailable because the intro source file was not found.")

        # 3 · Upload and trim
        with st.container(border=True):
            sec(
                3,
                "Upload and trim",
                "Add source videos and define how much of the original intro and outro to remove.",
                _workflow_states(brand)[2],
            )
            st.markdown(
                '<div class="upload-intro"><div class="upload-intro-icon">⇧</div>'
                '<div><strong>Drop your lesson videos below</strong><br>'
                'MP4, MOV, AVI and MKV are supported. Multiple files can be queued together.'
                '</div></div>',
                unsafe_allow_html=True,
            )
            uploads = st.file_uploader(
                "Upload source videos",
                ["mp4", "mov", "avi", "mkv"],
                accept_multiple_files=True,
                key="video_uploads",
                help="Choose one or more original SLC lesson videos.",
            )

            trim_left, trim_right = st.columns(2)
            with trim_left:
                trim_start = st.number_input(
                    "Remove from start (seconds)",
                    0.0,
                    300.0,
                    9.0,
                    0.5,
                    "%.1f",
                    key="trim_start_seconds",
                    help="Seconds of the old SLC intro to remove.",
                )
            with trim_right:
                outro_remove = st.number_input(
                    "Remove from end (seconds)",
                    0.0,
                    300.0,
                    10.0,
                    0.5,
                    "%.1f",
                    key="trim_end_seconds",
                    help="Seconds of the old SLC outro to remove.",
                )

            if uploads and st.button(
                "Preview trim points for the first video",
                width="stretch",
                key="preview_trim_points",
            ):
                with st.spinner("Preparing trim-point preview..."):
                    src0 = upload_cache_path(uploads[0], cache_dir)
                    first, last, source_duration = trim_preview(
                        src0, trim_start, outro_remove
                    )
                if trim_start + outro_remove >= source_duration:
                    st.error(
                        f"The current trim removes the entire video. Source duration: "
                        f"{fmt_time(source_duration)}."
                    )
                else:
                    kept_duration = source_duration - trim_start - outro_remove
                    duration_col, kept_col, size_col = st.columns(3)
                    duration_col.metric("Source duration", fmt_time(source_duration))
                    kept_col.metric("Kept duration", fmt_time(kept_duration))
                    size_col.metric(
                        "File size",
                        _format_size(int(getattr(uploads[0], "size", 0) or 0)),
                    )

                    preview_left, preview_right = st.columns(2)
                    with preview_left:
                        if first is not None:
                            st.image(
                                first,
                                width="stretch",
                                caption=f"First kept frame at {fmt_time(trim_start)}",
                            )
                        else:
                            st.warning("The first kept frame could not be read.")
                    with preview_right:
                        if last is not None:
                            st.image(
                                last,
                                width="stretch",
                                caption=(
                                    "Last kept frame at "
                                    f"{fmt_time(source_duration - outro_remove)}"
                                ),
                            )
                        else:
                            st.warning("The last kept frame could not be read.")
                    st.caption(
                        "If the last kept frame still shows lesson content rather than the "
                        "old outro, reduce the end-trim value."
                    )

        # 4 · Add to queue
        with st.container(border=True):
            sec(
                4,
                "Add to queue",
                "Confirm the output name and create one processing job per uploaded video.",
                _workflow_states(brand)[3],
            )
            st.markdown(
                f'<div class="info-box">The exact <strong>Intro.mp4</strong> animation and '
                f'audio are preserved. Only the course name, unit and chapter are replaced, '
                f'and the selected <strong>{html.escape(brand)}</strong> branding is applied. '
                f'Each uploaded video becomes a separate queue job.</div>',
                unsafe_allow_html=True,
            )

            custom_name = ""
            if uploads and len(uploads) == 1:
                default_stem = Path(uploads[0].name).stem
                custom_name = st.text_input(
                    "Output filename",
                    f"{default_stem}_{brand.lower()}_rebranded.mp4",
                )

            upload_count = len(uploads) if uploads else 0
            button_text = (
                f"Add {upload_count} video{'s' if upload_count != 1 else ''} to queue"
                if upload_count
                else "Upload videos to add them to the queue"
            )
            add = st.button(
                button_text,
                type="primary",
                width="stretch",
                disabled=not (uploads and logo and intro),
                key="add_to_queue",
            )
            if add and uploads:
                for uploaded in uploads:
                    src = upload_cache_path(uploaded, cache_dir)
                    stem = Path(uploaded.name).stem
                    if len(uploads) == 1 and custom_name:
                        out_name = safe_name(custom_name, stem, brand)
                    else:
                        out_name = safe_name("", stem, brand)
                    st.session_state.queue.append(
                        new_job(
                            src,
                            uploaded.name,
                            brand,
                            course,
                            unit,
                            chapter,
                            trim_start,
                            outro_remove,
                            out_name,
                            speed,
                        )
                    )
                st.toast(f"Added {len(uploads)} job(s) to the processing queue", icon="✅")
                st.rerun()

        # 5 · Processing queue
        with st.container(border=True):
            sec(
                5,
                "Processing queue",
                "Start pending jobs, monitor progress and download completed videos.",
                _workflow_states(brand)[4],
            )
            queue = st.session_state.queue

            if not queue:
                render_empty_queue()
            else:
                n_pending, n_done, n_error = render_queue_metrics(queue)

                start_col, clear_col = st.columns([2, 1])
                with start_col:
                    start = st.button(
                        f"Start queue ({n_pending} pending)",
                        type="primary",
                        width="stretch",
                        disabled=n_pending == 0,
                        key="start_processing_queue",
                    )
                with clear_col:
                    clear_finished = st.button(
                        "Clear finished",
                        width="stretch",
                        disabled=(n_done + n_error) == 0,
                        type="tertiary",
                        key="clear_finished_jobs",
                    )

                if clear_finished:
                    for job in queue:
                        if job["status"] in (STATUS_DONE, STATUS_ERROR) and job.get("result"):
                            Path(job["result"]).unlink(missing_ok=True)
                    st.session_state.queue = [
                        job for job in queue
                        if job["status"] in (STATUS_PENDING, STATUS_PROCESSING)
                    ]
                    st.rerun()

                if start:
                    process_queue()
                else:
                    for job in queue:
                        render_job_card(job)


if __name__ == "__main__":
    main()
