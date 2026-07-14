"""
Video Rebranding Tool v14
Uses the EXACT uploaded Intro.mp4 — only changes the course name, unit number and chapter name.
All animations, logo, 3D shapes and audio are preserved perfectly.
The unit and chapter appear together in the pill box: "UNIT 01 -  CHAPTER 01"
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

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

# ─────────────────────────────────────────────
#  Pixel-measured regions in Intro.mp4
#  (re-measured directly from the current Intro.mp4 at t=8.5s,
#   full render settled: line1 rows 380-436, line2 rows 478-565,
#   pill fill rows 622-708 / cols 642-1276)
# ─────────────────────────────────────────────
# Original title text lives at Y=380-565 (two lines).
# We erase a wider band for safety, staying clear of the pill below.
TITLE_ERASE_Y = 330
TITLE_ERASE_H = 280   # covers Y 330-610

# New title is rendered centred in that band.
TITLE_CENTER_Y = 472  # = (380+565)//2 vertical midpoint for new text block

# Original pill: Y=622-708, X=642-1276, centre X≈959
# We erase the full band and redraw.
PILL_ERASE_Y  = 595
PILL_ERASE_H  = 145   # covers Y 595-740

# New pill sits at the same vertical centre as original.
PILL_CENTER_Y = 665   # = (622+708)//2
PILL_MIN_W    = 580   # wide enough to cover the original

# ─────────────────────────────────────────────
#  Entrance-animation timing (measured directly from Intro.mp4 by sampling
#  brightness in each region frame-by-frame — the title fades in gradually,
#  the pill pops in almost instantly, then its text follows a beat later)
# ─────────────────────────────────────────────
TITLE_FADE_START = 0.90   # seconds — title begins fading in
TITLE_FADE_DUR   = 0.30   # seconds — fade duration
PILL_FADE_START  = 1.90   # seconds — pill box appears (near-instant in the original)
PILL_FADE_DUR    = 0.10   # seconds — kept short to match the original's snappy pop-in

# Background colours (sampled from sides of each band)
TITLE_BG_HEX  = "9B5EE1"
PILL_BG_HEX   = "945BE1"

# Content-video SLC logo box (unchanged from v11)
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
        return [get_ffmpeg(), *cmd[1:]]
    return cmd

def run(cmd: list[str], label: str) -> None:
    r = subprocess.run(ff(cmd), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{label} failed.\n\n{(r.stderr or '')[-4000:]}")

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
    return ["-c:v", "libx264", "-preset", p["preset"], "-crf", str(crf or p["crf"])]

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
        try:
            st.error(
                "⚠ Poppins-Bold.ttf was not found next to app.py and no system "
                "font was found either. Falling back to a tiny placeholder font — "
                "title/pill text will look wrong. Make sure Poppins-Bold.ttf ships "
                "in the same folder as app.py."
            )
        except Exception:
            pass
    return ImageFont.load_default()

def tsz(draw, text: str, font) -> Tuple[int, int]:
    bb = draw.textbbox((0, 0), text or " ", font=font)
    return bb[2]-bb[0], bb[3]-bb[1]

def wrap(draw, text: str, font, max_w: int, max_lines: int = 2) -> list[str]:
    text = text.strip().upper()
    words = text.split()
    lines, cur = [], ""
    for i, w in enumerate(words):
        trial = w if not cur else f"{cur} {w}"
        if tsz(draw, trial, font)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines - 1:
                cur = " ".join(words[i:])
                break
    if cur:
        lines.append(cur)
    lines = lines[:max_lines]
    out = []
    for ln in lines:
        if tsz(draw, ln, font)[0] <= max_w:
            out.append(ln)
        else:
            v = ln
            while v and tsz(draw, v+"...", font)[0] > max_w:
                v = v[:-1]
            out.append((v.rstrip()+"...") if v else ln[:20])
    return out or [text[:20]]

def fit_title(draw, text: str, max_w: int):
    for size in range(82, 34, -2):
        font = make_font(size)
        lines = wrap(draw, text, font, max_w)
        if all(tsz(draw, l, font)[0] <= max_w for l in lines):
            return font, lines
    font = make_font(34)
    return font, wrap(draw, text, font, max_w)


# ─────────────────────────────────────────────
#  Extract clean background rows from intro video
# ─────────────────────────────────────────────
def _extract_clean_bg(intro: Path) -> Optional[np.ndarray]:
    """
    Sample the intro at t=280ms (before any text animates in) to get
    clean background pixel rows for the erase zones.
    Returns a (1080, 1920, 3) uint8 array, or None on failure.
    """
    cap = cv2.VideoCapture(str(intro))
    cap.set(cv2.CAP_PROP_POS_MSEC, 280)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    arr = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    return arr.astype(np.uint8)


# ─────────────────────────────────────────────
#  Core: build the overlay PNG for one set of text
# ─────────────────────────────────────────────
def _clean_bg_layer(intro_path: Optional[Path], erase_y: int, erase_h: int) -> Image.Image:
    """A full-canvas transparent RGBA image with just one band filled in with
    real background pixels copied from the intro (or a flat fallback colour)."""
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
    """Returns (erase_layer, text_layer) for the title band, as separate
    images so they can be faded in on different schedules."""
    tc = BRANDS[brand]["title_color"]
    course_text = (course or "COURSE NAME").strip().upper()

    erase = _clean_bg_layer(intro_path, TITLE_ERASE_Y, TITLE_ERASE_H)
    text_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer, "RGBA")

    max_title_w = 1580
    t_font, t_lines = fit_title(draw, course_text, max_title_w)
    ascent, descent = t_font.getmetrics()
    line_h = ascent + descent          # natural font line-height (matches original spacing)
    total_h = len(t_lines) * line_h
    title_top = TITLE_CENTER_Y - total_h // 2

    for i, line in enumerate(t_lines):
        w, _ = tsz(draw, line, t_font)
        x = TARGET_W // 2 - w // 2
        y = title_top + i * line_h
        draw.text((x + 3, y + 5), line, font=t_font, fill=(0, 0, 0, 60))
        draw.text((x, y),         line, font=t_font, fill=(*tc, 255))

    return erase, text_layer


def build_pill_layers(brand: str, unit_no: str, chapter: str,
                       intro_path: Optional[Path] = None) -> Tuple[Image.Image, Image.Image]:
    """Returns (erase_layer, text_layer) for the pill band, as separate
    images so they can be faded in on different schedules."""
    pb = BRANDS[brand]["pill_bg"]
    pt = BRANDS[brand]["pill_text"]
    unit_text    = (unit_no or "UNIT 01").strip().upper()
    chapter_text = (chapter or "CHAPTER 01").strip().upper()
    pill_line    = f"{unit_text} -  {chapter_text}"

    erase = _clean_bg_layer(intro_path, PILL_ERASE_Y, PILL_ERASE_H)
    text_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer, "RGBA")

    p_font = make_font(46)
    bbox = draw.textbbox((0, 0), pill_line, font=p_font)
    top_offset = bbox[1]                     # gap between draw-anchor and true ink top
    uw, uh = bbox[2]-bbox[0], bbox[3]-bbox[1]
    pw = min(max(uw + 160, PILL_MIN_W), 1080)
    ph = 88
    px = TARGET_W // 2 - pw // 2
    py = PILL_CENTER_Y - ph // 2

    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=44, fill=(*pb, 255))
    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=44, outline=(*pt, 80), width=5)
    # True vertical centering: ink runs from (y_drawn + top_offset) to
    # (y_drawn + top_offset + uh), so subtract top_offset (not a magic
    # constant) to actually land the ink in the middle of the pill.
    y_drawn = py + (ph - uh) // 2 - top_offset
    draw.text((TARGET_W // 2 - uw // 2, y_drawn),
              pill_line, font=p_font, fill=(*pt, 255))

    return erase, text_layer


def build_title_overlay_png(out_png: Path, brand: str, course: str,
                             intro_path: Optional[Path] = None) -> None:
    """Static (erase+text already merged) title overlay — used for the still preview."""
    erase, text_layer = build_title_layers(brand, course, intro_path)
    Image.alpha_composite(erase, text_layer).save(out_png)


def build_pill_overlay_png(out_png: Path, brand: str, unit_no: str, chapter: str,
                            intro_path: Optional[Path] = None) -> None:
    """Static (erase+text already merged) pill overlay — used for the still preview."""
    erase, text_layer = build_pill_layers(brand, unit_no, chapter, intro_path)
    Image.alpha_composite(erase, text_layer).save(out_png)


def build_overlay_png(
    out_png: Path,
    brand: str,
    course: str,
    unit_no: str,
    chapter: str,
    intro_path: Optional[Path] = None,
) -> None:
    """Static (non-animated) combined overlay — used for the still preview
    image. The real video path uses build_title_overlay_png /
    build_pill_overlay_png separately so each can fade in at its own time."""
    title_png = out_png.with_suffix(".title.png")
    pill_png  = out_png.with_suffix(".pill.png")
    try:
        build_title_overlay_png(title_png, brand, course, intro_path)
        build_pill_overlay_png(pill_png, brand, unit_no, chapter, intro_path)
        title_img = Image.open(title_png).convert("RGBA")
        pill_img  = Image.open(pill_png).convert("RGBA")
        Image.alpha_composite(title_img, pill_img).save(out_png)
    finally:
        title_png.unlink(missing_ok=True)
        pill_png.unlink(missing_ok=True)


# ─────────────────────────────────────────────
#  Preview (static still from the video)
# ─────────────────────────────────────────────
def make_preview(brand: str, course: str, unit_no: str, chapter: str) -> Optional[Image.Image]:
    intro = _intro_path(brand)
    if not intro:
        return None

    # Use the same clean frame (t=280ms, before text animates in) as the base
    # so the preview is free of the original course/unit/chapter text.
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
# How fast the erase band snaps to fully opaque. Kept short and started
# slightly before the corresponding text fade so the original graphic is
# already fully hidden before it would start playing its own entrance
# animation — otherwise both the original (fading in on its own) and our
# semi-transparent replacement would be visible at once, producing a
# double-exposure ghost during the transition.
ERASE_SNAP_DUR = 0.08
ERASE_LEAD     = 0.05

def generate_intro_clip(
    out_path: Path,
    brand: str,
    course: str,
    unit_no: str,
    chapter: str,
    speed: str = DEFAULT_SPEED,
) -> None:
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
def process_content(
    src: Path, out: Path, media_info, trim_start: float, trim_end: float,
    logo: Path, speed: str,
) -> None:
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
    run(cmd, "Process content segment")


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
        # fallback re-encode
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
    # First try the uploaded generic Intro.mp4 (user-supplied template)
    generic = _base() / "Intro.mp4"
    if generic.exists():
        return generic
    # Fall back to brand-prefixed version
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
#  Single-video processing
# ─────────────────────────────────────────────
def _cache_dir() -> Path:
    d = Path(st.session_state.out_dir) / "_cache"
    d.mkdir(exist_ok=True)
    return d

def upload_cache_path(uploaded) -> Path:
    """Write (once per uploaded file) a cached on-disk copy, keyed by name+size,
    so re-running previews doesn't re-write the bytes every time."""
    key = hashlib.sha1(f"{uploaded.name}|{uploaded.size}".encode()).hexdigest()[:16]
    p = _cache_dir() / f"{key}.mp4"
    if not p.exists():
        p.write_bytes(uploaded.getvalue())
    return p

def get_upload_duration(uploaded) -> float:
    dur, _, _, _ = get_media_info(upload_cache_path(uploaded))
    return dur

def extract_frame_at(path: Path, t: float) -> Optional[Image.Image]:
    """Grab a single frame at time t (seconds) as a PIL Image, or None on failure."""
    t = max(t, 0.0)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def trim_preview(uploaded, trim_start: float, outro_remove: float):
    """Returns (first_kept_frame, last_kept_frame, duration) for the given
    trim settings, so the user can see exactly where the cut lands before
    processing — trimming a fixed number of seconds blindly risks either
    leaving old branding in or cutting off real lesson content."""
    src = upload_cache_path(uploaded)
    dur = get_upload_duration(uploaded)
    t0 = float(trim_start)
    t1 = max(dur - float(outro_remove), 0.0)
    first = extract_frame_at(src, t0)
    last  = extract_frame_at(src, max(t1 - 0.05, 0.0))
    return first, last, dur


def process_one(
    uploaded, brand: str, logo: Path, outro_norm: Path,
    trim_start: float, outro_remove: float,
    course: str, unit_no: str, chapter: str,
    out_name: str, tmpdir: Path, idx: int, speed: str,
) -> Path:
    src = tmpdir / f"src_{idx}.mp4"
    src.write_bytes(uploaded.getvalue())
    info = get_media_info(src)
    dur, w, h, _ = info
    if w == 0:
        raise ValueError(f"{uploaded.name}: no video stream found.")
    t0, t1 = float(trim_start), dur - float(outro_remove)
    if t1 <= t0:
        raise ValueError(f"Trim removes entire video (duration {fmt_time(dur)}).")

    intro_out = tmpdir / f"intro_{idx}.mp4"
    generate_intro_clip(intro_out, brand, course, unit_no, chapter, speed)

    content_out = tmpdir / f"content_{idx}.mp4"
    process_content(src, content_out, info, t0, t1, logo, speed)

    final = tmpdir / out_name
    concat_clips(intro_out, content_out, outro_norm, final, speed)
    if not final.exists() or final.stat().st_size == 0:
        raise RuntimeError("Output file not created.")
    return final


def run_single(uploaded, brand: str, logo: Path, course: str, unit_no: str,
                chapter: str, out_name: str, trim_start: float,
                outro_remove: float, speed: str) -> Path:
    out_dir = Path(st.session_state.out_dir); out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        st.session_state._progress_text = "Preparing outro…"
        outro_raw  = tmpdir / "outro_raw.mp4"
        outro_norm = tmpdir / "outro_norm.mp4"
        op = _outro_path(brand)
        if op:
            shutil.copy(op, outro_raw)
        else:
            logo_slate(logo, outro_raw, brand, 3.0, speed)
        normalize_clip(outro_raw, outro_norm, speed)

        result = process_one(
            uploaded, brand, logo, outro_norm,
            trim_start, outro_remove,
            course, unit_no, chapter,
            out_name, tmpdir, 1, speed,
        )
        dest = out_dir / out_name
        shutil.copy(result, dest)
        return dest


# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────
def setup_page():
    st.set_page_config(page_title="Video Rebranding Tool", page_icon="🎬", layout="centered")
    st.markdown("""
<style>
.block-container{max-width:820px;padding-top:1.2rem}
.hero{padding:20px 24px;border-radius:16px;background:linear-gradient(135deg,#15243F,#1A5D82);color:#fff;margin-bottom:14px}
.hero h1{margin:0;font-size:1.9rem;color:#fff}
.hero p{margin:6px 0 0;color:#EAF4FB;font-size:.95rem}
.info-box{background:#EAF4FB;border-left:4px solid #1565C0;padding:10px 13px;border-radius:8px;color:#16324F;font-size:.9rem}
</style>""", unsafe_allow_html=True)


def init_state():
    if "out_dir" not in st.session_state:
        st.session_state.out_dir = tempfile.mkdtemp(prefix="rebrand_")
    if "result_path" not in st.session_state:
        st.session_state.result_path = None
    if "result_error" not in st.session_state:
        st.session_state.result_error = None


def main():
    setup_page()
    init_state()
    try:
        get_ffmpeg()
    except RuntimeError as e:
        st.error(str(e)); st.stop()

    st.markdown("""
<div class="hero">
  <h1>🎬 Video Rebranding Tool</h1>
  <p>Uses your exact Intro.mp4 — only replaces the course name, unit number and chapter in the intro.</p>
</div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.header("Settings")
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

    # ── 1 · Video details ─────────────────────
    with st.container(border=True):
        st.subheader("1 · Video details")
        course = st.text_input("Course name", "LEVEL 4 DIPLOMA IN EDUCATION STUDIES (RQF)")
        c1, c2 = st.columns(2)
        with c1:
            unit = st.text_input("Unit number", "UNIT 01")
        with c2:
            chapter = st.text_input("Chapter name", "CHAPTER 01")

    # ── 2 · Intro preview ─────────────────────
    with st.container(border=True):
        st.subheader("2 · Intro preview")
        st.caption("Shows the real intro video with your text applied.")
        prev = make_preview(brand, course, unit, chapter)
        if prev:
            st.image(prev, use_container_width=True, caption="Real intro + your text overlay")
        else:
            st.info("Intro.mp4 not found.")

    # ── 3 · Upload + trim ─────────────────────
    with st.container(border=True):
        st.subheader("3 · Upload & trim")
        uploaded = st.file_uploader("Choose an SLC video", ["mp4", "mov", "avi", "mkv"])

        t1, t2 = st.columns(2)
        with t1:
            trim_start = st.number_input("Remove from start (s)", 0.0, 300.0, 9.0, 0.5, "%.1f",
                                          help="Seconds of the old SLC intro to cut.")
        with t2:
            outro_remove = st.number_input("Remove from end (s)", 0.0, 300.0, 10.0, 0.5, "%.1f",
                                            help="Seconds of the old SLC outro to cut.")

        if uploaded and st.button("👁 Preview trim points", use_container_width=True):
            with st.spinner("Grabbing frames…"):
                first, last, src_dur = trim_preview(uploaded, trim_start, outro_remove)
            if trim_start + outro_remove >= src_dur:
                st.error(f"Trim removes the entire video (source is only {fmt_time(src_dur)}).")
            else:
                p1, p2 = st.columns(2)
                with p1:
                    if first is not None:
                        st.image(first, use_container_width=True,
                                 caption=f"First kept frame @ {fmt_time(trim_start)}")
                    else:
                        st.warning("Couldn't read that frame.")
                with p2:
                    if last is not None:
                        st.image(last, use_container_width=True,
                                 caption=f"Last kept frame @ {fmt_time(src_dur - outro_remove)}")
                    else:
                        st.warning("Couldn't read that frame.")
                st.caption(f"Source duration: {fmt_time(src_dur)}. If the \"last kept frame\" "
                           "still shows real lesson content (not the old outro), lower "
                           "\"Remove from end\".")

    # ── 4 · Process ────────────────────────────
    with st.container(border=True):
        st.subheader("4 · Process")
        st.markdown(
            f'<div class="info-box">The exact <b>Intro.mp4</b> is used. '
            f'Only the course name, unit number and chapter name are replaced in the intro. '
            f'All animations, music and the {brand} logo are kept intact.</div>',
            unsafe_allow_html=True)
        st.write("")

        default_stem = Path(uploaded.name).stem if uploaded else "output"
        out_name_raw = st.text_input("Output filename", f"{default_stem}_{brand.lower()}_rebranded.mp4"
                                      if uploaded else "")
        out_name = safe_name(out_name_raw, default_stem, brand)

        go = st.button("▶ Process video", type="primary", use_container_width=True,
                        disabled=not (uploaded and logo))

        if go and uploaded and logo:
            st.session_state.result_path = None
            st.session_state.result_error = None
            try:
                with st.spinner("Processing… this can take a minute for longer videos."):
                    dest = run_single(uploaded, brand, logo, course, unit, chapter,
                                       out_name, trim_start, outro_remove, speed)
                st.session_state.result_path = str(dest)
            except Exception as e:
                st.session_state.result_error = str(e)

        if st.session_state.result_error:
            st.error(st.session_state.result_error)

        if st.session_state.result_path and Path(st.session_state.result_path).exists():
            p = Path(st.session_state.result_path)
            dur, _, _, _ = get_media_info(p)
            st.success(f"Ready · {p.stat().st_size/1_048_576:.1f} MB · {fmt_time(dur)}")
            st.download_button("⬇ Download", data=p.read_bytes(), file_name=p.name,
                                mime="video/mp4", type="primary", use_container_width=True)


if __name__ == "__main__":
    main()
