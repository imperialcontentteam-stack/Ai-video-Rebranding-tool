"""
Video Rebranding Tool v13
Uses the EXACT uploaded Intro.mp4 — only changes course name, unit number and chapter name.
All animations, logo, 3D shapes and audio are preserved perfectly.
"""
from __future__ import annotations

import io
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
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
#  (analysed at 1920×1080, t=5s)
# ─────────────────────────────────────────────
# Original title text lives at Y=290-455.
# We erase a slightly wider band for safety.
TITLE_ERASE_Y = 270
TITLE_ERASE_H = 200   # covers Y 270-470

# New title is rendered centred in that band.
TITLE_CENTER_Y = 370  # vertical midpoint for new text block

# Original pill:  Y=478-565, X=676-1241, centre X≈958
# We erase the full band and redraw.
PILL_ERASE_Y  = 455
PILL_ERASE_H  = 160   # covers Y 455-615

# New pill sits at the same vertical centre as original.
PILL_CENTER_Y = 521   # = (478+565)//2
PILL_MIN_W    = 580   # wide enough to cover the original

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
        os.environ.get("POPPINS_BOLD_FONT", "") if bold else "",
        str(base / ("Poppins-Bold.ttf" if bold else "Poppins-Regular.ttf")),
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

def make_font(size: int, bold: bool = True):
    fp = find_font(bold)
    return ImageFont.truetype(fp, size) if fp else ImageFont.load_default()

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
#  Core: build the overlay PNG for one set of text
# ─────────────────────────────────────────────
def build_overlay_png(
    out_png: Path,
    brand: str,
    course: str,
    unit_no: str,
    chapter: str,
) -> None:
    """
    Two-layer approach so the erase rectangles are never corrupted by
    semi-transparent pill strokes drawn on the same canvas:
      Layer A (erase)  — solid-colour rectangles that blank out the original text.
      Layer B (text)   — new title + new pill on a transparent canvas.
    Both are saved as a single flat PNG via alpha_composite(A, B).
    FFmpeg then composites this PNG over every frame of the intro video.
    """
    title_bg  = tuple(int(TITLE_BG_HEX[i:i+2], 16) for i in (0, 2, 4))
    pill_bg_c = tuple(int(PILL_BG_HEX[i:i+2],  16) for i in (0, 2, 4))

    tc = BRANDS[brand]["title_color"]
    pb = BRANDS[brand]["pill_bg"]
    pt = BRANDS[brand]["pill_text"]

    course_text  = (course  or "COURSE NAME").strip().upper()
    unit_text    = (unit_no or "UNIT 01").strip().upper()
    chapter_text = (chapter or "CHAPTER 01").strip().upper()
    pill_line    = f"{unit_text} -  {chapter_text}"

    # ── Layer A: erase ─────────────────────────────────
    erase = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    ed = ImageDraw.Draw(erase, "RGBA")
    ed.rectangle([0, TITLE_ERASE_Y, TARGET_W, TITLE_ERASE_Y + TITLE_ERASE_H],
                 fill=(*title_bg,  255))
    ed.rectangle([0, PILL_ERASE_Y,  TARGET_W, PILL_ERASE_Y  + PILL_ERASE_H],
                 fill=(*pill_bg_c, 255))

    # ── Layer B: new text + new pill ───────────────────
    text_layer = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer, "RGBA")

    # Title
    max_title_w = 1580
    t_font, t_lines = fit_title(draw, course_text, max_title_w)
    lh = tsz(draw, t_lines[0], t_font)[1]
    gap = 16
    total_h = len(t_lines) * lh + (len(t_lines) - 1) * gap
    title_top = TITLE_CENTER_Y - total_h // 2

    for i, line in enumerate(t_lines):
        w, _ = tsz(draw, line, t_font)
        x = TARGET_W // 2 - w // 2
        y = title_top + i * (lh + gap)
        draw.text((x + 3, y + 5), line, font=t_font, fill=(0, 0, 0, 60))
        draw.text((x, y),         line, font=t_font, fill=(*tc, 255))

    # Pill
    p_font = make_font(46)
    uw, uh = tsz(draw, pill_line, p_font)
    pw = min(max(uw + 160, PILL_MIN_W), 1080)
    ph = 88
    px = TARGET_W // 2 - pw // 2
    py = PILL_CENTER_Y - ph // 2

    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=44, fill=(*pb, 255))
    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=44, outline=(*pt, 80), width=5)
    draw.text((TARGET_W // 2 - uw // 2, py + (ph - uh) // 2 - 4),
              pill_line, font=p_font, fill=(*pt, 255))

    # ── Merge and save ─────────────────────────────────
    Image.alpha_composite(erase, text_layer).save(out_png)


# ─────────────────────────────────────────────
#  Preview (static still from the video)
# ─────────────────────────────────────────────
def make_preview(brand: str, course: str, unit_no: str, chapter: str) -> Optional[Image.Image]:
    intro = _intro_path(brand)
    if not intro:
        return None

    cap = cv2.VideoCapture(str(intro))
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
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
        build_overlay_png(tmp, brand, course, unit_no, chapter)
        # The overlay PNG already has erase+text merged; composite once onto base
        overlay = Image.open(tmp).convert("RGBA")
        return Image.alpha_composite(base, overlay).convert("RGB")
    finally:
        tmp.unlink(missing_ok=True)


# ─────────────────────────────────────────────
#  Generate the intro clip with new text
# ─────────────────────────────────────────────
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

    overlay_png = out_path.with_suffix(".overlay.png")
    try:
        build_overlay_png(overlay_png, brand, course, unit_no, chapter)

        # FFmpeg filter:
        #  [0] intro video   → scale to 1920×1080, fps=30
        #  [1] overlay PNG   → loop, composite with alpha
        vf = (
            f"[0:v]{scale_filter()}[base];"
            f"[1:v]format=rgba[ov];"
            f"[base][ov]overlay=0:0:format=auto,format=yuv420p[v]"
        )

        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", str(intro),
            "-loop", "1", "-i", str(overlay_png),
            "-filter_complex", vf,
            "-map", "[v]",
        ]
        if audio:
            cmd += ["-map", "0:a:0"]
        else:
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}",
                    "-i", f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
                    "-map", "2:a:0"]

        cmd += [
            *enc_args(speed),
            "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-ac", "2",
            "-movflags", "+faststart",
            "-t", f"{dur:.3f}",
            str(out_path),
        ]
        run(cmd, "Generate intro with text overlay")
    finally:
        overlay_png.unlink(missing_ok=True)


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
#  Queue processing
# ─────────────────────────────────────────────
@dataclass
class QFile:
    name: str
    data: bytes
    def getvalue(self): return self.data

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


# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────
def setup_page():
    st.set_page_config(page_title="Video Rebranding Tool", page_icon="🎬", layout="wide")
    st.markdown("""
<style>
.block-container{max-width:1200px;padding-top:1.2rem}
.hero{padding:20px 24px;border-radius:16px;background:linear-gradient(135deg,#15243F,#1A5D82);color:#fff;margin-bottom:14px}
.hero h1{margin:0;font-size:1.9rem;color:#fff}
.hero p{margin:6px 0 0;color:#EAF4FB;font-size:.95rem}
.badge{display:inline-block;padding:3px 9px;border-radius:999px;font-size:.78rem;font-weight:700}
.bq{background:#EEF2FF;color:#334155}.bp{background:#FFF4D6;color:#8A5200}
.bd{background:#EAF7EF;color:#166534}.bf{background:#FDECEC;color:#991B1B}
.info-box{background:#EAF4FB;border-left:4px solid #1565C0;padding:10px 13px;border-radius:8px;color:#16324F;font-size:.9rem}
</style>""", unsafe_allow_html=True)


def init_state():
    if "jobs" not in st.session_state:
        st.session_state.jobs = []
    if "out_dir" not in st.session_state:
        st.session_state.out_dir = tempfile.mkdtemp(prefix="rebrand_")


def badge(status: str) -> str:
    css = {"queued":"bq","processing":"bp","done":"bd","failed":"bf"}.get(status.lower(),"bq")
    return f'<span class="badge {css}">{status}</span>'


def job_id(name: str, data: bytes) -> str:
    return hashlib.sha1(f"{name}|{len(data)}|{hashlib.sha1(data).hexdigest()}".encode()).hexdigest()[:14]


def add_to_queue(uploads, brand, default_course, default_unit, default_chapter):
    existing = {j["id"] for j in st.session_state.jobs}
    added = skipped = 0
    start = len(st.session_state.jobs) + 1
    for i, up in enumerate(uploads or []):
        data = up.getvalue()
        jid  = job_id(up.name, data)
        if jid in existing:
            skipped += 1; continue
        stem = Path(up.name).stem
        st.session_state.jobs.append({
            "id": jid, "name": up.name, "data": data,
            "size": f"{len(data)/1_048_576:.1f} MB",
            "course":  default_course.strip() or "COURSE NAME",
            "unit":    f"UNIT {start+added:02d}",
            "chapter": default_chapter.strip() or "CHAPTER 01",
            "out_name": safe_name("", stem, brand),
            "status": "Queued", "error": "", "output": "",
            "dur": "", "file_size": "",
        })
        existing.add(jid); added += 1
    return added, skipped


def counts():
    c = {"Total":0,"Queued":0,"Processing":0,"Done":0,"Failed":0}
    for j in st.session_state.jobs:
        c["Total"] += 1; c[j["status"]] = c.get(j["status"],0)+1
    return c


def requeue(job):
    if job["status"] != "Processing":
        if job["output"] and Path(job["output"]).exists():
            Path(job["output"]).unlink(missing_ok=True)
        job.update({"status":"Queued","error":"","output":"","dur":"","file_size":""})


def run_queue(brand: str, logo: Path, trim_start: float, outro_remove: float, speed: str):
    pending = [j for j in st.session_state.jobs if j["status"] in ("Queued","Failed")]
    if not pending:
        st.info("Nothing to process."); return

    s_slot = st.empty(); t_slot = st.empty(); p_bar = st.progress(0)
    out_dir = Path(st.session_state.out_dir); out_dir.mkdir(exist_ok=True)

    def show(msg, pct):
        s_slot.info(msg); p_bar.progress(min(max(pct,0),100))
        rows = [{"#":i+1,"File":j["name"],"Unit":j["unit"],
                 "Chapter":j["chapter"],"Status":j["status"],
                 "Size":j["size"]}
                for i,j in enumerate(st.session_state.jobs)]
        t_slot.dataframe(rows, use_container_width=True, hide_index=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        show("Preparing outro…", 3)

        # Prepare outro
        outro_raw  = tmpdir / "outro_raw.mp4"
        outro_norm = tmpdir / "outro_norm.mp4"
        op = _outro_path(brand)
        if op:
            shutil.copy(op, outro_raw)
        else:
            logo_slate(logo, outro_raw, brand, 3.0, speed)
        normalize_clip(outro_raw, outro_norm, speed)

        total = len(pending)
        for i, job in enumerate(pending, 1):
            job["status"] = "Processing"
            show(f"Processing {i}/{total}: {job['name']}", 5 + int((i-1)*88/max(total,1)))
            try:
                out_name = safe_name(job["out_name"], Path(job["name"]).stem, brand)
                result = process_one(
                    QFile(job["name"], job["data"]),
                    brand, logo, outro_norm,
                    float(trim_start), float(outro_remove),
                    job["course"], job["unit"], job["chapter"],
                    out_name, tmpdir, i, speed,
                )
                dest = out_dir / f"{job['id']}_{out_name}"
                shutil.copy(result, dest)
                info_dur, _, _, _ = get_media_info(dest)
                job.update({
                    "status": "Done", "output": str(dest),
                    "out_name": out_name,
                    "dur": fmt_time(info_dur),
                    "file_size": f"{dest.stat().st_size/1_048_576:.1f} MB",
                })
            except Exception as e:
                job.update({"status":"Failed","error":str(e)})
            show(f"Done {i}/{total}: {job['name']}", 5 + int(i*88/max(total,1)))

    show("Queue complete.", 100)
    c = counts()
    if c["Failed"]:
        st.warning(f"{c['Failed']} job(s) failed.")
    else:
        st.success("All videos processed!")


def build_zip(brand: str) -> bytes:
    buf = io.BytesIO()
    seen: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for j in st.session_state.jobs:
            if j["status"] != "Done" or not j["output"]: continue
            p = Path(j["output"])
            if not p.exists(): continue
            name = safe_name(j["out_name"], p.stem, brand)
            if name in seen:
                stem, ext = Path(name).stem, Path(name).suffix
                n = 2
                while f"{stem}_{n}{ext}" in seen: n += 1
                name = f"{stem}_{n}{ext}"
            seen.add(name); zf.write(p, name)
    return buf.getvalue()


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
  <p>Uses your exact Intro.mp4 — only replaces the course name, unit and chapter text.</p>
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
            st.warning(f"Missing: {BRANDS[brand]['prefix']}_intro.mp4")

        st.divider()
        speed = st.selectbox("Processing speed", list(SPEED_PROFILES.keys()),
                             index=list(SPEED_PROFILES.keys()).index(DEFAULT_SPEED))

        st.divider()
        trim_start   = st.number_input("Remove from start (s)", 0.0, 300.0, 9.0, 0.5, "%.1f",
                                       help="Seconds of original SLC intro to cut")
        outro_remove = st.number_input("Remove from end (s)",   0.0, 300.0, 10.0, 0.5, "%.1f",
                                       help="Seconds of original SLC outro to cut")
        st.divider()
        fp = find_font()
        st.caption(f"Font: {Path(fp).name if fp else 'PIL default'}")
        st.caption(f"FFmpeg: {Path(get_ffmpeg()).name}")

    # ── Metrics row ───────────────────────────
    c = counts()
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total",      c["Total"])
    m2.metric("Queued",     c["Queued"])
    m3.metric("Done",       c["Done"])
    m4.metric("Failed",     c["Failed"])

    left, right = st.columns([1, 1.1], gap="large")

    # ── LEFT COLUMN ───────────────────────────
    with left:
        with st.container(border=True):
            st.subheader("1 · Add videos")
            default_course  = st.text_input("Default course name",
                                            "LEVEL 4 DIPLOMA IN EDUCATION STUDIES (RQF)")
            default_unit    = st.text_input("Default unit number", "UNIT 01")
            default_chapter = st.text_input("Default chapter name", "CHAPTER 01")
            uploads = st.file_uploader("Choose SLC videos", ["mp4","mov","avi","mkv"],
                                       accept_multiple_files=True)
            if st.button("Add to queue", type="primary", use_container_width=True,
                         disabled=not uploads):
                a, s = add_to_queue(uploads, brand, default_course, default_unit, default_chapter)
                if a: st.success(f"Added {a} video(s).")
                if s: st.info(f"Skipped {s} duplicate(s).")

        with st.container(border=True):
            st.subheader("2 · Process")
            st.markdown(
                '<div class="info-box">The exact <b>Intro.mp4</b> is used. '
                'Only the course title, unit number and chapter name are replaced. '
                'All animations, music and the Aspirex logo are kept intact.</div>',
                unsafe_allow_html=True)
            st.write("")
            col1, col2, col3 = st.columns(3)
            pend = c["Queued"] + c["Failed"]
            with col1:
                go = st.button("▶ Start / resume", type="primary",
                               use_container_width=True,
                               disabled=(pend == 0 or not logo))
            with col2:
                if st.button("↩ Retry failed", use_container_width=True,
                             disabled=c["Failed"]==0):
                    for j in st.session_state.jobs:
                        if j["status"]=="Failed": requeue(j)
                    st.rerun()
            with col3:
                if st.button("🗑 Clear all", use_container_width=True,
                             disabled=c["Total"]==0):
                    st.session_state.jobs.clear(); st.rerun()

            if go and logo:
                run_queue(brand, logo, trim_start, outro_remove, speed)

        with st.container(border=True):
            st.subheader("3 · Intro preview")
            st.caption("Shows the real intro video with your text applied.")
            first = st.session_state.jobs[0] if st.session_state.jobs else {}
            prev = make_preview(brand,
                                first.get("course", default_course),
                                first.get("unit",   default_unit),
                                first.get("chapter",default_chapter))
            if prev:
                st.image(prev, use_container_width=True,
                         caption="Real intro + your text overlay")
            else:
                st.info("Intro.mp4 not found or no videos added yet.")

    # ── RIGHT COLUMN ──────────────────────────
    with right:
        with st.container(border=True):
            st.subheader("Queue")
            rows = [{"#":i+1,"File":j["name"],"Unit":j["unit"],
                     "Chapter":j["chapter"],"Status":j["status"],
                     "Size":j["size"]}
                    for i,j in enumerate(st.session_state.jobs)]
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No videos yet — add some on the left.")

        with st.container(border=True):
            st.subheader("Edit each video")
            if not st.session_state.jobs:
                st.info("Queue is empty.")
            else:
                ac1, ac2 = st.columns(2)
                with ac1:
                    auto_start = st.number_input("Auto-number from", 1, 999, 1, 1)
                with ac2:
                    if st.button("Auto-number units", use_container_width=True):
                        for n, j in enumerate(st.session_state.jobs, int(auto_start)):
                            old = j["unit"]
                            j["unit"] = f"UNIT {n:02d}"
                            if j["status"]=="Done" and j["unit"]!=old: requeue(j)
                        st.success("Units renumbered.")

                for i, job in enumerate(st.session_state.jobs, 1):
                    status = job["status"]
                    with st.expander(f"{i}. {job['name']}  —  {status}",
                                     expanded=(i==1 and status!="Done")):
                        st.markdown(badge(status), unsafe_allow_html=True)
                        st.caption(f"Input: {job['size']}")

                        old = (job["course"], job["unit"], job["chapter"], job["out_name"])

                        course  = st.text_input("Course name",   job["course"],  key=f"c_{job['id']}")
                        u1, u2  = st.columns([.35, .65])
                        with u1:
                            unit = st.text_input("Unit number", job["unit"],    key=f"u_{job['id']}")
                        with u2:
                            chap = st.text_input("Chapter name", job["chapter"], key=f"ch_{job['id']}")
                        oname = st.text_input("Output filename", job["out_name"], key=f"o_{job['id']}")

                        new_oname = safe_name(oname, Path(job["name"]).stem, brand)
                        new = (course, unit, chap, new_oname)
                        if new != old and status == "Done": requeue(job)
                        job["course"], job["unit"], job["chapter"], job["out_name"] = new

                        b1, b2, b3 = st.columns(3)
                        with b1:
                            if st.button("Remove", key=f"rm_{job['id']}",
                                         use_container_width=True,
                                         disabled=status=="Processing"):
                                st.session_state.jobs = [
                                    j for j in st.session_state.jobs if j["id"]!=job["id"]]
                                st.rerun()
                        with b2:
                            if st.button("Requeue", key=f"rq_{job['id']}",
                                         use_container_width=True,
                                         disabled=status=="Processing"):
                                requeue(job); st.success("Requeued.")
                        with b3:
                            op = Path(job["output"]) if job["output"] else None
                            can_dl = status=="Done" and op and op.exists()
                            if can_dl:
                                st.download_button("⬇ Download",
                                                   data=op.read_bytes(),
                                                   file_name=Path(job["out_name"]).name,
                                                   mime="video/mp4",
                                                   key=f"dl_{job['id']}",
                                                   use_container_width=True)
                            else:
                                st.button("Download", disabled=True,
                                          key=f"dl_dis_{job['id']}",
                                          use_container_width=True)

                        if status=="Done":
                            st.success(f"Ready · {job['file_size']} · {job['dur']}")
                        if status=="Failed":
                            st.error(job["error"])

        with st.container(border=True):
            st.subheader("Download all")
            done = [j for j in st.session_state.jobs
                    if j["status"]=="Done" and j["output"]
                    and Path(j["output"]).exists()]
            if done:
                zb = build_zip(brand)
                st.download_button("⬇ Download all as ZIP",
                                   data=zb,
                                   file_name=f"{brand.lower()}_rebranded_videos.zip",
                                   mime="application/zip",
                                   type="primary",
                                   use_container_width=True)
                st.caption(f"{len(done)} video(s) ready.")
            else:
                st.info("Completed videos will appear here.")


if __name__ == "__main__":
    main()
