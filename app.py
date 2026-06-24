from __future__ import annotations

import io
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# -----------------------------------------------------------------------------
# Fixed tool settings
# -----------------------------------------------------------------------------

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30
AUDIO_RATE = 48000
DEFAULT_CRF = 18

SPEED_PROFILES = {
    "High quality": {"preset": "fast", "crf": 18, "note": "Best quality, slower processing."},
    "Fast (recommended)": {"preset": "veryfast", "crf": 24, "note": "Good balance for daily use."},
    "Very fast": {"preset": "ultrafast", "crf": 26, "note": "Fastest CPU mode, larger files."},
}
DEFAULT_SPEED_MODE = "Fast (recommended)"

# Hard-coded from the position requested in the UI screenshot.
# These coordinates are used on the final normalized 1920x1080 video.
FIXED_SLC_LOGO_BOX = (1640, 933, 272, 126)  # x, y, width, height
FIXED_SLC_COVER_COLOR = "#FFFFFF"
FIXED_SLC_COVER_BLEED = 8

INTRO_DURATION = 5.0

# Cover page template/layout copied from the provided Intro.mp4.
# This keeps the course title, unit pill, logo position, sizes, and spacing fixed.
INTRO_COVER_TEMPLATE_NAME = "intro_cover_template.png"
COVER_TITLE_FONT_SIZE = 78
COVER_TITLE_MIN_FONT_SIZE = 46
COVER_TITLE_MAX_WIDTH = 1300
COVER_TITLE_TOP_Y = 360
COVER_TITLE_LINE_STEP = 99
COVER_UNIT_FONT_SIZE = 50
COVER_UNIT_MIN_FONT_SIZE = 34
COVER_UNIT_TEXT_MAX_WIDTH = 980
COVER_PILL_Y = 612
COVER_PILL_HEIGHT = 102
COVER_PILL_MIN_WIDTH = 650
COVER_PILL_MAX_WIDTH = 1180
COVER_LOGO_CENTER = (960, 933)
COVER_LOGO_MAX_SIZE = (250, 70)
COVER_LOGO_CLEANUP_BOX = (760, 875, 400, 125)

BRANDS = {
    "Aspirex": {
        "prefix": "aspirex",
        "logo": "aspirex_logo.png",
        "bg": "0x9051D9",
        "intro_bg": "#9654DD",
        "intro_accent": "#6D32B5",
        "intro_text": "#FFFFFF",
        "intro_pill_text": "#6D32B5",
    },
    "GEL": {
        "prefix": "gel",
        "logo": "gel_logo.png",
        "bg": "0xF7FBFF",
        "intro_bg": "#F7FBFF",
        "intro_accent": "#29ABE2",
        "intro_text": "#1A2E4A",
        "intro_pill_text": "#1A2E4A",
    },
}


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    has_audio: bool

    @property
    def has_video(self) -> bool:
        return self.width > 0 and self.height > 0


# -----------------------------------------------------------------------------
# File and media helpers
# -----------------------------------------------------------------------------


def get_base() -> Path:
    return Path(__file__).resolve().parent


def brand_logo_path(brand: str) -> Optional[Path]:
    path = get_base() / BRANDS[brand]["logo"]
    return path if path.exists() else None


def bundled_clip_path(brand: str, kind: str) -> Optional[Path]:
    path = get_base() / f"{BRANDS[brand]['prefix']}_{kind}.mp4"
    return path if path.exists() else None


_FFMPEG_EXE: Optional[str] = None


def get_ffmpeg_exe() -> str:
    """Return a working FFmpeg executable.

    The old version required system ffmpeg + ffprobe on PATH. This version first
    tries system FFmpeg, then falls back to imageio-ffmpeg from requirements.txt.
    That makes the tool work on machines where FFmpeg is not installed globally.
    """
    global _FFMPEG_EXE
    if _FFMPEG_EXE:
        return _FFMPEG_EXE

    candidates: list[str] = []

    env_value = os.environ.get("FFMPEG_BINARY", "").strip()
    if env_value:
        candidates.append(env_value)

    system_value = shutil.which("ffmpeg")
    if system_value:
        candidates.append(system_value)

    try:
        import imageio_ffmpeg

        bundled_value = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_value:
            candidates.append(bundled_value)
    except Exception:
        pass

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                _FFMPEG_EXE = candidate
                return candidate
        except Exception:
            continue

    raise RuntimeError(
        "FFmpeg is not available. Run `pip install -r requirements.txt` first. "
        "This version includes imageio-ffmpeg as an automatic FFmpeg fallback. "
        "If your platform blocks that fallback, install FFmpeg manually and make sure `ffmpeg` is on PATH."
    )


def ensure_ffmpeg() -> None:
    get_ffmpeg_exe()


def prepare_ffmpeg_cmd(cmd: list[str]) -> list[str]:
    if cmd and cmd[0] == "ffmpeg":
        return [get_ffmpeg_exe(), *cmd[1:]]
    return cmd


def run_cmd(cmd: list[str], label: str) -> None:
    result = subprocess.run(prepare_ffmpeg_cmd(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[-5000:] if result.stderr else "No FFmpeg error text returned."
        raise RuntimeError(f"{label} failed.\n\n{stderr}")


def ffmpeg_probe_text(path: Path) -> str:
    result = subprocess.run(
        prepare_ffmpeg_cmd(["ffmpeg", "-hide_banner", "-i", str(path)]),
        capture_output=True,
        text=True,
    )
    return (result.stderr or "") + "\n" + (result.stdout or "")


def parse_duration_from_ffmpeg_output(text: str) -> float:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def parse_video_size_from_ffmpeg_output(text: str) -> tuple[int, int]:
    # Look only near Video stream lines to avoid matching bitrates or unrelated numbers.
    for line in text.splitlines():
        if "Video:" not in line:
            continue
        match = re.search(r"(\d{2,5})x(\d{2,5})", line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return 0, 0


def get_media_info(path: Path) -> MediaInfo:
    duration = 0.0
    width = 0
    height = 0

    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
    cap.release()

    probe_text = ffmpeg_probe_text(path)
    has_audio = bool(re.search(r"Stream #.*Audio:", probe_text))

    if duration <= 0:
        duration = parse_duration_from_ffmpeg_output(probe_text)

    if width <= 0 or height <= 0:
        probed_w, probed_h = parse_video_size_from_ffmpeg_output(probe_text)
        width = width or probed_w
        height = height or probed_h

    return MediaInfo(duration=duration, width=width, height=height, has_audio=has_audio)


def save_uploaded_file(uploaded_file, path: Path) -> None:
    path.write_bytes(uploaded_file.getvalue())


def seconds_label(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def safe_output_name(value: str, fallback_stem: str, brand: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raw = f"{fallback_stem}_{brand.lower()}_rebranded.mp4"
    raw = raw.replace("\\", "_").replace("/", "_").strip()
    raw = re.sub(r"[^A-Za-z0-9._ -]+", "_", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" ._")
    if not raw:
        raw = f"{fallback_stem}_{brand.lower()}_rebranded.mp4"
    if not raw.lower().endswith(".mp4"):
        raw += ".mp4"
    return raw


def ffmpeg_color_from_hex(hex_color: str, opacity: float = 1.0) -> str:
    value = (hex_color or "#FFFFFF").strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6 or any(c not in "0123456789abcdefABCDEF" for c in value):
        value = "FFFFFF"
    alpha = max(0.0, min(float(opacity), 1.0))
    return f"0x{value}@{alpha:.3f}"


def get_speed_profile(speed_mode: str) -> dict:
    return SPEED_PROFILES.get(speed_mode, SPEED_PROFILES[DEFAULT_SPEED_MODE])


def video_encoding_args(speed_mode: str = DEFAULT_SPEED_MODE, crf: Optional[int] = None, preset: Optional[str] = None) -> list[str]:
    profile = get_speed_profile(speed_mode)
    chosen_preset = preset or str(profile["preset"])
    chosen_crf = int(crf if crf is not None else profile["crf"])
    return ["-c:v", "libx264", "-preset", chosen_preset, "-crf", str(chosen_crf)]


# -----------------------------------------------------------------------------
# Preview helpers
# -----------------------------------------------------------------------------


def extract_frame(video_path: Path, seconds: float = 0.0):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(float(seconds), 0.0) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def normalize_frame_to_target(frame):
    if frame is None:
        return None
    img = Image.fromarray(frame).convert("RGB")
    scale = min(TARGET_WIDTH / img.width, TARGET_HEIGHT / img.height)
    scaled_w = max(1, int(round(img.width * scale)))
    scaled_h = max(1, int(round(img.height * scale)))
    resized = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0))
    canvas.paste(resized, ((TARGET_WIDTH - scaled_w) // 2, (TARGET_HEIGHT - scaled_h) // 2))
    return canvas


def draw_fixed_logo_box(frame):
    img = normalize_frame_to_target(frame)
    if img is None:
        return None
    x, y, w, h = FIXED_SLC_LOGO_BOX
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([x, y, x + w, y + h], fill=(255, 0, 0, 45), outline=(255, 0, 0, 255), width=4)
    draw.rectangle([x, max(y - 34, 0), min(x + 360, TARGET_WIDTH - 1), y], fill=(255, 0, 0, 220))
    draw.text((x + 8, max(y - 27, 3)), "Fixed SLC logo replacement area", fill=(255, 255, 255, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")


# -----------------------------------------------------------------------------
# Intro image helpers
# -----------------------------------------------------------------------------


def find_font(bold: bool = True) -> str | None:
    # Prefer Lato Heavy because it is close to the supplied Intro.mp4 cover title style.
    # Fall back to common system fonts on Linux, macOS, and Windows.
    candidates = [
        "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf" if bold else "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
        "/usr/share/fonts/truetype/lato/Lato-Black.ttf" if bold else "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
        "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf" if bold else "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None

def make_font(size: int, bold: bool = True):
    font_path = find_font(bold=bold)
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text or " ", font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int = 2) -> list[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_size(draw, trial, font)[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
            if len(lines) >= max_lines - 1:
                break
    if current:
        remaining = " ".join(words[len(" ".join(lines).split()) :]) if lines else current
        if lines and remaining and remaining != current:
            current = remaining
        lines.append(current)
    lines = lines[:max_lines]
    # If the last line is still too wide, reduce it with an ellipsis.
    fixed: list[str] = []
    for line in lines:
        if text_size(draw, line, font)[0] <= max_width:
            fixed.append(line)
            continue
        clipped = line
        while clipped and text_size(draw, clipped + "...", font)[0] > max_width:
            clipped = clipped[:-1]
        fixed.append((clipped.rstrip() + "...") if clipped else line[:20])
    return fixed


def draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font,
    center_x: int,
    top_y: int,
    fill: str,
    line_gap: int = 16,
    shadow: bool = False,
) -> int:
    y = top_y
    for line in lines:
        w, h = text_size(draw, line, font)
        x = center_x - w // 2
        if shadow:
            draw.text((x + 3, y + 4), line, font=font, fill=(0, 0, 0, 60))
        draw.text((x, y), line, font=font, fill=fill)
        y += h + line_gap
    return y


def fit_font_for_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int, bold: bool = True):
    for size in range(start_size, min_size - 1, -2):
        font = make_font(size, bold=bold)
        if text_size(draw, text, font)[0] <= max_width:
            return font
    return make_font(min_size, bold=bold)


def paste_logo_center(img: Image.Image, logo_path: Path, center_x: int, center_y: int, max_width: int, max_height: int) -> None:
    logo = Image.open(logo_path).convert("RGBA")
    scale = min(max_width / logo.width, max_height / logo.height, 1.0)
    size = (max(1, int(round(logo.width * scale))), max(1, int(round(logo.height * scale))))
    logo = logo.resize(size, Image.Resampling.LANCZOS)
    img.alpha_composite(logo, (center_x - logo.width // 2, center_y - logo.height // 2))


def remove_template_logo_patch(img: Image.Image) -> Image.Image:
    """Smooth the old logo area in the template before adding the chosen brand logo."""
    x, y, w, h = COVER_LOGO_CLEANUP_BOX
    x = max(0, min(int(x), img.width - 1))
    y = max(0, min(int(y), img.height - 1))
    w = max(1, min(int(w), img.width - x))
    h = max(1, min(int(h), img.height - y))

    patch = img.crop((x, y, x + w, y + h)).filter(ImageFilter.GaussianBlur(18))
    mask = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, w, h], radius=42, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(10))
    img.paste(patch, (x, y), mask)
    return img


def split_course_title_for_cover(draw: ImageDraw.ImageDraw, course: str, font, max_width: int, max_lines: int = 2) -> list[str]:
    text = (course or "COURSE NAME").strip().upper()
    if not text:
        return ["COURSE NAME"]

    words = text.split()
    lines: list[str] = []
    current = ""
    word_index = 0

    while word_index < len(words):
        word = words[word_index]
        trial = word if not current else f"{current} {word}"
        if text_size(draw, trial, font)[0] <= max_width or not current:
            current = trial
            word_index += 1
            continue

        lines.append(current)
        current = ""
        if len(lines) >= max_lines - 1:
            current = " ".join(words[word_index:])
            word_index = len(words)
            break

    if current:
        lines.append(current)

    lines = lines[:max_lines]
    clipped: list[str] = []
    for line in lines:
        if text_size(draw, line, font)[0] <= max_width:
            clipped.append(line)
            continue
        value = line
        while value and text_size(draw, value + "...", font)[0] > max_width:
            value = value[:-1]
        clipped.append((value.rstrip() + "...") if value else line[:24])
    return clipped or [text]


def fit_course_title_for_cover(draw: ImageDraw.ImageDraw, course: str) -> tuple[object, list[str]]:
    for size in range(COVER_TITLE_FONT_SIZE, COVER_TITLE_MIN_FONT_SIZE - 1, -2):
        font = make_font(size, bold=True)
        lines = split_course_title_for_cover(draw, course, font, COVER_TITLE_MAX_WIDTH, max_lines=2)
        if all(text_size(draw, line, font)[0] <= COVER_TITLE_MAX_WIDTH for line in lines):
            return font, lines
    font = make_font(COVER_TITLE_MIN_FONT_SIZE, bold=True)
    return font, split_course_title_for_cover(draw, course, font, COVER_TITLE_MAX_WIDTH, max_lines=2)


def draw_cover_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    center_x: int,
    top_y: int,
    fill: tuple[int, int, int, int],
) -> None:
    w, _ = text_size(draw, text, font)
    x = center_x - w // 2
    # Soft purple shadow from the supplied cover page.
    draw.text((x + 3, top_y + 5), text, font=font, fill=(72, 41, 120, 105))
    draw.text((x, top_y), text, font=font, fill=fill)


def create_intro_image(
    logo_path: Path,
    output_png: Path,
    brand: str,
    course_name: str,
    unit_number: str,
    unit_name: str,
) -> None:
    course = (course_name or "COURSE NAME").strip().upper()
    unit_no = (unit_number or "UNIT 01").strip().upper()
    unit_title = (unit_name or "CHAPTER 01").strip().upper()
    unit_line = unit_no if not unit_title else f"{unit_no} - {unit_title}"

    template_path = get_base() / INTRO_COVER_TEMPLATE_NAME
    if template_path.exists():
        img = Image.open(template_path).convert("RGBA").resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    else:
        # Fallback if the template image is accidentally removed.
        img = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), "#9654DD")

    img = remove_template_logo_patch(img)
    draw = ImageDraw.Draw(img, "RGBA")

    title_font, title_lines = fit_course_title_for_cover(draw, course)
    title_top_y = COVER_TITLE_TOP_Y
    if len(title_lines) == 1:
        # Keep a one-line title in the same vertical band as the supplied template.
        title_top_y = COVER_TITLE_TOP_Y

    for i, line in enumerate(title_lines):
        draw_cover_text_with_shadow(
            draw,
            line,
            title_font,
            center_x=TARGET_WIDTH // 2,
            top_y=title_top_y + i * COVER_TITLE_LINE_STEP,
            fill=(255, 255, 255, 255),
        )

    unit_font = fit_font_for_text(
        draw,
        unit_line,
        max_width=COVER_UNIT_TEXT_MAX_WIDTH,
        start_size=COVER_UNIT_FONT_SIZE,
        min_size=COVER_UNIT_MIN_FONT_SIZE,
        bold=True,
    )
    unit_w, unit_h = text_size(draw, unit_line, unit_font)
    pill_w = min(max(unit_w + 150, COVER_PILL_MIN_WIDTH), COVER_PILL_MAX_WIDTH)
    pill_h = COVER_PILL_HEIGHT
    pill_x = (TARGET_WIDTH - pill_w) // 2
    pill_y = COVER_PILL_Y

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        [pill_x + 8, pill_y + 10, pill_x + pill_w + 8, pill_y + pill_h + 10],
        radius=50,
        fill=(55, 20, 95, 95),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    img.alpha_composite(shadow)

    draw = ImageDraw.Draw(img, "RGBA")
    draw.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h], radius=50, fill=(255, 255, 255, 255))
    draw.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h], radius=50, outline=(220, 193, 246, 255), width=7)
    draw.text(
        (TARGET_WIDTH // 2 - unit_w // 2, pill_y + (pill_h - unit_h) // 2 - 8),
        unit_line,
        font=unit_font,
        fill=(91, 43, 130, 255),
    )

    if logo_path and logo_path.exists():
        logo_center_x, logo_center_y = COVER_LOGO_CENTER
        logo_max_w, logo_max_h = COVER_LOGO_MAX_SIZE
        paste_logo_center(img, logo_path, logo_center_x, logo_center_y, max_width=logo_max_w, max_height=logo_max_h)

    img.convert("RGB").save(output_png, quality=95)

def create_dynamic_intro_clip(
    logo_path: Path,
    output_path: Path,
    brand: str,
    course_name: str,
    unit_number: str,
    unit_name: str,
    speed_mode: str = DEFAULT_SPEED_MODE,
    crf: Optional[int] = None,
) -> None:
    png_path = output_path.with_suffix(".png")
    create_intro_image(logo_path, png_path, brand, course_name, unit_number, unit_name)
    fade_out_start = max(INTRO_DURATION - 0.45, 0.1)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loop",
        "1",
        "-framerate",
        str(TARGET_FPS),
        "-t",
        f"{INTRO_DURATION:.3f}",
        "-i",
        str(png_path),
        "-f",
        "lavfi",
        "-t",
        f"{INTRO_DURATION:.3f}",
        "-i",
        f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
        "-vf",
        f"fps={TARGET_FPS},fade=t=in:st=0:d=0.35,fade=t=out:st={fade_out_start:.3f}:d=0.45,format=yuv420p",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        *video_encoding_args(speed_mode, crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    try:
        run_cmd(cmd, "Generate dynamic intro")
    finally:
        png_path.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# FFmpeg processing helpers
# -----------------------------------------------------------------------------


def video_scale_filter(target_w: int = TARGET_WIDTH, target_h: int = TARGET_HEIGHT, fps: int = TARGET_FPS) -> str:
    return (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,fps={fps}"
    )


def process_content_segment(
    source_path: Path,
    output_path: Path,
    media: MediaInfo,
    trim_start: float,
    trim_end: float,
    brand_logo: Path,
    speed_mode: str = DEFAULT_SPEED_MODE,
    crf: Optional[int] = None,
) -> None:
    """Cut source, hard-hide the fixed SLC logo box, then add the brand logo in the same box."""
    duration = float(trim_end) - float(trim_start)
    if duration <= 0:
        raise ValueError("Trim settings leave no content. Reduce intro/outro removal.")
    if not brand_logo.exists():
        raise FileNotFoundError(f"Brand logo not found: {brand_logo}")

    x, y, w, h = FIXED_SLC_LOGO_BOX
    cover = ffmpeg_color_from_hex(FIXED_SLC_COVER_COLOR, 1.0)

    cover_x = max(0, x - FIXED_SLC_COVER_BLEED)
    cover_y = max(0, y - FIXED_SLC_COVER_BLEED)
    cover_w = min(TARGET_WIDTH - cover_x, w + (x - cover_x) + FIXED_SLC_COVER_BLEED)
    cover_h = min(TARGET_HEIGHT - cover_y, h + (y - cover_y) + FIXED_SLC_COVER_BLEED)

    filter_complex = (
        f"[0:v]{video_scale_filter()},"
        f"drawbox=x={cover_x}:y={cover_y}:w={cover_w}:h={cover_h}:color={cover}:t=fill,format=rgba[base];"
        f"[1:v]format=rgba,"
        f"scale={w}:{h}:force_original_aspect_ratio=decrease[brandlogo];"
        f"[base][brandlogo]overlay=x={x}+({w}-w)/2:y={y}+({h}-h)/2:format=auto,"
        f"format=yuv420p[v]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-accurate_seek",
        "-ss",
        f"{float(trim_start):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_path),
        "-loop",
        "1",
        "-i",
        str(brand_logo),
    ]

    if media.has_audio:
        cmd += [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "0:a:0",
            *video_encoding_args(speed_mode, crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(AUDIO_RATE),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
    else:
        cmd += [
            "-f",
            "lavfi",
            "-t",
            f"{duration:.3f}",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "2:a:0",
            *video_encoding_args(speed_mode, crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(AUDIO_RATE),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]

    run_cmd(cmd, "Cut content, remove SLC logo, add brand logo")


def normalize_clip(input_path: Path, output_path: Path, speed_mode: str = DEFAULT_SPEED_MODE, crf: Optional[int] = None) -> None:
    info = get_media_info(input_path)
    if not info.has_video:
        raise ValueError(f"No video stream found in {input_path.name}")

    filter_complex = f"[0:v]{video_scale_filter()},format=yuv420p[v]"
    if info.has_audio:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(input_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "0:a:0",
            *video_encoding_args(speed_mode, crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(AUDIO_RATE),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
    else:
        duration = max(info.duration, 0.1)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(input_path),
            "-f",
            "lavfi",
            "-t",
            f"{duration:.3f}",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            *video_encoding_args(speed_mode, crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(AUDIO_RATE),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]

    run_cmd(cmd, f"Normalize {input_path.name}")


def create_logo_slate(
    logo_path: Path,
    output_path: Path,
    brand: str,
    duration: float = 3.0,
    logo_width_px: int = 620,
    speed_mode: str = DEFAULT_SPEED_MODE,
    crf: Optional[int] = None,
) -> None:
    bg = BRANDS[brand]["bg"]
    duration = max(float(duration), 0.5)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-f",
        "lavfi",
        "-t",
        f"{duration:.3f}",
        "-i",
        f"color=c={bg}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:r={TARGET_FPS}",
        "-loop",
        "1",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(logo_path),
        "-f",
        "lavfi",
        "-t",
        f"{duration:.3f}",
        "-i",
        f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
        "-filter_complex",
        (
            f"[1:v]format=rgba,scale={logo_width_px}:-1[logo];"
            f"[0:v][logo]overlay=x=(W-w)/2:y=(H-h)/2:format=auto,format=yuv420p[v]"
        ),
        "-map",
        "[v]",
        "-map",
        "2:a:0",
        *video_encoding_args(speed_mode, crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    run_cmd(cmd, f"Generate {brand} logo slate")


def prepare_outro_clip(brand: str, logo_path: Path, tmpdir: Path, speed_mode: str = DEFAULT_SPEED_MODE) -> Path:
    raw = tmpdir / f"raw_{BRANDS[brand]['prefix']}_outro.mp4"
    bundled = bundled_clip_path(brand, "outro")
    if bundled:
        shutil.copy(bundled, raw)
    else:
        create_logo_slate(logo_path, raw, brand=brand, duration=3.0, speed_mode=speed_mode)
    norm = tmpdir / f"normalized_{BRANDS[brand]['prefix']}_outro.mp4"
    normalize_clip(raw, norm, speed_mode=speed_mode)
    return norm


def concat_three_clips(intro: Path, content: Path, outro: Path, output: Path, speed_mode: str = DEFAULT_SPEED_MODE, crf: Optional[int] = None) -> None:
    concat_file = output.with_suffix(".concat.txt")
    concat_file.write_text(
        f"file '{intro.as_posix()}'\n"
        f"file '{content.as_posix()}'\n"
        f"file '{outro.as_posix()}'\n",
        encoding="utf-8",
    )

    copy_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]

    result = subprocess.run(prepare_ffmpeg_cmd(copy_cmd), capture_output=True, text=True)
    if result.returncode == 0:
        concat_file.unlink(missing_ok=True)
        return

    fallback_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(intro),
        "-i",
        str(content),
        "-i",
        str(outro),
        "-filter_complex",
        "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]",
        "-map",
        "[v]",
        "-map",
        "[a]",
        *video_encoding_args(speed_mode, crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        run_cmd(fallback_cmd, "Final concat")
    finally:
        concat_file.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Processing orchestration
# -----------------------------------------------------------------------------


def process_one_video(
    uploaded_file,
    brand: str,
    logo_path: Path,
    outro_norm: Path,
    trim_start: float,
    outro_remove: float,
    course_name: str,
    unit_number: str,
    unit_name: str,
    output_name: str,
    tmpdir: Path,
    index: int,
    speed_mode: str = DEFAULT_SPEED_MODE,
) -> Path:
    source_path = tmpdir / f"source_{index}.mp4"
    save_uploaded_file(uploaded_file, source_path)
    media = get_media_info(source_path)
    if not media.has_video:
        raise ValueError(f"{uploaded_file.name} has no readable video stream.")

    trim_start_f = float(trim_start)
    trim_end_f = media.duration - float(outro_remove)
    if trim_start_f < 0 or trim_end_f <= trim_start_f:
        raise ValueError(
            f"Trim settings remove the whole video for {uploaded_file.name}. "
            f"Duration is {seconds_label(media.duration)}."
        )

    intro_path = tmpdir / f"intro_{index}.mp4"
    create_dynamic_intro_clip(
        logo_path=logo_path,
        output_path=intro_path,
        brand=brand,
        course_name=course_name,
        unit_number=unit_number,
        unit_name=unit_name,
        speed_mode=speed_mode,
    )

    content_path = tmpdir / f"content_{index}.mp4"
    process_content_segment(
        source_path=source_path,
        output_path=content_path,
        media=media,
        trim_start=trim_start_f,
        trim_end=trim_end_f,
        brand_logo=logo_path,
        speed_mode=speed_mode,
    )

    final_path = tmpdir / output_name
    concat_three_clips(intro_path, content_path, outro_norm, final_path, speed_mode=speed_mode)
    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError(f"Output was not created for {uploaded_file.name}.")
    return final_path



# -----------------------------------------------------------------------------
# Streamlit UI and queue helpers
# -----------------------------------------------------------------------------


@dataclass
class QueuedUploadedFile:
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


def setup_style() -> None:
    st.set_page_config(page_title="Video Rebranding Queue", page_icon="🎬", layout="wide")
    st.markdown(
        """
<style>
.block-container {max-width: 1220px; padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {padding: 22px 26px; border-radius: 18px; background: linear-gradient(135deg, #15243F 0%, #1A5D82 100%); color: white; margin-bottom: 16px;}
.hero h1 {margin: 0; color: white; font-size: 2rem; line-height: 1.2;}
.hero p {margin: 8px 0 0 0; color: #EAF4FB; font-size: 1rem;}
.badge {display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.80rem; font-weight: 700;}
.badge-queued {background:#EEF2FF; color:#334155;}
.badge-processing {background:#FFF4D6; color:#8A5200;}
.badge-done {background:#EAF7EF; color:#166534;}
.badge-failed {background:#FDECEC; color:#991B1B;}
.muted {color:#64748B; font-size: 0.92rem;}
.fixed-logo-note {background:#EAF4FB; border-left: 4px solid #1565C0; padding: 11px 13px; border-radius: 10px; color:#16324F;}
.queue-tip {background:#F8FAFC; border:1px solid #E2E8F0; padding: 12px 14px; border-radius: 12px;}
hr {margin-top: 1.1rem; margin-bottom: 1.1rem;}
</style>
""",
        unsafe_allow_html=True,
    )


def init_queue_state() -> None:
    if "queue_jobs" not in st.session_state:
        st.session_state.queue_jobs = []
    if "queue_output_dir" not in st.session_state:
        st.session_state.queue_output_dir = tempfile.mkdtemp(prefix="video_rebranding_queue_")


def output_dir() -> Path:
    init_queue_state()
    path = Path(st.session_state.queue_output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_job_id(filename: str, data: bytes) -> str:
    digest = hashlib.sha1(data).hexdigest()
    return hashlib.sha1(f"{filename}|{len(data)}|{digest}".encode("utf-8", errors="ignore")).hexdigest()[:16]


def status_badge(status: str) -> str:
    key = (status or "Queued").lower()
    css = {
        "queued": "badge-queued",
        "processing": "badge-processing",
        "done": "badge-done",
        "failed": "badge-failed",
    }.get(key, "badge-queued")
    return f'<span class="badge {css}">{status}</span>'


def remove_job_output(job: dict) -> None:
    path_text = job.get("output_path") or ""
    if path_text:
        try:
            Path(path_text).unlink(missing_ok=True)
        except Exception:
            pass
    job["output_path"] = ""
    job["duration_label"] = ""
    job["output_size_label"] = ""


def remove_job(job_id: str) -> None:
    init_queue_state()
    kept = []
    for job in st.session_state.queue_jobs:
        if job.get("job_id") == job_id:
            remove_job_output(job)
        else:
            kept.append(job)
    st.session_state.queue_jobs = kept


def clear_queue() -> None:
    init_queue_state()
    for job in st.session_state.queue_jobs:
        remove_job_output(job)
    st.session_state.queue_jobs = []


def requeue_job(job: dict) -> None:
    if job.get("status") != "Processing":
        remove_job_output(job)
        job["status"] = "Queued"
        job["error"] = ""


def requeue_failed_jobs() -> None:
    init_queue_state()
    for job in st.session_state.queue_jobs:
        if job.get("status") == "Failed":
            requeue_job(job)


def requeue_all_jobs() -> None:
    init_queue_state()
    for job in st.session_state.queue_jobs:
        requeue_job(job)


def add_uploads_to_queue(uploaded_videos, brand: str, default_course: str, default_unit_name: str) -> tuple[int, int]:
    init_queue_state()
    existing = {job.get("job_id") for job in st.session_state.queue_jobs}
    added = 0
    skipped = 0
    start_index = len(st.session_state.queue_jobs) + 1

    for offset, uploaded in enumerate(uploaded_videos or []):
        data = uploaded.getvalue()
        job_id = make_job_id(uploaded.name, data)
        if job_id in existing:
            skipped += 1
            continue
        stem = Path(uploaded.name).stem
        unit_number = f"UNIT {start_index + added:02d}"
        output_name = safe_output_name(f"{stem}_{brand.lower()}_rebranded.mp4", stem, brand)
        st.session_state.queue_jobs.append(
            {
                "job_id": job_id,
                "name": uploaded.name,
                "data": data,
                "size_label": f"{len(data) / 1_048_576:.1f} MB",
                "course_name": default_course.strip() or "LEVEL 4 DIPLOMA IN EDUCATION STUDIES (RQF)",
                "unit_number": unit_number,
                "unit_name": default_unit_name.strip() or "CHAPTER 01",
                "output_name": output_name,
                "status": "Queued",
                "error": "",
                "output_path": "",
                "duration_label": "",
                "output_size_label": "",
            }
        )
        existing.add(job_id)
        added += 1
    return added, skipped


def queue_counts() -> dict[str, int]:
    init_queue_state()
    counts = {"Total": 0, "Queued": 0, "Processing": 0, "Done": 0, "Failed": 0}
    for job in st.session_state.queue_jobs:
        status = job.get("status", "Queued")
        counts["Total"] += 1
        counts[status] = counts.get(status, 0) + 1
    return counts


def make_queue_rows() -> list[dict]:
    rows = []
    for i, job in enumerate(st.session_state.queue_jobs, start=1):
        rows.append(
            {
                "#": i,
                "Video": job.get("name", ""),
                "Unit": job.get("unit_number", ""),
                "Chapter / unit name": job.get("unit_name", ""),
                "Status": job.get("status", "Queued"),
                "Input size": job.get("size_label", ""),
                "Output": Path(job.get("output_name", "")).name,
            }
        )
    return rows


def render_queue_metrics() -> None:
    counts = queue_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", counts.get("Total", 0))
    c2.metric("Queued", counts.get("Queued", 0))
    c3.metric("Done", counts.get("Done", 0))
    c4.metric("Failed", counts.get("Failed", 0))


def build_completed_zip(brand: str) -> bytes:
    buffer = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for job in st.session_state.queue_jobs:
            if job.get("status") != "Done":
                continue
            output_path_text = job.get("output_path") or ""
            if not output_path_text:
                continue
            output_path = Path(output_path_text)
            if not output_path.exists() or not output_path.is_file():
                continue
            arcname = safe_output_name(job.get("output_name", output_path.name), output_path.stem, brand)
            if arcname in used_names:
                stem = Path(arcname).stem
                suffix = Path(arcname).suffix or ".mp4"
                counter = 2
                while f"{stem}_{counter}{suffix}" in used_names:
                    counter += 1
                arcname = f"{stem}_{counter}{suffix}"
            used_names.add(arcname)
            zf.write(output_path, arcname=arcname)
    return buffer.getvalue()


def get_first_queued_video_preview(trim_start: float) -> Optional[Image.Image]:
    if not st.session_state.queue_jobs:
        return None
    job = st.session_state.queue_jobs[0]
    preview_path: Optional[Path] = None
    try:
        suffix = Path(job.get("name", "video.mp4")).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(job.get("data", b""))
            preview_path = Path(tmp_file.name)
        info = get_media_info(preview_path)
        if not info.has_video:
            return None
        preview_time = min(max(float(trim_start) + 2.0, 0.0), max(info.duration - 0.1, 0.0))
        frame = extract_frame(preview_path, preview_time)
        return draw_fixed_logo_box(frame)
    except Exception:
        return None
    finally:
        if preview_path is not None:
            preview_path.unlink(missing_ok=True)


def create_intro_cover_preview_image(
    logo_path: Path,
    brand: str,
    course_name: str,
    unit_number: str,
    unit_name: str,
) -> Optional[Image.Image]:
    preview_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            preview_path = Path(tmp_file.name)
        create_intro_image(logo_path, preview_path, brand, course_name, unit_number, unit_name)
        return Image.open(preview_path).convert("RGB").copy()
    except Exception:
        return None
    finally:
        if preview_path is not None:
            preview_path.unlink(missing_ok=True)


def update_queue_table(slot) -> None:
    rows = make_queue_rows()
    if rows:
        slot.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        slot.info("Queue is empty.")


def process_queue(
    brand: str,
    logo_path: Path,
    trim_start: float,
    outro_remove: float,
    speed_mode: str,
) -> None:
    pending = [job for job in st.session_state.queue_jobs if job.get("status") in {"Queued", "Failed"}]
    if not pending:
        st.info("Nothing queued. Add videos or requeue completed items first.")
        return

    status_slot = st.empty()
    table_slot = st.empty()
    progress_slot = st.empty()
    progress_bar = progress_slot.progress(0)

    def show(message: str, pct: int) -> None:
        status_slot.info(message)
        progress_bar.progress(min(max(int(pct), 0), 100))
        update_queue_table(table_slot)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmpdir = Path(tmp_name)
        show(f"Preparing {brand} outro...", 3)
        outro_norm = prepare_outro_clip(brand, logo_path, tmpdir, speed_mode=speed_mode)

        total = len(pending)
        for idx, job in enumerate(pending, start=1):
            job["status"] = "Processing"
            job["error"] = ""
            show(f"Processing {idx}/{total}: {job.get('name', '')}", 5 + int((idx - 1) * 90 / max(total, 1)))
            try:
                stem = Path(job.get("name", f"video_{idx}.mp4")).stem
                output_name = safe_output_name(job.get("output_name", ""), stem, brand)
                queued_upload = QueuedUploadedFile(name=job.get("name", f"video_{idx}.mp4"), data=job.get("data", b""))
                result_path = process_one_video(
                    uploaded_file=queued_upload,
                    brand=brand,
                    logo_path=logo_path,
                    outro_norm=outro_norm,
                    trim_start=float(trim_start),
                    outro_remove=float(outro_remove),
                    course_name=job.get("course_name", ""),
                    unit_number=job.get("unit_number", ""),
                    unit_name=job.get("unit_name", ""),
                    output_name=output_name,
                    tmpdir=tmpdir,
                    index=idx,
                    speed_mode=speed_mode,
                )

                final_disk_path = output_dir() / f"{job.get('job_id')}_{output_name}"
                shutil.copy(result_path, final_disk_path)
                info = get_media_info(final_disk_path)
                job["output_name"] = output_name
                job["output_path"] = str(final_disk_path)
                job["duration_label"] = seconds_label(info.duration)
                job["output_size_label"] = f"{final_disk_path.stat().st_size / 1_048_576:.1f} MB"
                job["status"] = "Done"
            except Exception as ex:
                job["status"] = "Failed"
                job["error"] = str(ex)
            show(f"Finished {idx}/{total}: {job.get('name', '')}", 5 + int(idx * 90 / max(total, 1)))

    show("Queue finished.", 100)
    counts = queue_counts()
    if counts.get("Failed", 0):
        st.warning(f"Queue finished with {counts['Failed']} failed item(s). Open the failed item to see the error and retry.")
    else:
        st.success("Queue complete. All videos were processed.")


def main() -> None:
    setup_style()
    init_queue_state()

    try:
        ensure_ffmpeg()
    except Exception as ex:
        st.error(str(ex))
        st.stop()

    st.markdown(
        """
<div class="hero">
  <h1>Video Rebranding Queue</h1>
  <p>Upload many SLC videos, edit intro text per video, then process the queue one by one.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Global settings")
        brand = st.radio("Brand", list(BRANDS.keys()), index=0)
        logo_path = brand_logo_path(brand)
        if logo_path:
            try:
                logo_img = Image.open(logo_path).convert("RGBA")
                bg = Image.new("RGB", logo_img.size, (255, 255, 255))
                alpha = logo_img.split()[3] if len(logo_img.split()) == 4 else None
                bg.paste(logo_img, mask=alpha)
                st.image(bg, width=165)
            except Exception:
                st.caption(logo_path.name)
        else:
            st.error(f"Missing logo file for {brand}.")

        st.divider()
        speed_mode = st.selectbox("Processing speed", list(SPEED_PROFILES.keys()), index=list(SPEED_PROFILES.keys()).index(DEFAULT_SPEED_MODE))
        st.caption(get_speed_profile(speed_mode)["note"])

        st.divider()
        trim_start = st.number_input(
            "Remove SLC intro from start",
            min_value=0.0,
            max_value=300.0,
            value=9.0,
            step=0.1,
            format="%.1f",
            help="Seconds removed from the beginning of every queued video.",
        )
        outro_remove = st.number_input(
            "Remove SLC outro from end",
            min_value=0.0,
            max_value=300.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            help="Seconds removed from the end of every queued video.",
        )

        st.divider()
        x, y, w, h = FIXED_SLC_LOGO_BOX
        st.caption(f"Fixed SLC logo box: X={x}, Y={y}, W={w}, H={h}")
        st.caption(f"FFmpeg: {Path(get_ffmpeg_exe()).name}")

    render_queue_metrics()

    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("1. Add videos to queue")
            st.markdown('<p class="muted">Set default intro text for new videos. You can still edit each video later.</p>', unsafe_allow_html=True)
            default_course = st.text_input(
                "Default course name",
                value="LEVEL 4 DIPLOMA IN EDUCATION STUDIES (RQF)",
                key="default_course_name",
            )
            default_unit_name = st.text_input("Default unit/chapter name", value="CHAPTER 01", key="default_unit_name")
            uploaded_videos = st.file_uploader(
                "Choose SLC videos",
                type=["mp4", "mov", "avi", "mkv"],
                accept_multiple_files=True,
                help="Files are added to the queue only after you click Add to queue.",
            )
            add_disabled = not uploaded_videos
            if st.button("Add to queue", type="primary", use_container_width=True, disabled=add_disabled):
                added, skipped = add_uploads_to_queue(uploaded_videos, brand, default_course, default_unit_name)
                if added:
                    st.success(f"Added {added} video(s) to the queue.")
                if skipped:
                    st.info(f"Skipped {skipped} duplicate video(s).")

        with st.container(border=True):
            st.subheader("2. Queue controls")
            st.markdown(
                '<div class="fixed-logo-note">The tool always hides the old SLC logo at the hard-coded position, then adds the selected brand logo in that same place.</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            c1, c2, c3 = st.columns(3)
            pending_count = queue_counts().get("Queued", 0) + queue_counts().get("Failed", 0)
            with c1:
                start_clicked = st.button("Start / resume queue", type="primary", use_container_width=True, disabled=(pending_count == 0 or logo_path is None))
            with c2:
                st.button("Retry failed", use_container_width=True, disabled=queue_counts().get("Failed", 0) == 0, on_click=requeue_failed_jobs)
            with c3:
                st.button("Clear queue", use_container_width=True, disabled=queue_counts().get("Total", 0) == 0, on_click=clear_queue)

            if start_clicked and logo_path is not None:
                process_queue(
                    brand=brand,
                    logo_path=logo_path,
                    trim_start=float(trim_start),
                    outro_remove=float(outro_remove),
                    speed_mode=speed_mode,
                )

        with st.container(border=True):
            st.subheader("3. Preview")
            st.caption("Cover page layout is hard-coded from your Intro.mp4 template: same title size, unit pill, logo position, and spacing.")

            first_job = st.session_state.queue_jobs[0] if st.session_state.queue_jobs else {}
            cover_course = first_job.get("course_name", default_course)
            cover_unit_number = first_job.get("unit_number", "UNIT 01")
            cover_unit_name = first_job.get("unit_name", default_unit_name)
            if logo_path is not None:
                intro_preview = create_intro_cover_preview_image(logo_path, brand, cover_course, cover_unit_number, cover_unit_name)
            else:
                intro_preview = None

            if intro_preview is not None:
                st.image(intro_preview, caption="Cover page preview for the first queued video", use_container_width=True)
            else:
                st.info("Add a video to preview the generated cover page.")

            preview = get_first_queued_video_preview(float(trim_start))
            if preview is not None:
                st.image(preview, caption="Fixed SLC logo replacement box on the first queued video", use_container_width=True)
            else:
                st.info("Add a video to see the fixed SLC logo replacement preview.")

    with right:
        with st.container(border=True):
            st.subheader("Queue list")
            rows = make_queue_rows()
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    '<div class="queue-tip">No videos in the queue yet. Upload videos on the left, then click <b>Add to queue</b>.</div>',
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            st.subheader("Edit intro text video by video")
            if not st.session_state.queue_jobs:
                st.info("Queue is empty.")
            else:
                auto_col1, auto_col2, auto_col3 = st.columns([1, 1, 1])
                with auto_col1:
                    auto_start = st.number_input("Auto unit start", min_value=1, max_value=999, value=1, step=1)
                with auto_col2:
                    if st.button("Auto-number units", use_container_width=True):
                        for i, job in enumerate(st.session_state.queue_jobs, start=int(auto_start)):
                            old = job.get("unit_number", "")
                            job["unit_number"] = f"UNIT {i:02d}"
                            if job.get("status") == "Done" and job["unit_number"] != old:
                                requeue_job(job)
                        st.success("Unit numbers updated.")
                with auto_col3:
                    st.button("Requeue all", use_container_width=True, on_click=requeue_all_jobs)

                st.caption("Changing a completed item will automatically move it back to Queued so the output is regenerated.")

                for i, job in enumerate(st.session_state.queue_jobs, start=1):
                    status = job.get("status", "Queued")
                    title = f"{i}. {job.get('name', '')} — {status}"
                    with st.expander(title, expanded=(i == 1 and status != "Done")):
                        st.markdown(status_badge(status), unsafe_allow_html=True)
                        st.caption(f"Input size: {job.get('size_label', '')}")
                        old_values = (
                            job.get("course_name", ""),
                            job.get("unit_number", ""),
                            job.get("unit_name", ""),
                            job.get("output_name", ""),
                        )

                        course_name = st.text_input("Course name", value=job.get("course_name", ""), key=f"course_{job['job_id']}")
                        unit_col1, unit_col2 = st.columns([0.35, 0.65])
                        with unit_col1:
                            unit_number = st.text_input("Unit number", value=job.get("unit_number", ""), key=f"unit_no_{job['job_id']}")
                        with unit_col2:
                            unit_name = st.text_input("Unit/chapter name", value=job.get("unit_name", ""), key=f"unit_name_{job['job_id']}")
                        output_name = st.text_input("Output file name", value=job.get("output_name", ""), key=f"output_{job['job_id']}")

                        new_values = (course_name, unit_number, unit_name, safe_output_name(output_name, Path(job.get("name", "video")).stem, brand))
                        if new_values != old_values and job.get("status") == "Done":
                            requeue_job(job)
                        job["course_name"], job["unit_number"], job["unit_name"], job["output_name"] = new_values

                        action_col1, action_col2, action_col3 = st.columns(3)
                        with action_col1:
                            st.button("Remove", key=f"remove_{job['job_id']}", use_container_width=True, on_click=remove_job, args=(job["job_id"],), disabled=(job.get("status") == "Processing"))
                        with action_col2:
                            if st.button("Requeue", key=f"requeue_{job['job_id']}", use_container_width=True, disabled=(job.get("status") == "Processing")):
                                requeue_job(job)
                                st.success("Moved back to Queued.")
                        with action_col3:
                            output_path_text = job.get("output_path") or ""
                            output_path = Path(output_path_text) if output_path_text else Path("__missing_output__")
                            can_download = job.get("status") == "Done" and bool(output_path_text) and output_path.exists() and output_path.is_file()
                            if can_download:
                                st.download_button(
                                    "Download",
                                    data=output_path.read_bytes(),
                                    file_name=Path(job.get("output_name", output_path.name)).name,
                                    mime="video/mp4",
                                    key=f"download_{job['job_id']}",
                                    use_container_width=True,
                                )
                            else:
                                st.button("Download", key=f"download_disabled_{job['job_id']}", disabled=True, use_container_width=True)

                        if job.get("status") == "Done":
                            st.success(f"Output ready: {job.get('output_size_label', '')} | {job.get('duration_label', '')}")
                        if job.get("status") == "Failed":
                            st.error(job.get("error", "Unknown error"))

        with st.container(border=True):
            st.subheader("Downloads")
            completed = [
                job
                for job in st.session_state.queue_jobs
                if job.get("status") == "Done"
                and bool(job.get("output_path"))
                and Path(job.get("output_path")).exists()
                and Path(job.get("output_path")).is_file()
            ]
            if completed:
                zip_bytes = build_completed_zip(brand)
                st.download_button(
                    "Download all completed videos as ZIP",
                    data=zip_bytes,
                    file_name=f"{brand.lower()}_completed_rebranded_videos.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                )
                st.caption(f"{len(completed)} completed video(s) available.")
            else:
                st.info("Completed videos will appear here after the queue finishes.")


if __name__ == "__main__":
    main()
