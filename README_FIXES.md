# Corrected Video Rebranding Tool

## Fixes included

- Course title is now white for both Aspirex and GEL intros.
- Unit/chapter text is now purple for both brands.
- The title uses a soft blurred drop shadow instead of a sharp duplicate.
- Title wrapping, visual line spacing and vertical centring were adjusted to match the source intro.
- The unit/chapter pill now has the original-style height, rounded shape, lavender border and soft shadow.
- The extra space after the hyphen was removed.
- Brand-specific intro files are preferred before the generic `Intro.mp4` fallback.
- The logo embedded in the intro is dynamically removed and replaced with the selected brand logo.
- The preview uses the same corrected styling and selected brand logo.
- The intro cache version was updated so older incorrectly styled cached intros are not reused.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py`, the MP4 assets, logo PNG files and `Poppins-Bold.ttf` in the same folder.
