# Video Rebranding Queue Tool v11

This version keeps the queue workflow and updates the logo replacement, cover-title typography, ZIP download, and Google Drive save workflow.

## What it does

- Removes the SLC intro by seconds from the start.
- Removes the SLC outro by seconds from the end.
- Hides the exact rendered SLC logo footprint:
  - X = 1722
  - Y = 966
  - Width = 106
  - Height = 60
- Adds the selected brand logo at the same exact position and rendered size: 106 x 60 px at X=1722, Y=966.
- Generates a new intro cover page for each video using that video's course name, unit number, and unit/chapter name.
- Course title sizing is fixed to:
  - Default / max size = 52 px
  - Minimum auto-shrink size = 28 px
  - Preferred font = Poppins-Bold.ttf
- Processes many uploaded videos through a queue.
- Shows Queued, Processing, Done, and Failed statuses.
- Allows retrying failed jobs.
- Downloads all completed videos as one ZIP.
- Can save the completed ZIP to a local Google Drive sync folder or mounted Google Drive path.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app uses FFmpeg. If system FFmpeg is not available, it will try to use the FFmpeg binary installed by `imageio-ffmpeg` from `requirements.txt`.

## Poppins font

The app is set to use `Poppins-Bold.ttf` for the cover-page course title. The font file is not bundled in this package.

To use the exact font, place `Poppins-Bold.ttf` in one of these locations before running Streamlit:

- Next to `app.py`
- `assets/fonts/Poppins-Bold.ttf`
- `fonts/Poppins-Bold.ttf`

You can also set an environment variable named `POPPINS_BOLD_FONT` to the full path of the font file. If the font is not found, the app uses a bold fallback font and shows that in the sidebar.

## Google Drive save option

The **Save ZIP to Google Drive folder** button saves the completed ZIP to a folder path that the Streamlit server can access. This works with:

- A local Google Drive desktop sync folder
- A mounted Google Drive path such as `/content/drive/MyDrive/...`
- A server folder that is already connected/synced to Google Drive

For hosted cloud deployment, connect or mount Google Drive on the server first.

## Recommended settings

- Processing speed: `Fast (recommended)`
- Intro remove: use the SLC intro length in seconds.
- Outro remove: use the SLC outro length in seconds.

## Queue workflow

1. Select brand and speed in the sidebar.
2. Set the intro/outro cut seconds in the sidebar.
3. Upload videos.
4. Click **Add to queue**.
5. Edit course name, unit number, unit/chapter name, and output filename for each queued video.
6. Check the cover-page preview.
7. Click **Start / resume queue**.
8. Download the completed ZIP or save it to your Google Drive folder.
