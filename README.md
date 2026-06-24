# Video Rebranding Queue Tool v9

This version keeps the workflow simple and adds a processing queue.

## What it does

- Removes the SLC intro by seconds from the start.
- Removes the SLC outro by seconds from the end.
- Hides the old SLC logo at the fixed position:
  - X = 1640
  - Y = 933
  - Width = 272
  - Height = 126
- Adds the selected brand logo in the same fixed place.
- Generates a new intro for each video using that video's course name, unit number, and unit/chapter name.
- Adds the selected brand outro.
- Processes many uploaded videos through a queue.
- Shows Queued, Processing, Done, and Failed statuses.
- Allows retrying failed jobs and downloading all completed videos as a ZIP.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app uses FFmpeg. If system FFmpeg is not available, it will try to use the FFmpeg binary installed by `imageio-ffmpeg` from `requirements.txt`.

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
6. Click **Start / resume queue**.
7. Download individual outputs or all completed videos as one ZIP.
