# salsabachatadownloader

Download lesson videos from [salsabachata.es](https://alumnos.salsabachata.es/) using Playwright.

The school's built-in download system is unreliable and limits you to one video per day. Lessons are also deleted after 6 months. This script lets you bulk-download your lesson recordings before they disappear.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/USERNAME/salsabachatadownloader.git
cd salsabachatadownloader
uv sync
uv run playwright install firefox
```

## Usage

```sh
uv run python salsabachatadownloader.py LESSON_IDS...
```

Credentials can be passed via flags or hardcoded in the script's `DEFAULT_EMAIL` and `DEFAULT_PASSWORD` constants.

Lesson IDs are the numeric identifiers from the lesson page URL. For example, if the lesson page is `https://alumnos.salsabachata.es/mis-videos/88693`, the lesson ID is `88693`.

Lesson IDs can be individual values or ranges:

```sh
# Single lessons
uv run python salsabachatadownloader.py 87933 87935

# Ranges
uv run python salsabachatadownloader.py 87933-87940

# Mix of both
uv run python salsabachatadownloader.py 87933 87935-87940 87950

# With explicit credentials
uv run python salsabachatadownloader.py -e me@example.com -p secret 87933-87940
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-e`, `--email` | | Login email |
| `-p`, `--password` | | Login password |
| `-o`, `--output` | `salsabachata` | Output directory |
| `-w`, `--workers` | `4` | Concurrent browser pages |
| `--visible` | off | Show the browser window |

## Notes

For large batch downloads, consider running inside a `tmux` or `screen` session so a disconnected terminal doesn't kill the process.

Videos are loaded fully into memory before writing to disk. Based on 218 downloaded lessons, most videos are small (median 15 MB, 95th percentile 60 MB). The headless Firefox instance uses roughly 200-300 MB on its own. With the default 4 workers, expect peak memory usage around 300-500 MB. Adjust `-w` if memory is tight.

## What it does

1. Logs into the student portal
2. Visits each lesson page and extracts metadata (instructor, style, level, date)
3. Downloads videos from Cloudflare Stream embeds
4. Skips files that already exist locally
5. Retries failed downloads with exponential backoff

Videos are organized into subdirectories by dance style: `{output}/{style}/{instructor}_{style}{level}_{date}_{lessonId}v{n}.mp4`