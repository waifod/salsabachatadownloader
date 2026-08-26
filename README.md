# salsabachatadownloader

Download lesson videos from [salsabachata.es](https://alumnos.salsabachata.es/) using Playwright.

The school's built-in download system is unreliable and limits you to one video per day. Recordings are also removed over time: the portal currently states that videos from before October 2025 are gone. This script lets you bulk-download your lesson recordings before they disappear.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/USERNAME/salsabachatadownloader.git
cd salsabachatadownloader
uv sync
uv run playwright install firefox
```

## Usage

Credentials can be passed via flags or hardcoded in the script's `DEFAULT_EMAIL` and `DEFAULT_PASSWORD` constants.

The script works from your [Mis clases](https://alumnos.salsabachata.es/mis-clases) listing. That listing is newest first and paginated: `/mis-clases` (equivalently `?page=1`) holds your most recent classes, `?page=2` the ones before those, and so on. With no arguments the script downloads every class with videos on that most recent page:

```sh
uv run python salsabachatadownloader.py
```

To widen or narrow that:

```sh
# The 10 most recent classes that have videos
uv run python salsabachatadownloader.py --latest 10

# Everything still available, across all listing pages
uv run python salsabachatadownloader.py --all

# A date range
uv run python salsabachatadownloader.py --since 2026-07-01 --until 2026-07-31

# See what would be downloaded, without downloading
uv run python salsabachatadownloader.py --latest 10 --list-only
```

Classes are selected by date. The listing is the only place a class's month and year appear, since a lesson page shows just a weekday and a day number (`lun 10`), which is why every run is driven by the listing.

`--since D --until D` narrows to a single day and fetches every class you attended that day.

`--latest` and `--since` stop the crawl as soon as they are satisfied, so they usually read one page. `--until` on its own has no such stopping point and reads the listing to the end, since older classes live on later pages. Cap it with `--index-pages` if you only want recent ones.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-e`, `--email` | | Login email |
| `-p`, `--password` | | Login password |
| `-o`, `--output` | `salsabachata` | Output directory |
| `-w`, `--workers` | `4` | Lesson pages scanned concurrently |
| `-d`, `--download-workers` | `4` | Videos downloaded concurrently |
| `-n`, `--latest` | | Download the N most recent classes with videos |
| `--all` | off | Download from every listing page, not just the most recent |
| `--since` | | Only classes on or after `YYYY-MM-DD` |
| `--until` | | Only classes on or before `YYYY-MM-DD` |
| `--index-pages` | 1 with no arguments, else all | Read only N listing pages, working back from the most recent |
| `-l`, `--list-only` | off | List the selected classes and exit |
| `--visible` | off | Show the browser window |

## Notes

For large batch downloads, consider running inside a `tmux` or `screen` session so a disconnected terminal doesn't kill the process.

The run has three steps. It scans the selected lesson pages through the browser and records what videos exist, then closes the browser and compares that against what's on disk, then downloads what's missing. The comparison is deliberately separate from the scan: what the site holds and what you already have are different questions, and only the first needs a browser.

```
Indexed page 1 of 10: 41 classes with videos.

Scanning 41 lessons, 4 pages at a time
[98252]  2026-08-18 19:00  Salsa Cubana 1         Ale                     Sol - Sala 1       1 video    2.15s
[97950]  2026-08-10 21:00  Bachata 2              Tijana                  Sol - Sala 4       2 videos   3.07s
[96560]  2026-06-30 18:00  Bachata Dominicana 1   Deniol                  Ventas - Sala 1    no videos  1.90s
  [Scanned] 41 lessons in 33.0s (3.05s each), 0 errors, ids 95548-98252
    66 videos, per lesson [1:23, 2:11, 3:7]
  [Plan] 4 to download, 62 already on disk
    to download  [bachata:4]
    on disk      [bachata:58, ruedacubana:2, salsacubana:1, salsaenlinea:1]

Downloading 4 videos, 4 at a time
[1/4]  45.6 MB    1.58s   28.8 MB/s   school      bachata/tij_bachata1_260810t20_97952v1.mp4

  [Downloaded] 4 of 4, 104.8 MB in 1.78s, 59.0 MB/s aggregate
    sources      [school:4]

Done in 41.3s.
```

Scan lines read left to right as identity, then class metadata, then what the run found: lesson id and when it was, the style, teacher and room, then the video count and how long the page took. Every column is left-aligned so values start at the same offset down the page. Counts under `Plan` are videos, not lessons.

The split exists because response bodies fetched through Playwright travel over a single connection to its driver process, where they serialize against each other and against every page operation. Measured on a 2-vCPU VPS: 5.64 MB/s through that connection versus 1073 MB/s for a plain HTTP fetch of the same file, and four concurrent transfers finished in a staircase (7.0s, 12.2s, 18.9s, 25.2s) instead of together. A large download would stall unrelated page loads for minutes. Downloads therefore run outside Playwright, reusing the session cookies, which also means they stream to disk in chunks rather than buffering whole videos in memory.

`-w` sets how many lesson pages are scanned at once and is bound by browser CPU and memory. `-d` sets how many videos download at once and is bound by the network. They're separate because scanning is usually the slower phase now.

Each download announces itself, then reports its size, duration, rate and which host served it, named rather than assumed so an unexpected one shows up as itself. Failures are listed again at the end so they aren't lost in the scroll, and a failed download makes the script exit non-zero.

Files are written to a `.part` file and renamed on completion, so an interrupted run never leaves a truncated file that a later run mistakes for finished. Style directories are created only when a video is about to be written into one.

## What it does

1. Logs into the student portal
2. Crawls the `Mis clases` listing for each class's video-page link and metadata (style, level, date, time, venue, instructor), taking the month and year from each month heading and cross-checking every date against the weekday printed beside it
3. Visits each lesson page and queues the videos that aren't on disk yet
4. Closes the browser, then downloads the queue, preferring the site's own `/descargar-video` endpoint and falling back to the Cloudflare Stream URL
5. Checks the content type and MP4 header before writing, so an error page never lands as a `.mp4`

Videos are organized into subdirectories by dance style, and every name is lowercased:

```
{output}/{style}/{instructor}_{style}{level}_{yymmdd}T{hour}_{lessonId}v{n}.mp4
salsabachata/bachata/tij_bachata2_260810t21_97950v1.mp4
```

`{instructor}` is the first three letters of each name in the credit, so "Nico y Estefi" becomes `nicest`. Accented letters fold to their base letter, so "Salsa en Línea" gives a `salsaenlinea` directory and "Forró" gives `forro`.

Already-downloaded videos are recognised by name and skipped, so the scheme is worth keeping stable. If you change it, existing files are re-downloaded under the new names and the old ones are left behind.

Not every video is downloadable. The school attaches a download button to some of them only, and the script names the ones without. Those fall back to the Stream URL, which may refuse.