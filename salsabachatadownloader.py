"""Download lesson videos from salsabachata.es."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from urllib.parse import urljoin, urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeout,
    async_playwright,
)

# --- CONFIGURATION ---
DEFAULT_EMAIL = ""
DEFAULT_PASSWORD = ""
DEFAULT_OUTPUT_DIR = "salsabachata"
DEFAULT_WORKERS = 4
DEFAULT_DOWNLOAD_WORKERS = 4
# ---------------------

BASE_URL = "https://alumnos.salsabachata.es"
LOGIN_URL = f"{BASE_URL}/"
CLASSES_URL = f"{BASE_URL}/mis-clases"
LESSON_URL = f"{BASE_URL}/mis-videos/"

# Read size for streaming a video to disk.
CHUNK_BYTES = 256 * 1024
# Anything smaller than this is an error page, not a video.
MIN_VIDEO_BYTES = 1024
# Per socket operation, not per download, so it does not cap a large file.
SOCKET_TIMEOUT = 60

# Matches both the iframe embed and any other Stream reference in the page HTML.
STREAM_RE = re.compile(
    r"(https://(?:customer-[0-9a-z]+\.cloudflarestream\.com|videodelivery\.net))"
    r"/([0-9a-f]{32})"
)
VIDEO_COUNT_RE = re.compile(r"V[ií]deo\s+\d+\s+de\s+(\d+)", re.IGNORECASE)
PAGER_RE = re.compile(r"(\d+)\s+de\s+(\d+)")

# Safety net for the case where the listing stops reporting its page total.
MAX_INDEX_PAGES = 50

MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

# Weekday abbreviations the listing prints beside each day number, for
# cross-checking a parsed date.
WEEKDAYS = {"lun": 0, "mar": 1, "mie": 2, "jue": 3, "vie": 4, "sab": 5, "dom": 6}

MONTH_LABEL_RE = re.compile(r"^([^\W\d_]+)\s+de\s+(\d{4})$", re.UNICODE)

# Reads one class row out of the /mis-clases listing.
ROW_READER_JS = """
const readRow = (el) => {
    const grow = el.querySelector('.d-list-col-grow') || el;
    const title = grow.querySelector('.font-semibold');
    if (!title) return null;
    const dayEl = el.querySelector('.tabular-nums');
    const weekdayEl = el.querySelector('.opacity-60');
    const meta = Array.from(grow.querySelectorAll('.opacity-70'))
        .map((n) => n.textContent.trim())
        .find((t) => t.includes('\\u00b7')) || null;
    const link = el.querySelector('a[href*="/mis-videos/"]');
    const badge = el.querySelector('.d-badge');
    return {
        title: title.textContent.trim(),
        day: dayEl ? dayEl.textContent.trim() : null,
        weekday: weekdayEl ? weekdayEl.textContent.trim() : null,
        meta: meta,
        href: link ? link.getAttribute('href') : null,
        badge: badge ? badge.textContent.trim() : null,
    };
};
"""

# Walks <main> in document order so each row keeps the month heading above it.
INDEX_JS = f"""
() => {{
    {ROW_READER_JS}
    const root = document.querySelector('main') || document.body;
    const rows = [];
    let monthLabel = null;
    for (const node of root.querySelectorAll('h2, li')) {{
        if (node.tagName === 'H2') {{
            monthLabel = node.textContent.trim();
            continue;
        }}
        const row = readRow(node);
        if (row) {{
            row.monthLabel = monthLabel;
            rows.push(row);
        }}
    }}
    // The pager reads like "1 de 10", which gives the total page count. Scoped
    // to a nav because row day-numbers share its .tabular-nums class.
    let pager = null;
    for (const nav of root.querySelectorAll('nav')) {{
        const text = nav.textContent.replace(/\\s+/g, ' ').trim();
        if (/\\d+\\s+de\\s+\\d+/.test(text)) {{
            pager = text;
            break;
        }}
    }}
    return {{ rows: rows, pager: pager }};
}}
"""

# Each video sits in its own card: a Stream iframe plus, when the school allows
# downloading it, an Alpine component holding the first-party download URL.
VIDEOS_JS = """
() => {
    const root = document.querySelector('main') || document.body;
    const out = [];
    for (const card of root.querySelectorAll('.d-card')) {
        const iframe = card.querySelector('iframe[src]');
        let downloadUrl = null;
        for (const holder of card.querySelectorAll('[x-data]')) {
            const raw = holder.getAttribute('x-data') || '';
            const m = raw.match(/videoDownload\\(\\s*['"]([^'"]+)['"]/);
            if (m) {
                downloadUrl = m[1];
                break;
            }
        }
        let label = null;
        for (const el of card.querySelectorAll('.opacity-70')) {
            const text = el.textContent.trim();
            if (/^V[\\u00edi]deo\\s+\\d+/i.test(text)) {
                label = text;
                break;
            }
        }
        if (!iframe && !downloadUrl) continue;
        out.push({
            iframeSrc: iframe ? iframe.getAttribute('src') : null,
            downloadUrl: downloadUrl,
            label: label,
        });
    }
    return out;
}
"""


def strip_accents(text: str) -> str:
    """
    Fold accented letters to their base letter: 'Línea' -> 'Linea'.

    Used both to match Spanish month and weekday names and to build filenames.
    """
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def sanitize_filename(name: str) -> str:
    """Remove filesystem-unsafe characters from a filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def format_instructor_name(raw_name: str) -> str:
    """
    Split on separators and take the first 3 letters of each name.
    Ex: "Jose y Maria" -> 'josmar', "Valentín y Angy" -> 'valang'.

    Accents fold to their base letter, so "Fornié" contributes 'for' and not a
    truncated 'fo'.
    """
    clean = re.sub(r" y | & |-|/", " ", strip_accents(raw_name).lower())
    clean = re.sub(r"[^a-z\s]", "", clean)
    parts = clean.split()
    return "".join([part[:3] for part in parts if part])


def format_style(raw_style: str) -> str:
    """
    Reduce a class title to a bare style name, dropping the level digits.
    Ex: "Bachata 2" -> 'bachata', "Salsa en Línea 1" -> 'salsaenlinea'.
    """
    return re.sub(r"[^a-zA-Z]", "", strip_accents(raw_style)).lower()


def parse_month_label(label: str | None) -> tuple[int | None, int | None]:
    """Parse a listing heading like 'Agosto de 2026' into (month, year)."""
    if not label:
        return None, None
    match = MONTH_LABEL_RE.match(label.strip())
    if not match:
        return None, None
    month = MONTHS.get(strip_accents(match.group(1)).lower())
    return month, int(match.group(2))


@dataclass
class Lesson:
    """One class from the /mis-clases listing."""

    lesson_id: int
    title: str = ""
    day: int | None = None
    weekday: str | None = None
    month: int | None = None
    year: int | None = None
    time: str | None = None
    venue: str | None = None
    instructor: str | None = None
    video_count: int | None = None

    @property
    def date(self) -> date | None:
        """
        Class date, or None when the listing did not yield a full one.

        The day comes from the row and the month and year from the heading above
        it. None means one of those was unreadable.
        """
        if not (self.day and self.month and self.year):
            return None
        try:
            return date(self.year, self.month, self.day)
        except ValueError:
            return None

    def weekday_matches(self) -> bool | None:
        """
        Check the parsed date against the weekday the page printed next to it.

        The two are independent fields, so disagreement means the month heading
        or the day number was misread. Returns None when there is nothing to
        compare.
        """
        when = self.date
        if when is None or not self.weekday:
            return None
        expected = WEEKDAYS.get(strip_accents(self.weekday).lower()[:3])
        if expected is None:
            return None
        return when.weekday() == expected

    @property
    def style_code(self) -> str:
        """Dance style with the level digits stripped, e.g. 'bachata'."""
        return format_style(self.title) or "unknown"

    @property
    def level_code(self) -> str:
        """Trailing level digits from the title, e.g. '2' for 'Bachata 2'."""
        match = re.search(r"(\d+)\s*$", self.title)
        return match.group(1) if match else ""

    @property
    def hour(self) -> str:
        """
        Hour portion of the class time, e.g. '21'.

        Taken as printed, so it is not zero-padded. Classes run in the evening,
        so this is two digits in practice.
        """
        if self.time and ":" in self.time:
            return self.time.split(":")[0].strip()
        return "xx"

    def filename_prefix(self) -> str:
        """
        Build the shared filename prefix for this lesson's videos.

        Already-downloaded files are recognised by name, so changing the layout
        makes the script re-download everything under the new one.
        """
        instructor = format_instructor_name(self.instructor or "") or "unknown"
        when = self.date
        yymmdd = when.strftime("%y%m%d") if when else "nodate"
        return f"{instructor}_{self.style_code}{self.level_code}_{yymmdd}T{self.hour}"

    @property
    def when(self) -> str:
        """Date and time as a fixed-width column."""
        when = self.date
        return f"{when.isoformat() if when else '????-??-??'} {self.time or '--:--'}"

    def columns(self) -> str:
        """
        Style, instructor and venue as fixed-width columns.

        Widths come from the longest real values: 'Bachata Dominicana 1' at 20
        characters, 'Julio Marquetti y Moni' at 22 and 'América - Sala 1' at 16.
        Venue is padded rather than left ragged, because the counts printed
        after it would otherwise not line up. Callers rstrip the line they
        build, so a class with no venue leaves no trailing space.
        """
        return (
            f"{self.title or '?':<22}"
            f" {self.instructor or '?':<23}"
            f" {self.venue or '':<17}"
        )


def lesson_from_row(row: dict) -> Lesson | None:
    """
    Turn a raw listing row into a Lesson.

    Returns None for a class with no link to a video page, which is how the
    listing shows classes whose recordings have been deleted.
    """
    match = re.search(r"/mis-videos/(\d+)", row.get("href") or "")
    if not match:
        return None
    resolved_id = int(match.group(1))

    month, year = parse_month_label(row.get("monthLabel"))

    day = None
    if row.get("day"):
        digits = re.sub(r"\D", "", row["day"])
        day = int(digits) if digits else None

    time_str = venue = instructor = None
    meta = row.get("meta")
    if meta:
        parts = [p.strip() for p in meta.split("\u00b7") if p.strip()]
        for part in parts:
            if re.fullmatch(r"\d{1,2}:\d{2}", part):
                time_str = part
                break
        rest = [p for p in parts if p != time_str]
        if rest:
            instructor = rest[-1]
        if len(rest) > 1:
            venue = " ".join(rest[:-1])

    video_count = None
    if row.get("badge"):
        digits = re.sub(r"\D", "", row["badge"])
        video_count = int(digits) if digits else None

    return Lesson(
        lesson_id=resolved_id,
        title=row.get("title") or "",
        day=day,
        weekday=row.get("weekday"),
        month=month,
        year=year,
        time=time_str,
        venue=venue,
        instructor=instructor,
        video_count=video_count,
    )


def parse_pager_total(text: str | None) -> int | None:
    """Read the total page count from the listing's '1 de 10' indicator."""
    if not text:
        return None
    match = PAGER_RE.search(text)
    if not match:
        return None
    total = int(match.group(2))
    return total if total > 0 else None


def listing_url(number: int) -> str:
    """
    URL of a listing page.

    Page 1 is the bare path, which is the form the site links to. '?page=1'
    serves the same page.
    """
    return CLASSES_URL if number <= 1 else f"{CLASSES_URL}?page={number}"


async def iter_index(page: Page, max_pages: int | None = None):
    """
    Yield pages of the /mis-clases listing, walking back from the most recent.

    Page 1 holds the newest classes, and its '1 de 10' pager gives the total up
    front. Yields (number, total, lessons) one page at a time so a caller with
    enough classes can stop without reading the rest.
    """
    total: int | None = None
    number = 1

    while max_pages is None or number <= max_pages:
        await page.goto(listing_url(number), wait_until="domcontentloaded")
        result = await page.evaluate(INDEX_JS)

        if total is None:
            total = parse_pager_total(result.get("pager"))

        # Rows include classes whose videos are gone, so an empty row set means
        # the page itself is empty: either past the end or an empty listing.
        rows = result.get("rows", [])
        if not rows:
            if number == 1:
                print(
                    "Warning: no classes found on the listing at all."
                    " If you expected some, the login may not have taken effect."
                )
            break

        lessons = []
        for row in rows:
            lesson = lesson_from_row(row)
            if lesson:
                lessons.append(lesson)

        yield number, total, lessons

        number += 1
        if total is not None:
            if number > total:
                break
        elif number > MAX_INDEX_PAGES:
            print(
                f"Warning: no page total found; stopping after"
                f" {MAX_INDEX_PAGES} listing pages."
            )
            break


@dataclass
class VideoSource:
    """One video on a lesson page, with its download URLs in priority order."""

    urls: list[str]
    label: str | None = None
    has_button: bool = False


def stream_download_url(src: str) -> str | None:
    """Turn a Stream iframe src into its direct download URL."""
    match = STREAM_RE.search(src)
    if not match:
        return None
    base, uid = match.group(1), match.group(2)
    return f"{base}/{uid}/downloads/default.mp4"


async def find_videos(page: Page) -> tuple[list[VideoSource], int | None]:
    """
    Collect download candidates from a lesson page.

    Each video card carries a Stream iframe and, when the school permits saving
    it, a "Guardar" button holding a first-party /descargar-video URL. The
    first-party URL is tried first: it is the path the site itself uses, and
    Stream's direct download is not enabled for every video.

    Also returns the count the page claims ("Vídeo 1 de 2"), which lets the
    caller notice when fewer videos were found than the page advertises.
    """
    try:
        cards = await page.evaluate(VIDEOS_JS)
    except PlaywrightError as e:
        print(f"Warning: could not read the video cards ({e}); using raw HTML.")
        cards = []

    sources: list[VideoSource] = []
    for card in cards:
        urls: list[str] = []
        if card.get("downloadUrl"):
            urls.append(urljoin(BASE_URL, card["downloadUrl"]))

        if card.get("iframeSrc"):
            stream_url = stream_download_url(card["iframeSrc"])
            if stream_url:
                urls.append(stream_url)

        if urls:
            sources.append(
                VideoSource(
                    urls=urls,
                    label=card.get("label"),
                    has_button=bool(card.get("downloadUrl")),
                )
            )

    html = await page.content()

    expected = None
    counts = VIDEO_COUNT_RE.findall(html)
    if counts:
        expected = max(int(c) for c in counts)

    # Fallback for when no card yields a URL: take any Stream reference in the
    # raw HTML. Loses the pairing with the download buttons, but finds videos
    # even if the card markup stops matching.
    if not sources:
        seen: set[str] = set()
        for base, uid in STREAM_RE.findall(html):
            if uid in seen:
                continue
            seen.add(uid)
            sources.append(VideoSource(urls=[f"{base}/{uid}/downloads/default.mp4"]))
        if sources:
            print("Warning: read videos from raw HTML; the card markup has moved.")

    return sources, expected


def fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    return f"{n / 1024 / 1024 / 1024:.2f} GB"


def fmt_seconds(seconds: float) -> str:
    """Format a duration, keeping precision on short ones."""
    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 600:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def plural(n: int, noun: str, many: str | None = None) -> str:
    """
    '1 video' / '2 videos', so counts never read as '1 video(s)'.

    Irregular plurals are given rather than derived: 'class' would otherwise
    come out as 'classs'.
    """
    if n == 1:
        return f"{n} {noun}"
    return f"{n} {many or noun + 's'}"


def source_label(url: str) -> str:
    """
    Name the host a URL points at, for reporting which one served a video.

    Named rather than guessed, so an unexpected host shows up as itself instead
    of being reported as one of the two known ones.
    """
    if url.startswith(BASE_URL):
        return "school"
    host = urlparse(url).hostname or "?"
    if host.endswith(("cloudflarestream.com", "videodelivery.net")):
        return "cloudflare"
    return host


@dataclass
class ScannedLesson:
    """One lesson page and the videos it offered."""

    lesson: Lesson
    sources: list[VideoSource]


@dataclass
class Scan:
    """
    What the browser phase found.

    Holds no opinion about what to download: it records the site's state, and
    plan_downloads decides what is missing locally.
    """

    lessons: list[ScannedLesson] = field(default_factory=list)
    cookies: list[dict] = field(default_factory=list)
    user_agent: str = ""
    visited: int = 0
    errors: int = 0
    sum_elapsed: float = 0.0
    elapsed: float = 0.0
    video_dist: Counter = field(default_factory=Counter)
    style_dist: Counter = field(default_factory=Counter)
    min_id: int | None = None
    max_id: int | None = None

    @property
    def videos(self) -> int:
        """Total videos found across every lesson."""
        return sum(len(item.sources) for item in self.lessons)

    def record(
        self,
        lesson: Lesson,
        sources: list[VideoSource],
        elapsed: float,
        error: str | None = None,
    ) -> None:
        """Record one visited lesson page."""
        self.visited += 1
        self.sum_elapsed += elapsed
        self.video_dist[len(sources)] += 1
        if error:
            self.errors += 1
        if sources:
            self.lessons.append(ScannedLesson(lesson=lesson, sources=sources))
            self.style_dist[lesson.style_code] += 1
            if self.min_id is None or lesson.lesson_id < self.min_id:
                self.min_id = lesson.lesson_id
            if self.max_id is None or lesson.lesson_id > self.max_id:
                self.max_id = lesson.lesson_id

    def print_summary(self) -> None:
        """
        Report what the site holds, before any local comparison.

        Styles are left to the plan, which counts videos rather than lessons,
        so the two are not reported in units that look comparable but are not.
        """
        if self.visited == 0:
            return
        avg = self.sum_elapsed / self.visited
        span = f", ids {self.min_id}-{self.max_id}" if self.min_id else ""
        dist = ", ".join(f"{k}:{self.video_dist[k]}" for k in sorted(self.video_dist))
        print(
            f"  [Scanned] {plural(self.visited, 'lesson')}"
            f" in {fmt_seconds(self.elapsed)} ({fmt_seconds(avg)} each),"
            f" {plural(self.errors, 'error')}{span}\n"
            f"    {plural(self.videos, 'video')}, per lesson [{dist}]"
        )


@dataclass
class DownloadJob:
    """One video to fetch, carrying the lesson that named it."""

    lesson: Lesson
    urls: list[str]
    filepath: str
    filename: str
    style_code: str
    index: int
    of: int
    has_button: bool

    @property
    def where(self) -> str:
        """Path as reported, relative to the output directory."""
        return f"{self.style_code}/{self.filename}"


def plan_downloads(
    scan: Scan, output_path: str
) -> tuple[list[DownloadJob], list[DownloadJob]]:
    """
    Turn scanned lessons into jobs, split into (to fetch, already on disk).

    Kept out of the browser phase on purpose: deciding what is missing is a
    local filesystem question, so it needs no page open and is verifiable
    without one.
    """
    pending: list[DownloadJob] = []
    present: list[DownloadJob] = []

    for item in scan.lessons:
        prefix = item.lesson.filename_prefix()
        lesson_id = item.lesson.lesson_id
        style_code = item.lesson.style_code

        for index, source in enumerate(item.sources, start=1):
            filename = sanitize_filename(f"{prefix}_{lesson_id}v{index}.mp4".lower())
            job = DownloadJob(
                lesson=item.lesson,
                urls=source.urls,
                filepath=os.path.join(output_path, style_code, filename),
                filename=filename,
                style_code=style_code,
                index=index,
                of=len(item.sources),
                has_button=source.has_button,
            )
            (present if os.path.exists(job.filepath) else pending).append(job)

    return pending, present


def style_counts(jobs: list[DownloadJob]) -> str:
    """Videos per style, for reporting a plan."""
    counts = Counter(job.style_code for job in jobs)
    return ", ".join(f"{style}:{counts[style]}" for style in sorted(counts))


def print_plan(pending: list[DownloadJob], present: list[DownloadJob]) -> None:
    """
    Report what will be fetched against what is already here.

    The total is derived from the two lists rather than passed in, since every
    video found lands in exactly one of them.
    """
    print(f"  [Plan] {len(pending)} to download, {len(present)} already on disk")
    if pending:
        print(f"    to download  [{style_counts(pending)}]")
    if present:
        print(f"    on disk      [{style_counts(present)}]")


@dataclass
class DownloadResult:
    """Outcome of one job, including which source served it."""

    job: DownloadJob
    size: int = 0
    elapsed: float = 0.0
    source: str | None = None
    problems: list[str] = field(default_factory=list)


async def scrape_lesson(
    context: BrowserContext,
    lesson: Lesson,
    scan: Scan,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Open one lesson page and record the videos it offers.

    Touches no filesystem: what is already downloaded is decided afterwards by
    plan_downloads.
    """
    async with semaphore:
        lesson_id = lesson.lesson_id
        t_start = time.monotonic()

        page = await context.new_page()
        try:
            await page.goto(f"{LESSON_URL}{lesson_id}", wait_until="domcontentloaded")
            sources, expected = await find_videos(page)
            elapsed = time.monotonic() - t_start
            scan.record(lesson, sources, elapsed)

            if not sources:
                print(
                    f"[{lesson_id}]  {lesson.when}  {lesson.columns()}"
                    f"  {'no videos':<9}  {fmt_seconds(elapsed):<6}".rstrip()
                )
                return

            if expected and expected != len(sources):
                print(
                    f"[{lesson_id}] Warning: page says {expected} video(s),"
                    f" found {len(sources)}."
                )

            no_button = [
                s.label or f"video {i + 1}"
                for i, s in enumerate(sources)
                if not s.has_button
            ]
            note = f"  no download button: {', '.join(no_button)}" if no_button else ""
            print(
                f"[{lesson_id}]  {lesson.when}  {lesson.columns()}"
                f"  {plural(len(sources), 'video'):<9}"
                f"  {fmt_seconds(elapsed):<6}{note}".rstrip()
            )

        except Exception as e:
            elapsed = time.monotonic() - t_start
            scan.record(lesson, [], elapsed, str(e))
            print(
                f"[{lesson_id}]  {lesson.when}  {lesson.columns()}"
                f"  {'ERROR':<9}  {fmt_seconds(elapsed):<6}  {e}".rstrip()
            )
        finally:
            await page.close()


class DropCookieOnRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the school session cookie from following a redirect off-site."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            new is not None
            and urlparse(newurl).hostname != urlparse(req.full_url).hostname
        ):
            new.headers.pop("Cookie", None)
            new.unredirected_hdrs.pop("Cookie", None)
        return new


OPENER = urllib.request.build_opener(DropCookieOnRedirect())


def cookie_header(cookies: list[dict], url: str) -> str:
    """
    Build a Cookie header for one URL out of the browser's cookies.

    Scoped by domain, so the school's session cookie is never handed to
    Cloudflare.
    """
    host = urlparse(url).hostname or ""
    pairs = []
    for cookie in cookies:
        domain = (cookie.get("domain") or "").lstrip(".")
        if domain and (host == domain or host.endswith(f".{domain}")):
            pairs.append(f"{cookie['name']}={cookie['value']}")
    return "; ".join(pairs)


def video_rejection(head: bytes, content_type: str) -> str | None:
    """
    Say why a response is not an MP4, or None if it looks like one.

    A lapsed session or a rate limit answers with an HTML page and a 200
    status. Writing that to a .mp4 would leave a file later runs skip as
    already downloaded, so the start of the body is checked first.
    """
    kind = content_type.split(";")[0].strip().lower()
    if kind and not (kind.startswith("video/") or "octet-stream" in kind):
        return f"served {kind}"
    # MP4 and friends put an 'ftyp' box at the start of the file.
    if b"ftyp" not in head[:64]:
        return "no MP4 header"
    return None


def download_to_file(
    url: str, headers: dict[str, str], filepath: str
) -> tuple[int, str | None]:
    """
    Stream one URL to disk. Returns (bytes written, error) with error None on
    success.

    Bytes go to a '.part' file that is renamed only once the transfer finishes,
    so an interrupted run cannot leave a truncated file that the next run
    mistakes for a complete download.
    """
    part = f"{filepath}.part"
    request = urllib.request.Request(url, headers=headers)
    try:
        with OPENER.open(request, timeout=SOCKET_TIMEOUT) as response:
            head = response.read(CHUNK_BYTES)
            reason = video_rejection(head, response.headers.get("content-type", ""))
            if reason:
                return 0, reason

            # Only once the response is known to be a video, so a failed run
            # leaves no empty style directory behind.
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            written = 0
            with open(part, "wb") as f:
                f.write(head)
                written = len(head)
                while chunk := response.read(CHUNK_BYTES):
                    f.write(chunk)
                    written += len(chunk)

        if written < MIN_VIDEO_BYTES:
            return 0, f"only {written} bytes"

        os.replace(part, filepath)
        return written, None
    except urllib.error.HTTPError as e:
        return 0, f"HTTP {e.code}"
    except Exception as e:
        return 0, str(e) or e.__class__.__name__
    finally:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass


def download_job(
    job: DownloadJob, cookies: list[dict], user_agent: str
) -> DownloadResult:
    """
    Fetch one video, trying each candidate URL in turn.

    Runs in a worker thread: it does no printing, so the caller can report
    results in a readable order.
    """
    result = DownloadResult(job=job)
    for url in job.urls:
        source = source_label(url)
        headers = {"User-Agent": user_agent, "Accept": "*/*"}
        jar = cookie_header(cookies, url)
        if jar:
            headers["Cookie"] = jar

        t_start = time.monotonic()
        size, error = download_to_file(url, headers, job.filepath)
        if size:
            result.size = size
            result.elapsed = max(time.monotonic() - t_start, 0.001)
            result.source = source
            return result
        result.problems.append(f"{source}: {error}")

    return result


def download_phase(
    jobs: list[DownloadJob], cookies: list[dict], user_agent: str, workers: int
) -> int:
    """
    Fetch every queued video, several at a time. Returns the number that failed.

    Deliberately outside Playwright: response bodies routed through its driver
    connection serialize against each other and against every page operation.
    """
    total = len(jobs)
    width = len(str(total))
    print(f"\nDownloading {plural(total, 'video')}, {workers} at a time")

    t_start = time.monotonic()
    written = 0
    failures: list[DownloadResult] = []
    by_source: Counter = Counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_job, job, cookies, user_agent) for job in jobs]
        for n, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            # Padded inside the brackets so they stay a fixed width and the
            # column after them does not move.
            counter = f"[{n:>{width}}/{total}]"

            if result.size:
                written += result.size
                by_source[result.source] += 1
                rate = f"{result.size / result.elapsed / 1024 / 1024:.1f} MB/s"
                print(
                    f"{counter}  {fmt_bytes(result.size):<9}"
                    f"  {fmt_seconds(result.elapsed):<6}  {rate:<10}"
                    f"  {result.source:<10}  {result.job.where}"
                )
            else:
                failures.append(result)
                print(
                    f"{counter}  {'FAILED':<9}  {'':<6}  {'':<10}"
                    f"  {'':<10}  {result.job.where}"
                )

    elapsed = max(time.monotonic() - t_start, 0.001)
    sources = ", ".join(f"{s}:{by_source[s]}" for s in sorted(by_source))
    print(
        f"  [Downloaded] {total - len(failures)} of {total},"
        f" {fmt_bytes(written)} in {fmt_seconds(elapsed)},"
        f" {written / elapsed / 1024 / 1024:.1f} MB/s aggregate\n"
        f"    sources      [{sources}]"
    )

    # Repeated at the end so a failure is not lost in the scroll above.
    if failures:
        print(f"    {len(failures)} failed:")
        for result in failures:
            print(f"      {result.job.where}: {'; '.join(result.problems)}")

    return len(failures)


async def login(page: Page, email: str, password: str) -> None:
    """Perform login and wait for the redirect out of the login page."""
    selector = 'input[name="email"], input[type="email"]'
    await page.wait_for_selector(selector, state="visible", timeout=10000)
    await page.fill(selector, email)
    await page.fill('input[name="password"], input[type="password"]', password)
    await asyncio.sleep(0.5)
    await page.keyboard.press("Enter")

    try:
        await page.wait_for_url("**/dashboard", timeout=20000)
    except PlaywrightTimeout:
        # Landing somewhere other than the dashboard is fine; still seeing the
        # login form is not.
        if await page.locator(selector).count():
            raise
        print(f"Note: landed on {page.url} instead of the dashboard.")


def select_targets(args: argparse.Namespace, lessons: list[Lesson]) -> list[Lesson]:
    """Apply date filters and the --latest cap to indexed lessons."""
    selected = []
    for lesson in lessons:
        when = lesson.date
        if args.since and (when is None or when < args.since):
            continue
        if args.until and (when is None or when > args.until):
            continue
        selected.append(lesson)
        if args.latest and len(selected) >= args.latest:
            break
    return selected


async def collect_targets(page: Page, args: argparse.Namespace) -> list[Lesson]:
    """
    Work out which lessons to download.

    The listing is the only source of lessons. It supplies the lesson ID and,
    through its month heading ('Agosto de 2026'), the date: nothing on the
    lesson page itself carries a month or year.
    """
    seen: set[int] = set()
    candidates: list[Lesson] = []
    mismatched: list[Lesson] = []

    async for number, total, lessons in iter_index(page, args.index_pages):
        where = f"page {number} of {total}" if total else f"page {number}"
        print(f"Indexed {where}: {len(lessons)} class(es) with videos.")

        for lesson in lessons:
            if lesson.lesson_id in seen:
                continue
            seen.add(lesson.lesson_id)
            if lesson.weekday_matches() is False:
                mismatched.append(lesson)
            # Having a video page is the qualifier. The count badge beside it is
            # only reported, so a change to it cannot silence the whole run.
            candidates.append(lesson)

        if args.latest and len(select_targets(args, candidates)) >= args.latest:
            break
        if args.since and lessons:
            oldest = lessons[-1].date
            if oldest and oldest < args.since:
                break

    targets = select_targets(args, candidates)

    # Both of these mean the listing was misread, not that the data is odd: the
    # weekday is parsed independently of the date, and every listing row carries
    # a month heading above it.
    if mismatched:
        ids = ", ".join(str(item.lesson_id) for item in mismatched)
        print(
            f"Warning: {len(mismatched)} class(es) have a date that disagrees with"
            f" their printed weekday ({ids}). The listing markup may have changed."
        )

    # Checked across everything indexed, not just the selection: a date filter
    # drops undated classes, which would hide the very problem being reported.
    undated = [item for item in candidates if item.date is None]
    if undated:
        ids = ", ".join(str(item.lesson_id) for item in undated)
        print(
            f"Warning: {len(undated)} indexed class(es) have no readable date"
            f" ({ids}). The month headings may have changed. Date filters skip"
            " them, and without a filter their files are named 'nodate'."
        )

    return targets


async def fill_scan(browser: Browser, args: argparse.Namespace, scan: Scan) -> bool:
    """
    Log in, choose the lessons, and record what videos they hold.

    Returns False when there is nothing to hand on.
    """
    context = await browser.new_context()
    page = await context.new_page()

    await page.goto(LOGIN_URL)

    try:
        await login(page, args.email, args.password)
        print("Logged in.")
    except (PlaywrightError, PlaywrightTimeout) as e:
        print(f"Login failed: {e}")
        sys.exit(1)

    # Captured while a page is still open; the downloads run without one.
    scan.user_agent = await page.evaluate("() => navigator.userAgent")

    targets = await collect_targets(page, args)
    await page.close()

    if not targets:
        print("No classes selected.")
        return False

    if args.list_only:
        print(f"\n{plural(len(targets), 'class', 'classes')} selected:")
        for lesson in targets:
            count = lesson.video_count
            label = "? videos" if count is None else plural(count, "video")
            print(
                f"[{lesson.lesson_id}]  {lesson.when}"
                f"  {lesson.columns()}  {label:<9}".rstrip()
            )
        return False

    print(
        f"\nScanning {plural(len(targets), 'lesson')}, {args.workers} pages at a time"
    )

    t_start = time.monotonic()
    semaphore = asyncio.Semaphore(args.workers)
    await asyncio.gather(
        *(scrape_lesson(context, lesson, scan, semaphore) for lesson in targets)
    )
    scan.elapsed = time.monotonic() - t_start

    # Taken after scraping so any refreshed session cookie is the one used.
    scan.cookies = await context.cookies()
    return True


async def scrape_phase(args: argparse.Namespace) -> Scan | None:
    """
    Scan the selected lessons, then shut the browser down.

    Firefox and the Playwright driver are both gone before this returns, on
    every path including failure, so the download phase runs with that memory
    released. Returns None when there is nothing to hand on.
    """
    scan = Scan()

    async with async_playwright() as p:
        print("Launching Firefox...")
        browser = await p.firefox.launch(headless=not args.visible)
        try:
            ready = await fill_scan(browser, args, scan)
        finally:
            await browser.close()

    if not ready:
        return None

    scan.print_summary()
    return scan


def apply_default_scope(args: argparse.Namespace) -> str | None:
    """
    Fill in the default choice of classes, returning a note to print, or None.

    Choosing nothing means every class with videos on the most recent listing
    page, which is page 1 and is reachable as /mis-clases with no query string.
    An explicit --index-pages widens that, so the note reports the page count
    actually in force. The note names the flags it stood in for, since 'no
    selection' means nothing to anyone reading the output.
    """
    if args.latest or args.all or args.since or args.until:
        return None

    args.all = True
    if args.index_pages is None:
        args.index_pages = 1

    pages = args.index_pages
    scope = "the most recent page" if pages == 1 else f"the {pages} most recent pages"
    return (
        f"Taking every class with videos from {scope} of your listing"
        " (no --latest, --since, --until or --all given)."
    )


def parse_date_arg(value: str) -> date:
    """Parse a YYYY-MM-DD command line date."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got '{value}'") from e


def run() -> None:
    """Parse arguments and kick off the download."""
    parser = argparse.ArgumentParser(
        description="Download videos from Salsabachata.",
        epilog=(
            "Run with no arguments to download every class with videos on the most"
            " recent page of your /mis-clases listing. Use --latest, --since,"
            " --until or --all to widen or narrow that."
        ),
    )
    parser.add_argument("-e", "--email", default=DEFAULT_EMAIL, help="Login email")
    parser.add_argument(
        "-p", "--password", default=DEFAULT_PASSWORD, help="Login password"
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output folder"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent browser pages while scanning (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "-d",
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        metavar="N",
        help=f"Concurrent downloads (default: {DEFAULT_DOWNLOAD_WORKERS})",
    )

    parser.add_argument(
        "-n",
        "--latest",
        type=int,
        metavar="N",
        help="Download the N most recent classes that have videos",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download from every listing page, not just the most recent one",
    )
    parser.add_argument(
        "--since",
        type=parse_date_arg,
        metavar="YYYY-MM-DD",
        help="Only classes on or after this date",
    )
    parser.add_argument(
        "--until",
        type=parse_date_arg,
        metavar="YYYY-MM-DD",
        help="Only classes on or before this date",
    )
    parser.add_argument(
        "--index-pages",
        type=int,
        metavar="N",
        help=(
            "Read only N listing pages, working back from the most recent"
            " (default: 1 with no arguments)"
        ),
    )
    parser.add_argument(
        "-l",
        "--list-only",
        action="store_true",
        help="List the selected classes and exit without downloading",
    )
    parser.add_argument("--visible", action="store_true", help="Show browser window")

    args = parser.parse_args()

    # Zero is rejected rather than tolerated: --workers 0 would wait on a
    # semaphore that never opens, and --latest 0 would look like no selection.
    for flag, value in (
        ("--workers", args.workers),
        ("--download-workers", args.download_workers),
        ("--latest", args.latest),
        ("--index-pages", args.index_pages),
    ):
        if value is not None and value < 1:
            parser.error(f"{flag} needs a positive number, got {value}")

    if args.since and args.until and args.since > args.until:
        parser.error(f"--since {args.since} is after --until {args.until}")

    note = apply_default_scope(args)
    if note:
        print(note)

    if not args.email or not args.password:
        print("Error: Email and Password are required.")
        print("Please edit DEFAULT_EMAIL in the script or pass -e and -p arguments.")
        sys.exit(1)

    output_path = os.path.expanduser(args.output)

    if not args.list_only and not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
            print(f"Created directory: {output_path}")
        except OSError as e:
            print(f"Error creating directory: {e}")
            sys.exit(1)

    t_start = time.monotonic()

    scan = asyncio.run(scrape_phase(args))
    if scan is None:
        return

    pending, present = plan_downloads(scan, output_path)
    print_plan(pending, present)

    if not pending:
        print("Nothing to do; everything selected is already downloaded.")
        return

    failed = download_phase(
        pending, scan.cookies, scan.user_agent, args.download_workers
    )
    print(f"\nDone in {fmt_seconds(time.monotonic() - t_start)}.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run()
