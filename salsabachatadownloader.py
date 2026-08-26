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
# Credentials come from the environment so they are never stored in this file,
# which is tracked. -e/-p override them; a missing password is prompted for.
DEFAULT_EMAIL = os.environ.get("SALSABACHATA_EMAIL", "")
DEFAULT_PASSWORD = os.environ.get("SALSABACHATA_PASSWORD", "")
DEFAULT_OUTPUT_DIR = "salsabachata"
DEFAULT_SCAN_WORKERS = 4
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
# A lesson recording is a few hundred MB. Past this the response is broken
# or hostile, not a class video, and the write loop would fill the disk.
MAX_VIDEO_BYTES = 4 * 1024 * 1024 * 1024
# Per socket operation, not per download, so it does not cap a large file.
SOCKET_TIMEOUT = 60

# Matches both the iframe embed and any other Stream reference in the page HTML.
STREAM_RE = re.compile(
    r"(https://(?:[0-9A-Za-z-]+\.)*(?:cloudflarestream\.com|videodelivery\.net))"
    r"/([0-9a-f]{32})"
)
VIDEO_COUNT_RE = re.compile(r"V[ií]deo\s+\d+\s+de\s+(\d+)", re.IGNORECASE)
# The same label read for its own number rather than the total. Both groups are
# taken from one match so 'Vídeo 2 de 3' cannot be read as video 3.
VIDEO_LABEL_RE = re.compile(r"V[ií]deo\s+(\d+)(?:\s+de\s+(\d+))?", re.IGNORECASE)
# Digit runs are bounded: the pager text is page-controlled, and an unbounded
# \d+ either side of the literal backtracks quadratically on a long run.
PAGER_RE = re.compile(r"(\d{1,6})\s+de\s+(\d{1,6})")

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
        if (/\\d{{1,6}}\\s+de\\s+\\d{{1,6}}/.test(text)) {{
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
        // Innermost cards only. A wrapper card around the per-video ones would
        // otherwise also match and repeat the first video it contains.
        if (card.querySelector('.d-card')) continue;
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
    // Every iframe src on the page, for the fallback when no card matched.
    // Taken from the DOM rather than the serialised HTML so it carries no
    // entity escaping, and scoped to iframes so a Stream URL mentioned in a
    // script or a link cannot pass for a video.
    const frames = [];
    for (const frame of root.querySelectorAll('iframe[src]')) {
        frames.push(frame.getAttribute('src'));
    }
    return {cards: out, frames: frames};
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
    First 3 letters of each name: "Valentín y Angy" -> 'valang'.

    Accents fold to their base letter before the letters are taken, which is
    load-bearing when the accent falls inside the first three: "Cándido" gives
    'can', and 'cnd' without the fold, since stripping would drop the 'á'.

    Capped at 32 because the name is page-controlled and a filename past 255
    bytes fails the write; real values are under 10, so the cap never bites.
    """
    clean = re.sub(r" y | & |-|/", " ", strip_accents(raw_name).lower())
    clean = re.sub(r"[^a-z\s]", "", clean)
    parts = clean.split()
    return "".join([part[:3] for part in parts if part])[:32]


def format_style(raw_style: str) -> str:
    """
    Style name without the level digits: "Salsa en Línea 1" -> 'salsaenlinea'.

    Capped at 32 for the same reason as the instructor name.
    """
    return re.sub(r"[^a-zA-Z]", "", strip_accents(raw_style)).lower()[:32]


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
        """
        Trailing level digits from the title: '2' for 'Bachata 2'.

        Only the tail is searched. The title is page-controlled and this pattern
        backtracks quadratically over a long digit run that ends in anything but
        a digit or space, which is what defeats the anchor (7.4s on a 32k title,
        0.0001s sliced).
        """
        match = re.search(r"(\d+)\s*$", self.title[-32:])
        return match.group(1) if match else ""

    @property
    def hour(self) -> str:
        """
        Hour of the class time, unpadded as printed: '21'.

        'xx' when there is no readable time, which reaches the filename.
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

        Widths clear the longest values seen: 'Técnica de Bachata' at 18,
        'Julio Marquetti y Moni' at 22 and 'Quevedo - Sala 1' at 16. The
        instructor column has only one character of slack, so widen it before
        adding a longer credit. Venue is padded too, or the counts printed after
        it would not line up; callers rstrip the result.
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


def parse_pager(text: str | None) -> tuple[int | None, int | None]:
    """
    Read current and total page from the listing's '1 de 10' indicator.

    The current page is what tells us the site honoured '?page=N'. Both are
    optional: a missing pager is handled, a misread one is not guessed at.
    """
    if not text:
        return None, None
    match = PAGER_RE.search(text)
    if not match:
        return None, None
    current = int(match.group(1))
    total = int(match.group(2))
    return (current or None), (total or None)


def listing_url(number: int) -> str:
    """
    URL of a listing page.

    Page 1 is the bare path, which is the form the site links to. '?page=1'
    serves the same page.
    """
    return CLASSES_URL if number <= 1 else f"{CLASSES_URL}?page={number}"


class ListingError(RuntimeError):
    """
    The listing could not be read as expected.

    Raised for the cases that would otherwise pass for an empty result: a page
    that will not load, pagination ignored, or rows whose video links all
    stopped matching. Each ends the run non-zero.
    """


@dataclass
class IndexPage:
    """One page of the /mis-clases listing."""

    number: int
    total: int | None
    lessons: list[Lesson]
    # Rows on the page, including classes whose videos are gone. Compared
    # against the lessons to tell an empty page from an unreadable one.
    rows: int


async def iter_index(page: Page, max_pages: int | None = None):
    """
    Yield pages of the /mis-clases listing, walking back from the most recent.

    Page 1 holds the newest classes and its '1 de 10' pager gives the total.
    Yielded one page at a time so a caller with enough classes can stop early.
    """
    total: int | None = None
    number = 1

    while max_pages is None or number <= max_pages:
        try:
            await page.goto(listing_url(number), wait_until="domcontentloaded")
            result = await page.evaluate(INDEX_JS)
        except (PlaywrightError, PlaywrightTimeout) as e:
            # Raised rather than swallowed: stopping here silently would look
            # exactly like reaching the end of the listing.
            raise ListingError(f"could not read listing page {number}: {e}") from e

        current, page_total = parse_pager(result.get("pager"))
        if total is None:
            total = page_total

        # The pager is the only confirmation that '?page=N' was honoured. If the
        # site re-serves page 1 instead, the duplicate rows dedupe away and it
        # looks like a short listing rather than a broken crawl.
        if current is not None and current != number:
            raise ListingError(
                f"asked for listing page {number} but the site served page"
                f" {current}; pagination is not working as expected"
            )

        # Rows include classes whose videos are gone, so an empty row set means
        # the page itself is empty: either past the end or an empty listing.
        rows = result.get("rows", [])
        if not rows:
            if number == 1:
                raise ListingError(
                    "the listing has no classes at all; the login may not have"
                    " taken effect, or the row markup has changed"
                )
            break

        lessons = []
        for row in rows:
            lesson = lesson_from_row(row)
            if lesson:
                lessons.append(lesson)

        # The row count travels with the lessons so the caller can tell "these
        # classes have no videos left" from "the link markup changed".
        yield IndexPage(number=number, total=total, lessons=lessons, rows=len(rows))

        number += 1
        if total is not None and number > total:
            break
        # Applied even when a total was found: the total comes from the page, so
        # a wrong one would otherwise crawl without a bound.
        if number > MAX_INDEX_PAGES:
            print(f"Warning: stopping after {MAX_INDEX_PAGES} listing pages.")
            break


@dataclass
class VideoSource:
    """One video on a lesson page, with its download URLs in priority order."""

    urls: list[str]
    label: str | None = None
    has_button: bool = False
    # The number the page gives this video, from its 'Vídeo 2 de 3' label. This
    # is its identity: see video_numbers for why position will not do.
    number: int | None = None
    # Set when the card markup stopped matching and the video came from a bare
    # iframe instead, which means no label and so no reliable number.
    from_fallback: bool = False

    @classmethod
    def numbered(cls, label: str | None, **kw) -> VideoSource:
        """Build a source, reading its number out of the label."""
        match = VIDEO_LABEL_RE.search(label) if label else None
        return cls(label=label, number=int(match.group(1)) if match else None, **kw)


def video_numbers(sources: list[VideoSource]) -> list[int]:
    """
    Choose the v-number for each source, which becomes part of its filename.

    The page's own numbering is the stable identity; position is not. If one
    card stops yielding a URL, every later video shifts down one, renaming files
    already on disk and leaving the last one looking fetched when it is not.

    Falls back to position when labels are missing or repeat, since two videos
    sharing a number is worse than a shifted one.
    """
    numbers = [source.number for source in sources]
    if all(n is not None for n in numbers) and len(set(numbers)) == len(numbers):
        return [n for n in numbers if n is not None]
    return list(range(1, len(sources) + 1))


def stream_download_url(src: str) -> str | None:
    """Turn a Stream iframe src into its direct download URL."""
    match = STREAM_RE.search(src)
    if not match:
        return None
    base, uid = match.group(1), match.group(2)
    return f"{base}/{uid}/downloads/default.mp4"


SCHOOL_HOST = "alumnos.salsabachata.es"


def safe_download_url(url: str) -> str | None:
    """
    Return the URL if it is https on a host we expect, else None.

    The download URL is read out of an x-data attribute, so its scheme and
    host are page-controlled, and urllib's opener will happily fetch file:
    and ftp: targets.
    """
    parts = urlparse(url)
    if parts.scheme != "https":
        return None
    host = (parts.hostname or "").lower()
    if host in (SCHOOL_HOST, "videodelivery.net", "cloudflarestream.com"):
        return url
    # Any subdomain of the two Stream domains, matching STREAM_RE. Checked as a
    # dotted suffix so a lookalike like 'cloudflarestream.com.evil.net' fails.
    return (
        url if host.endswith((".cloudflarestream.com", ".videodelivery.net")) else None
    )


def dedupe_sources(sources: list[VideoSource]) -> list[VideoSource]:
    """
    Drop sources that would fetch the same URL twice.

    Keyed on the first URL, the one actually requested. Two cards for one video
    would otherwise be saved under two names.
    """
    seen: set[str] = set()
    unique = []
    for source in sources:
        key = source.urls[0]
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


async def find_videos(page: Page) -> tuple[list[VideoSource], int | None]:
    """
    Collect download candidates from a lesson page.

    Each card carries a Stream iframe and, where saving is allowed, a "Guardar"
    button holding a /descargar-video URL. The first-party URL goes first: it is
    the path the site uses, and Stream's direct download is not always enabled.

    Also returns the count the page claims, so the caller can notice finding
    fewer videos than advertised.
    """
    js_failed = False
    try:
        result = await page.evaluate(VIDEOS_JS)
    except PlaywrightError as e:
        print(f"Warning: could not read the video cards ({e}); using raw HTML.")
        result = {}
        js_failed = True

    cards = result.get("cards", [])
    frames = result.get("frames", [])

    sources: list[VideoSource] = []
    for card in cards:
        urls: list[str] = []
        if card.get("downloadUrl"):
            resolved = urljoin(BASE_URL, card["downloadUrl"])
            safe = safe_download_url(resolved)
            if safe:
                urls.append(safe)
            else:
                print(
                    f"Warning: ignoring an off-site download URL from the page"
                    f" ({card['downloadUrl']!r})."
                )

        if card.get("iframeSrc"):
            stream_url = stream_download_url(card["iframeSrc"])
            if stream_url:
                urls.append(stream_url)

        if urls:
            sources.append(
                VideoSource.numbered(
                    card.get("label"),
                    urls=urls,
                    has_button=bool(card.get("downloadUrl")),
                )
            )

    # Two cards pointing at the same video would otherwise be downloaded twice,
    # to two names. Keyed on the preferred URL, which is what actually gets
    # fetched.
    sources = dedupe_sources(sources)

    # Read from the card labels rather than the whole page, so the count and the
    # per-video numbers come from the same place, and a 'Vídeo 1 de 2' appearing
    # anywhere outside a video card cannot change it.
    totals = [
        int(match.group(2))
        for card in cards
        if (match := VIDEO_LABEL_RE.search(card.get("label") or "")) and match.group(2)
    ]
    expected = max(totals) if totals else None

    # Fallback for when no card yields a URL: take the Stream iframes directly.
    # Loses the pairing with the download buttons, and with it the labels, so
    # these carry no number and fall back to position.
    if not sources:
        for src in frames:
            stream_url = stream_download_url(src or "")
            if stream_url:
                sources.append(VideoSource(urls=[stream_url], from_fallback=True))
        sources = dedupe_sources(sources)
        if sources:
            print(
                f"Warning: read {plural(len(sources), 'video')} from the iframes"
                " directly; the card markup has moved."
            )

    # Last resort, only when the DOM could not be queried at all: scrape the
    # serialised HTML. Unscoped, so it is not used while the DOM is readable.
    if not sources and js_failed:
        html = await page.content()
        for base, uid in STREAM_RE.findall(html):
            url = f"{base}/{uid}/downloads/default.mp4"
            sources.append(VideoSource(urls=[url], from_fallback=True))
        sources = dedupe_sources(sources)
        if expected is None:
            counts = VIDEO_COUNT_RE.findall(html)
            expected = max(int(c) for c in counts) if counts else None

    return sources, expected


def fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    return f"{n / 1024 / 1024 / 1024:.2f} GB"


def redact(text: str, *secrets: str) -> str:
    """
    Blank out secrets before printing text that came from somewhere else.

    Precautionary rather than fixing a known leak: Playwright quotes some failed
    calls' arguments in the error message, but on 1.62 a page.fill timeout, the
    one call here handed the password, reports only its selector. This guards
    against that changing, or a future call passing the password somewhere that
    is echoed, since the message is printed verbatim on the login failure path.

    A short secret peppers the message with '***', which is the harmless
    direction to fail in. Only the password is passed, never the email, which
    shares substrings with the URLs alongside it.
    """
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


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

    Irregulars are passed in, since 'class' would derive as 'classs'.
    """
    if n == 1:
        return f"{n} {noun}"
    return f"{n} {many or noun + 's'}"


def source_label(url: str) -> str:
    """
    Name the host a URL points at, to report which one served a video.

    Unknown hosts report as themselves rather than as one of the known two.
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
    """
    One video to fetch.

    Carries only what the download needs. The lesson it came from is not kept:
    the filename already names it, and that is what failures are reported by.
    """

    urls: list[str]
    filepath: str
    filename: str
    style_code: str

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

        numbers = video_numbers(item.sources)

        for index, source in zip(numbers, item.sources, strict=True):
            filename = sanitize_filename(f"{prefix}_{lesson_id}v{index}.mp4".lower())
            job = DownloadJob(
                urls=source.urls,
                filepath=os.path.join(output_path, style_code, filename),
                filename=filename,
                style_code=style_code,
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
        page: Page | None = None
        sources: list[VideoSource] = []
        expected: int | None = None
        error: str | None = None

        try:
            # Inside the try: opening a page is itself a failure worth
            # surviving, since it is where a browser short of memory gives out,
            # and there are scan_workers of these in flight.
            page = await context.new_page()
            await page.goto(f"{LESSON_URL}{lesson_id}", wait_until="domcontentloaded")
            sources, expected = await find_videos(page)
        # Deliberately broad: these run concurrently under one gather, and one
        # unreadable lesson page must not cancel the rest of the scan.
        except Exception as e:  # noqa: BLE001
            error = str(e) or e.__class__.__name__
        finally:
            if page is not None:
                try:
                    await page.close()
                except PlaywrightError:
                    # Already gone, which is the state close was after.
                    pass

        elapsed = time.monotonic() - t_start

        # Exactly once, and before anything that could raise. Recording inside
        # the try meant a failing print counted the lesson a second time and as
        # an error, which reaches the exit status.
        scan.record(lesson, sources, elapsed, error)

        if error is not None:
            print(
                f"[{lesson_id}]  {lesson.when}  {lesson.columns()}"
                f"  {'ERROR':<9}  {fmt_seconds(elapsed):<6}  {error}".rstrip()
            )
            return

        if not sources:
            print(
                f"[{lesson_id}]  {lesson.when}  {lesson.columns()}"
                f"  {'no videos':<9}  {fmt_seconds(elapsed):<6}".rstrip()
            )
            return

        if expected and expected != len(sources):
            print(
                f"[{lesson_id}] Warning: page says {plural(expected, 'video')},"
                f" found {len(sources)}."
            )

        no_button = [
            s.label or f"video {n}"
            for n, s in zip(video_numbers(sources), sources, strict=True)
            if not s.has_button
        ]
        note = f"  no download button: {', '.join(no_button)}" if no_button else ""
        print(
            f"[{lesson_id}]  {lesson.when}  {lesson.columns()}"
            f"  {plural(len(sources), 'video'):<9}"
            f"  {fmt_seconds(elapsed):<6}{note}".rstrip()
        )


class DropCookieOnRedirect(urllib.request.HTTPRedirectHandler):
    """
    Refuse a non-https redirect, and drop the cookie when the target changes.

    Redirect targets are server-controlled: a hop to http: would carry the
    session cookie in cleartext and urllib would follow an ftp: one, so anything
    but https is refused. Dropping the cookie on any host or scheme change
    leaves Cloudflare's cross-host hop working but unauthenticated.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlparse(newurl)
        if target.scheme != "https":
            return None
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        current = urlparse(req.full_url)
        if new is not None and (
            target.hostname != current.hostname or target.scheme != current.scheme
        ):
            new.headers.pop("Cookie", None)
            new.unredirected_hdrs.pop("Cookie", None)
        return new


OPENER = urllib.request.build_opener(DropCookieOnRedirect())


def cookie_header(cookies: list[dict], url: str) -> str:
    """
    Build a Cookie header for one URL out of the browser's cookies.

    Scoped by domain, so the school's session cookie never reaches Cloudflare,
    and by the secure flag, so a Secure cookie never goes out in cleartext.

    A leading dot matches the domain and its subdomains; a dotless one is
    host-only, so '.salsabachata.es' and 'alumnos.salsabachata.es' differ.
    """
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    is_https = parts.scheme == "https"
    pairs = []
    for cookie in cookies:
        if cookie.get("secure") and not is_https:
            continue
        domain = (cookie.get("domain") or "").lower()
        if not domain:
            continue
        if domain.startswith("."):
            bare = domain[1:]
            matches = host == bare or host.endswith(domain)
        else:
            matches = host == domain
        if matches:
            pairs.append(f"{cookie['name']}={cookie['value']}")
    return "; ".join(pairs)


def video_rejection(head: bytes, content_type: str) -> str | None:
    """
    Say why a response is not an MP4, or None if it looks like one.

    A lapsed session or a rate limit answers with HTML and a 200 status. Saved
    as a .mp4 that becomes a file every later run skips as already downloaded,
    so the start of the body is checked and not just the content type.
    """
    kind = content_type.split(";")[0].strip().lower()
    accepted = (
        "video/",
        "application/mp4",
        "application/octet-stream",
        "binary/octet-stream",
    )
    if kind and not kind.startswith(accepted):
        return f"served {kind}"
    # MP4 and friends put an 'ftyp' box at the start of the file.
    if b"ftyp" not in head[:64]:
        return "no MP4 header"
    return None


def download_to_file(
    url: str, headers: dict[str, str], filepath: str
) -> tuple[int, str | None, bool]:
    """
    Stream one URL to disk. Returns (bytes written, error, retryable), with
    error None on success.

    Bytes go to a '.part' file that is renamed only once the transfer finishes,
    so an interrupted run cannot leave a truncated file that the next run
    mistakes for a complete download.

    'retryable' is set for a server or transport fault that another attempt
    could get past, and cleared for a response that was understood and refused.
    """
    part = f"{filepath}.part"
    request = urllib.request.Request(url, headers=headers)
    try:
        with OPENER.open(request, timeout=SOCKET_TIMEOUT) as response:
            head = response.read(CHUNK_BYTES)
            reason = video_rejection(head, response.headers.get("content-type", ""))
            if reason:
                return 0, reason, False

            declared = response.headers.get("content-length", "")
            expected_bytes = int(declared) if declared.isdigit() else None

            if expected_bytes is not None and expected_bytes > MAX_VIDEO_BYTES:
                return 0, f"declares {fmt_bytes(expected_bytes)}", False

            # Only once the response is known to be a video, so a failed run
            # leaves no empty style directory behind.
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            written = 0
            with open(part, "wb") as f:
                f.write(head)
                written = len(head)
                while chunk := response.read(CHUNK_BYTES):
                    written += len(chunk)
                    if written > MAX_VIDEO_BYTES:
                        return 0, f"over {fmt_bytes(MAX_VIDEO_BYTES)}", False
                    f.write(chunk)

        # http.client does not raise when a body ends early with a clean FIN:
        # the read loop just stops. Comparing against the declared length is
        # the only truncation check there is.
        if expected_bytes is not None and written != expected_bytes:
            return 0, f"truncated: {written} of {expected_bytes} bytes", True

        if written < MIN_VIDEO_BYTES:
            return 0, f"only {written} bytes", False

        os.replace(part, filepath)
        return written, None, False
    except urllib.error.HTTPError as e:
        return 0, f"HTTP {e.code}", e.code >= 500 or e.code == 429
    # Deliberately broad: every transport failure becomes a retryable result
    # rather than an exception, so one video cannot end the download phase.
    except Exception as e:  # noqa: BLE001
        return 0, str(e) or e.__class__.__name__, True
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

    Each URL gets up to three attempts with a backoff, so a single transient
    502, 429 or reset does not lose a video for the whole run. A refusal the
    server meant is not retried. Only the last problem per URL is recorded, to
    keep the end-of-run failure list readable.

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
        for attempt in range(3):
            size, error, retryable = download_to_file(url, headers, job.filepath)
            if size or not retryable:
                break
            time.sleep(2**attempt)

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
    attempted = 0
    interrupted = False
    failures: list[DownloadResult] = []
    saved: list[DownloadJob] = []
    by_source: Counter = Counter()

    # Built without 'with' so the shutdown can cancel what has not started: the
    # context manager waits without cancelling, which makes Ctrl-C look inert
    # while the workers drain the whole queue.
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {
            pool.submit(download_job, job, cookies, user_agent): job for job in jobs
        }
        for n, future in enumerate(as_completed(futures), start=1):
            # A raising worker must not take the summary and the tallies of
            # every job that already succeeded down with it.
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001
                result = DownloadResult(
                    job=futures[future], problems=[f"worker: {e!r}"]
                )
            attempted += 1
            # Padded inside the brackets so they stay a fixed width and the
            # column after them does not move.
            counter = f"[{n:>{width}}/{total}]"

            if result.size:
                written += result.size
                saved.append(result.job)
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
    except KeyboardInterrupt:
        interrupted = True
        print(
            "\nInterrupted: downloads that had not started were abandoned."
            " Transfers already in flight cannot be cut short, so this waits"
            " for them."
        )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    elapsed = max(time.monotonic() - t_start, 0.001)
    sources = ", ".join(f"{s}:{by_source[s]}" for s in sorted(by_source))
    # After an interrupt most jobs never ran, so the count is out of what was
    # attempted rather than out of the queue.
    scope = f"{attempted} attempted" if interrupted else f"{total}"
    print(
        f"  [Downloaded] {attempted - len(failures)} of {scope},"
        f" {fmt_bytes(written)} in {fmt_seconds(elapsed)},"
        f" {written / elapsed / 1024 / 1024:.1f} MB/s aggregate"
    )
    # Styles of what actually landed, so a partly failed run does not report the
    # plan's breakdown as though all of it arrived. Same helper as the plan, so
    # the two lines can be read against each other.
    # One guard for both: a job that lands is added to saved and counted in
    # by_source together, so the two are never populated independently.
    if saved:
        print(f"    styles       [{style_counts(saved)}]")
        print(f"    sources      [{sources}]")

    # Repeated at the end so a failure is not lost in the scroll above.
    if failures:
        print(f"    {len(failures)} failed:")
        for result in failures:
            print(f"      {result.job.where}: {'; '.join(result.problems)}")

    # 130 is the conventional SIGINT status, and raising here skips the caller's
    # exit-1 path, which would report an interrupt as a download failure.
    if interrupted:
        raise SystemExit(130)

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
        # Anywhere else is allowed through. A session that is not really logged
        # in gets an empty listing, and that is where it is caught, on evidence
        # rather than on the nav markup happening to hold a logout link.
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
    rows_read = 0

    satisfied = False
    last: IndexPage | None = None

    async for index in iter_index(page, args.index_pages):
        last = index
        number, total, lessons = index.number, index.total, index.lessons
        rows_read += index.rows
        where = f"page {number} of {total}" if total else f"page {number}"
        print(
            f"Indexed {where}: {plural(len(lessons), 'class', 'classes')}"
            f" with videos, out of {index.rows} listed."
        )

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
            satisfied = True
            break
        if args.since and lessons:
            oldest = lessons[-1].date
            if oldest and oldest < args.since:
                satisfied = True
                break

    # The page cap wins over a date range, so --since with --index-pages can
    # stop before reaching the oldest class asked for. Worth saying, since the
    # result looks like a complete answer to the filter.
    if (
        not satisfied
        and last is not None
        and args.index_pages is not None
        and last.number >= args.index_pages
        and (last.total is None or last.total > args.index_pages)
    ):
        remaining = f" of {last.total}" if last.total else ""
        print(
            f"Note: stopped at the --index-pages limit of {args.index_pages}"
            f"{remaining}. Older classes were not looked at."
        )

    # Rows but no lessons means every video link was missed, which is what a
    # changed link pattern looks like. Left alone it reports '0 classes' and
    # exits successfully, indistinguishable from having nothing to download.
    if rows_read and not candidates:
        raise ListingError(
            f"read {rows_read} listing row(s) but found no video links in any of"
            " them; the link pattern has probably changed"
        )

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

    try:
        # Inside the try: reaching the login page can fail the same way filling
        # it in can, and an unhandled traceback here says nothing useful.
        await page.goto(LOGIN_URL)
        await login(page, args.email, args.password)
        print("Logged in.")
    except (PlaywrightError, PlaywrightTimeout) as e:
        print(f"Login failed: {redact(str(e), args.password)}")
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
        f"\nScanning {plural(len(targets), 'lesson')},"
        f" {args.scan_workers} pages at a time"
    )

    t_start = time.monotonic()
    semaphore = asyncio.Semaphore(args.scan_workers)
    # return_exceptions so one task raising outside its own handler, which now
    # means a failure while printing, does not cancel every other page mid-load
    # and lose the whole scan.
    outcomes = await asyncio.gather(
        *(scrape_lesson(context, lesson, scan, semaphore) for lesson in targets),
        return_exceptions=True,
    )
    scan.elapsed = time.monotonic() - t_start

    for lesson, outcome in zip(targets, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            print(f"[{lesson.lesson_id}] Scan task failed: {outcome!r}")
            scan.errors += 1

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
        except ListingError as e:
            print(f"Error: {e}.")
            sys.exit(1)
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
        # Naive on purpose: this is compared against listing dates, which carry
        # no zone either, and only the date part survives.
        return datetime.strptime(value, "%Y-%m-%d").date()  # noqa: DTZ007
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got '{value}'") from e


def build_parser() -> argparse.ArgumentParser:
    """Define the command line."""
    parser = argparse.ArgumentParser(
        description="Download videos from Salsabachata.",
        epilog=(
            "Run with no arguments to download every class with videos on the most"
            " recent page of your /mis-clases listing. Use --latest, --since,"
            " --until or --all to widen or narrow that."
        ),
    )
    parser.add_argument(
        "-e",
        "--email",
        default=DEFAULT_EMAIL,
        help="Login email (default: $SALSABACHATA_EMAIL)",
    )
    parser.add_argument(
        "-p",
        "--password",
        default=DEFAULT_PASSWORD,
        help="Login password (default: $SALSABACHATA_PASSWORD)",
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output folder"
    )
    parser.add_argument(
        "-s",
        "--scan-workers",
        type=int,
        metavar="N",
        default=DEFAULT_SCAN_WORKERS,
        help=(
            "Lesson pages scanned at once, bound by browser memory"
            f" (default: {DEFAULT_SCAN_WORKERS})"
        ),
    )
    parser.add_argument(
        "-d",
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        metavar="N",
        help=(
            "Videos downloaded at once, bound by the network"
            f" (default: {DEFAULT_DOWNLOAD_WORKERS})"
        ),
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
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, str | None]:
    """
    Parse and check the command line, filling in the default selection.

    Separate from run so the whole command line can be exercised without
    starting a browser. Returns the note about the default scope rather than
    printing it, so exercising it stays silent.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Zero is rejected rather than tolerated: --scan-workers 0 would wait on a
    # semaphore that never opens, and --latest 0 would look like no selection.
    for flag, value in (
        ("--scan-workers", args.scan_workers),
        ("--download-workers", args.download_workers),
        ("--latest", args.latest),
        ("--index-pages", args.index_pages),
    ):
        if value is not None and value < 1:
            parser.error(f"{flag} needs a positive number, got {value}")

    if args.since and args.until and args.since > args.until:
        parser.error(f"--since {args.since} is after --until {args.until}")

    missing = [
        name
        for name, value in (
            ("SALSABACHATA_EMAIL (-e)", args.email),
            ("SALSABACHATA_PASSWORD (-p)", args.password),
        )
        if not value
    ]
    if missing:
        parser.error(f"no credentials: set {' and '.join(missing)}")

    return args, apply_default_scope(args)


def output_problem(path: str) -> str | None:
    """
    Say why videos could not be written under path, or None if they could.

    Checked against the nearest existing ancestor, since the directory itself is
    only created later, once there is something to put in it.
    """
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not os.path.isdir(probe):
        return f"{probe} is not a directory"
    if not os.access(probe, os.W_OK):
        return f"{probe} is not writable"
    return None


def run() -> None:
    """Scan, plan, and download."""
    args, note = parse_args()
    if note:
        print(note)

    output_path = os.path.expanduser(args.output)

    # Before the browser starts. Creating the directory waits until there is
    # something to write, but finding out it is unusable should not: a scan can
    # take minutes, and failing afterwards wastes all of it.
    if not args.list_only:
        problem = output_problem(output_path)
        if problem:
            print(f"Error: cannot write to --output {args.output}: {problem}.")
            sys.exit(1)

    t_start = time.monotonic()

    scan = asyncio.run(scrape_phase(args))
    if scan is None:
        return

    # A lesson page that failed to scan is a video that will not be downloaded,
    # so it counts towards the exit status even if every download succeeds.
    failed = scan.errors

    pending, present = plan_downloads(scan, output_path)
    print_plan(pending, present)

    if pending:
        # Created here, once there is something to put in it, so a run that
        # finds nothing missing leaves no empty directory behind. isdir, not
        # exists: an output path that is a file would pass an exists check and
        # then fail once per video, deep inside the retrying download loop.
        if not os.path.isdir(output_path):
            try:
                os.makedirs(output_path)
                print(f"Created directory: {output_path}")
            except OSError as e:
                print(f"Error creating directory: {e}")
                sys.exit(1)
        failed += download_phase(
            pending, scan.cookies, scan.user_agent, args.download_workers
        )
    elif present:
        print("Nothing to do; everything selected is already downloaded.")
    else:
        # Distinguished, because 'already downloaded' next to a plan of 0 and 0
        # says the opposite of what happened.
        print("Nothing to do; the selected classes had no videos.")

    print(f"\nDone in {fmt_seconds(time.monotonic() - t_start)}.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        # Ctrl-C during the browser phase would otherwise print an asyncio
        # traceback. 130 is the shell's convention for it.
        print("\nInterrupted.")
        sys.exit(130)
