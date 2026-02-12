"""Download lesson videos from salsabachata.es."""

import re
import argparse
import sys
import os
import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse
from playwright.async_api import (
    Error as PlaywrightError,
    async_playwright,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeout,
)

# --- CONFIGURATION ---
DEFAULT_EMAIL = ""
DEFAULT_PASSWORD = ""
DEFAULT_OUTPUT_DIR = "salsabachata"
DEFAULT_WORKERS = 4
# ---------------------

BASE_LESSON_URL = "https://alumnos.salsabachata.es/mis-videos/"
LOGIN_URL = "https://alumnos.salsabachata.es/"
CF_DOWNLOAD_BASE = "https://customer-1t3py9uv9w0onl4z.cloudflarestream.com"


def sanitize_filename(name: str) -> str:
    """Remove filesystem-unsafe characters from a filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def format_instructor_name(raw_name: str) -> str:
    """
    Splits by separators and takes the first 3 letters of each name.
    Ex: "Jose y Maria" -> "josmar"
    """
    clean = re.sub(r" y | & |-|/", " ", raw_name.lower())
    clean = re.sub(r"[^a-z\s]", "", clean)
    parts = clean.split()
    return "".join([part[:3] for part in parts if part])


def format_style(raw_style: str) -> str:
    """Removes non-alphabetic characters and lowercases."""
    return re.sub(r"[^a-zA-Z]", "", raw_style).lower()


def expand_lesson_ids(raw_ids: list[str]) -> list[int]:
    """
    Expands a list of IDs containing ranges into a flat list of ints.
    Ex: ['100', '105-107'] -> [100, 105, 106, 107]
    """
    expanded = []
    for item in raw_ids:
        if "-" in item:
            try:
                parts = item.split("-")
                if len(parts) != 2:
                    print(f"Warning: Invalid range format '{item}'. Skipping.")
                    continue

                start = int(parts[0])
                end = int(parts[1])

                if start > end:
                    start, end = end, start

                for i in range(start, end + 1):
                    expanded.append(i)
            except ValueError:
                print(
                    f"Warning: Could not parse range '{item}'. Ensure they are numbers. Skipping."
                )
        else:
            expanded.append(int(item))

    # Deduplicate while preserving order
    return list(dict.fromkeys(expanded))


@dataclass
class JobContext:
    """Shared context for all lesson processing tasks."""

    output_path: str
    semaphore: asyncio.Semaphore
    t_start: float = field(default_factory=time.monotonic)
    total_lessons: int = 0
    processed: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    total_bytes: int = 0
    errors: int = 0
    sum_videos: int = 0
    sum_elapsed: float = 0.0
    video_dist: Counter = field(default_factory=Counter)
    style_dist: Counter = field(default_factory=Counter)
    min_id: int | None = None
    max_id: int | None = None

    def record(
        self,
        lesson_id: int,
        videos: int,
        elapsed: float,
        error: str | None = None,
        style: str | None = None,
    ) -> None:
        """Record a stat entry and print checkpoint if needed."""
        self.processed += 1
        self.sum_videos += videos
        self.sum_elapsed += elapsed
        self.video_dist[videos] += 1
        if style:
            self.style_dist[style] += 1
        if error:
            self.errors += 1
        if videos > 0:
            if self.min_id is None or lesson_id < self.min_id:
                self.min_id = lesson_id
            if self.max_id is None or lesson_id > self.max_id:
                self.max_id = lesson_id
        if self.processed % 100 == 0:
            self.print_summary(f"{self.processed}/{self.total_lessons}")

    def _fmt_bytes(self, n: int) -> str:
        """Format byte count as human-readable string."""
        if n < 1024 * 1024:
            return f"{n / 1024:.0f} KB"
        if n < 1024 * 1024 * 1024:
            return f"{n / 1024 / 1024:.1f} MB"
        return f"{n / 1024 / 1024 / 1024:.2f} GB"

    def print_summary(self, label: str = "Progress") -> None:
        """Print a summary of processing stats."""
        if self.processed == 0:
            return
        avg = self.sum_elapsed / self.processed
        wall = time.monotonic() - self.t_start
        range_str = f", range: {self.min_id}-{self.max_id}" if self.min_id else ""
        dist_str = ", ".join(
            f"{k}:{self.video_dist[k]}" for k in sorted(self.video_dist)
        )
        style_str = ", ".join(
            f"{s}:{self.style_dist[s]}" for s in sorted(self.style_dist)
        )
        print(
            f"  [{label}] {self.processed} lessons,"
            f" {self.errors} errors,"
            f" {wall:.1f}s wall, {avg:.1f}s avg{range_str}\n"
            f"    {self.sum_videos} video(s),"
            f" {self.downloaded} downloaded"
            f" ({self._fmt_bytes(self.total_bytes)}),"
            f" {self.skipped} skipped,"
            f" {self.failed} failed,"
            f" distribution: [{dist_str}]\n"
            f"    styles: [{style_str}]"
        )


async def extract_metadata(page: Page, lesson_id: int) -> tuple[str, str]:
    """Extract instructor, style, level, and date from the lesson page. Returns (filename_prefix, style_code)."""
    try:
        raw_instructor = await page.locator(
            r"main .bg-gradient-to-br .space-y-2\.5 > div:last-child span"
        ).inner_text()
        instructor_code = format_instructor_name(raw_instructor.strip())

        raw_style = await page.locator("main h3").first.inner_text()
        style_code = format_style(raw_style.strip())

        try:
            raw_level = await page.locator("main h3 + span").first.inner_text()
            level_code = re.sub(r"[^0-9]", "", raw_level.strip())
        except (PlaywrightError, PlaywrightTimeout):
            level_code = ""

        date_text = await page.locator(
            r"main .bg-gradient-to-br .space-y-2\.5 > div:nth-child(2) span"
        ).inner_text()
        parts = date_text.strip().split()
        date_part = next(pt for pt in parts if "/" in pt)
        time_part = next(pt for pt in parts if ":" in pt)

        day, month, year = date_part.split("/")
        hour_clean = time_part.split(":")[0]
        yymmdd = f"{year[2:]}{month}{day}"

        prefix = f"{instructor_code}_{style_code}{level_code}_{yymmdd}T{hour_clean}"
        print(
            f"[{lesson_id}] {raw_instructor.strip()} -> {instructor_code} | "
            f"{raw_style.strip()} {level_code} | {yymmdd}T{hour_clean}"
        )
        return prefix, style_code

    except Exception as e:
        print(f"[{lesson_id}] Metadata extraction failed ({e}). Using fallback.")
        return "unknown_lesson", "unknown"


async def process_lesson(
    context: BrowserContext, lesson_id: int, job: JobContext
) -> None:
    """Process a single lesson: open page, extract metadata, download videos."""
    async with job.semaphore:
        target_url = f"{BASE_LESSON_URL}{lesson_id}"
        print(f"[{lesson_id}] Processing...")
        t_start = time.monotonic()

        page = await context.new_page()
        try:
            await page.goto(target_url)

            try:
                await page.wait_for_selector("iframe", timeout=8000)
            except PlaywrightTimeout:
                elapsed = time.monotonic() - t_start
                job.record(lesson_id, 0, elapsed)
                print(f"[{lesson_id}] No video(s) found. Skipping. ({elapsed:.1f}s)")
                return

            base_filename_prefix, style_code = await extract_metadata(page, lesson_id)

            style_dir = os.path.join(job.output_path, style_code)
            os.makedirs(style_dir, exist_ok=True)

            iframes = await page.locator('iframe[src*="cloudflarestream.com"]').all()
            print(f"[{lesson_id}] Found {len(iframes)} video(s).")

            for i, frame in enumerate(iframes):
                src = await frame.get_attribute("src")
                if not src:
                    continue
                video_id = urlparse(src).path.strip("/").split("/")[0]
                download_link = f"{CF_DOWNLOAD_BASE}/{video_id}/downloads/default.mp4"

                suffix = f"_{lesson_id}v{i + 1}"
                filename = sanitize_filename(
                    f"{base_filename_prefix}{suffix}.mp4".lower()
                )
                filepath = os.path.join(style_dir, filename)

                if os.path.exists(filepath):
                    print(f"[{lesson_id}]    {filename}: Already exists. Skipped.")
                    job.skipped += 1
                    continue

                nbytes = await download_video(
                    page, download_link, filepath, filename, lesson_id
                )
                if nbytes:
                    job.downloaded += 1
                    job.total_bytes += nbytes
                else:
                    job.failed += 1

            elapsed = time.monotonic() - t_start
            job.record(lesson_id, len(iframes), elapsed, style=style_code)
            print(f"[{lesson_id}] Done. {len(iframes)} video(s) in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.monotonic() - t_start
            job.record(lesson_id, 0, elapsed, str(e))
            print(f"[{lesson_id}] Critical error: {e} ({elapsed:.1f}s)")
        finally:
            await page.close()


async def download_video(
    page: Page,
    url: str,
    filepath: str,
    filename: str,
    lesson_id: int,
    max_retries: int = 3,
) -> int:
    """Download a single video file with retry logic. Returns bytes written, 0 on failure."""
    for attempt in range(max_retries):
        try:
            response = await page.request.get(url)
            if response.status == 200:
                body = await response.body()
                size = len(body)
                with open(filepath, "wb") as f:
                    f.write(body)
                print(
                    f"[{lesson_id}]    {filename}: Saved ({size / 1024 / 1024:.1f} MB)."
                )
                return size

            if response.status >= 500:
                # Server error, worth retrying
                print(
                    f"[{lesson_id}]    {filename}: HTTP {response.status}, retrying..."
                )
            else:
                # Client error (4xx), don't retry
                print(f"[{lesson_id}]    {filename}: HTTP {response.status}")
                return 0
        except Exception as e:
            print(f"[{lesson_id}]    {filename}: Error: {e}")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass

        if attempt < max_retries - 1:
            delay = 2**attempt  # 1s, 2s, 4s
            await asyncio.sleep(delay)

    print(f"[{lesson_id}]    {filename}: Failed after {max_retries} attempts.")
    return 0


async def login(page: Page, email: str, password: str) -> None:
    """Perform login and wait for dashboard redirect."""
    await page.wait_for_selector('input[name="email"]', state="visible", timeout=10000)
    await page.fill('input[name="email"]', email)
    await page.fill('input[name="password"]', password)
    await asyncio.sleep(0.5)
    await page.keyboard.press("Enter")
    await page.wait_for_url("**/dashboard", timeout=20000)


async def async_main(
    args: argparse.Namespace, target_ids: list[int], output_path: str
) -> None:
    """Log in and process all lessons concurrently."""
    async with async_playwright() as p:
        print("Launching Firefox...")
        browser = await p.firefox.launch(headless=not args.visible)
        context = await browser.new_context()
        page = await context.new_page()

        print("Navigating to login...")
        await page.goto(LOGIN_URL)

        try:
            await login(page, args.email, args.password)
            print("Login successful!")
        except (PlaywrightError, PlaywrightTimeout) as e:
            print(f"Login failed: {e}")
            await browser.close()
            sys.exit(1)

        await page.close()

        # --- PROCESS LESSONS IN PARALLEL ---
        job = JobContext(
            output_path=output_path,
            semaphore=asyncio.Semaphore(args.workers),
            total_lessons=len(target_ids),
        )
        print(
            f"Processing {len(target_ids)} lessons ({args.workers} concurrent pages)..."
        )
        print("================================================")

        tasks = [process_lesson(context, lid, job) for lid in target_ids]
        await asyncio.gather(*tasks)

        await browser.close()

    print("================================================")
    job.print_summary("Final")
    print("Done.")


def run() -> None:
    """Parse arguments and kick off the download."""
    parser = argparse.ArgumentParser(description="Download videos from Salsabachata.")
    parser.add_argument(
        "lesson_ids",
        nargs="+",
        help="List of Lesson IDs or ranges (e.g. 87933 87935-87940)",
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
        help=f"Concurrent browser pages (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument("--visible", action="store_true", help="Show browser window")

    args = parser.parse_args()

    # Expand ranges here
    target_ids = expand_lesson_ids(args.lesson_ids)
    print(f"Targeting {len(target_ids)} lessons: {target_ids}")

    output_path = os.path.expanduser(args.output)

    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
            print(f"Created directory: {output_path}")
        except OSError as e:
            print(f"Error creating directory: {e}")
            sys.exit(1)

    if not args.email or not args.password:
        print("Error: Email and Password are required.")
        print("Please edit DEFAULT_EMAIL in the script or pass -e and -p arguments.")
        sys.exit(1)

    asyncio.run(async_main(args, target_ids, output_path))


if __name__ == "__main__":
    run()
