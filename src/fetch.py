"""Polite, cached downloader for Pro-Football-Reference pages (NOT USED: see docs/METHODS.md).

PFR now answers scripts with a Cloudflare challenge (HTTP 403), so the data
was collected from Stathead exports instead. Kept to document the original plan.


Every page is saved under data/raw/ using the same path as the URL, e.g.
    https://www.pro-football-reference.com/years/2012/draft.htm
    -> data/raw/years/2012/draft.htm
If the file is already on disk it is returned without contacting PFR, so a
scrape can be stopped and restarted without downloading anything twice.

PFR allows roughly 20 requests per minute. We wait at least DELAY_SECONDS
between real requests (about 10 per minute) and back off on HTTP 429.
"""

import time
from pathlib import Path
from urllib.parse import urlparse

import requests

BASE_URL = "https://www.pro-football-reference.com"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DELAY_SECONDS = 6        # minimum gap between two real requests
MAX_ATTEMPTS = 5         # tries per page before giving up
RATE_LIMIT_WAIT = 120    # seconds to wait after a 429 if PFR gives no Retry-After
SERVER_ERROR_WAIT = 30   # seconds to wait after a 5xx or network error

HEADERS = {"User-Agent": "bounded-draft-value academic research scraper (UNC Charlotte)"}

_session = requests.Session()
_session.headers.update(HEADERS)
_last_request = 0.0


def cache_path(url: str) -> Path:
    """Where a URL's page is stored on disk."""
    path = urlparse(url).path.lstrip("/")
    return RAW_DIR / path


def _wait_for_turn() -> None:
    """Sleep until DELAY_SECONDS have passed since the previous request."""
    global _last_request
    remaining = DELAY_SECONDS - (time.monotonic() - _last_request)
    if remaining > 0:
        time.sleep(remaining)
    _last_request = time.monotonic()


def get(url: str) -> str | None:
    """Return the page's HTML, from the cache if possible.

    Returns None if the page does not exist (HTTP 404). Raises RuntimeError
    if the page still cannot be downloaded after MAX_ATTEMPTS tries.
    """
    if url.startswith("/"):
        url = BASE_URL + url
    path = cache_path(url)
    if path.exists():
        return path.read_text(encoding="utf-8")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        _wait_for_turn()
        try:
            resp = _session.get(url, timeout=30)
        except requests.RequestException as err:
            print(f"  network error ({err}); waiting {SERVER_ERROR_WAIT}s "
                  f"[attempt {attempt}/{MAX_ATTEMPTS}]")
            time.sleep(SERVER_ERROR_WAIT)
            continue

        if resp.status_code == 200:
            resp.encoding = "utf-8"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file first so an interrupted run never leaves
            # a half-written page in the cache.
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_text(resp.text, encoding="utf-8")
            tmp.replace(path)
            return resp.text

        if resp.status_code == 404:
            print(f"  404 not found: {url}")
            return None

        if resp.status_code == 403:
            # Cloudflare bot challenge or a block. Retrying will not help and
            # repeated hits can extend the block, so stop immediately.
            reason = "Cloudflare bot challenge" if resp.headers.get("cf-mitigated") else "forbidden"
            raise RuntimeError(f"HTTP 403 ({reason}) for {url}; not retrying")

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else RATE_LIMIT_WAIT * attempt
            print(f"  429 too many requests; waiting {wait}s [attempt {attempt}/{MAX_ATTEMPTS}]")
            time.sleep(wait)
            continue

        print(f"  HTTP {resp.status_code}; waiting {SERVER_ERROR_WAIT}s "
              f"[attempt {attempt}/{MAX_ATTEMPTS}]")
        time.sleep(SERVER_ERROR_WAIT)

    raise RuntimeError(f"Could not download {url} after {MAX_ATTEMPTS} attempts")
