"""
host_cache_builder.py
Crawls the FastAPI documentation sitemap, extracts Python code blocks
from technical pages, and saves the result to static_docs_cache.json
in the same directory as this script.

Usage:
    python server/host_cache_builder.py
"""

import json
import os
import re
import sys

import httpx
from bs4 import BeautifulSoup

SITEMAP_URL = "https://fastapi.tiangolo.com/sitemap.xml"
BASE_URL = "https://fastapi.tiangolo.com"
ALLOWED_PREFIXES = ("/tutorial/", "/advanced/", "/reference/", "/how-to/")
MIN_SNIPPET_LEN = 50
MAX_SNIPPETS_PER_PAGE = 5
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "static_docs_cache.json")


def fetch_sitemap_urls() -> list[str]:
    """Fetch all page URLs from the FastAPI sitemap that match allowed prefixes."""
    print(f"Fetching sitemap from {SITEMAP_URL} ...")
    response = httpx.get(SITEMAP_URL, timeout=30, follow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml-xml")
    urls = []
    for loc in soup.find_all("loc"):
        href = loc.get_text(strip=True)
        path = href.replace(BASE_URL, "")
        if any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            urls.append(href)

    print(f"  Found {len(urls)} technical pages.")
    return urls


def extract_python_snippets(html: str) -> list[str]:
    """Extract Python code blocks longer than MIN_SNIPPET_LEN characters."""
    soup = BeautifulSoup(html, "lxml")
    snippets = []

    for code_tag in soup.find_all("code", class_=re.compile(r"language-python")):
        text = code_tag.get_text()
        if len(text.strip()) >= MIN_SNIPPET_LEN:
            snippets.append(text.strip())

    # Also try <pre><code> blocks without an explicit language class
    if not snippets:
        for pre in soup.find_all("pre"):
            code = pre.find("code")
            if code:
                text = code.get_text()
                if len(text.strip()) >= MIN_SNIPPET_LEN and (
                    "def " in text or "import " in text or "async " in text
                ):
                    snippets.append(text.strip())

    return snippets[:MAX_SNIPPETS_PER_PAGE]


def url_to_key(url: str) -> str:
    """Convert a full URL to a short cache key, e.g. tutorial/path-params."""
    path = url.replace(BASE_URL, "").strip("/")
    return path


def build_cache(urls: list[str]) -> dict[str, str]:
    cache: dict[str, str] = {}
    total = len(urls)

    with httpx.Client(timeout=20, follow_redirects=True) as client:
        for i, url in enumerate(urls, 1):
            key = url_to_key(url)
            print(f"  [{i}/{total}] {key} ...", end=" ", flush=True)
            try:
                resp = client.get(url)
                resp.raise_for_status()
                snippets = extract_python_snippets(resp.text)
                if snippets:
                    formatted_snippets = [f"```python\n{s}\n```" for s in snippets]
                    cache[key] = "\n\n---\n\n".join(formatted_snippets)
                    print(f"{len(snippets)} snippet(s)")
            except Exception as exc:
                print(f"ERROR: {exc}")

    return cache


def main():
    urls = fetch_sitemap_urls()
    if not urls:
        print("No URLs found. Exiting.")
        sys.exit(1)

    print(f"\nBuilding cache for {len(urls)} pages ...")
    cache = build_cache(urls)

    print(f"\nSaving {len(cache)} entries to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"Done. Cache saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
