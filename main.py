import requests
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
import numpy as np

START_URL = "https://en.wikipedia.org/wiki/GitHub"
END_URL = "https://en.wikipedia.org/wiki/Warsaw"

MAX_LINKS_SCAN = 140
TOP_N = 8
BATCH_SIZE = 64
PREFETCH_N = 8

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0"
executor = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 8))

cache = {}
pending = {}
embedding_cache = {}

SKIP = (
    "File:", "Wikipedia:", "Help:", "Special:", "Talk:",
    "Template:", "Category:", "Portal:", "Main_Page"
)

LINK_PATTERN = re.compile(
    r'<a[^>]*href="/wiki/([^"#:]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE
)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean(text):
    return re.sub(r"[_%27]+", " ", text)


def get_page(url):
    return url[url.find("/wiki/") + 6:]


def fetch(page):
    url = "https://en.wikipedia.org/wiki/" + page
    try:
        r = session.get(url, timeout=8)
        if r.ok:
            return page, r.text
    except:
        pass
    return page, ""


def get_html(page):
    done_pages = []
    for p, future in pending.items():
        if future.done():
            done_pages.append(p)

    for p in done_pages:
        future = pending.pop(p)
        try:
            result = future.result()
            cache[p] = result[1]
        except:
            cache[p] = ""

    if page in cache:
        return cache[page]

    if page in pending:
        future = pending.pop(page)
        try:
            result = future.result(timeout=15)
            cache[page] = result[1]
            return result[1]
        except:
            return ""

    result = fetch(page)
    cache[page] = result[1]
    return result[1]


def prefetch(pages):
    for p in pages:
        if p not in cache and p not in pending:
            pending[p] = executor.submit(fetch, p)


def strip_tags(text):
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def extract_links_fast(html):
    links = {}
    count = 0

    for match in LINK_PATTERN.finditer(html):
        link = match.group(1)
        raw_text = match.group(2)

        if not link or link.startswith(SKIP):
            continue
        if link in links:
            continue

        anchor = strip_tags(raw_text)
        title = clean(link)

        # shorter context than before: just title + anchor
        if anchor and anchor.lower() != title.lower():
            context = f"{title} | {anchor}"
        else:
            context = title

        links[link] = context[:100]

        count += 1
        if count >= MAX_LINKS_SCAN:
            break

    return links


def find_closest(links, target_embed, top_n=TOP_N):
    keys = list(links.keys())
    contexts = list(links.values())

    uncached = [c for c in contexts if c not in embedding_cache]
    if uncached:
        embeds = model.encode(
            uncached,
            show_progress_bar=False,
            batch_size=BATCH_SIZE
        )
        for c, e in zip(uncached, embeds):
            embedding_cache[c] = e

    all_embeds = np.vstack([embedding_cache[c] for c in contexts])
    scores = util.cos_sim(all_embeds, target_embed).squeeze()

    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()

    if np.ndim(scores) == 0:
        return keys[0], float(scores), keys[:top_n]

    k = min(top_n, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    best = top_idx[0]
    top_links = [keys[i] for i in top_idx]

    return keys[best], float(scores[best]), top_links


if __name__ == "__main__":
    start = get_page(START_URL)
    end = get_page(END_URL)

    current = start
    path = [start]
    visited = {start}

    target_embed = model.encode(clean(end))

    print(f"Starting: {clean(start)}")
    print(f"Target: {clean(end)}\n")

    start_time = time.time()

    while current != end:
        step_time = time.time()

        html = get_html(current)
        all_links = extract_links_fast(html)

        links = {}
        for k, v in all_links.items():
            if k not in visited:
                links[k] = v

        if not links:
            print("dead end")
            break

        if end in links:
            path.append(end)
            elapsed = time.time() - step_time
            print(f"-> {clean(end)} (FOUND) [{elapsed:.2f}s]")
            break

        closest, score, top = find_closest(links, target_embed, top_n=TOP_N)

        prefetch([c for c in top[:PREFETCH_N] if c not in visited])

        visited.add(closest)
        path.append(closest)
        current = closest

        elapsed = time.time() - step_time
        print(f"-> {clean(closest)} (score: {score:.3f}) [{elapsed:.2f}s]")

    total_time = time.time() - start_time
    steps = len(path) - 1

    print(f"\n{'='*60}")
    print(f"Time: {total_time:.2f}s | Steps: {steps}")
    print(f"{'='*60}")

    print("\nPath taken:")
    for p in path:
        print(f"  https://en.wikipedia.org/wiki/{p}")

    executor.shutdown(wait=False)