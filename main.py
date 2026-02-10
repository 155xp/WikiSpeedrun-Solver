import requests
import re
import time
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
import numpy as np
from bs4 import BeautifulSoup
import torch

START_URL = "https://en.wikipedia.org/wiki/GitHub"
END_URL = "https://en.wikipedia.org/wiki/Warsaw"
MAX_LINKS_PER_PAGE = 150
ENCODE_BATCH_SIZE = 128
STEP_DELAY_SECONDS = 0.2

# setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0"
executor = ThreadPoolExecutor(max_workers=4)

# caches
cache = {}
pending = {}
embedding_cache = {}


def fetch(page):
    # download a wiki page
    url = "https://en.wikipedia.org/wiki/" + page
    try:
        r = session.get(url, timeout=10)
        if r.ok:
            return page, r.text
        return page, ""
    except:
        return page, ""


def get_html(page):
    # save any finished downloads
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
            pass
    
    # check cache first
    if page in cache:
        return cache.pop(page)
    
    # wait if already downloading
    if page in pending:
        future = pending.pop(page)
        try:
            result = future.result(timeout=15)
            return result[1]
        except:
            pass
    
    # download now
    result = fetch(page)
    return result[1]


def prefetch(pages):
    # start downloading pages in background
    for p in pages:
        if p not in cache and p not in pending:
            pending[p] = executor.submit(fetch, p)


def extract_links(html):
    # pages to skip
    skip = ('File:', 'Wikipedia:', 'Help:', 'Special:', 'Talk:', 
            'Template:', 'Category:', 'Portal:', 'Main_Page')

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    links = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/wiki/"):
            continue
        link = href[len("/wiki/"):]

        # skip bad links
        if link.startswith(skip):
            continue
        if ":" in link or "#" in link:
            continue
        if link in links:
            continue

        text = a.get_text(" ", strip=True)
        parent_text = a.parent.get_text(" ", strip=True) if a.parent else text
        if not parent_text:
            continue

        context = parent_text
        if len(context) > 320:
            if text:
                idx = context.lower().find(text.lower())
            else:
                idx = -1
            if idx >= 0:
                start = max(0, idx - 120)
                end = min(len(context), idx + len(text) + 120)
                context = context[start:end]
            else:
                context = context[:240]

        context = re.sub(r"\s+", " ", context)
        links[link] = context.strip()
    
    return links


def find_closest(links, target_embed, top_n=5):
    keys = list(links.keys())
    contexts = list(links.values())

    # encode only uncached contexts
    uncached = [c for c in contexts if c not in embedding_cache]
    if uncached:
        new_embeds = model.encode(
            uncached,
            show_progress_bar=False,
            batch_size=ENCODE_BATCH_SIZE,
        )
        for c, e in zip(uncached, new_embeds):
            embedding_cache[c] = e
    embeds = np.vstack([embedding_cache[c] for c in contexts])
    
    # compare to target
    scores = util.cos_sim(embeds, target_embed)
    scores = scores.squeeze().cpu().numpy()
    
    # handle single result
    if scores.ndim == 0:
        return keys[0], float(scores), keys[:top_n]
    
    # get top matches
    sorted_idx = np.argsort(scores)
    top_idx = sorted_idx[-top_n:][::-1]  # best first
    
    best = top_idx[0]
    top_links = [keys[i] for i in top_idx]
    
    return keys[best], float(scores[best]), top_links


def clean(text):
    # make url text readable
    return re.sub(r"[_%27]+", " ", text)


def get_page(url):
    # get page name from url
    return url[url.find("/wiki/") + 6:]


# main
if __name__ == "__main__":
    start = get_page(START_URL)
    end = get_page(END_URL)
    
    current = start
    path = [start]
    visited = {start}
    
    # encode target for comparison
    target_embed = model.encode(clean(end))
    
    print(f"Starting: {clean(start)}")
    print(f"Target: {clean(end)}\n")
    
    start_time = time.time()
    
    while current != end:
        step_time = time.time()
        
        # get page and extract links
        html = get_html(current)
        all_links = extract_links(html)
        
        # filter out visited pages
        links = {}
        for k, v in all_links.items():
            if k not in visited:
                links[k] = v

        # keep only the first N links to speed up each step
        if len(links) > MAX_LINKS_PER_PAGE:
            links = dict(list(links.items())[:MAX_LINKS_PER_PAGE])
        
        # dead end check
        if not links:
            print("dead end")
            break
        
        # check if target is directly linked
        if end in links:
            path.append(end)
            elapsed = time.time() - step_time
            print(f"-> {clean(end)} (FOUND) [{elapsed:.2f}s]")
            break
        
        # find best link
        closest, score, top = find_closest(links, target_embed, top_n=8)
        
        # prefetch top candidates
        to_prefetch = []
        for c in top:
            if c not in visited:
                to_prefetch.append(c)
        prefetch(to_prefetch)
        
        # move to next page
        visited.add(closest)
        path.append(closest)
        current = closest
        
        # print progress
        elapsed = time.time() - step_time
        print(f"-> {clean(closest)} (score: {score:.3f}) [{elapsed:.2f}s]")
        time.sleep(STEP_DELAY_SECONDS)
    
    # done
    total_time = time.time() - start_time
    steps = len(path) - 1
    
    print(f"\n{'='*60}")
    print(f"Time: {total_time:.2f}s | Steps: {steps}")
    print(f"{'='*60}")
    
    print("\nPath taken:")
    for p in path:
        print(f"  https://en.wikipedia.org/wiki/{p}")
    
    executor.shutdown(wait=False)
