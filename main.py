import requests
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
import numpy as np

START_URL = "https://en.wikipedia.org/wiki/GitHub"
END_URL = "https://en.wikipedia.org/wiki/YouTube"

# setup
model = SentenceTransformer("BAAI/bge-small-en-v1.5") 
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0"
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 8)

# caches
cache = {}
pending = {}


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
    
    # remove script and style tags
    html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<style.*?</style>', '', html, flags=re.DOTALL)
    
    # find all wiki links
    pattern = r'(.{0,150})<a[^>]*href="/wiki/([^"#:]+)"[^>]*>([^<]+)</a>(.{0,150})'
    matches = re.findall(pattern, html, re.DOTALL)
    
    links = {}
    for before, link, text, after in matches:
        # skip bad links
        if link.startswith(skip):
            continue
        if link in links:
            continue
        
        # get context around link
        context = before + text + after
        context = re.sub(r'<[^>]+>', ' ', context)  # remove html tags
        context = re.sub(r'\s+', ' ', context)      # fix whitespace
        links[link] = context.strip()
    
    return links


def find_closest(links, target_embed, top_n=5):
    keys = list(links.keys())
    contexts = list(links.values())
    
    # encode all contexts
    embeds = model.encode(contexts, show_progress_bar=False, batch_size=64)
    
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
