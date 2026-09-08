import sys
import time
from html.parser import HTMLParser
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests

START_URL = "https://en.wikipedia.org/wiki/GitHub"
END_URL = "https://en.wikipedia.org/wiki/Warsaw"
BASE_URL = "https://en.wikipedia.org/wiki/"

# reuse the connection for each page
session = requests.Session()
session.headers["User-Agent"] = "WikiSpeedrun-Solver/1.0 (https://github.com/155xp/WikiSpeedrun-Solver)"

# these pages are not game articles
SKIP = ("File:", "Image:", "Wikipedia:", "Help:", "Special:", "Talk:",
        "Template:", "Category:", "Portal:", "Draft:", "Module:", "MediaWiki:",
        "User:", "Media:")


def page_name(value):
    # accept an article title or an english wikipedia url
    value = value.strip()
    if value.startswith(('https://', 'http://')):
        url = urlsplit(value)
        if url.hostname != 'en.wikipedia.org' or not url.path.startswith('/wiki/'):
            raise ValueError('use an english wikipedia article url')
        value = url.path[6:]
    value = unquote(value).replace(' ', '_')
    if not value:
        raise ValueError('article title cannot be empty')
    return value[0].upper() + value[1:]


def page_url(page):
    # keep punctuation in titles from changing the url
    return BASE_URL + quote(page, safe='/:')


class ArticleLinks(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.url = page_url(page)
        self.links = {}

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        href = dict(attrs).get('href') or ''
        # resolve both ./article and /wiki/article links
        url = urlsplit(urljoin(self.url, href))
        if not href or href.startswith('#') or url.hostname != 'en.wikipedia.org':
            return
        if not url.path.startswith('/wiki/') or url.path == '/wiki/' or url.query:
            return
        title = page_name(url.path[6:])
        # filter site pages after decoding escaped names
        if title == 'Main_Page' or title.startswith(SKIP) or '_talk:' in title:
            return
        self.links[title] = title.replace('_', ' ')


def extract_links(html, page):
    parser = ArticleLinks(page)
    parser.feed(html)
    return parser.links


def get_links(page):
    response = session.get(page_url(page), timeout=15)
    # show request failures instead of calling them dead ends
    response.raise_for_status()
    return extract_links(response.text, page)


def solve(start, end, model, max_steps=50):
    path = [start]
    target = model.encode([end.replace('_', ' ')], normalize_embeddings=True)[0]
    for _ in range(max_steps):
        if path[-1] == end:
            return path
        # read the whole page and never revisit an article
        links = {k: v for k, v in get_links(path[-1]).items() if k not in path}
        if end in links:
            print(f'-> {end.replace("_", " ")} (FOUND)', flush=True)
            return path + [end]
        if not links:
            raise RuntimeError(f'dead end at {path[-1]}: no unvisited article links')
        vectors = model.encode(list(links.values()), normalize_embeddings=True,
                               batch_size=64, show_progress_bar=False)
        # normalized vectors let a dot product measure similarity
        scores = [sum(a * b for a, b in zip(vector, target)) for vector in vectors]
        best = max(range(len(scores)), key=scores.__getitem__)
        page = list(links)[best]
        path.append(page)
        print(f'-> {page.replace("_", " ")} (score: {scores[best]:.3f})', flush=True)
    # ponytail: greedy search can miss a route; use graph search if that matters
    raise RuntimeError(f'step limit reached ({max_steps}) without finding {end}')


def main():
    # load the model only when running a search
    from sentence_transformers import SentenceTransformer

    try:
        if len(sys.argv) not in (1, 3):
            raise ValueError('usage: python main.py [start_article target_article]')
        start, end = map(page_name, sys.argv[1:] or [START_URL, END_URL])
        print(f'Starting: {start.replace("_", " ")}\nTarget: {end.replace("_", " ")}', flush=True)
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        started = time.monotonic()
        path = solve(start, end, model)
        print(f'\nTime: {time.monotonic() - started:.2f}s | Steps: {len(path) - 1}')
        print('\nPath taken:')
        for page in path:
            print(f'  {page_url(page)}')
    except (requests.RequestException, ValueError, RuntimeError) as error:
        # failed runs should also report failure to the shell
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
