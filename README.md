# Wikipedia Speedrun Solver

## What is a Wikipedia Speedrun?

A Wikipedia speedrun is a game where you try to navigate from one Wikipedia article to another using only the hyperlinks on each page—in as few clicks as possible. The key rule: **no going back**. Once you leave a page, you can't return to it. This solver follows that rule and automates the process using machine learning.

## How It Works

### 1. Semantic Embeddings
The solver uses a sentence transformer model *(`BAAI/bge-small-en-v1.5`)* to convert text into numerical vectors that capture meaning. Two pieces of text with similar meanings will have similar vectors.

### 2. Context Extraction
For each Wikipedia page, the solver:
- Extracts all outgoing links to other Wikipedia articles and the surrounding context of each link
- Filters out non-article links (files, categories, help pages, etc.)

### 3.  Navigation
At each step, the solver:
1. Encodes the target article name into a vector
2. Encodes the context around every link on the current page
3. Calculates cosine similarity between each link's context and the target
4. Picks the link with the highest similarity score

This means it doesn't just match keywords—it understands *meaning*. A link mentioning "video platform" will score high when trying to reach "YouTube" even if "YouTube" isn't explicitly mentioned.

### 4. Prefetching for Speed
To minimize wait times, the solver downloads the top candidate pages in the background using parallel threads. If the AI's first choice leads somewhere unexpected, the backup options are often already loaded.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Edit `main.py` to set your start and end URLs:

```python
START_URL = "https://en.wikipedia.org/wiki/GitHub"
END_URL = "https://en.wikipedia.org/wiki/YouTube"
```

Then run:

```bash
python main.py
```

## Example Output

```
Starting: GitHub
Target: YouTube

-> Google (score: 0.612) [0.45s]
-> Alphabet Inc. (score: 0.583) [0.31s]
-> YouTube (FOUND) [0.22s]

============================================================
Time: 1.23s | Steps: 3
============================================================

Path taken:
  https://en.wikipedia.org/wiki/GitHub
  https://en.wikipedia.org/wiki/Google
  https://en.wikipedia.org/wiki/Alphabet_Inc.
  https://en.wikipedia.org/wiki/YouTube
```

## How the Score Works

The score (0.0 to 1.0) represents cosine similarity between the link's context and the target:
- **> 0.7**: Strong semantic match
- **0.5 - 0.7**: Moderate relevance  
- **< 0.5**: Weak match (the solver is exploring)

## Requirements

- Python 3.8+
- `requests` - HTTP requests
- `sentence-transformers` - Semantic embeddings
- `numpy` - Numerical operations
- `tf-keras` - Keras 3 compatibility fix

