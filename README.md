# Wikipedia Speedrun Solver

Follow links between English Wikipedia articles using semantic similarity.
The solver uses `BAAI/bge-small-en-v1.5` to rank article titles, takes a direct
link to the target when available, and never revisits an article.

## Run

```sh
pip install -r requirements.txt
python main.py GitHub Warsaw
```

You can pass article titles or full English Wikipedia URLs. Quote titles
containing spaces. With no arguments, the script uses `START_URL` and `END_URL`
in `main.py`. The first run downloads the model from Hugging Face.

The parser handles both `/wiki/Article` and `./Article` links and checks the
whole page. HTTP failures are reported as errors. Greedy ranking does not
guarantee the shortest path or a successful route; searches stop after 50 clicks.
Failed searches exit with status 1.

## Check

```sh
python -m unittest -q
```

The tests run without downloading the model or contacting Wikipedia.
