# askrepo

askrepo answers natural-language questions about a local codebase. It indexes files on disk, retrieves a few relevant chunks with a lexical scorer that knows about paths and simple symbol hints, and can send those chunks to an OpenAI-compatible chat model.

The index is the product boundary. Embeddings and a code graph are not in this version. Both can be added as retrievers that consume the same chunks, without changing ingest or the chat step.

## Why keep structure in the loop

A bag of embedding windows is enough for "find code that looks like this sentence" and weak for "where is this function defined?" or "what imports this module?". Those questions are answered by the path, the definition name, and the import line. askrepo stores that information while it chunks:

- every chunk has a repo-relative path and a line range
- a short language guess comes from the filename
- symbol and import hints are extracted with regexes and stored on the chunk

Retrieval uses those fields today. A later graph pass can turn the same hints into edges instead of throwing the index away and starting from raw text.

Lexical search will miss paraphrases. That is the reason to add embeddings later. It is not a reason to drop paths and symbols now.

## Quickstart

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

askrepo index examples/tiny_repo
askrepo ask "where is the invoice total calculated?" --index examples/tiny_repo
```

`index` writes `<path>/.askrepo/index.json`. `ask` loads that file. With no `ASKREPO_API_KEY`, it prints the retrieved chunks and a short offline note. It does not open a network connection.

```bash
export ASKREPO_API_KEY=...
export ASKREPO_BASE_URL=https://api.openai.com/v1   # optional; API root, usually ending in /v1
export ASKREPO_MODEL=gpt-4o-mini                     # optional
askrepo ask "where is the invoice total calculated?" --index examples/tiny_repo
```

`examples/tiny_repo` is a small billing tree plus prose notes. It also contains `node_modules/`, `build/`, and `.env` so you can see that those are not indexed. Nothing in that tree is a real secret.

To index the project you are sitting in:

```bash
askrepo index .
askrepo ask "where is the index written?"
```

`ask` with no `--index` walks upward from the current directory until it finds `.askrepo/index.json`.

## Environment

| Variable | Default | Role |
| --- | --- | --- |
| `ASKREPO_API_KEY` | unset | Bearer token. Unset means offline mode. |
| `ASKREPO_BASE_URL` | `https://api.openai.com/v1` | API root. The client posts to `{base}/chat/completions`. |
| `ASKREPO_MODEL` | `gpt-4o-mini` | Model name sent in the chat request. |
| `ASKREPO_RETRIEVER` | `lexical` | Backend name. `embedding` and `graph` are reserved and error if selected. |

The key is read from the environment only. It is not written into the index.

## CLI

```text
askrepo index [PATH]              # default PATH is the current directory
askrepo ask QUESTION [--k 5] [--index PATH] [--retriever lexical]
askrepo eval [PATH] [--k 5] [--cases PATH] [--retriever lexical]
askrepo --version
```

`--k` is capped at 50. While other files still have hits, at most two chunks from one file are kept; if the window is still short, further chunks from a file already used are filled in. That stops one long module from crowding out the rest of the tree when more than one file matched.

## Layout

```text
src/askrepo/
  cli.py        commands and stdout
  ingest.py     walk, skip, chunk, hints, index IO
  retrieve.py   Retriever protocol and the lexical scorer
  answer.py     offline note, prompt, chat HTTP call
  eval.py       gold-set hit@k for retrievers
  types.py      Chunk, Index, Hit, Answer
  config.py     environment settings
```

```text
index PATH
  walk the tree
  drop junk directories, lockfiles, binary files, and secret-shaped names
  split text into line windows
  attach language, symbols, and file-level imports
  write .askrepo/index.json

ask QUESTION
  load the index
  Retriever.retrieve(query, index, k) -> hits
  if there are no hits: stop, do not call a model
  if ASKREPO_API_KEY is unset: print hits and an offline note
  else: POST the hits as chat context and print the reply
```

`ingest` does not know the question. `retrieve` does not call a model. `answer` does not rank. A new backend implements `Retriever.retrieve` and returns `list[Hit]`.

## Index format (version 1)

Path: `<root>/.askrepo/index.json`. Chunk paths are POSIX and relative to the indexed root. `root` is the absolute path at index time and is informational; moving the repo does not rewrite chunk paths.

```json
{
  "version": 1,
  "tool": "askrepo",
  "tool_version": "0.1.0",
  "created_at": "2026-11-04T12:00:00+00:00",
  "root": "/absolute/path/to/repo",
  "files_indexed": 4,
  "files_skipped": 1,
  "dirs_pruned": 2,
  "chunks": [
    {
      "id": "src/billing/invoice.py:1-11",
      "path": "src/billing/invoice.py",
      "start_line": 1,
      "end_line": 11,
      "language": "python",
      "symbols": ["calculate_invoice_total"],
      "imports": ["billing.tax"],
      "text": "..."
    }
  ]
}
```

- Chunk id is `path:start-end`. Keep that stable if you add sidecars.
- Windows are 80 lines with a 10-line overlap. A file under that size is one chunk.
- Files larger than 400 KB, files with a line over 2000 characters, and files that are not valid UTF-8 or that contain a NUL are skipped.
- `symbols` and `imports` are best-effort regex hints. Python, JavaScript/TypeScript, Go, Rust, and Java get definitions; other languages still get a language label and text. Imports are file-level and copied onto each chunk of that file.
- There is no vector field and no edge list. Add those beside this file (for example `.askrepo/vectors` or `.askrepo/graph.json`) keyed by chunk id, and bump `version` only when the chunk record itself changes.

`.askrepo/` is in this repo's `.gitignore`. Add the same ignore in any project you index. The file contains source text.

The junk list is `SKIP_DIRS` in `ingest.py`. It includes `.git`, `node_modules`, `venv`, `.venv`, `dist`, `build`, `target`, and `vendor`, and directory names ending in `.egg-info`. Names such as `.env`, `id_rsa`, and `*.pem` are skipped. The project's own `.gitignore` is not read.

## How scoring works today

`LexicalRetriever` tokenizes the question, the chunk text, the path, symbol names, and imports. Snake case and camel case are split. English function words (`how`, `where`, `the`, …) are dropped.

The score is the mean IDF of question tokens that appear in the chunk, plus a fixed bonus when those tokens hit the path, a symbol name, or an import. IDF is computed over the in-memory chunk list at query time. There is no inverted index. Ties break by path, then start line.

That bonus is why "calculate invoice total" should rank `invoice.py` above a markdown file that only says "invoice total". It will not match "how do we bill customers?" unless those words are actually in the tree.

## Retrieval eval

`evals/tiny_repo.json` is the baseline for retrieval experiments. Each case is a question plus the repo-relative file that should answer it. `askrepo eval` builds an in-memory index of `examples/tiny_repo` (the same ingest path as `askrepo index`, nothing written), runs `Retriever.retrieve` for every question, and prints hit@1 and hit@5. It does not call a model and does not read `ASKREPO_API_KEY`.

```bash
askrepo eval
```

From a checkout, `python -m askrepo eval` is the same command. The process exits non-zero if a required case misses its rank cutoff. `pytest` runs that check.

Required cases are lookups the lexical scorer should already get: symbol names, "where is X defined", a few exact sentences, and one import hint (`notify.receipt`) that is stored on the chunk. Most of them must be rank 1 (`"at": 1`). `caller-apply-tax` only has to stay inside the top 5, because the definition outranks the call site.

Challenge cases are expected lexical misses. Two of them need an import or call edge into a file that shares no question words (`gateway-hand-off`, `payment-message-callee`). One is a paraphrase of `apply_tax` (`gross-up-paraphrase`). A later graph or embedding retriever should raise the challenge hit rate without dropping required cases. On the lexical baseline, `pytest` fails if a challenge case starts hitting; move it to `required` when that is intentional.

Add a case by appending an object to `evals/tiny_repo.json`. Every path in `expect` has to appear in the top k, and the path has to be a file the indexer actually keeps:

```json
{
  "id": "define-apply-tax",
  "question": "where is apply_tax defined",
  "expect": ["src/billing/tax.py"],
  "tier": "required",
  "at": 1,
  "note": "Symbol lookup."
}
```

`tier` is `required` or `challenge`. `at` is the worst rank that still passes for a required case. Omit it to use the file's `k`.

## Roadmap

The comparison point for the retrievers below is `askrepo eval`.

1. **Embeddings.** Implement `Retriever` under the name `embedding`. Embed each chunk once, store vectors in a sidecar keyed by chunk id, and embed the query with the same model at ask time. Keep `lexical` available: symbol-shaped queries are often better as keyword hits. A hybrid can sum normalized scores. Ingest remains the source of chunk text. Select the backend with `ASKREPO_RETRIEVER` or `--retriever`. Both names already exist and raise `NotImplementedError`.
2. **Code graph.** Do not bury this in the chunker. A separate pass should read `symbols` and `imports`, then replace the regexes with real parses. Write `.askrepo/graph.json` as edges (`src`, `dst`, `kind`) for imports, defines, and later calls. A `graph` retriever expands or reranks the lexical or embedding hits by one hop and still returns `list[Hit]`, with an `edge:…` reason. `answer.py` stays unaware of the graph.
3. **tree-sitter and LSP.** Use tree-sitter queries for definitions, imports, and calls, and prefer LSP `documentSymbol` / references when a language server is already running. Chunk on function and class boundaries instead of an 80-line window. The index format can keep the same chunk ids if the line range is still the identity.
4. **Smaller gaps.** Honor `.gitignore`, make the skip list configurable, reindex a changed file instead of the whole tree, and stream an inverted index once the chunk count makes a full scan unpleasant.

## Limitations

- Paraphrase search does not work. If the words are not in the path, the symbol, the import, or the text, the chunk is invisible. Tokens are not stemmed: `calculated` does not match `calculate`. A definition can still rank first when the other words hit the symbol or the path.
- Regex hints miss names hidden by decorators, factories, re-exports, macros, and most dynamic calls. They are ranking hints, not a compiler.
- JavaScript hints cover `function`, `class`, and simple `const name = (...) =>` forms. Many other assignment shapes are ignored.
- Line windows split long functions. The overlap only reduces how often a signature lands on the cut.
- `build`, `vendor`, and `env` are always pruned. A source directory with one of those names will not be indexed. This is not configurable yet.
- `.gitignore`, nested gitlinks, and symlink targets are not followed. Symlinks are skipped.
- Offline mode does not summarize the code. It shows what retrieval returned and which files would have been sent.
- With a key, the model is instructed to stay inside the excerpts and can still over-claim. Empty retrieval does not call the API.
- Secret-shaped filenames are skipped. A token pasted into a `.py` or `.md` file is copied into the index and, if you ask with a key, is sent to `ASKREPO_BASE_URL`. Review the index before pointing it at a hosted model.
- The client trusts `ASKREPO_BASE_URL`. Only the bearer header and the chat payload leave the machine, and only from `answer.complete_chat`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests index `examples/tiny_repo` and a few temporary trees. They check skip rules, chunking, ranking, the retrieval gold set, the offline answer path, and the shape of the chat request via `httpx.MockTransport`. They do not call a hosted model and do not need an API key.

## License

MIT. See [LICENSE](LICENSE).
