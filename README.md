# molab

Marimo notebooks developed locally (or in a cloud session), versioned here, and
run in [molab](https://docs.marimo.io/guides/molab/).

GitHub is the source of truth: push here, and synced molab notebooks pick the
change up automatically.

## Notebooks

| Notebook | Run |
|---|---|
| [`hello.py`](notebooks/hello.py) — round-trip smoke test | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/josephsmann/molab/blob/main/notebooks/hello.py) [(wasm preview)](https://molab.marimo.io/github/josephsmann/molab/blob/main/notebooks/hello.py/wasm) |

## Conventions

- Every notebook declares its dependencies **inline** (PEP 723 `# /// script`
  header), so it reproduces its own environment with no `requirements.txt`.
- Run one locally with `uvx marimo@latest edit --sandbox notebooks/hello.py`.
- `marimo check --fix <file>` before committing.

## Scope

This repo is **public** — molab pulls from GitHub unauthenticated, so anything
synced has to be. Nothing private, personal, or client-related belongs here.
