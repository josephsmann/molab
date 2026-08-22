# molab

Marimo notebooks developed locally (or in a cloud session), versioned here, and
run in [molab](https://docs.marimo.io/guides/molab/).

GitHub is the source of truth: push here, and synced molab notebooks pick the
change up automatically.

## Notebooks

| Notebook | Run |
|---|---|
| [`hello.py`](notebooks/hello.py) — round-trip smoke test | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/josephsmann/molab/blob/main/notebooks/hello.py) [(wasm preview)](https://molab.marimo.io/github/josephsmann/molab/blob/main/notebooks/hello.py/wasm) |
| [`plate_model.py`](notebooks/plate_model.py) — write a plate model as text, get the diagram (and TikZ/daft/DOT/Mermaid) | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/josephsmann/molab/blob/main/notebooks/plate_model.py) [(wasm preview)](https://molab.marimo.io/github/josephsmann/molab/blob/main/notebooks/plate_model.py/wasm) |

## Pairing with agents

Agents (Claude Code, Codex, OpenCode) can drive a *running* notebook — inspect
live variables, test code in the kernel scratchpad, add and run cells — via
[`marimo pair`](https://docs.marimo.io/guides/generate_with_ai/marimo_pair/).

```sh
npx skills add marimo-team/marimo-pair   # installs the skill (tracked in skills-lock.json)
```

Then either pair on a local notebook:

```sh
uvx marimo@latest edit --sandbox notebooks/hello.py --no-token
```

...or on a molab sandbox: start the notebook, choose **Pair with an agent**
from the actions panel, and hand the agent the kernel URL and token it shows.

While a session is live the **kernel is the source of truth** — edit cells
through the agent, not by writing to the `.py` file, or the kernel will
overwrite your changes on save.

## Conventions

- Every notebook declares its dependencies **inline** (PEP 723 `# /// script`
  header), so it reproduces its own environment with no `requirements.txt`.
- Run one locally with `uvx marimo@latest edit --sandbox notebooks/hello.py`.
- `marimo check --fix <file>` before committing.

## Scope

This repo is **public** — molab pulls from GitHub unauthenticated, so anything
synced has to be. Nothing private, personal, or client-related belongs here.
