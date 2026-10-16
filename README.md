# Learning reusable structure from temporal experience

**Status: replacement suite; development must pass gates before confirmation.**
The broader question is whether learners discover reusable structure and recode
old experience while preserving useful knowledge. Robustness and postlearning
drift are separate questions. No ordering of neural stability across layers,
necessary sparsification, or necessary suppression of smaller memories is assumed.

## Historical preservation

Last pre-reset SHA: **cd95d2cf8c1e76d6a79d8a692238a2df858c7272**.
EXP001 and EXP002 are removed from the active tree by a normal descendant commit.
Their complete files/results remain in reachable history and published GitHub:

- [EXP001 measured limitations](https://github.com/OgnjenX/chunking-drift-stability/blob/af5272bf88b569268a3a993264a7bba16c38645f/RESULTS.md)
- [EXP002 development failures](https://github.com/OgnjenX/chunking-drift-stability/blob/7a5fc86af4abb2a6380df3f239817a83d9b2d528/exp002/RESULTS.md)
- [Entire pre-reset tree](https://github.com/OgnjenX/chunking-drift-stability/tree/cd95d2cf8c1e76d6a79d8a692238a2df858c7272)

These are inadequate operationalizations of the broader hypothesis, not its
scientific falsification. Their actual negative observations remain valid for
the tested models/tasks. No history was rewritten. To inspect: `git show SHA:path`;
to run old code, use a separate checkout of that SHA.

## Run

Python 3.11, pinned environment with `uv`:

```sh
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked python -m srclearner.reference --output results/reference-v1
uv run --locked python -m srclearner.run --output results/development-v1
uv run --locked python -m srclearner.plot
uv run --locked python -m srclearner.audit
```

Runs refuse to overwrite existing attempts. For reproduction, specify a fresh
`--output` directory; the audit/plot commands default to the archived development
attempt. No command launches reserved confirmation or downstream stages.

- [Mechanism and source boundary](docs/mechanism.md)
- [Development protocol and gates](docs/development.md)
- [Prospective confirmation and downstream protocol](docs/confirmation.md)
- [Actual findings](RESULTS.md)

`srclearner/` contains the new implementation; `tests/` checks components.
`configs/` fixes exposures and independent development worlds. `results/`
contains only attempts actually run, raw trial predictions, full input streams,
weights, aggregate decisions and figures. Evaluation truth stays in the runner.
The model is a deliberately simplified mechanistic pilot, not a replication of
STORE2/Masking Field or a validated account of general relational abstraction.
