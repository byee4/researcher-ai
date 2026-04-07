# Read the Docs Deployment

This repository includes first-class Read the Docs config.

## Included config files

- `.readthedocs.yaml`
- `docs/requirements.txt`
- `docs/conf.py`

## One-time RTD setup

1. Push this repository to GitHub/GitLab/Bitbucket.
2. In Read the Docs, click **Import a Project**.
3. Select the repository.
4. Confirm it detects `.readthedocs.yaml`.
5. Set default branch and click **Build Version**.

## Expected build behavior

RTD will:
- create a Python 3.11 environment,
- install docs requirements,
- install package in editable mode (`pip install -e .`),
- run `sphinx` with `docs/conf.py` and `docs/index.rst`.

## Local parity check

Run locally before pushing:

```bash
python -m pip install -r docs/requirements.txt
python -m pip install -e .
python -m sphinx -b html docs docs/_build/html
```

## Troubleshooting

## Import errors during autodoc

If runtime-heavy deps are unavailable in RTD, `docs/conf.py` uses `autodoc_mock_imports` for common optional dependencies.

## Build fails on missing docs page

Ensure every entry in `docs/index.rst` `toctree` exists.

## Theme/parser issues

Confirm `docs/requirements.txt` includes:
- `sphinx`
- `furo`
- `myst-parser`
