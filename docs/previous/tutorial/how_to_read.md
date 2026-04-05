# How To Read The Function Docs

Use this workflow in the `researcher-ai` conda environment.

## 1) Install docs dependencies

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate researcher-ai
cd /Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai
python -m pip install -e ".[docs]"
```

## 2) Build docs locally

```bash
sphinx-build -b html docs docs/_build/html
```

Open the generated site at:

`/Users/brianyee/Documents/work/01_active/researcher-ai/researcher-ai/docs/_build/html/index.html`

## 3) Live-reload while editing docs

```bash
python -m pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

Then browse [http://127.0.0.1:8000](http://127.0.0.1:8000).

## 4) Publish on Read the Docs

1. Import the repository in Read the Docs.
2. Keep the default config path `.readthedocs.yaml`.
3. Trigger a build.
4. Open the generated site and navigate:
   - `API Reference` for functions/classes
   - Module pages for parser/pipeline/model internals

