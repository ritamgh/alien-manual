---
title: Alien Manual demo
sdk: gradio
app_file: app.py
python_version: 3.12
---

# Alien Manual demo

Pick a held-out task and see the generated tool documentation, the model's predicted call, the
oracle call (what the documentation asks for), the habit call (what a model with habits would
do) and a plain verdict. The Space only reads: it trains and logs nothing, and it has one page.

## Layout

`app.py` puts its own folder and the folder above it on `sys.path`, so `alien/` is found either
way. In the Space repo the layout is:

```
app.py
requirements.txt
README.md
alien/        (copied from this repo)
```

## How to deploy (Ritam does this)

1. Create the Space (SDK: Gradio) on Hugging Face.
2. Copy the `alien/` package and everything in `space/` (`app.py`, `requirements.txt`,
   `README.md`) into the Space repo, with `alien/` next to `app.py`.
3. Add the Space secrets: `ADAPTER_REPO` (hub id of the tuned adapter), `HF_TOKEN` (only if the
   adapter or base model is private; it is read from the environment and never printed), and
   optionally `MODEL_ID` (base model, default is the one in `alien/eval.py`).
4. To preview the page before the adapter exists, set `DEMO_FAKE` to `oracle` or `habit`: the
   page then answers from the tasks themselves, with no model.

Without `ADAPTER_REPO` and `DEMO_FAKE` the page opens and shows which setting is missing.
