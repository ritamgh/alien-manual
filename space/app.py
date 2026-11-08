"""Alien Manual demo Space (PRD S5, FR-006, NG-4).

The user picks one of the demo tasks, sees the generated tool doc, and sees the model's
predicted call next to the oracle call (what the doc says) and the habit call (what a model
with habits would do), with a plain verdict. The demo only reads: it trains and logs nothing.
It is the only UI; there are no other pages and no auth.

Layout: this file expects the ``alien/`` package next to it (in the Space repo) or one level
up (in this repo, where it lives in ``space/``). Both are put on ``sys.path``.

Environment (set as Space secrets, never written to a file):

* ``DEMO_FAKE=oracle|habit``: answer from the tasks themselves, no model (shows the UI early).
* ``ADAPTER_REPO``: hub id of the tuned LoRA adapter; needed unless ``DEMO_FAKE`` is set.
* ``MODEL_ID``: base model id, default the one ``alien/eval.py`` uses.
* ``HF_TOKEN``: read by the hub library from the environment only, for a private adapter.

Public API:

* ``load_demo_tasks(seed=0, n=24) -> list[Task]``: held-out trap and normal tasks.
* ``predict_vs_oracle(task, generate) -> dict``: the comparison and the verdict.
* ``answer_request(request, tasks, generate) -> dict``: what the UI shows for a request.
* ``make_generate() -> Callable[[str], str]``; ``ConfigError`` when it is not configured.
* ``build_demo(generate)``: the Gradio app (gradio is imported inside).
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from alien.eval import DECODING, MODEL_ID  # noqa: E402
from alien.generator import Call, Task  # noqa: E402
from alien.oracle import call_to_json, render_prompt  # noqa: E402
from alien.scoring import parse_calls, score_example  # noqa: E402
from alien.splits import make_splits  # noqa: E402

GenerateOne = Callable[[str], str]

PICK_MESSAGE = "pick one of the demo tasks"
_VERDICT_NOTES = {
    "matches the oracle": "The model made the call the tool documentation asks for.",
    "followed the habit instead of the doc": (
        "The model made the call its habits suggest, not the one the tool documentation asks for."
    ),
    "wrong call": "The model made a call, but it is not the one the tool documentation asks for.",
    "no valid call": "The model's reply did not contain a call in the expected JSON format.",
}


class ConfigError(RuntimeError):
    """The Space is not configured; the message says which setting to add."""


def load_demo_tasks(seed: int = 0, n: int = 24) -> list[Task]:
    """Half trap, half normal tasks from the held-out split, interleaved, seeded."""
    heldout = make_splits(seed).heldout
    rng = random.Random(f"demo:{seed}")
    trap = [t for t in heldout if t.is_trap]
    normal = [t for t in heldout if not t.is_trap]
    n_trap = min(n // 2, len(trap))
    picked_trap = rng.sample(trap, n_trap)
    picked_normal = rng.sample(normal, min(n - n_trap, len(normal)))
    mixed: list[Task] = []
    for i in range(max(len(picked_trap), len(picked_normal))):
        mixed += picked_trap[i:i + 1] + picked_normal[i:i + 1]
    return mixed


def predict_vs_oracle(task: Task, generate: GenerateOne) -> dict[str, Any]:
    prompt = render_prompt(task)
    model_output = generate(prompt)
    calls = parse_calls(model_output)
    score = score_example(task, model_output)  # same rules as the scorer (FR-007)
    if not calls:
        verdict = "no valid call"
    elif score.prior_trap:
        verdict = "followed the habit instead of the doc"
    elif score.tp == 1 and score.fp == 0:
        verdict = "matches the oracle"
    else:
        verdict = "wrong call"
    return {
        "prompt": prompt,
        "tool_doc": task.tool.doc,
        "request": task.request,
        "model_output": model_output,
        "predicted_calls": calls,
        "oracle_call": task.oracle_call,
        "habit_call": task.habit_call,
        "verdict": verdict,
        "verdict_note": _VERDICT_NOTES[verdict],
    }


def _pretty(call: Call) -> str:
    return json.dumps(call, indent=2)


def answer_request(request: str, tasks: list[Task], generate: GenerateOne) -> dict[str, str]:
    """Display strings for a request; only a demo task's exact request is answered."""
    task = next((t for t in tasks if t.request == request.strip()), None)
    if task is None:
        return {"tool_doc": "", "model": "", "oracle": "", "habit": "",
                "verdict": PICK_MESSAGE, "verdict_note": ""}
    shown = {"tool_doc": task.tool.doc, "oracle": _pretty(task.oracle_call),
             "habit": _pretty(task.habit_call)}
    try:
        result = predict_vs_oracle(task, generate)
    except ConfigError as exc:
        return {**shown, "model": "", "verdict": str(exc), "verdict_note": ""}
    except Exception as exc:  # the UI shows a message, never a stack trace
        message = f"The model could not run ({type(exc).__name__})."
        return {**shown, "model": "", "verdict": message, "verdict_note": ""}
    return {**shown, "model": result["model_output"], "verdict": result["verdict"],
            "verdict_note": result["verdict_note"]}


def _fake_generate(kind: str) -> GenerateOne:
    answers = {}
    for task in load_demo_tasks():
        call = task.oracle_call if kind == "oracle" else task.habit_call
        answers[render_prompt(task)] = call_to_json(call)

    def generate(prompt: str) -> str:
        if prompt not in answers:
            raise ConfigError("DEMO_FAKE only answers the demo tasks.")
        return answers[prompt]

    return generate


def _hf_generate(model_id: str, adapter: str) -> GenerateOne:
    """Load the base model plus the tuned adapter. The hub library reads HF_TOKEN itself."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def generate(prompt: str) -> str:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        batch = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=DECODING["max_new_tokens"],
                do_sample=DECODING["do_sample"],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def make_generate() -> GenerateOne:
    fake = os.environ.get("DEMO_FAKE", "").strip()
    if fake:
        if fake not in ("oracle", "habit"):
            raise ConfigError("DEMO_FAKE must be 'oracle' or 'habit' (or unset).")
        return _fake_generate(fake)
    adapter = os.environ.get("ADAPTER_REPO", "").strip()
    if not adapter:
        raise ConfigError(
            "This Space is not configured: set the ADAPTER_REPO secret to the tuned adapter's "
            "hub id, or set DEMO_FAKE to 'oracle' or 'habit' to preview the page without a model."
        )
    return _hf_generate(os.environ.get("MODEL_ID", "").strip() or MODEL_ID, adapter)


def build_demo(generate: GenerateOne):
    import gradio as gr

    tasks = load_demo_tasks()
    by_id = {t.id: t for t in tasks}

    def pick(task_id: str) -> str:
        return by_id[task_id].request

    def run(request: str) -> tuple[str, str, str, str, str]:
        r = answer_request(request, tasks, generate)
        verdict = f"**{r['verdict']}**" + (f" - {r['verdict_note']}" if r["verdict_note"] else "")
        return r["tool_doc"], r["model"], r["oracle"], r["habit"], verdict

    with gr.Blocks(title="Alien Manual demo") as demo:
        gr.Markdown(
            "# Alien Manual demo\n"
            "Pick a task on a tool the model never saw in tuning. The tool documentation may "
            "contradict what the tool's name suggests; see whether the model follows the doc."
        )
        dropdown = gr.Dropdown(
            choices=[(f"{t.id}: {t.tool.name}", t.id) for t in tasks],
            value=tasks[0].id, label="Demo task",
        )
        request = gr.Textbox(value=tasks[0].request, label="Request")
        run_button = gr.Button("Run", variant="primary")
        doc = gr.Textbox(label="Tool documentation", lines=8, interactive=False)
        with gr.Row():
            model_box = gr.Code(label="Model prediction", interactive=False)
            oracle_box = gr.Code(label="Oracle (correct)", language="json", interactive=False)
            habit_box = gr.Code(
                label="Habit (what a model with habits would do)", language="json",
                interactive=False,
            )
        verdict = gr.Markdown()
        dropdown.change(pick, dropdown, request)
        run_button.click(run, request, [doc, model_box, oracle_box, habit_box, verdict])
    return demo


if __name__ == "__main__":
    try:
        _generate = make_generate()
    except ConfigError as _exc:
        _message = str(_exc)

        def _generate(prompt: str) -> str:
            raise ConfigError(_message)

    build_demo(_generate).launch()
