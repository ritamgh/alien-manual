"""Tests for the demo Space logic in space/app.py (PRD S5, FR-006, NG-4, B-3, B-5).

No gradio, torch or model: the UI and the model loader are never exercised here. Each group
names the production change that would make it fail.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from alien.oracle import call_to_json, render_prompt
from alien.splits import make_splits

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"


def _load_app():
    spec = importlib.util.spec_from_file_location("space_app", SPACE / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return _load_app()


@pytest.fixture(scope="module")
def tasks(app):
    return app.load_demo_tasks()


@pytest.fixture(scope="module")
def trap_task(tasks):
    return next(t for t in tasks if t.is_trap)


@pytest.fixture(scope="module")
def normal_task(tasks):
    return next(t for t in tasks if not t.is_trap)


# load_demo_tasks: fails if it draws from train/dev, returns only one kind, or ignores n.
def test_demo_tasks_are_held_out_and_mix_trap_with_normal(tasks):
    held_ids = {t.id for t in make_splits(0).heldout}
    assert len(tasks) == 24
    assert {t.id for t in tasks} <= held_ids
    assert all(t.id.startswith("heldout-") for t in tasks)
    assert sum(t.is_trap for t in tasks) == 12
    assert sum(not t.is_trap for t in tasks) == 12


def test_demo_tasks_are_deterministic_and_respect_n(app, tasks):
    again = app.load_demo_tasks(seed=0, n=24)
    assert [t.id for t in again] == [t.id for t in tasks]
    small = app.load_demo_tasks(seed=0, n=6)
    assert len(small) == 6
    assert sum(t.is_trap for t in small) == 3


# predict_vs_oracle verdicts: fails if the comparison is not by canonical call.
def test_oracle_output_matches_the_oracle(app, trap_task):
    result = app.predict_vs_oracle(trap_task, lambda prompt: call_to_json(trap_task.oracle_call))
    assert result["verdict"] == "matches the oracle"


def test_habit_output_on_trap_task_is_flagged_as_habit(app, trap_task):
    assert trap_task.habit_call != trap_task.oracle_call
    result = app.predict_vs_oracle(trap_task, lambda prompt: call_to_json(trap_task.habit_call))
    assert result["verdict"] == "followed the habit instead of the doc"


def test_integral_float_argument_still_matches(app, tasks):
    task = next(t for t in tasks
                if any(isinstance(v, int) for v in t.oracle_call["arguments"].values()))
    key = next(k for k, v in task.oracle_call["arguments"].items() if isinstance(v, int))
    floaty = json.loads(call_to_json(task.oracle_call))
    floaty["arguments"][key] = float(floaty["arguments"][key])
    text = json.dumps(floaty)
    assert f"{task.oracle_call['arguments'][key]}.0" in text
    result = app.predict_vs_oracle(task, lambda prompt: text)
    assert result["verdict"] == "matches the oracle"


def test_wrong_argument_is_a_wrong_call(app, trap_task):
    wrong = json.loads(call_to_json(trap_task.oracle_call))
    wrong["arguments"][next(iter(wrong["arguments"]))] = "NOT-A-REAL-ID"
    result = app.predict_vs_oracle(trap_task, lambda prompt: json.dumps(wrong))
    assert result["verdict"] == "wrong call"
    assert result["predicted_calls"] == [wrong]


def test_garbage_is_no_valid_call(app, trap_task):
    result = app.predict_vs_oracle(trap_task, lambda prompt: "I cannot help with that.")
    assert result["verdict"] == "no valid call"
    assert result["predicted_calls"] == []


def test_every_verdict_has_a_plain_note(app, trap_task):
    outputs = [call_to_json(trap_task.oracle_call), call_to_json(trap_task.habit_call),
               '{"name": "x", "arguments": {}}', "nope"]
    notes = [app.predict_vs_oracle(trap_task, lambda p, o=o: o)["verdict_note"] for o in outputs]
    assert all(isinstance(n, str) and n.endswith(".") and len(n) > 20 for n in notes)
    assert len(set(notes)) == 4


def test_habit_is_only_checked_for_trap_tasks(app, normal_task):
    # On a normal task habit == oracle, so the oracle verdict wins; a wrong call stays wrong.
    assert normal_task.habit_call == normal_task.oracle_call
    ok = app.predict_vs_oracle(normal_task, lambda p: call_to_json(normal_task.habit_call))
    assert ok["verdict"] == "matches the oracle"


# QA F-10. The Space applies the scorer's rules (alien.scoring.score_example): "matches" only for
# exactly the oracle call, prior_trap (habit emitted on a trap, even next to the oracle) is the
# habit verdict, anything else with a call is wrong. Fails if the verdict only looks for the
# oracle call among the predicted ones.
def test_scorer_rule_oracle_plus_habit_call_on_trap_is_the_habit_verdict(app, trap_task):
    for text in (
        call_to_json(trap_task.oracle_call) + "\n" + call_to_json(trap_task.habit_call),
        call_to_json(trap_task.habit_call) + "\n" + call_to_json(trap_task.oracle_call),
    ):
        result = app.predict_vs_oracle(trap_task, lambda prompt, t=text: t)
        assert result["verdict"] == "followed the habit instead of the doc"


def test_scorer_rule_oracle_plus_extra_wrong_call_is_a_wrong_call(app, trap_task):
    extra = json.loads(call_to_json(trap_task.oracle_call))
    extra["arguments"][next(iter(extra["arguments"]))] = "NOT-A-REAL-ID"
    text = call_to_json(trap_task.oracle_call) + "\n" + json.dumps(extra)
    result = app.predict_vs_oracle(trap_task, lambda prompt: text)
    assert result["verdict"] == "wrong call"


def test_scorer_rule_duplicated_oracle_call_is_a_wrong_call(app, trap_task):
    text = call_to_json(trap_task.oracle_call) * 2
    result = app.predict_vs_oracle(trap_task, lambda prompt: text)
    assert result["verdict"] == "wrong call"


def test_scorer_rule_oracle_call_in_a_fence_with_prose_still_matches(app, trap_task):
    text = "Here:\n```json\n" + call_to_json(trap_task.oracle_call) + "\n```\nDone."
    result = app.predict_vs_oracle(trap_task, lambda prompt: text)
    assert result["verdict"] == "matches the oracle"


def test_scorer_rule_exact_oracle_on_normal_task_matches(app, normal_task):
    # On a normal task habit == oracle: scorer's prior_trap is False, so exact oracle matches.
    text = call_to_json(normal_task.oracle_call)
    result = app.predict_vs_oracle(normal_task, lambda prompt: text)
    assert result["verdict"] == "matches the oracle"


def test_result_carries_the_literal_doc_request_prompt_and_calls(app, trap_task):
    result = app.predict_vs_oracle(trap_task, lambda prompt: "raw model text")
    assert result["tool_doc"] == trap_task.tool.doc
    assert result["request"] == trap_task.request
    assert result["prompt"] == render_prompt(trap_task)
    assert result["model_output"] == "raw model text"
    assert result["oracle_call"] == trap_task.oracle_call
    assert result["habit_call"] == trap_task.habit_call


def test_generate_receives_the_rendered_prompt(app, trap_task):
    seen = []

    def generate(prompt):
        seen.append(prompt)
        return ""

    app.predict_vs_oracle(trap_task, generate)
    assert seen == [render_prompt(trap_task)]


# make_generate: fails if the fake does not look the prompt up or if torch is loaded eagerly.
@pytest.mark.parametrize("kind", ["oracle", "habit"])
def test_fake_generate_returns_the_chosen_call_json(app, tasks, monkeypatch, kind):
    monkeypatch.setenv("DEMO_FAKE", kind)
    monkeypatch.delenv("ADAPTER_REPO", raising=False)
    generate = app.make_generate()
    trap = next(t for t in tasks if t.is_trap)
    expected = trap.oracle_call if kind == "oracle" else trap.habit_call
    assert generate(render_prompt(trap)) == call_to_json(expected)


def test_missing_adapter_repo_is_a_named_config_error_without_torch(app, monkeypatch):
    monkeypatch.delenv("DEMO_FAKE", raising=False)
    monkeypatch.delenv("ADAPTER_REPO", raising=False)
    with pytest.raises(app.ConfigError, match="ADAPTER_REPO"):
        app.make_generate()
    assert "torch" not in sys.modules


def test_unknown_demo_fake_value_is_rejected(app, monkeypatch):
    monkeypatch.setenv("DEMO_FAKE", "banana")
    with pytest.raises(app.ConfigError, match="DEMO_FAKE"):
        app.make_generate()


# answer_request: the free-text rule and the UI error path.
def test_request_text_must_equal_a_demo_task_request(app, tasks):
    task = tasks[0]
    result = app.answer_request(task.request, tasks, lambda p: call_to_json(task.oracle_call))
    assert result["verdict"] == "matches the oracle"
    assert result["tool_doc"] == task.tool.doc
    blocked = app.answer_request("Please do something else", tasks, lambda p: "x")
    assert blocked["verdict"] == "pick one of the demo tasks"
    assert blocked["tool_doc"] == ""


def test_generate_failure_becomes_a_message_not_a_crash(app, tasks):
    def broken(prompt):
        raise app.ConfigError("ADAPTER_REPO is not set")

    result = app.answer_request(tasks[0].request, tasks, broken)
    assert "ADAPTER_REPO is not set" in result["verdict"]
    assert json.loads(result["oracle"]) == tasks[0].oracle_call


# Import hygiene (separate interpreter so other tests cannot pollute sys.modules).
def test_importing_app_pulls_in_neither_gradio_nor_torch():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('space_app', {str(SPACE / 'app.py')!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "bad = [n for n in ('gradio', 'torch', 'transformers', 'peft') if n in sys.modules]\n"
        "print('LOADED', bad)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "LOADED []"


# Deploy files.
def test_requirements_names_the_stack_and_holds_no_secret():
    text = (SPACE / "requirements.txt").read_text(encoding="utf-8")
    names = {re.split(r"[<>=!~ ]", line.strip())[0].lower()
             for line in text.splitlines() if line.strip() and not line.startswith("#")}
    assert names == {"gradio", "transformers", "peft", "torch", "accelerate"}
    assert not re.search(r"hf_[A-Za-z0-9]{8,}|\btokens?\b|api[_-]?key|secret", text, re.IGNORECASE)


def test_readme_front_matter_and_deploy_steps():
    text = (SPACE / "README.md").read_text(encoding="utf-8")
    front = text.split("---")[1]
    assert "sdk: gradio" in front
    assert "app_file: app.py" in front
    assert "python_version: 3.12" in front
    assert "emoji" not in front
    assert "How to deploy (Ritam does this)" in text
    for name in ("ADAPTER_REPO", "HF_TOKEN", "DEMO_FAKE"):
        assert name in text
    assert not re.search(r"hf_[A-Za-z0-9]{8,}", text)
