"""Tests for the base-model development probe and its CLI (PRD S1, E-1, R-1, B-2, B-6).

Everything runs on fake deterministic generators: no torch, no GPU, no download. Each test
names the production change that would make it fail.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import check_results
import probe_base
from alien.eval import build_prompts
from alien.oracle import call_to_json
from alien.persist import StaleProgressError
from alien.splits import make_splits

REPO = Path(__file__).resolve().parent.parent
SCRIPT = str(REPO / "scripts" / "probe_base.py")


@pytest.fixture(scope="module")
def splits():
    return make_splits(0)


@pytest.fixture(scope="module")
def dev(splits):
    return splits.dev


class Interrupted(Exception):
    """Stands in for a free Colab session dying mid-run."""


def metrics_with_rate(rate):
    def metric(value):
        return {"value": value, "ci_low": 0.0, "ci_high": 1.0, "n": 79}
    return {"prior_trap_rate": metric(rate), "tool_call_f1": metric(0.5),
            "tool_call_validity": metric(1.0), "normal_tool_f1": metric(0.5)}


def run_main(tmp_path, *extra, name="base_probe.json"):
    out = tmp_path / name
    code = probe_base.main(["--out", str(out), "--progress-dir", str(tmp_path / "progress"),
                            "--n-resamples", "50", *extra])
    return code, out


def test_probe_never_reads_the_heldout_split(monkeypatch, splits, tmp_path):
    # Breaks if main/run_probe touch splits.heldout (B-2/B-6): the sentinel raises on access.
    class DevOnly:
        dev = splits.dev

        @property
        def heldout(self):
            raise AssertionError("the probe read splits.heldout")

    monkeypatch.setattr(probe_base, "make_splits", lambda seed, hardness=0: DevOnly())
    code, out = run_main(tmp_path, "--fake", "habit")
    assert code == 0
    assert json.loads(out.read_text())["split"] == "dev"


def test_habit_model_is_fully_trapped_and_the_example_is_its_habit_call(dev, tmp_path, capsys):
    # Breaks if prior_trap_rate, headroom or the quoted example are wired to the wrong thing.
    code, out = run_main(tmp_path, "--fake", "habit")
    doc = json.loads(out.read_text())
    assert code == 0
    assert doc["kind"] == "base_probe"
    assert doc["n_examples"] == 100
    assert doc["prior_trap_rate"] == 1.0
    assert doc["metrics"]["prior_trap_rate"]["n"] == 79
    assert doc["headroom"] == "ok"
    example = doc["trap_example"]
    assert example["task_id"] == "dev-0001"
    assert example["request"] == "Look up the record for pallet P-83."
    assert example["model_output"] == '{"name": "read_runru", "arguments": {"item": "P-83"}}'
    assert example["habit_call"] == {"name": "read_runru", "arguments": {"item": "P-83"}}
    assert example["oracle_call"] == {"name": "grab_runru", "arguments": {"item": "P-83"}}
    assert check_results.validate(doc) == []
    shown = capsys.readouterr().out
    assert "Prior-trap rate on 79 dev trap tasks: 100.0%" in shown
    assert "request: Look up the record for pallet P-83." in shown
    assert "followed its habit instead of the doc" in shown
    assert "Always call `grab_runru`" in shown
    assert "rerun with --hardness 1" not in shown


def test_oracle_model_has_low_headroom_and_cli_still_exits_zero(tmp_path, capsys):
    # Breaks if low headroom is treated as an error or the E-1 instruction is not printed.
    code, out = run_main(tmp_path, "--fake", "oracle")
    doc = json.loads(out.read_text())
    assert code == 0
    assert doc["prior_trap_rate"] == 0.0
    assert doc["headroom"] == "low"
    assert doc["trap_example"]["task_id"] == "dev-0001"
    assert doc["trap_example"]["model_output"] == (
        '{"name": "grab_runru", "arguments": {"item": "P-83"}}')
    assert check_results.validate(doc) == []
    shown = capsys.readouterr().out
    assert ("Base model already follows the docs: rerun with --hardness 1 (harder twists) "
            "BEFORE freezing the held-out split; do not move the core to Gremlin Gym "
            "without Ritam's yes (B-6)") in shown
    assert "followed the doc" in shown


def test_headroom_threshold_is_inclusive_at_ten_percent(dev):
    # Breaks if the comparison becomes > instead of >= (check_results uses >=).
    outputs = [call_to_json(t.habit_call) for t in dev]
    def headroom(rate):
        doc = probe_base.build_probe_results(
            dev, outputs, metrics_with_rate(rate), seed=0, model="m", model_revision="r")
        return doc["headroom"], doc["prior_trap_rate"]
    assert headroom(0.10) == ("ok", 0.10)
    assert headroom(0.0999) == ("low", 0.0999)


@pytest.mark.parametrize(("rate", "hardness", "expected"), [
    (0.0, 0, 1),
    (0.0999, 1, 2),
    (0.0, 2, 3),
    (0.10, 0, None),
    (0.8, 1, None),
    (0.0, 3, None),
])
def test_suggest_next_hardness(rate, hardness, expected):
    # Breaks if the threshold, the +1 step or the cap at hardness 3 changes.
    assert probe_base.suggest_next_hardness(rate, hardness) == expected


def test_capped_low_headroom_asks_the_user(dev, capsys):
    # Breaks if the cap case prints no instruction to ask Ritam.
    outputs = [call_to_json(t.oracle_call) for t in dev]
    doc = probe_base.build_probe_results(
        dev, outputs, metrics_with_rate(0.0), seed=0, model="m", model_revision="r")
    text = probe_base.format_verdict(doc, dev, hardness=3)
    assert "Hardness 3 is the highest level" in text
    assert "Hardness 3 is the highest level" not in probe_base.format_verdict(doc, dev, 2)


def test_example_prefers_prior_trap_then_wrong_then_first_trap(dev):
    # Breaks if the fallback order of the quoted example changes.
    traps = [t for t in dev if t.is_trap]
    first, second, third = traps[0], traps[1], traps[2]
    by_id = {t.id: call_to_json(t.oracle_call) for t in dev}
    by_id[second.id] = "I cannot help with that."
    by_id[third.id] = call_to_json(third.habit_call)
    outputs = [by_id[t.id] for t in dev]
    assert probe_base.pick_trap_example(dev, outputs)["task_id"] == third.id
    by_id[third.id] = call_to_json(third.oracle_call)
    by_id[second.id] = ""
    outputs = [by_id[t.id] for t in dev]
    example = probe_base.pick_trap_example(dev, outputs)
    assert example["task_id"] == second.id
    assert example["model_output"] == "<empty output>"
    outputs = [call_to_json(t.oracle_call) for t in dev]
    assert probe_base.pick_trap_example(dev, outputs)["task_id"] == first.id


def test_example_ignores_non_trap_tasks_even_when_they_are_wrong(dev):
    # Breaks if a wrong non-trap answer is quoted as the trap example.
    normal = next(t for t in dev if not t.is_trap)
    outputs = [call_to_json(t.oracle_call) for t in dev]
    outputs[dev.index(normal)] = "garbage"
    first_trap = next(t for t in dev if t.is_trap)
    assert probe_base.pick_trap_example(dev, outputs)["task_id"] == first_trap.id


def test_interrupted_run_resumes_without_repeating_or_duplicating(dev, tmp_path):
    # Breaks if progress is not recorded per batch or a rerun regenerates finished tasks.
    answers = {}
    for task in dev:
        answers[build_prompts([task], "base")[0]] = call_to_json(
            task.habit_call)
    seen = []

    def flaky(prompts):
        if len(seen) >= 2:
            raise Interrupted
        seen.append(list(prompts))
        return [answers[p] for p in prompts]

    progress = tmp_path / "progress.jsonl"
    with pytest.raises(Interrupted):
        probe_base.run_probe(dev, flaky, progress, n_resamples=20, batch_size=10)
    assert len(progress.read_text().splitlines()) == 20

    rerun = []

    def healthy(prompts):
        rerun.append(list(prompts))
        return [answers[p] for p in prompts]

    outputs, metrics = probe_base.run_probe(dev, healthy, progress, n_resamples=20, batch_size=10)
    assert sum(len(batch) for batch in rerun) == 80
    assert outputs == [call_to_json(t.habit_call) for t in dev]
    assert metrics["prior_trap_rate"]["value"] == 1.0
    ids = [json.loads(line)["id"] for line in progress.read_text().splitlines()]
    assert len(ids) == len(set(ids)) == 100


def test_hardness_rerun_does_not_reuse_progress_of_another_hardness(tmp_path):
    # Breaks if the progress file name ignores hardness: task ids repeat across hardness levels.
    progress = tmp_path / "progress"
    for hardness in ("0", "1"):
        code = probe_base.main(["--fake", "habit", "--hardness", hardness, "--n-resamples", "20",
                                "--out", str(tmp_path / f"out{hardness}.json"),
                                "--progress-dir", str(progress)])
        assert code == 0
    assert sorted(p.name for p in progress.iterdir() if p.suffix == ".jsonl") == [
        "fake-habit_seed0_h0.jsonl", "fake-habit_seed0_h1.jsonl"]


def test_cli_output_is_accepted_by_check_results_script(tmp_path):
    # Breaks if the written file is not what `python scripts/check_results.py` accepts.
    out = tmp_path / "base_probe.json"
    done = subprocess.run(
        [sys.executable, SCRIPT, "--fake", "habit", "--n-resamples", "50", "--out", str(out),
         "--progress-dir", str(tmp_path / "progress")],
        capture_output=True, text=True, timeout=120, cwd=REPO)
    assert done.returncode == 0, done.stderr
    checked = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_results.py"), str(out)],
        capture_output=True, text=True, timeout=60, cwd=REPO)
    assert checked.returncode == 0, checked.stdout
    assert checked.stdout.strip() == "OK base_probe"


def test_help_exits_zero_without_torch():
    # Breaks if torch/transformers are imported at module level.
    done = subprocess.run([sys.executable, SCRIPT, "--help"], capture_output=True, text=True,
                          timeout=60, cwd=REPO)
    assert done.returncode == 0
    for flag in ("--seed", "--hardness", "--model-id", "--revision", "--batch-size",
                 "--progress-dir", "--out", "--fake", "--n-resamples"):
        assert flag in done.stdout


def test_progress_of_another_revision_is_refused_with_exit_2_and_no_traceback(tmp_path, capsys):
    # Breaks if the probe keys its progress file only by seed and hardness: a second model
    # revision would silently reuse the first one's outputs.
    code, _ = run_main(tmp_path, "--fake", "habit", "--revision", "r1")
    assert code == 0
    progress_file = tmp_path / "progress" / "fake-habit_seed0_h0.jsonl"
    before = progress_file.read_text()
    capsys.readouterr()
    code, out = run_main(tmp_path, "--fake", "habit", "--revision", "r2", name="second.json")
    assert code == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "fresh --progress-dir" in err
    assert str(progress_file) in err
    assert "Traceback" not in err
    assert progress_file.read_text() == before


def test_same_revision_rerun_still_resumes_without_generating(tmp_path):
    # Breaks if the run key includes something that changes between sessions (a timestamp).
    assert run_main(tmp_path, "--fake", "habit", "--revision", "r1")[0] == 0
    assert run_main(tmp_path, "--fake", "habit", "--revision", "r1", name="again.json")[0] == 0


def test_run_probe_refuses_a_different_run_key_and_resumes_the_same_one(dev, tmp_path):
    # Breaks if run_probe drops the run key, or refuses a matching one.
    answers = {build_prompts([t], "base")[0]: call_to_json(t.habit_call) for t in dev}
    calls = []

    def generate(prompts):
        calls.append(len(prompts))
        return [answers[p] for p in prompts]

    progress = tmp_path / "progress.jsonl"
    probe_base.run_probe(dev, generate, progress, n_resamples=20, run_key="m@r1")
    assert sum(calls) == 100
    probe_base.run_probe(dev, generate, progress, n_resamples=20, run_key="m@r1")
    assert sum(calls) == 100
    with pytest.raises(StaleProgressError, match="fresh --progress-dir"):
        probe_base.run_probe(dev, generate, progress, n_resamples=20, run_key="m@r2")
    assert sum(calls) == 100
