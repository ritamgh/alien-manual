"""Tests for crash-safe eval progress, checkpoints and upload retries (FR-002, E-6).

Each test group names the production change that would make it fail.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from alien import persist
from alien.persist import (
    CorruptProgressError,
    ProgressStore,
    StaleProgressError,
    UploadError,
    latest_checkpoint,
    load_checkpoints,
    resume_plan,
    upload_with_retry,
    write_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


# --- ProgressStore ---------------------------------------------------------------------------


def test_record_twice_counts_once(tmp_path):
    # Breaks if record() appends without checking the id (double count on re-run).
    path = tmp_path / "progress.jsonl"
    store = ProgressStore(path)
    assert store.record("t1", {"f1": 1.0}) is True
    assert store.record("t1", {"f1": 0.0}) is False
    assert _lines(path) == ['{"id": "t1", "f1": 1.0}']
    assert store.done_ids() == {"t1"}
    assert store.results() == {"t1": {"f1": 1.0}}


def test_results_survive_reopen_in_first_seen_order(tmp_path):
    # Breaks if reload loses order or leaves the "id" key inside the payload.
    path = tmp_path / "progress.jsonl"
    store = ProgressStore(path)
    store.record("b", {"n": 2})
    store.record("a", {"n": 1})
    reopened = ProgressStore(path)
    assert list(reopened.results().items()) == [("b", {"n": 2}), ("a", {"n": 1})]


def test_payload_with_id_key_is_rejected(tmp_path):
    # Breaks if a payload "id" is allowed to overwrite the record id.
    store = ProgressStore(tmp_path / "progress.jsonl")
    with pytest.raises(ValueError, match="id"):
        store.record("t1", {"id": "other"})
    assert store.done_ids() == set()


def test_truncated_last_line_is_ignored_and_repaired(tmp_path):
    # Breaks if a half-written last line crashes loading, or is left in the file so the
    # next appended record is glued onto it.
    path = tmp_path / "progress.jsonl"
    good = ['{"id": "t1", "n": 1}', '{"id": "t2", "n": 2}', '{"id": "t3", "n": 3}']
    path.write_text("\n".join(good) + '\n{"id": "t4", "n"')
    store = ProgressStore(path)
    assert store.done_ids() == {"t1", "t2", "t3"}
    assert store.record("t4", {"n": 4}) is True
    lines = _lines(path)
    assert len(lines) == 4
    assert [json.loads(line) for line in lines] == [
        {"id": "t1", "n": 1},
        {"id": "t2", "n": 2},
        {"id": "t3", "n": 3},
        {"id": "t4", "n": 4},
    ]
    assert ProgressStore(path).done_ids() == {"t1", "t2", "t3", "t4"}


def test_garbage_last_line_with_newline_is_ignored(tmp_path):
    # Breaks if only a missing-newline tail is tolerated and a corrupt final line raises.
    path = tmp_path / "progress.jsonl"
    path.write_text('{"id": "t1", "n": 1}\n\x00\x00garbage\n')
    store = ProgressStore(path)
    assert store.done_ids() == {"t1"}
    store.record("t2", {"n": 2})
    assert _lines(path) == ['{"id": "t1", "n": 1}', '{"id": "t2", "n": 2}']


def test_middle_corruption_names_the_line(tmp_path):
    # Breaks if mid-file damage is silently skipped (would hide lost results).
    path = tmp_path / "progress.jsonl"
    path.write_text('{"id": "t1", "n": 1}\nnot json\n{"id": "t3", "n": 3}\n')
    with pytest.raises(CorruptProgressError, match="line 2"):
        ProgressStore(path)


def test_duplicate_ids_on_disk_first_wins(tmp_path):
    # Breaks if a later duplicate line overwrites the first recorded result.
    path = tmp_path / "progress.jsonl"
    path.write_text('{"id": "t1", "n": 1}\n{"id": "t2", "n": 2}\n{"id": "t1", "n": 99}\n')
    store = ProgressStore(path)
    assert store.results() == {"t1": {"n": 1}, "t2": {"n": 2}}


# --- interruption (real process, SIGKILL) ----------------------------------------------------

ALL_IDS = [f"t{i:02d}" for i in range(12)]
PAYLOADS = {
    "t00": {"valid": True, "f1": 1.0},
    "t01": {"valid": True, "f1": 0.5},
    "t02": {"valid": False, "f1": 0.0},
    "t03": {"valid": True, "f1": 1.0},
    "t04": {"valid": True, "f1": 0.75},
    "t05": {"valid": False, "f1": 0.0},
    "t06": {"valid": True, "f1": 1.0},
    "t07": {"valid": True, "f1": 0.25},
    "t08": {"valid": True, "f1": 1.0},
    "t09": {"valid": False, "f1": 0.0},
    "t10": {"valid": True, "f1": 0.5},
    "t11": {"valid": True, "f1": 1.0},
}

WORKER = textwrap.dedent(
    """
    import json, sys, time
    from alien.persist import ProgressStore

    store = ProgressStore(sys.argv[1])
    payloads = json.loads(sys.argv[2])
    for n, (example_id, payload) in enumerate(payloads.items(), start=1):
        store.record(example_id, payload)
        if n == 5:
            print("ready", flush=True)
        if n >= 5:
            time.sleep(0.5)
    """
)


def test_sigkill_mid_eval_then_resume_loses_and_double_counts_nothing(tmp_path):
    # Breaks if progress is only buffered/flushed at exit, or if resume re-scores done ids.
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    path = tmp_path / "progress.jsonl"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.Popen(
        [sys.executable, str(script), str(path), json.dumps(PAYLOADS)],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.kill()
    finally:
        proc.kill()
        proc.wait()
    assert proc.returncode != 0

    store = ProgressStore(path)
    already = store.done_ids()
    assert ALL_IDS[:5] == [i for i in ALL_IDS if i in already][:5]
    assert len(already) < len(ALL_IDS)

    remaining = resume_plan(ALL_IDS, store)
    assert remaining == [i for i in ALL_IDS if i not in already]
    assert remaining[-1] == "t11"
    for example_id in remaining:
        assert store.record(example_id, PAYLOADS[example_id]) is True

    final = ProgressStore(path)
    assert final.results() == PAYLOADS
    ids_on_disk = [json.loads(line)["id"] for line in _lines(path)]
    assert ids_on_disk == ALL_IDS
    assert resume_plan(ALL_IDS, final) == []


def test_resume_plan_keeps_original_order_and_skips_done(tmp_path):
    # Breaks if the plan is sorted, or includes ids already scored.
    store = ProgressStore(tmp_path / "progress.jsonl")
    store.record("c", {"n": 3})
    store.record("a", {"n": 1})
    assert resume_plan(["d", "c", "b", "a"], store) == ["d", "b"]


# --- checkpoints + manifest ------------------------------------------------------------------


def _writer(text: str):
    def write(directory: Path) -> None:
        (directory / "weights.txt").write_text(text)

    return write


def test_checkpoint_is_listed_after_manifest_written(tmp_path):
    # Breaks if the checkpoint is not renamed into place or not listed in the manifest.
    root = tmp_path / "ckpt"
    out = write_checkpoint(root, "sft", _writer("w1"))
    assert out == root / "sft"
    assert (out / "weights.txt").read_text() == "w1"
    assert json.loads((root / "MANIFEST.json").read_text()) == {
        "checkpoints": ["sft"],
        "complete": True,
    }
    assert load_checkpoints(root) == [root / "sft"]
    assert latest_checkpoint(root) == root / "sft"
    assert not (root / "sft.tmp").exists()


def test_crash_before_manifest_leaves_checkpoint_unlisted(tmp_path, monkeypatch):
    # Breaks if the manifest is written before the checkpoint dir is complete, or if
    # loading trusts directories that the manifest does not list.
    root = tmp_path / "ckpt"
    write_checkpoint(root, "sft", _writer("w1"))

    def crash(src, dst):
        raise KeyboardInterrupt("session killed before manifest replace")

    monkeypatch.setattr(persist.os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        write_checkpoint(root, "grpo", _writer("w2"))
    monkeypatch.undo()

    assert (root / "grpo" / "weights.txt").read_text() == "w2"
    assert load_checkpoints(root) == [root / "sft"]
    assert latest_checkpoint(root) == root / "sft"
    assert json.loads((root / "MANIFEST.json").read_text())["checkpoints"] == ["sft"]


def test_rewriting_an_unlisted_checkpoint_after_crash_succeeds(tmp_path, monkeypatch):
    # Breaks if a stale unlisted dir blocks the restart from ever re-writing that name.
    root = tmp_path / "ckpt"
    monkeypatch.setattr(persist.os, "replace", lambda s, d: (_ for _ in ()).throw(OSError))
    with pytest.raises(OSError):
        write_checkpoint(root, "sft", _writer("old"))
    monkeypatch.undo()
    write_checkpoint(root, "sft", _writer("new"))
    assert (root / "sft" / "weights.txt").read_text() == "new"
    assert load_checkpoints(root) == [root / "sft"]


def test_leftover_tmp_dir_and_missing_manifest_are_ignored(tmp_path):
    # Breaks if *.tmp dirs or a missing manifest are treated as loadable checkpoints.
    root = tmp_path / "ckpt"
    (root / "half.tmp").mkdir(parents=True)
    (root / "orphan").mkdir()
    assert load_checkpoints(root) == []
    assert latest_checkpoint(root) is None
    write_checkpoint(root, "sft", _writer("w1"))
    assert load_checkpoints(root) == [root / "sft"]


def test_leftover_tmp_dir_does_not_leak_into_new_checkpoint(tmp_path):
    # Breaks if a half-written sft.tmp from a killed session is reused as-is.
    root = tmp_path / "ckpt"
    (root / "sft.tmp").mkdir(parents=True)
    (root / "sft.tmp" / "stale.txt").write_text("stale")
    out = write_checkpoint(root, "sft", _writer("w1"))
    assert sorted(p.name for p in out.iterdir()) == ["weights.txt"]


def test_two_checkpoints_load_in_order_and_latest_is_last(tmp_path):
    # Breaks if checkpoints are sorted by name or the manifest is overwritten, not extended.
    root = tmp_path / "ckpt"
    write_checkpoint(root, "stage2", _writer("a"))
    write_checkpoint(root, "stage1", _writer("b"))
    assert load_checkpoints(root) == [root / "stage2", root / "stage1"]
    assert latest_checkpoint(root) == root / "stage1"


def test_listed_checkpoint_missing_on_disk_is_skipped(tmp_path):
    # Breaks if a manifest entry whose directory vanished is returned.
    root = tmp_path / "ckpt"
    write_checkpoint(root, "a", _writer("1"))
    write_checkpoint(root, "b", _writer("2"))
    os.rename(root / "b", root / "b-moved")
    assert load_checkpoints(root) == [root / "a"]


def test_rewriting_a_listed_checkpoint_name_is_refused(tmp_path):
    # Breaks if a completed, listed checkpoint can be silently overwritten.
    root = tmp_path / "ckpt"
    write_checkpoint(root, "sft", _writer("w1"))
    with pytest.raises(FileExistsError):
        write_checkpoint(root, "sft", _writer("w2"))
    assert (root / "sft" / "weights.txt").read_text() == "w1"


@pytest.mark.parametrize("name", ["", "a/b", "x.tmp", "MANIFEST.json", ".."])
def test_bad_checkpoint_names_are_rejected(tmp_path, name):
    # Breaks if a name can escape root or collide with the tmp/manifest conventions.
    with pytest.raises(ValueError):
        write_checkpoint(tmp_path / "ckpt", name, _writer("w"))


# --- upload_with_retry -----------------------------------------------------------------------


def test_flaky_upload_succeeds_with_exponential_backoff():
    # Breaks if the delay is constant, not doubled, or sleeps after the successful try.
    calls = []
    sleeps = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("hub unreachable")
        return "uploaded"

    assert upload_with_retry(flaky, sleep=sleeps.append) == "uploaded"
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]


def test_always_failing_upload_raises_after_exactly_attempts():
    # Breaks if retries are unbounded, off by one, or the cause is not chained.
    calls = []
    sleeps = []

    def broken():
        calls.append(1)
        raise TimeoutError("secret-looking-detail hf_abc123")

    with pytest.raises(UploadError) as info:
        upload_with_retry(broken, attempts=4, base_delay=1.0, sleep=sleeps.append)
    assert len(calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]
    assert "4 attempts" in str(info.value)
    assert isinstance(info.value.__cause__, TimeoutError)
    assert "hf_abc123" not in str(info.value)


def test_upload_attempts_below_one_is_rejected():
    # Breaks if attempts=0 silently returns nothing or loops forever.
    with pytest.raises(ValueError):
        upload_with_retry(lambda: "x", attempts=0, sleep=lambda s: None)


def test_upload_first_try_never_sleeps():
    # Breaks if a sleep happens before the first attempt.
    sleeps = []
    assert upload_with_retry(lambda: 7, sleep=sleeps.append) == 7
    assert sleeps == []


# --- ProgressStore run fingerprint (F-5) -----------------------------------------------------


def test_fingerprinted_store_resumes_when_the_fingerprint_matches(tmp_path):
    # Breaks if the stored fingerprint is not read back, so every restart looks like a new run.
    path = tmp_path / "p" / "base.jsonl"
    store = ProgressStore(path, fingerprint="abc")
    assert not (tmp_path / "p").exists()  # opening writes nothing until the first record
    store.record("t1", {"output": "x"})
    assert (tmp_path / "p" / "base.jsonl.key").read_text() == "abc"
    again = ProgressStore(path, fingerprint="abc")
    assert again.results() == {"t1": {"output": "x"}}
    assert again.fingerprint == "abc"


def test_fingerprinted_store_refuses_progress_of_another_run(tmp_path):
    # Breaks if the same example ids from a different run are served as this run's results.
    path = tmp_path / "base.jsonl"
    ProgressStore(path, fingerprint="abc").record("t1", {"output": "x"})
    with pytest.raises(StaleProgressError) as info:
        ProgressStore(path, fingerprint="xyz")
    assert str(path) in str(info.value)
    assert "use a fresh --progress-dir or delete this file" in str(info.value)
    assert ProgressStore(path).results() == {"t1": {"output": "x"}}  # nothing was touched


def test_fingerprinted_store_refuses_progress_without_a_key_file(tmp_path):
    # Breaks if a file from older code (no fingerprint) is trusted for a new run.
    path = tmp_path / "base.jsonl"
    ProgressStore(path).record("t1", {"output": "x"})
    with pytest.raises(StaleProgressError, match="no run fingerprint"):
        ProgressStore(path, fingerprint="abc")


def test_store_without_a_fingerprint_keeps_the_old_behaviour(tmp_path):
    # Breaks if plain stores (build_report reads them) start writing key files or refusing.
    path = tmp_path / "base.jsonl"
    ProgressStore(path).record("t1", {"output": "x"})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base.jsonl"]
    assert ProgressStore(path).done_ids() == {"t1"}


def test_deleted_progress_file_with_a_leftover_key_starts_the_new_run_cleanly(tmp_path):
    # Breaks if a key left behind after deleting only the .jsonl blocks the new run forever.
    path = tmp_path / "base.jsonl"
    ProgressStore(path, fingerprint="abc").record("t1", {"output": "x"})
    path.unlink()
    fresh = ProgressStore(path, fingerprint="xyz")
    fresh.record("t9", {"output": "y"})
    assert (tmp_path / "base.jsonl.key").read_text() == "xyz"
    assert ProgressStore(path, fingerprint="xyz").done_ids() == {"t9"}


# --- torn MANIFEST.json (F-4) ----------------------------------------------------------------


@pytest.mark.parametrize("content", ["", "{", '{"checkpoints": ["step-100", "ste'])
def test_torn_or_empty_manifest_is_no_usable_manifest_and_nothing_is_touched(tmp_path, content):
    # Breaks if a half-synced MANIFEST.json crashes the restart with a raw JSONDecodeError, or
    # if reading it deletes or rewrites anything.
    root = tmp_path / "ckpt"
    (root / "step-100").mkdir(parents=True)
    (root / "step-100" / "weights.txt").write_text("w")
    (root / "MANIFEST.json").write_text(content)
    assert latest_checkpoint(root) is None
    assert load_checkpoints(root) == []
    assert (root / "MANIFEST.json").read_text() == content
    assert (root / "step-100" / "weights.txt").read_text() == "w"
    assert sorted(p.name for p in root.iterdir()) == ["MANIFEST.json", "step-100"]


def test_torn_manifest_is_reported_through_the_warn_callback(tmp_path):
    # Breaks if the problem is swallowed with no way for the caller to tell the user.
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "MANIFEST.json").write_text("{")
    messages = []
    assert latest_checkpoint(root, warn=messages.append) is None
    assert len(messages) == 1
    assert str(root / "MANIFEST.json") in messages[0]
    assert "ignoring" in messages[0]


def test_manifest_with_one_good_and_one_missing_directory_returns_the_good_one(tmp_path):
    # Breaks if a missing listed directory hides the good checkpoint listed before it.
    root = tmp_path / "ckpt"
    (root / "step-50").mkdir(parents=True)
    (root / "MANIFEST.json").write_text(
        json.dumps({"checkpoints": ["step-50", "step-100"], "complete": True}))
    assert load_checkpoints(root) == [root / "step-50"]
    assert latest_checkpoint(root) == root / "step-50"
