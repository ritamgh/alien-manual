"""Crash-safe persistence for long jobs on interruptible free GPU sessions (FR-002, E-6, R-4).

Public API:

* ``ProgressStore(path, fingerprint=None)``: append-only JSONL of per-example results,
  fsynced per record, deduplicated by example id. A torn last line (process killed
  mid-write) is ignored and truncated away on the next ``record``; corruption anywhere else
  raises ``CorruptProgressError``. Example ids repeat across seeds, splits and models, so a
  caller that passes a ``fingerprint`` (a string naming the run) gets it stored in the
  sibling file ``<path>.key`` before the first record; reopening a file that holds records
  with a different fingerprint, or with no ``.key`` file (older code), raises
  ``StaleProgressError`` instead of serving another run's results. Without a fingerprint
  nothing is checked and no ``.key`` file is written.
* ``resume_plan(task_ids, store)``: ids still to score, in the original order.
* ``write_checkpoint(root, name, writer)``: write into ``root/name.tmp/``, rename to
  ``root/name/``, then atomically rewrite ``root/MANIFEST.json`` LAST. A crash before the
  manifest leaves the checkpoint unlisted.
* ``load_checkpoints(root, warn)`` / ``latest_checkpoint(root, warn)``: only manifest-listed
  directories that exist. An empty or torn ``MANIFEST.json`` counts as no manifest: nothing is
  listed, nothing is deleted or rewritten, and ``warn(message)`` is called once if given.
* ``upload_with_retry(upload, ...)``: bounded retries with exponential backoff; raises
  ``UploadError``. The upload callable is passed in, so this module imports nothing from
  huggingface and never sees a token.

Manifest shape: ``{"checkpoints": ["<name>", ...], "complete": true}`` (names in write order;
``complete`` is true because only finished checkpoints are ever listed).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

MANIFEST_NAME = "MANIFEST.json"


class CorruptProgressError(ValueError):
    """A progress file has a damaged line that is not the (tolerated) torn last line."""


class StaleProgressError(ValueError):
    """A progress file belongs to a different run, or to unknown older code (no fingerprint)."""


class UploadError(RuntimeError):
    """An upload still failed after the allowed number of attempts."""


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _parse_record(raw: bytes) -> tuple[str, dict]:
    obj = json.loads(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
        raise ValueError("record is not a JSON object with a string id")
    return obj["id"], {k: v for k, v in obj.items() if k != "id"}


class ProgressStore:
    """Append-only JSONL file of per-example results, keyed by example id."""

    def __init__(self, path: str | os.PathLike[str], fingerprint: str | None = None) -> None:
        self._path = Path(path)
        self._key_path = self._path.with_name(self._path.name + ".key")
        self._fingerprint = fingerprint
        self._key_saved = False
        self._results: dict[str, dict] = {}
        # Byte length of the valid prefix, set only when a torn tail must be cut off.
        self._truncate_to: int | None = None
        if fingerprint is not None:
            self._check_fingerprint(fingerprint)
        self._load()

    @property
    def fingerprint(self) -> str | None:
        return self._fingerprint

    def _check_fingerprint(self, fingerprint: str) -> None:
        stored = self._key_path.read_text() if self._key_path.exists() else None
        if stored == fingerprint:
            self._key_saved = True
            return
        if not self._path.exists() or self._path.stat().st_size == 0:
            return  # no records to protect: the key is (re)written with the first record
        if stored is None:
            why = "has no run fingerprint (it was written by older code)"
        else:
            why = "belongs to a different run (seed, split, prompts, model or adapter changed)"
        raise StaleProgressError(
            f"{self._path} {why}; use a fresh --progress-dir or delete this file"
        )

    def _save_key(self) -> None:
        tmp = self._key_path.with_name(self._key_path.name + ".tmp")
        with open(tmp, "w") as handle:
            handle.write(self._fingerprint or "")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._key_path)
        self._key_saved = True

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = self._path.read_bytes()
        lines = data.split(b"\n")
        # data ending in "\n" yields a final empty piece: not a line.
        if lines and lines[-1] == b"":
            lines.pop()
            last_unterminated = False
        else:
            last_unterminated = True
        offset = 0
        for number, raw in enumerate(lines, start=1):
            is_last = number == len(lines)
            torn = is_last and last_unterminated
            try:
                if torn:
                    raise ValueError("unterminated last line")
                example_id, payload = _parse_record(raw)
            except ValueError as exc:
                if not is_last:
                    raise CorruptProgressError(
                        f"{self._path}: corrupted line {number}: {exc}"
                    ) from exc
                self._truncate_to = offset
                return
            self._results.setdefault(example_id, payload)
            offset += len(raw) + 1

    def done_ids(self) -> set[str]:
        return set(self._results)

    def results(self) -> dict[str, dict]:
        """Example id to payload (without the id key), in first-seen order."""
        return {example_id: dict(payload) for example_id, payload in self._results.items()}

    def record(self, example_id: str, payload: dict) -> bool:
        """Append one result durably. Returns False, writing nothing, if already recorded."""
        if example_id in self._results:
            return False
        if "id" in payload:
            raise ValueError("payload must not contain the reserved key 'id'")
        line = json.dumps({"id": example_id, **payload}) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._fingerprint is not None and not self._key_saved:
            self._save_key()  # before the first record, so records never lack their key
        created = not self._path.exists()
        if self._truncate_to is not None:
            with open(self._path, "r+b") as handle:
                handle.truncate(self._truncate_to)
                handle.flush()
                os.fsync(handle.fileno())
            self._truncate_to = None
        with open(self._path, "ab") as handle:
            handle.write(line.encode())
            handle.flush()
            os.fsync(handle.fileno())
        if created:
            _fsync_dir(self._path.parent)
        self._results[example_id] = dict(payload)
        return True


def resume_plan(task_ids: Sequence[str], store: ProgressStore) -> list[str]:
    """Ids not yet scored, in the original order."""
    done = store.done_ids()
    return [task_id for task_id in task_ids if task_id not in done]


def _read_manifest(root: Path, warn: Callable[[str], None] | None = None) -> list[str]:
    """Names listed in ``MANIFEST.json``; ``[]`` if it is missing, empty, torn or malformed."""
    manifest = root / MANIFEST_NAME
    if not manifest.exists():
        return []
    try:
        names = json.loads(manifest.read_text())["checkpoints"]
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise TypeError("'checkpoints' is not a list of names")
    except (ValueError, KeyError, TypeError) as exc:
        if warn is not None:
            warn(f"{manifest}: unreadable ({type(exc).__name__}); ignoring it")
        return []
    return names


def _fsync_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    for sub in [directory, *(p for p in directory.rglob("*") if p.is_dir())]:
        _fsync_dir(sub)


def write_checkpoint(
    root: str | os.PathLike[str], name: str, writer: Callable[[Path], None]
) -> Path:
    """Write checkpoint ``name`` under ``root``; list it in the manifest last."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"invalid checkpoint name: {name!r}")
    if name.endswith(".tmp") or name == MANIFEST_NAME:
        raise ValueError(f"reserved checkpoint name: {name!r}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    listed = _read_manifest(root)
    if name in listed:
        raise FileExistsError(f"checkpoint {name!r} is already complete in {root}")
    tmp_dir = root / f"{name}.tmp"
    final_dir = root / name
    # Leftovers of a killed session: a half-written tmp dir, or a renamed-but-unlisted dir.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(final_dir, ignore_errors=True)
    tmp_dir.mkdir()
    writer(tmp_dir)
    _fsync_tree(tmp_dir)
    os.rename(tmp_dir, final_dir)
    _fsync_dir(root)
    manifest_tmp = root / f"{MANIFEST_NAME}.tmp"
    with open(manifest_tmp, "w") as handle:
        json.dump({"checkpoints": [*listed, name], "complete": True}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(manifest_tmp, root / MANIFEST_NAME)
    _fsync_dir(root)
    return final_dir


def load_checkpoints(
    root: str | os.PathLike[str], warn: Callable[[str], None] | None = None
) -> list[Path]:
    """Manifest-listed checkpoint directories that exist, in write order."""
    root = Path(root)
    paths = (root / name for name in _read_manifest(root, warn))
    return [path for path in paths if path.is_dir()]


def latest_checkpoint(
    root: str | os.PathLike[str], warn: Callable[[str], None] | None = None
) -> Path | None:
    checkpoints = load_checkpoints(root, warn)
    return checkpoints[-1] if checkpoints else None


def upload_with_retry(
    upload: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``upload`` up to ``attempts`` times, sleeping ``base_delay * 2**i`` between.

    The error message carries only the attempt count and the last exception's type name, so
    a token that leaks into an exception text is not copied into logs; the original exception
    stays chained as ``__cause__``.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    for attempt in range(attempts):
        try:
            return upload()
        except Exception as exc:
            if attempt == attempts - 1:
                raise UploadError(
                    f"upload failed after {attempts} attempts ({type(exc).__name__})"
                ) from exc
            sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover
