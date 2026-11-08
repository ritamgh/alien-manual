"""Validate a results JSON file against the shared results schema (PRD FR-006, T-009).

Usage: python scripts/check_results.py results/<kind>.json

Exits 0 and prints "OK <kind>" when the file is valid. Otherwise exits 1 and prints one
problem per line, each starting with the dotted field path and saying what was expected.
Stdlib only, so it runs on a bare Colab runtime and on the laptop alike.
"""

import json
import math
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

KINDS = ("base_probe", "eval_baselines", "sft", "grpo", "rft", "spike", "final")
METRIC_KEYS = ("prior_trap_rate", "tool_call_f1", "tool_call_validity", "normal_tool_f1")
HEADROOM_THRESHOLD = 0.10
BUDGET_GPU_SECONDS = 10800
MAX_NEW_TOKENS = 256
SC001_GAP = 0.10
EPS = 1e-9
SECRET_TEXT = re.compile(
    r"hf_[A-Za-z0-9_]{20,}"  # Hugging Face token
    r"|(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"  # sk- API key; "task-..." is not one
    r"|Bearer [A-Za-z0-9._~+/=-]{20,}"  # Authorization header value
)
SECRET_KEY = re.compile(r"(authorization|api[_-]?key|token)$", re.IGNORECASE)

_MISSING = object()


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _show(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, str) and SECRET_TEXT.search(value):
        return "<redacted>"
    try:
        return repr(value)
    except ValueError:  # an int of more than 4300 digits has no repr
        return "<value too large to show>"
    except RecursionError:
        return "<too deeply nested to show>"


def _is_num(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int beyond float range, e.g. 10**400
        return False


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _get(obj: dict, key: str, path: str, errs: list[str]) -> Any:
    """Return obj[key], or record '<path>.<key>: missing' and return _MISSING."""
    if key not in obj:
        errs.append(f"{_join(path, key)}: missing")
        return _MISSING
    return obj[key]


def _get_obj(obj: dict, key: str, path: str, errs: list[str]) -> dict | None:
    value = _get(obj, key, path, errs)
    if value is _MISSING:
        return None
    if not isinstance(value, dict):
        errs.append(f"{_join(path, key)}: expected an object, got {type(value).__name__}")
        return None
    return value


def _check_text(obj: dict, key: str, path: str, errs: list[str]) -> str | None:
    value = _get(obj, key, path, errs)
    if value is _MISSING:
        return None
    if not _is_text(value):
        errs.append(f"{_join(path, key)}: expected a non-empty string, got {_show(value)}")
        return None
    return value


def _check_bool(obj: dict, key: str, path: str, errs: list[str]) -> bool | None:
    value = _get(obj, key, path, errs)
    if value is _MISSING:
        return None
    if not isinstance(value, bool):
        errs.append(f"{_join(path, key)}: expected a boolean, got {_show(value)}")
        return None
    return value


def _check_int(obj: dict, key: str, path: str, errs: list[str], minimum: int) -> int | None:
    value = _get(obj, key, path, errs)
    if value is _MISSING:
        return None
    if not _is_int(value) or value < minimum:
        errs.append(f"{_join(path, key)}: expected an integer >= {minimum}, got {_show(value)}")
        return None
    return value


def _check_number(
    obj: dict, key: str, path: str, errs: list[str], minimum: float | None = None
) -> float | None:
    value = _get(obj, key, path, errs)
    if value is _MISSING:
        return None
    if not _is_num(value) or (minimum is not None and value < minimum):
        floor = "" if minimum is None else f" >= {minimum:g}"
        errs.append(f"{_join(path, key)}: expected a number{floor}, got {_show(value)}")
        return None
    return value


def _check_equal(obj: dict, key: str, path: str, errs: list[str], expected: Any) -> None:
    value = _get(obj, key, path, errs)
    if value is not _MISSING and value != expected:
        errs.append(f"{_join(path, key)}: expected {_show(expected)}, got {_show(value)}")


def _check_unit(obj: dict, key: str, path: str, errs: list[str]) -> float | None:
    """A float in [0, 1]. NaN or null means a metric had no eligible examples (E-5)."""
    value = _get(obj, key, path, errs)
    if value is _MISSING:
        return None
    where = _join(path, key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        errs.append(
            f"{where}: expected a number in [0, 1], got {_show(value)} "
            "(zero eligible examples, E-5: the metric must not be reported as 0 or NaN)"
        )
        return None
    if not _is_num(value) or not 0 <= value <= 1:
        errs.append(f"{where}: expected a number in [0, 1], got {_show(value)}")
        return None
    return value


def _check_interval(
    obj: dict, key: str, path: str, errs: list[str], low_name: str, high_name: str
) -> None:
    """Check obj[low_name] <= obj[key] <= obj[high_name] once all three are numbers."""
    low, mid, high = obj.get(low_name), obj.get(key), obj.get(high_name)
    if all(_is_num(v) for v in (low, mid, high)) and not low - EPS <= mid <= high + EPS:
        errs.append(
            f"{path}: expected {low_name} <= {key} <= {high_name}, got {low}, {mid}, {high}"
        )


def _check_metric(metric: Any, path: str, errs: list[str]) -> None:
    if not isinstance(metric, dict):
        errs.append(f"{path}: expected an object, got {type(metric).__name__}")
        return
    for key in ("value", "ci_low", "ci_high"):
        _check_unit(metric, key, path, errs)
    _check_interval(metric, "value", path, errs, "ci_low", "ci_high")
    _check_int(metric, "n", path, errs, 1)


def _check_metrics(metrics: Any, path: str, errs: list[str]) -> None:
    if not isinstance(metrics, dict):
        errs.append(f"{path}: expected an object, got {type(metrics).__name__}")
        return
    for key in metrics:
        if key not in METRIC_KEYS:
            errs.append(f"{path}: unexpected key {key!r}, expected only {', '.join(METRIC_KEYS)}")
    for key in METRIC_KEYS:
        value = _get(metrics, key, path, errs)
        if value is not _MISSING:
            _check_metric(value, _join(path, key), errs)


def _check_models(data: dict, names: Sequence[str], errs: list[str]) -> None:
    models = _get_obj(data, "models", "", errs)
    if models is None:
        return
    for name in names:
        value = _get(models, name, "models", errs)
        if value is not _MISSING:
            _check_metrics(value, _join("models", name), errs)


def _check_checkpoint(data: dict, errs: list[str]) -> None:
    checkpoint = _get_obj(data, "checkpoint", "", errs)
    if checkpoint is None:
        return
    _check_text(checkpoint, "hub_repo", "checkpoint", errs)
    revision = _get(checkpoint, "revision", "checkpoint", errs)
    if revision is not _MISSING and revision is not None and not _is_text(revision):
        errs.append(f"checkpoint.revision: expected a string or null, got {_show(revision)}")


def _check_envelope(data: dict, errs: list[str]) -> None:
    version = _get(data, "schema_version", "", errs)
    if version is not _MISSING and (version != 1 or isinstance(version, bool)):
        errs.append(f"schema_version: expected 1, got {_show(version)}")
    created = _get(data, "created_at", "", errs)
    if created is not _MISSING:
        try:
            if not isinstance(created, str):
                raise ValueError(created)
            datetime.fromisoformat(created[:-1] + "+00:00" if created.endswith("Z") else created)
        except ValueError:
            errs.append(
                f"created_at: expected an ISO8601 timestamp such as 2026-11-07T18:30:00Z, "
                f"got {_show(created)}"
            )
    sha = _get(data, "git_sha", "", errs)
    if sha is not _MISSING and sha is not None and not _is_text(sha):
        errs.append(f"git_sha: expected a string or null, got {_show(sha)}")
    seed = _get(data, "seed", "", errs)
    if seed is not _MISSING and not _is_int(seed):
        errs.append(f"seed: expected an integer, got {_show(seed)}")


def _check_base_probe(data: dict, errs: list[str]) -> None:
    _check_equal(data, "n_examples", "", errs, 100)
    _check_equal(data, "split", "", errs, "dev")
    _check_text(data, "model", "", errs)
    _check_text(data, "model_revision", "", errs)
    metrics = _get(data, "metrics", "", errs)
    if metrics is not _MISSING:
        _check_metrics(metrics, "metrics", errs)
    rate = _check_unit(data, "prior_trap_rate", "", errs)
    inner = metrics.get("prior_trap_rate") if isinstance(metrics, dict) else None
    inner_value = inner.get("value") if isinstance(inner, dict) else None
    if rate is not None and _is_num(inner_value) and abs(rate - inner_value) > EPS:
        errs.append(
            f"prior_trap_rate: expected {_show(inner_value)} "
            f"(metrics.prior_trap_rate.value), got {_show(rate)}"
        )
    headroom = _get(data, "headroom", "", errs)
    if headroom is not _MISSING and rate is not None:
        want, op = ("ok", ">=") if rate >= HEADROOM_THRESHOLD else ("low", "<")
        if headroom != want:
            errs.append(
                f"headroom: expected {want!r} (prior_trap_rate {rate} {op} "
                f"{HEADROOM_THRESHOLD:.2f}), got {_show(headroom)}"
            )
    example = _get_obj(data, "trap_example", "", errs)
    if example is not None:
        for key in ("task_id", "request", "model_output"):
            _check_text(example, key, "trap_example", errs)
        for key in ("habit_call", "oracle_call"):
            value = _get(example, key, "trap_example", errs)
            usable = _is_text(value) or (isinstance(value, dict) and value)
            if value is not _MISSING and not usable:
                errs.append(
                    f"trap_example.{key}: expected a non-empty string or object, "
                    f"got {_show(value)}"
                )


def _check_eval_baselines(data: dict, errs: list[str]) -> None:
    _check_equal(data, "n_examples", "", errs, 200)
    _check_equal(data, "split", "", errs, "heldout")
    _check_text(data, "model_revision", "", errs)
    decoding = _get_obj(data, "decoding", "", errs)
    if decoding is not None:
        tokens = _get(decoding, "max_new_tokens", "decoding", errs)
        if tokens is not _MISSING:
            if not _is_int(tokens) or tokens < 1:
                errs.append(
                    f"decoding.max_new_tokens: expected a positive integer, got {_show(tokens)}"
                )
            elif tokens > MAX_NEW_TOKENS:
                errs.append(
                    f"decoding.max_new_tokens: expected an integer <= {MAX_NEW_TOKENS}, "
                    f"got {_show(tokens)}"
                )
        _check_number(decoding, "temperature", "decoding", errs, 0)
    _check_models(data, ("base", "prompt_only"), errs)
    seconds = _check_number(data, "gpu_seconds", "", errs, 0)
    budget = _check_bool(data, "within_budget", "", errs)
    if seconds is not None and budget is not None:
        under = seconds < BUDGET_GPU_SECONDS
        if budget != under:
            op = "<" if under else ">="
            errs.append(
                f"within_budget: expected {str(under).lower()} "
                f"(gpu_seconds {seconds} {op} {BUDGET_GPU_SECONDS}), got {str(budget).lower()}"
            )


def _check_sft(data: dict, errs: list[str]) -> None:
    _check_text(data, "model_revision", "", errs)
    _check_checkpoint(data, errs)
    first = _check_number(data, "loss_first", "", errs)
    last = _check_number(data, "loss_last", "", errs)
    if first is not None and last is not None and not last < first:
        errs.append(f"loss_last: expected a number < loss_first ({first}), got {last}")
    _check_int(data, "steps", "", errs, 1)
    _check_models(data, ("base", "prompt_only", "sft"), errs)


def _check_rl(data: dict, errs: list[str]) -> None:
    """Shared by grpo.json and rft.json; both record which path actually ran."""
    path = _get(data, "path", "", errs)
    if path is not _MISSING:
        if data.get("kind") == "rft" and path != "rft":
            errs.append(f"path: expected 'rft' for a results/rft.json file, got {_show(path)}")
        elif path not in ("grpo", "rft"):
            errs.append(f"path: expected one of 'grpo', 'rft', got {_show(path)}")
    _check_checkpoint(data, errs)
    generations = _check_int(data, "generations_per_prompt", "", errs, 1)
    if path == "grpo" and generations is not None and generations < 4:
        errs.append(
            "generations_per_prompt: expected an integer >= 4 for path grpo (FR-003), "
            f"got {_show(generations)}"
        )
    signal = _get_obj(data, "reward_signal", "", errs)
    if signal is not None:
        total = _check_int(signal, "groups_total", "reward_signal", errs, 0)
        variance = _check_int(signal, "groups_with_variance", "reward_signal", errs, 0)
        if variance is not None and total is not None and variance > total:
            errs.append(
                "reward_signal.groups_with_variance: expected an integer between 0 and "
                f"groups_total ({_show(total)}), got {_show(variance)}"
            )
        elif variance == 0 and path == "grpo":
            errs.append(
                "reward_signal.groups_with_variance: expected >= 1 for path grpo, got 0: "
                "every group had zero reward variance, so the file must say path rft (E-2)"
            )
    _check_models(data, ("base", "prompt_only", "sft", "tuned"), errs)


def _check_spike(data: dict, errs: list[str]) -> None:
    _check_bool(data, "install_ok", "", errs)
    _check_int(data, "sft_steps", "", errs, 1)
    if _check_bool(data, "fp16_nan_loss", "", errs):
        errs.append("fp16_nan_loss: spike failed: NaN loss in fp16, see AS-2 fallback")
    if _check_bool(data, "grpo_step_ok", "", errs) is False:
        errs.append("grpo_step_ok: spike failed: the GRPO step did not run (AS-3 fallback)")
    _check_bool(data, "reward_sees_full_conversation", "", errs)
    _check_text(data, "completions_0", "", errs)
    keys = _get(data, "kwargs_keys", "", errs)
    if keys is not _MISSING:
        if not isinstance(keys, list) or not keys:
            errs.append(f"kwargs_keys: expected a non-empty list of strings, got {_show(keys)}")
        else:
            for i, key in enumerate(keys):
                if not _is_text(key):
                    errs.append(f"kwargs_keys[{i}]: expected a non-empty string, got {_show(key)}")
    _check_text(data, "python_version", "", errs)
    versions = _get_obj(data, "package_versions", "", errs)
    for name, version in (versions or {}).items():
        if not isinstance(version, str):
            errs.append(f"package_versions.{name}: expected a string, got {_show(version)}")


def _check_final(data: dict, errs: list[str]) -> None:
    _check_models(data, ("base", "prompt_only", "sft", "tuned"), errs)
    tuned_path = _get(data, "tuned_path", "", errs)
    if tuned_path is not _MISSING and tuned_path not in ("grpo", "rft", "sft"):
        errs.append(f"tuned_path: expected one of 'grpo', 'rft', 'sft', got {_show(tuned_path)}")
    _check_text(data, "report_path", "", errs)
    sc001 = _get_obj(data, "sc001", "", errs)
    if sc001 is not None:
        _check_sc001(sc001, errs)


def _check_sc001(sc001: dict, errs: list[str]) -> None:
    verdict = _get(sc001, "verdict", "sc001", errs)
    if verdict is not _MISSING and verdict not in ("met", "not_met"):
        errs.append(f"sc001.verdict: expected 'met' or 'not_met', got {_show(verdict)}")
        verdict = _MISSING
    before = len(errs)
    gaps = _get_obj(sc001, "gaps", "sc001", errs)
    failures: list[str] = []
    for baseline in ("vs_base", "vs_prompt_only"):
        group = _get_obj(gaps, baseline, "sc001.gaps", errs) if gaps is not None else None
        for metric in ("prior_trap_rate", "tool_call_f1"):
            if group is None:  # an empty {} is still a group: its metrics are reported missing
                continue
            item = _get_obj(group, metric, f"sc001.gaps.{baseline}", errs)
            if item is None:
                continue
            where = f"sc001.gaps.{baseline}.{metric}"
            item_before = len(errs)
            for key in ("gap", "ci_low", "ci_high"):
                value = _get(item, key, where, errs)
                if value is not _MISSING and not (_is_num(value) and -1 <= value <= 1):
                    errs.append(f"{where}.{key}: expected a number in [-1, 1], got {_show(value)}")
            _check_interval(item, "gap", where, errs, "ci_low", "ci_high")
            if len(errs) != item_before:
                continue
            if item["gap"] < SC001_GAP - EPS:
                failures.append(f"{baseline}.{metric} gap {item['gap']} < {SC001_GAP:.2f}")
            if item["ci_low"] <= 0:
                failures.append(f"{baseline}.{metric} ci_low {item['ci_low']} is not > 0")
    if len(errs) != before or verdict is _MISSING:
        return
    if failures and verdict == "met":
        errs.append(f'sc001.verdict: expected "not_met", got "met": {"; ".join(failures)}')
    elif not failures and verdict == "not_met":
        errs.append(
            'sc001.verdict: expected "met", got "not_met": all four gaps are >= 0.10 '
            "with ci_low > 0"
        )


_KIND_CHECKS: dict[str, Callable[[dict, list[str]], None]] = {
    "base_probe": _check_base_probe,
    "eval_baselines": _check_eval_baselines,
    "sft": _check_sft,
    "grpo": _check_rl,
    "rft": _check_rl,
    "spike": _check_spike,
    "final": _check_final,
}


def _secret_errors(root: Any) -> list[str]:
    """One error per secret in the decoded document, naming its path and never its value.

    Covers string values and key names that look like a credential, and a secret-named key
    (api_key, token, ...) holding a non-empty string. Iterative: deeply nested JSON is safe.
    """
    found: list[str] = []
    stack: list[tuple[Any, str]] = [(root, "")]
    while stack:
        node, path = stack.pop()
        if isinstance(node, dict):
            children = []
            for key, value in node.items():
                name = str(key)
                key_leak = SECRET_TEXT.search(name) is not None
                where = _join(path, "<redacted>" if key_leak else name)
                if key_leak or (
                    SECRET_KEY.search(name) and isinstance(value, str) and value.strip()
                ):
                    found.append(f"{where}: contains what looks like a secret (B-3)")
                else:
                    children.append((value, where))
            stack.extend(reversed(children))
        elif isinstance(node, list):
            stack.extend(reversed([(v, f"{path}[{i}]") for i, v in enumerate(node)]))
        elif isinstance(node, str) and SECRET_TEXT.search(node):
            found.append(f"{path or '$'}: contains what looks like a secret (B-3)")
    return found


def _check_document(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return [f"$: expected a JSON object, got {type(data).__name__}"]
    errs: list[str] = []
    kind = data.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        errs.append(f"kind: expected one of {', '.join(KINDS)}, got {_show(kind)}")
    _check_envelope(data, errs)
    if isinstance(kind, str) and kind in _KIND_CHECKS:
        _KIND_CHECKS[kind](data, errs)
    return errs


def validate(data: Any) -> list[str]:
    """Return one message per problem in a parsed results document (empty list = valid).

    The secret scan runs first; if it finds anything those errors are returned alone, so no
    other message can echo a credential (B-3).
    """
    return _secret_errors(data) or _check_document(data)


def validate_file(path: str | Path) -> list[str]:
    """Like validate(), for a file on disk; unreadable or malformed files give one error."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
        return [f"{path}: cannot read file: {reason}"]
    secret = []
    if SECRET_TEXT.search(text):
        secret = [f"{path}: contains what looks like a secret (B-3)"]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        where = f"line {exc.lineno} column {exc.colno}"
        return [f"{path}: invalid JSON: {exc.msg} at {where}"] + secret
    except RecursionError:
        return [f"{path}: invalid JSON: nested too deeply"] + secret
    except ValueError as exc:  # e.g. an integer of more than 4300 digits
        return [f"{path}: invalid JSON: {exc}"] + secret
    # The decoded scan names the field; the raw-text hit is the fallback (e.g. a duplicate key).
    found = _secret_errors(data) or secret
    if found:
        return found
    if not isinstance(data, dict):
        got = type(data).__name__
        return [f"{path}: expected a JSON object at the top level, got {got}"]
    return _check_document(data)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    usage = "usage: python scripts/check_results.py <results/X.json>"
    if args in (["--help"], ["-h"]):
        print(usage)
        return 0
    if len(args) != 1:
        print(usage, file=sys.stderr)
        return 2
    errors = validate_file(args[0])
    if errors:
        encoding = sys.stdout.encoding or "utf-8"  # keys can hold lone surrogates
        print("\n".join(errors).encode(encoding, "backslashreplace").decode(encoding))
        return 1
    with open(args[0], encoding="utf-8") as handle:
        print(f"OK {json.load(handle)['kind']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
