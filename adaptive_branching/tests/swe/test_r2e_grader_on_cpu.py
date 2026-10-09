import json
import ast
from pathlib import Path

import pytest

from adaptive_branching.src.swe import r2e_grader

LOG = """
============================= short test summary info =============================
PASSED r2e_tests/test_fix.py::test_ok
FAILED r2e_tests/test_fix.py::test_regression - AssertionError: old behavior
ERROR r2e_tests/test_fix.py::test_setup - RuntimeError: broken
==================== 1 failed, 1 passed, 1 error in 0.20s ====================
"""


def test_grader_source_parses_with_python_37_grammar():
    source = Path(r2e_grader.__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 7))


def test_grade_log_matches_r2e_status_map_semantics():
    expected = {"test_ok": "PASSED", "test_regression": "FAILED", "test_setup": "ERROR"}

    reward, report = r2e_grader.grade_log(LOG, expected)

    assert reward == 1.0
    assert report["actual"] == expected


def test_grade_log_requires_an_exact_mapping():
    expected = {"test_ok": "PASSED", "test_regression": "PASSED", "test_setup": "ERROR"}
    assert r2e_grader.grade_log(LOG, expected)[0] == 0.0
    assert r2e_grader.grade_log(LOG, {"test_ok": "PASSED"})[0] == 0.0
    assert r2e_grader.grade_log("no pytest summary", expected)[0] == 0.0


def test_normalize_statuses_removes_ansi_codes_and_error_suffixes():
    statuses = {"\x1b[31mtest_one - details\x1b[0m": "FAILED"}
    assert r2e_grader.normalize_statuses(statuses) == {"test_one": "FAILED"}


def test_grade_log_rejects_empty_expected_map():
    with pytest.raises(ValueError, match="must not be empty"):
        r2e_grader.grade_log(LOG, {})


def test_infrastructure_failure_is_structured_and_rejects_empty_reasons():
    expected = {"test_ok": "PASSED"}

    reward, report = r2e_grader.infrastructure_failure(expected, "patch is missing")

    assert reward == 0.0
    assert report == {
        "reward": 0.0,
        "actual": {},
        "expected": expected,
        "infrastructure_error": "patch is missing",
    }
    with pytest.raises(ValueError, match="non-empty"):
        r2e_grader.infrastructure_failure(expected, "  ")
    with pytest.raises(TypeError, match="must be a dict"):
        r2e_grader.infrastructure_failure(["test_ok"], "patch is missing")


def test_grader_cli_writes_harbor_artifacts(tmp_path, monkeypatch):
    log = tmp_path / "test.log"
    expected = tmp_path / "expected.json"
    reward = tmp_path / "reward.txt"
    report = tmp_path / "report.json"
    log.write_text(LOG, encoding="utf-8")
    expected.write_text(json.dumps({"test_ok": "PASSED", "test_regression": "FAILED", "test_setup": "ERROR"}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "r2e_grader.py",
            "--log",
            str(log),
            "--expected",
            str(expected),
            "--reward",
            str(reward),
            "--report",
            str(report),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        r2e_grader.main()

    assert exc.value.code == 0
    assert reward.read_text() == "1\n"
    assert json.loads(report.read_text())["reward"] == 1.0


def test_grader_cli_records_infrastructure_failure_with_zero_exit(tmp_path, monkeypatch):
    expected = tmp_path / "expected.json"
    reward = tmp_path / "reward.txt"
    report = tmp_path / "report.json"
    expected.write_text(json.dumps({"test_ok": "PASSED"}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "r2e_grader.py",
            "--infra-error",
            "submission patch is missing",
            "--expected",
            str(expected),
            "--reward",
            str(reward),
            "--report",
            str(report),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        r2e_grader.main()

    assert exc.value.code == 0
    assert reward.read_text() == "0\n"
    assert json.loads(report.read_text())["infrastructure_error"] == "submission patch is missing"
