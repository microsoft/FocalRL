import copy
import subprocess

import pytest

from adaptive_branching.src.swe import smith_runtime as runtime


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    runtime.run(["git", "init", "--quiet", "-b", "main"], root=root)
    runtime.run(["git", "config", "user.name", "test"], root=root)
    runtime.run(["git", "config", "user.email", "test@invalid"], root=root)
    (root / "answer.py").write_text("VALUE = 1\n")
    (root / "test_answer.py").write_text(
        "import answer\ndef test_bug(): assert answer.VALUE == 1\ndef test_ok(): assert True\n"
    )
    runtime.run(["git", "add", "."], root=root)
    runtime.run(["git", "commit", "--quiet", "-m", "Initial commit"], root=root)
    runtime.run(["git", "checkout", "-b", "repo.task"], root=root)
    (root / "answer.py").write_text("VALUE = 0\n")
    runtime.run(["git", "commit", "--quiet", "-am", "Bug Patch"], root=root)
    runtime.run(["git", "rm", "test_answer.py"], root=root)
    runtime.run(["git", "commit", "--quiet", "-m", "Remove F2P Tests"], root=root)
    row = {
        "instance_id": "repo.task",
        "image_name": "owner/image:latest",
        "repo": "owner/repo",
        "problem_statement": "Fix answer.",
        "FAIL_TO_PASS": ["test_answer.py::test_bug"],
        "PASS_TO_PASS": ["test_answer.py::test_ok"],
    }
    return root, row


def test_reference_is_inverse_source_patch_without_hidden_tests(task):
    root, row = task
    ref = runtime.extract_reference(row, root=root)
    assert ref["golden_files"] == ["answer.py"]
    assert ref["deleted_test_files"] == ["test_answer.py"]
    assert "-VALUE = 0" in ref["golden_patch"] and "+VALUE = 1" in ref["golden_patch"]
    assert "test_answer.py" not in ref["golden_patch"]


@pytest.mark.parametrize("mutation", ["missing", "wrong_parent", "extra_source_change"])
def test_reject_unproven_layout(task, mutation):
    root, row = task
    if mutation == "missing":
        row["instance_id"] = "absent"
    elif mutation == "wrong_parent":
        runtime.run(["git", "commit", "--amend", "--quiet", "-m", "Unexpected"], root=root)
    else:
        (root / "answer.py").write_text("VALUE = 99\n")
        runtime.run(["git", "add", "."], root=root)
        runtime.run(["git", "commit", "--amend", "--no-edit", "--quiet"], root=root)
    with pytest.raises(ValueError):
        runtime.extract_reference(row, root=root)


@pytest.mark.parametrize("value", ["../secret", "/absolute", "a/.git/config", "a\\b", "a/../b", "", "a//b"])
def test_path_boundaries(value):
    with pytest.raises(ValueError):
        runtime.safe_path(value)


def test_test_policy_matches_files_not_string_prefix(task):
    _, row = task
    row["PASS_TO_PASS"].append("test_answer.py_extra::test")
    assert runtime.selected_tests(row, "f2p-files")[1] == ["test_answer.py::test_ok"]
    assert len(runtime.selected_tests(row, "all")[1]) == 2
    row["PASS_TO_PASS"] = []
    assert runtime.selected_tests(row, "all")[1] == []
    with pytest.raises(ValueError):
        runtime.selected_tests(row, "other")
    row["FAIL_TO_PASS"] = []
    with pytest.raises(ValueError):
        runtime.validate_row(row)


@pytest.mark.parametrize("golden", [False, True])
def test_clean_verifier_zero_then_golden(task, tmp_path, golden):
    root, row = task
    ref = runtime.extract_reference(row, root=root)
    assets = tmp_path / "assets"
    assets.mkdir()
    config = {"row": row, "pins": ref, "role": "verifier", "test_policy": "all", "test_timeout": 30}
    runtime.bootstrap(config, root=root, assets=assets)
    assert runtime.run(["git", "rev-list", "--all", "--count"], root=root).strip() == b"1"
    assert not runtime.run(["git", "remote"], root=root).strip()
    assert not (root / "test_answer.py").exists()
    assert (assets / "pristine/test_answer.py").is_file()
    patch = tmp_path / "submission.patch"
    patch.write_text(ref["golden_patch"] if golden else "")
    report, output = runtime.verify(config, root=root, assets=assets, patch=patch)
    assert report["reward"] == float(golden), output
    assert report["p2p_pass"] == 1
    assert report["missing"] == []
    with pytest.raises(ValueError, match="second bootstrap"):
        runtime.bootstrap(config, root=root, assets=assets)


def test_agent_has_no_history_or_pristine_test_copy(task, tmp_path):
    root, row = task
    ref = runtime.extract_reference(row, root=root)
    assets = tmp_path / "assets"
    assets.mkdir()
    config = {"row": row, "pins": ref, "role": "agent", "test_policy": "all"}
    runtime.bootstrap(config, root=root, assets=assets)
    assert not (assets / "pristine").exists()
    assert not (root / "test_answer.py").exists()
    with pytest.raises(subprocess.CalledProcessError):
        runtime.run(["git", "show", ref["base"] + ":answer.py"], root=root)


def test_invalid_pin_fails_before_changing_repository(task, tmp_path):
    root, row = task
    ref = runtime.extract_reference(row, root=root)
    ref["head"] = "invalid"
    with pytest.raises(ValueError, match="pinned head"):
        runtime.bootstrap({"row": row, "pins": ref, "role": "agent"}, root=root, assets=tmp_path)
    assert (root / ".git").exists()


def test_grader_missing_timeout_regression_and_space_in_node():
    f2p, p2p = ["t.py::test[x y]"], ["t.py::ok"]
    log = "PASSED t.py::test[x y]\nXFAIL t.py::ok - known failure\n"
    assert runtime.grade(log, f2p, p2p, 0)["reward"] == 1
    assert runtime.grade(log, f2p, p2p, 124, timed_out=True)["reward"] == 0
    assert runtime.grade(log, f2p, p2p, 2)["reward"] == 0
    assert runtime.grade("", f2p, p2p, 0)["missing"] == f2p + p2p
    assert runtime.grade("PASSED t.py::test[x y]\nFAILED t.py::ok", f2p, p2p, 1)["reward"] == 0
    with pytest.raises(ValueError, match="conflicting"):
        runtime.grade(log + "FAILED t.py::ok\n", f2p, p2p, 1)
    with pytest.raises(ValueError):
        runtime.grade(log, [], [], 0)


def test_duplicate_tests_fail(task):
    _, row = task
    row = copy.deepcopy(row)
    row["PASS_TO_PASS"] = row["FAIL_TO_PASS"]
    with pytest.raises(ValueError, match="overlap"):
        runtime.validate_row(row)
