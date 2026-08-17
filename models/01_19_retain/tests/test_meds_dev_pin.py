"""The MEDS-DEV revision guard, tested against real git checkouts rather than mocks.

`provision_meds_dev` pins the revision only when it clones from the URL. Supplying `MEDS_DEV_DIR` skips
the pin and takes the source's HEAD, and `assert_meds_dev_predicates_capability` does not catch it because
it tests for the *presence* of the predicates passthrough -- a newer revision still carrying it passes. A
gate run against an unpinned MEDS-DEV is not the gate it reports being.

These tests build actual repositories at right and wrong revisions, because a guard that silently passes is
the same class of defect as the check it replaces, and only an end-to-end exercise distinguishes the two.
No network, no MEDS-DEV; they run in the default suite.
"""

from __future__ import annotations

import subprocess

import pytest

from .e2e import resolved_meds_dev_revision


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    """A checkout with two commits, so HEAD can be pinned or wrong."""
    root = tmp_path / "MEDS-DEV"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "marker.txt").write_text("first\n")
    _git(root, "add", "marker.txt")
    _git(root, "commit", "-qm", "first")
    first = _git(root, "rev-parse", "HEAD")
    (root / "marker.txt").write_text("second\n")
    _git(root, "add", "marker.txt")
    _git(root, "commit", "-qm", "second")
    second = _git(root, "rev-parse", "HEAD")
    return root, first, second


def test_the_pinned_revision_is_accepted_and_returned(repository, monkeypatch):
    """A checkout detached at the pinned commit, which is what `provision_meds_dev` produces."""
    root, first, _ = repository
    _git(root, "checkout", "-q", "--detach", first)
    monkeypatch.setenv("MEDS_DEV_REF", first)
    assert resolved_meds_dev_revision(root) == first


def test_an_unpinned_head_is_refused(repository, monkeypatch):
    """The case `MEDS_DEV_DIR` creates: the checkout is simply at some other revision."""
    root, first, second = repository
    monkeypatch.setenv("MEDS_DEV_REF", first)
    with pytest.raises(AssertionError, match="not the pinned"):
        resolved_meds_dev_revision(root)
    # The message has to name both revisions, or a failing gate cannot be diagnosed from its output.
    try:
        resolved_meds_dev_revision(root)
    except AssertionError as error:
        assert second in str(error) and first in str(error)


def test_a_branch_name_is_resolved_rather_than_string_compared(repository, monkeypatch):
    """`MEDS_DEV_REF` may name a branch or tag; comparing it literally would always fail."""
    root, _, second = repository
    _git(root, "tag", "pinned-tag")
    monkeypatch.setenv("MEDS_DEV_REF", "pinned-tag")
    assert resolved_meds_dev_revision(root) == second


def test_a_directory_that_is_not_a_checkout_is_refused(tmp_path):
    with pytest.raises(AssertionError, match="not a git checkout"):
        resolved_meds_dev_revision(tmp_path)


def test_the_default_expectation_is_the_ref_integration_pins(repository, monkeypatch):
    """With no override the guard must compare against `tests/integration.py`'s pin, not a local default."""
    from .integration import MEDS_DEV_PREDICATES_REF

    root, _, second = repository
    monkeypatch.delenv("MEDS_DEV_REF", raising=False)
    with pytest.raises(AssertionError) as error:
        resolved_meds_dev_revision(root)
    assert MEDS_DEV_PREDICATES_REF in str(error.value)
    assert second in str(error.value)
