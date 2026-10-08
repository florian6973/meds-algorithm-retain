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


@pytest.mark.parametrize("kind", ["branch", "lightweight-tag", "annotated-tag"])
def test_a_symbolic_override_is_resolved_to_its_commit(repository, monkeypatch, kind):
    """`MEDS_DEV_REF` may name a branch or a tag, and the three forms do not behave alike.

    A plain `git rev-parse` already resolves a branch and a *lightweight* tag to the commit, so those two
    pass whether or not the override is peeled. An **annotated** tag is a git object in its own right:
    `rev-parse` returns the tag object's SHA, which never equals HEAD, and only `^{commit}` peels it to
    the commit. Measured on a scratch repository -- lightweight `cat-file -t` reports `commit`, annotated
    reports `tag` and resolves to a different SHA.

    All three forms are exercised because a test covering only the forms that work either way cannot tell
    whether the peel is present. That is the same gap as testing an override with a SHA alone.
    """
    root, _, second = repository
    if kind == "branch":
        _git(root, "branch", "pinned-branch")
        reference = "pinned-branch"
    elif kind == "lightweight-tag":
        _git(root, "tag", "pinned-light")
        reference = "pinned-light"
    else:
        _git(root, "tag", "-a", "pinned-annotated", "-m", "annotated")
        reference = "pinned-annotated"
        assert _git(root, "cat-file", "-t", reference) == "tag"
        assert _git(root, "rev-parse", reference) != second, (
            "an annotated tag must resolve to its own object, or this case is not testing the peel"
        )
    monkeypatch.setenv("MEDS_DEV_REF", reference)
    assert resolved_meds_dev_revision(root) == second


def test_a_directory_that_is_not_a_checkout_is_refused(tmp_path):
    """And says what could not be established, not merely that a command failed.

    A typo'd path is the case that would otherwise pass as a clean run, so the failure has to name the
    thing it could not determine rather than surface a bare subprocess error.
    """
    with pytest.raises(AssertionError) as error:
        resolved_meds_dev_revision(tmp_path)
    message = str(error.value)
    assert "Could not establish which MEDS-DEV revision" in message
    assert str(tmp_path) in message
    assert "unverified" in message


def test_the_default_expectation_is_the_ref_integration_pins(repository, monkeypatch):
    """With no override the guard must compare against `tests/integration.py`'s pin, not a local default."""
    from .integration import MEDS_DEV_PREDICATES_REF

    root, _, second = repository
    monkeypatch.delenv("MEDS_DEV_REF", raising=False)
    with pytest.raises(AssertionError) as error:
        resolved_meds_dev_revision(root)
    assert MEDS_DEV_PREDICATES_REF in str(error.value)
    assert second in str(error.value)
