"""Run with the installed wheel's Python using -I; no source checkout on sys.path.

This checks packaging, not scientific correctness. Extend for required algorithm resources.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

from retain_meds import RETAIN, RETAINConfig
from retain_meds.retain_meds import create_repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration", action="store_true")
    args = parser.parse_args()
    assert RETAIN.__module__.startswith("retain_meds.")
    assert RETAINConfig.__module__.startswith("retain_meds.")
    repository = create_repository()
    resources = files("retain_meds") / "resources"
    assert (resources / "runs" / "run.yaml").is_file()
    for profile in repository.profiles:
        assert (resources / "algorithms" / profile / "model.yaml").is_file()
    executable = Path(sys.executable).parent / ("retain-meds" + (".exe" if sys.platform == "win32" else ""))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = subprocess.run([str(executable), "profiles"], cwd=root, check=True, text=True, capture_output=True)
        assert result.stdout.splitlines() == list(repository.profiles)
        for profile in repository.profiles:
            result = subprocess.run([str(executable), "commands", "--profile", profile], cwd=root, check=True, text=True, capture_output=True)
            assert result.stdout.splitlines() == list(repository.profile_commands(profile))
        if args.orchestration:
            # A preprocessing-only dry run requires no fitted artifacts or task-specific labels.
            # Algorithms with mandatory configuration fields should add representative overrides here.
            (root / "inputs").mkdir()
            (root / "predicates.yaml").write_text("predicates: {}\n", encoding="utf-8")
            result = subprocess.run(
                [str(executable), "run", "--dry-run", f"profile={next(iter(repository.profiles))}",
                 "stages=[preprocess_data]", f"output_dir={root / 'planned'}",
                 f"inputs.meds_dir={root / 'inputs'}", f"inputs.predicates_file={root / 'predicates.yaml'}"],
                cwd=root, check=True, text=True, capture_output=True,
            )
            plan = json.loads(result.stdout)
            assert [stage["name"] for stage in plan["stages"]] == ["preprocess_data"]
            assert not (root / "planned").exists()
    print("Installed imports, commands, packaged resources and requested dry-run checks passed.")


if __name__ == "__main__":
    main()
