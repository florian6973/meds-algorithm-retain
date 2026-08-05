"""The ``meds-model`` entry point for PORTER."""

from importlib.resources import files

from meds_model_base.dispatch import make_cli

from .commands import COMMANDS

cli = make_cli(COMMANDS, str(files("porter").joinpath("configs")))

if __name__ == "__main__":  # pragma: no cover
    cli()
