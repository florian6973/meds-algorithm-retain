"""Which of the five commands this model supports — the ``supervised`` DAG.

The ``COMMANDS`` registry maps each supported
:class:`~meds_model_base.commands.base.CommandName` to the class implementing it. ``meds-model`` consults
it: an unsupported command exits with a message naming what *is* supported.

This is the one piece of the DAG that has to be Python rather than configuration — the dispatcher needs it
before Hydra composes anything. Edit it freely: swap in your own subclass, add a command, or drop one.
"""

from meds_model_base.commands import (
    CommandName,
    DefaultPreprocessDataCommand,
    DefaultSupervisedTrainCommand,
    SupervisedPredictCommand,
)

COMMANDS = {
    CommandName.preprocess_data: DefaultPreprocessDataCommand,
    CommandName.supervised_train: DefaultSupervisedTrainCommand,
    CommandName.predict: SupervisedPredictCommand,
}
