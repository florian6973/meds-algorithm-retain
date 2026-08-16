from meds_model_contract import DAGS, ModelRepository
from retain.commands import COMMANDS


def test_command_registry_is_exact_selected_dag():
    repository = ModelRepository("supervised", COMMANDS)
    assert tuple(repository.dag) == DAGS["supervised"]
