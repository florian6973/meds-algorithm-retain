from meds_model_contract import MODEL_PROFILES, ModelRepository
from retain.commands import COMMANDS


def test_command_registry_is_exact_selected_dag():
    repository = ModelRepository("supervised", COMMANDS)
    assert tuple(repository.profile_commands) == MODEL_PROFILES["supervised"]
