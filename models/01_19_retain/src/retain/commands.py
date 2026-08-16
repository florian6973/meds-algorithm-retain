"""Model-owned command implementations for the supervised DAG.

Replace the bodies below. The contract deliberately does not prescribe batching, frameworks, payload
formats, or checkpointing. Keep ``IS_STUB`` true until the rendered end-to-end test runs the real DAG.
"""

from pathlib import Path

from meds_model_contract import ArtifactType, Command, CommandName, measured_artifact

IS_STUB = True


class StubCommand(Command):
    artifact_type: ArtifactType
    output_argument: str

    def resolve_output(self, arguments):
        return arguments.get(self.output_argument)

    def run(self, arguments):
        output = self.resolve_output(arguments)
        if output is None:
            raise ValueError(f"{self.name.value} requires {self.output_argument}=<path>")
        with measured_artifact(
            Path(output),
            artifact_type=self.artifact_type.value,
            command=self.name.value,
            producer=f"retain.commands.{type(self).__name__}",
            payload_format="stub/v1",
            external_predicates_file=arguments.get("external_predicates_file"),
            overwrite=bool(arguments.get("do_overwrite", False)),
        ):
            raise NotImplementedError(
                f"Implement {type(self).__name__}.run in src/retain/commands.py"
            )



class PreprocessData(StubCommand):
    name = CommandName.preprocess_data
    artifact_type = ArtifactType.patients
    output_argument = "output_data_dir"





class SupervisedTrain(StubCommand):
    name = CommandName.supervised_train
    artifact_type = ArtifactType.supervised_model
    output_argument = "output_supervised_model_dir"


class Predict(StubCommand):
    name = CommandName.predict
    artifact_type = ArtifactType.predictions
    output_argument = "output_predictions_dir"


COMMANDS = {
    CommandName.preprocess_data: PreprocessData,
    CommandName.supervised_train: SupervisedTrain,
    CommandName.predict: Predict,
}
