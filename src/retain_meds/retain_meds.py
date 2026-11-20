"""Algorithm-owned algorithm. Implement typed methods; core owns staging and publication."""

from pathlib import Path

from meds_algorithm import AlgorithmRepository, PayloadMetadata
from meds_algorithm import SupervisedAlgorithm
from meds_algorithm.requests import PreprocessDataRequest
from meds_algorithm.requests import TrainSupervisedRequest
from meds_algorithm.requests import PredictSupervisedRequest

from .config import RETAINConfig


class RETAIN(SupervisedAlgorithm):
    name = "retain_meds"
    config_type = RETAINConfig

    # Extend validate_inputs(request) for payload/vocabulary compatibility. Call super() unless
    # explicitly documenting support for compatible external algorithms or different configurations.

    def preprocess_data(self, request: PreprocessDataRequest) -> PayloadMetadata:
        """Write payload into request.output.directory; return its format and version."""
        raise NotImplementedError("Implement RETAIN.preprocess_data")

    def train_supervised(self, request: TrainSupervisedRequest) -> PayloadMetadata:
        """Write payload into request.output.directory; return its format and version."""
        raise NotImplementedError("Implement RETAIN.train_supervised")

    def predict_supervised(self, request: PredictSupervisedRequest) -> PayloadMetadata:
        """Write payload into request.output.directory; return its format and version."""
        raise NotImplementedError("Implement RETAIN.predict_supervised")












# Declarative implementation/completion metadata, never fitted algorithms or mutable runtime state.
IMPLEMENTATIONS = {
    "supervised": RETAIN,
}
PROFILES = tuple(IMPLEMENTATIONS)
STUB_PROFILES = set(PROFILES)


def create_repository():
    return AlgorithmRepository(IMPLEMENTATIONS, repo=Path(__file__).resolve().parents[2])
