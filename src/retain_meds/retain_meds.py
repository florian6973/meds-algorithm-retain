"""RETAIN behind the MEDS algorithm contract: `preprocess_data -> train_supervised -> predict_supervised`.

That is RETAIN's actual lifecycle: the embedding, both attention GRUs and the risk head are trained jointly
against the labelled endpoint by `train_RETAIN`, and there is no task-independent pretraining or inference
stage to place anywhere else.

- `preprocess_data` tokenises the MEDS code stream. It reads no labels and no predicates, and fits the
  vocabulary on the dataset's training split only.
- `train_supervised` fits RETAIN on the `train` labels and selects the epoch on the `tuning` labels (or on a
  seeded hold-out of the training rows when the task supplies none).
- `predict_supervised` scores every requested key. It receives the index only; the runner attaches ground
  truth after the scores are written.

Histories are cut at each prediction time before visits are formed (`_windows.build_sequences`), so nothing
recorded after a prediction time can reach its prediction.

This port is predicate-free: RETAIN's input is a sequence of medical codes, which the MEDS `code` column
already is. `external_predicates_file` is accepted, recorded by the runner and not read.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from meds_algorithm import AlgorithmRepository, PayloadMetadata, SupervisedAlgorithm
from meds_algorithm.requests import PredictSupervisedRequest, PreprocessDataRequest, TrainSupervisedRequest

from . import _preprocess
from ._preprocess import HELD_OUT_SPLIT, SETTINGS_FILE, TRAIN_SPLIT, TUNING_SPLIT, VOCABULARY_SETTINGS
from ._windows import KEY_COLUMNS, PatientStore, build_sequences
from .config import RETAINConfig

_KEYS = list(KEY_COLUMNS)
_PREPARED = PayloadMetadata(_preprocess.PAYLOAD_FORMAT, 1)
_FITTED = PayloadMetadata("retain-supervised-model", 1)
_PREDICTIONS = PayloadMetadata("retain-predictions", 1)
_EXPECTED_INPUTS = {"input_data_dir": _PREPARED, "input_supervised_algorithm_dir": _FITTED}

_MODEL_FILE = "model.pt"
_TRAINING_FILE = "training.json"
_PREPROCESS_REPORT = "preprocess_report.json"
_PREDICTION_REPORT = "prediction_report.json"
_INTERPRETATION_FILE = "interpretation.ndjson"


def _format(metadata: PayloadMetadata) -> str:
    return f"{metadata.format}/v{metadata.version}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _vocabulary_digest(data_dir: Path) -> str:
    """Identity of the vocabulary a model was fitted on: its codes in index order."""
    vocabulary = pl.read_parquet(Path(data_dir) / "vocabulary.parquet").sort("code_index")
    return "sha256:" + hashlib.sha256("\n".join(vocabulary["code"].to_list()).encode()).hexdigest()


class RETAIN(SupervisedAlgorithm):
    name = "retain_meds"
    config_type = RETAINConfig

    def validate_inputs(self, request) -> None:
        """Reject an incompatible input artifact before any expensive work happens."""
        super().validate_inputs(request)
        for argument, artifact in request.artifact_inputs():
            found = artifact.read_manifest()["artifact"]["payload_format"]
            if found != _format(_EXPECTED_INPUTS[argument]):
                raise ValueError(f"{argument} has payload {found}; expected {_format(_EXPECTED_INPUTS[argument])}")
        data = getattr(request, "data", None)
        if data is not None:
            # The vocabulary is fixed by preprocess_data; changing its filters later would describe a
            # vocabulary that was never built.
            published = json.loads((data.directory / SETTINGS_FILE).read_text())
            for name in VOCABULARY_SETTINGS:
                if published.get(name) != getattr(self.config, name):
                    raise ValueError(
                        f"{name}={getattr(self.config, name)!r} disagrees with input_data_dir, which was "
                        f"built with {name}={published.get(name)!r}. It is a preprocess_data setting; "
                        "rebuild the prepared data to change it."
                    )
        fitted = getattr(request, "fitted_algorithm", None)
        if fitted is not None and data is not None:
            recorded = json.loads((fitted.directory / _TRAINING_FILE).read_text())["vocabulary_digest"]
            if recorded != _vocabulary_digest(data.directory):
                raise ValueError(
                    "input_data_dir holds a different vocabulary from the one the model was fitted on; "
                    "pass the prepared data used by train_supervised"
                )

    def preprocess_data(self, request: PreprocessDataRequest) -> PayloadMetadata:
        """Tokenise a MEDS dataset into RETAIN's code stream; task- and label-independent."""
        report = _preprocess.preprocess(request.meds_dir, request.output.directory, self.config)
        _write_json(request.output.directory / _PREPROCESS_REPORT, report)
        return _PREPARED

    def train_supervised(self, request: TrainSupervisedRequest) -> PayloadMetadata:
        """Fit RETAIN on the train labels; the tuning labels select the epoch."""
        import torch

        from ._training import resolve_device, train

        config = self.config
        seed = request.execution.seed
        store = PatientStore(request.data.directory)

        # Row order of a label file carries no meaning; a canonical key order keeps the fit a function of
        # the labels' content (it decides how equal-length ties resolve when minibatches are built).
        fit = request.labels.read(TRAIN_SPLIT).sort(_KEYS)
        _require_split(store, fit, TRAIN_SPLIT)
        try:
            validation = request.labels.read(TUNING_SPLIT).sort(_KEYS)
        except FileNotFoundError:
            validation = None
        validation_source = TUNING_SPLIT
        if validation is not None:
            _require_split(store, validation, TUNING_SPLIT)
        elif fit.height > 1 and config.validation_fraction > 0:
            # No tuning labels: a seeded hold-out of the training rows, as `load_data_simple` does upstream.
            shuffled = np.random.default_rng(seed).permutation(fit.height)
            n_validation = min(max(1, round(config.validation_fraction * fit.height)), fit.height - 1)
            validation = fit[np.sort(shuffled[:n_validation])]
            fit = fit[np.sort(shuffled[n_validation:])]
            validation_source = f"seeded {config.validation_fraction:g} hold-out of {TRAIN_SPLIT}"
        else:
            validation = fit.clear()
            validation_source = "none"
        if config.max_train_rows is not None:
            fit = fit.head(config.max_train_rows)

        # Sequences are built from the keys alone; the targets never reach sequence construction.
        train_batch = build_sequences(store, fit.select(_KEYS), config)
        validation_batch = build_sequences(store, validation.select(_KEYS), config)
        device = resolve_device(config.device)
        model, history = train(
            vocabulary_size=store.vocabulary_size,
            train_batch=train_batch,
            train_labels=fit["boolean_value"].to_numpy().astype(np.float32),
            validation_batch=validation_batch,
            validation_labels=validation["boolean_value"].to_numpy().astype(np.float32),
            config=config,
            device=device,
            seed=seed,
        )

        output = request.output.directory
        torch.save(
            {"state_dict": model.state_dict(), "vocabulary_size": store.vocabulary_size}, output / _MODEL_FILE
        )
        _write_json(
            output / _TRAINING_FILE,
            {
                "vocabulary_size": store.vocabulary_size,
                "vocabulary_digest": _vocabulary_digest(request.data.directory),
                "n_parameters": model.n_parameters(),
                "device": str(device),
                "seed": seed,
                "training": history.summary(),
                "validation_source": validation_source,
                "fit_rows": fit.height,
                "validation_rows": validation.height,
                "train_visits": int(train_batch.lengths.sum()),
                "empty_history_fit_rows": train_batch.n_empty,
                "empty_history_validation_rows": validation_batch.n_empty,
            },
        )
        return _FITTED

    def predict_supervised(self, request: PredictSupervisedRequest) -> PayloadMetadata:
        """Score every requested key; scores carry their own keys and are never matched by row order."""
        import torch

        from ._model import Retain
        from ._training import resolve_device, score

        config = self.config
        index = request.index.read().sort(_KEYS)
        store = PatientStore(request.data.directory)
        fitted = request.fitted_algorithm.directory
        device = resolve_device(config.device)
        checkpoint = torch.load(fitted / _MODEL_FILE, map_location=device, weights_only=True)
        model = Retain(checkpoint["vocabulary_size"], config).to(device)
        model.load_state_dict(checkpoint["state_dict"])

        # Every key yields a sequence: a key with no history is scored by the learned intercept.
        batch = build_sequences(store, index.select(_KEYS), config)
        probabilities = score(model, batch, config, device)
        predictions = pl.DataFrame(
            {
                "subject_id": pl.Series(batch.keys[:, 0]).cast(index.schema["subject_id"]),
                "prediction_time": pl.Series(batch.keys[:, 1]).cast(pl.Datetime("us")),
                "predicted_boolean_probability": pl.Series(probabilities, dtype=pl.Float64),
            }
        ).with_columns(
            pl.col("prediction_time").cast(index.schema["prediction_time"]),
            (pl.col("predicted_boolean_probability") > 0.5).alias("predicted_boolean_value"),
        )
        missing = index.join(predictions, on=_KEYS, how="anti").height
        extra = predictions.join(index, on=_KEYS, how="anti").height
        if missing or extra or predictions.height != index.height:
            raise ValueError(f"RETAIN scoring missed {missing} and added {extra} requested keys")

        output = request.output.directory
        predictions.write_parquet(output / "predictions.parquet")
        report = {
            "scored_keys": predictions.height,
            "unique_subjects": predictions["subject_id"].n_unique(),
            "visits": int(batch.lengths.sum()),
            "empty_history_keys": batch.n_empty,
            "exclusions": "none: every requested key is scored",
            "device": str(device),
        }
        if config.interpretation_top_k > 0:
            from ._interpret import write_contributions

            report["interpretation_records"] = write_contributions(
                output / _INTERPRETATION_FILE,
                model=model,
                batch=batch,
                config=config,
                device=device,
                code_strings=store.code_strings(),
                top_k=config.interpretation_top_k,
            )
        _write_json(output / _PREDICTION_REPORT, report)
        return _PREDICTIONS


def _require_split(store: PatientStore, labels: pl.DataFrame, split: str) -> None:
    """Training labels may only name subjects of their own MEDS split, so held-out subjects are never fitted."""
    assigned = store.split_of(labels["subject_id"])
    outside = labels.filter(assigned.fill_null("unassigned") != split)
    if outside.height:
        raise ValueError(
            f"{outside['subject_id'].n_unique()} subject(s) of the {split} labels are not in the MEDS "
            f"{split} split (e.g. {HELD_OUT_SPLIT} subjects must never be fitted)"
        )


# Declarative implementation/completion metadata, never fitted algorithms or mutable runtime state.
IMPLEMENTATIONS = {
    "supervised": RETAIN,
}
PROFILES = tuple(IMPLEMENTATIONS)
STUB_PROFILES: set[str] = set()


def create_repository():
    return AlgorithmRepository(IMPLEMENTATIONS, repo=Path(__file__).resolve().parents[2])
