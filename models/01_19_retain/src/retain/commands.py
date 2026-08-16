"""Model-owned command implementations for the supervised DAG.

`preprocess_data -> supervised_train -> predict`, which is RETAIN's actual lifecycle: the embedding, both
attention GRUs and the risk head are trained jointly against the labelled endpoint by `train_RETAIN`, and
there is no task-independent pretraining or inference stage to place anywhere else.

Intermediate payloads are implementation-owned. Each command declares the payload format it writes and
rejects an incompatible input before doing any expensive work. Every command does its work *inside*
`measured_artifact`, so the recorded wall time and memory describe the work rather than the bookkeeping;
the `extra` mapping handed to the context manager is mutated in place as results become known.

This port is rendered `uses_predicates=false` and is predicate-free by construction: RETAIN's input is a
sequence of medical codes, which the MEDS `code` column already is. No command reads
`external_predicates_file`, and passing one is an error rather than a silent no-op.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from meds_model_contract import ArtifactType, Command, CommandName, measured_artifact, read_manifest

from . import preprocess as preprocessing
from .config import RESERVED_ARGUMENTS, RetainConfig
from .labels import boolean_targets, prediction_keys, read_labels, requested_splits
from .preprocess import PAYLOAD_FORMAT as PATIENTS_FORMAT
from .preprocess import TRAIN_SPLIT, TUNING_SPLIT
from .windows import PatientStore, build_sequences

IS_STUB = False

SUPERVISED_MODEL_FORMAT = "retain-supervised-model/v1"
PREDICTIONS_FORMAT = "meds-evaluation-predictions/v1"

PREDICTED_PROBABILITY_FIELD = "predicted_boolean_probability"
PREDICTED_VALUE_FIELD = "predicted_boolean_value"

#: Columns of the MEDS prediction schema that may be carried into the published output.
_PREDICTION_COLUMNS = (
    "subject_id",
    "prediction_time",
    "boolean_value",
    "integer_value",
    "float_value",
    "categorical_value",
    PREDICTED_PROBABILITY_FIELD,
    PREDICTED_VALUE_FIELD,
)

#: Settings that describe the published vocabulary and therefore cannot be changed after preprocessing.
_PREPROCESS_ONLY = ("min_code_count", "max_vocab_size")

#: Settings a prediction run may override without invalidating the trained model.
_PREDICT_OVERRIDABLE = frozenset({"device", "batch_size", "interpretation_top_k"})


def _require(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if value in (None, ""):
        raise ValueError(f"{name} is required")
    return str(value)


def _reject_predicates(arguments: Mapping[str, Any]) -> None:
    value = arguments.get("external_predicates_file")
    if value in (None, "", "null", "None"):
        return
    raise ValueError(
        "This RETAIN port is predicate-free: it builds visits from the MEDS code stream directly and "
        f"never interprets dataset predicates, so external_predicates_file={value!r} would have no "
        "effect. Remove the argument rather than recording a digest for an unused input."
    )


def _config(arguments: Mapping[str, Any]) -> RetainConfig:
    return RetainConfig.from_arguments(arguments)


def _check_artifact(directory: Path, expected_type: str, expected_format: str) -> dict[str, Any]:
    """Reject an incompatible input artifact before any expensive work happens."""
    manifest = read_manifest(directory)
    artifact = manifest.get("artifact", {})
    if artifact.get("type") != expected_type:
        raise ValueError(f"{directory} is a {artifact.get('type')!r} artifact; {expected_type!r} is required")
    actual = str(artifact.get("payload_format", ""))
    if actual != expected_format:
        raise ValueError(
            f"{directory} declares payload format {actual!r}; this build reads {expected_format!r}"
        )
    return manifest


def _write_json(directory: Path, name: str, payload: Mapping[str, Any]) -> None:
    (directory / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


class PreprocessData(Command):
    """Tokenise a MEDS dataset into RETAIN's code stream. Reads no labels and no predicates."""

    name = CommandName.preprocess_data
    artifact_type = ArtifactType.patients

    def run(self, arguments: Mapping[str, Any]) -> Path:
        _reject_predicates(arguments)
        meds_dir = Path(_require(arguments, "external_meds_dir")).resolve()
        output = Path(_require(arguments, "output_data_dir"))
        config = _config(arguments)

        extra: dict[str, Any] = {"config": config.to_dict()}
        with measured_artifact(
            output,
            artifact_type=self.artifact_type.value,
            command=self.name.value,
            producer="retain.commands.PreprocessData",
            payload_format=PATIENTS_FORMAT,
            config_digest=config.digest(),
            inputs={"external_meds_dir": {"path": str(meds_dir)}},
            overwrite=bool(arguments.get("do_overwrite", False)),
            extra=extra,
        ) as staging:
            extra["work_units"] = preprocessing.preprocess(meds_dir, staging, config)
        return output


class SupervisedTrain(Command):
    """Fit RETAIN for the task supplied through `external_labels_dir`."""

    name = CommandName.supervised_train
    artifact_type = ArtifactType.supervised_model

    def run(self, arguments: Mapping[str, Any]) -> Path:
        import torch

        from .training import gpu_metrics, resolve_device, train

        _reject_predicates(arguments)
        data_dir = Path(_require(arguments, "input_data_dir")).resolve()
        labels_dir = Path(_require(arguments, "external_labels_dir")).resolve()
        output = Path(_require(arguments, "output_supervised_model_dir"))
        config = _config(arguments)

        data_manifest = _check_artifact(data_dir, ArtifactType.patients.value, PATIENTS_FORMAT)
        store = PatientStore(data_dir)
        _check_vocabulary_settings(config, store.config)

        device = resolve_device(config.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        extra: dict[str, Any] = {"config": config.to_dict()}
        with measured_artifact(
            output,
            artifact_type=self.artifact_type.value,
            command=self.name.value,
            producer="retain.commands.SupervisedTrain",
            payload_format=SUPERVISED_MODEL_FORMAT,
            config_digest=config.digest(),
            inputs={
                "input_data_dir": {
                    "path": str(data_dir),
                    "payload_format": PATIENTS_FORMAT,
                    "config_digest": data_manifest["command"].get("config_digest"),
                },
                "external_labels_dir": {"path": str(labels_dir)},
            },
            overwrite=bool(arguments.get("do_overwrite", False)),
            gpu_metrics=lambda: gpu_metrics(device),
            extra=extra,
        ) as staging:
            labels = read_labels(labels_dir)
            targets = boolean_targets(labels).to_numpy().astype(np.float32)
            keys = prediction_keys(labels)
            assignments = store.split_of(keys.get_column("subject_id"))

            train_rows, validation_rows, split_report = _training_rows(assignments, config)
            if not train_rows.size:
                raise ValueError(
                    f"No training rows among {labels.height} labelled keys; observed splits "
                    f"{split_report['observed']}. RETAIN fits on the '{TRAIN_SPLIT}' split."
                )
            if config.max_train_rows is not None:
                train_rows = train_rows[: config.max_train_rows]

            train_batch = build_sequences(store, keys[train_rows], config)
            validation_batch = build_sequences(store, keys[validation_rows], config)
            model, history = train(
                vocabulary_size=store.vocabulary_size,
                train_batch=train_batch,
                train_labels=targets[train_rows],
                validation_batch=validation_batch,
                validation_labels=targets[validation_rows],
                config=config,
                device=device,
            )

            torch.save(
                {"state_dict": model.state_dict(), "vocabulary_size": store.vocabulary_size},
                staging / "model.pt",
            )
            _write_json(
                staging,
                "model.json",
                {
                    "config": config.to_dict(),
                    "vocabulary_size": store.vocabulary_size,
                    "input_data_dir": str(data_dir),
                    "training": history.summary(),
                    "splits": split_report,
                },
            )
            extra["training"] = history.summary()
            extra["splits"] = split_report
            extra["work_units"] = {
                "labelled_rows": labels.height,
                "train_rows": int(train_rows.size),
                "validation_rows": int(validation_rows.size),
                "optimizer_steps": history.optimizer_steps,
                "vocabulary_size": store.vocabulary_size,
                "n_parameters": model.n_parameters(),
                "train_visits": int(train_batch.lengths.sum()),
                "empty_history_train_rows": train_batch.n_empty,
                "empty_history_validation_rows": validation_batch.n_empty,
            }
        return output


class Predict(Command):
    """Score a task index. Ground truth is separated before sequences are built and never reaches them."""

    name = CommandName.predict
    artifact_type = ArtifactType.predictions

    def run(self, arguments: Mapping[str, Any]) -> Path:
        import torch

        from .interpret import write_contributions
        from .model import Retain
        from .training import gpu_metrics, resolve_device, score

        _reject_predicates(arguments)
        model_dir = Path(_require(arguments, "input_supervised_model_dir")).resolve()
        labels_dir = Path(_require(arguments, "external_labels_dir")).resolve()
        output = Path(_require(arguments, "output_predictions_dir"))

        model_manifest = _check_artifact(
            model_dir, ArtifactType.supervised_model.value, SUPERVISED_MODEL_FORMAT
        )
        payload = json.loads((model_dir / "model.json").read_text())
        config = _prediction_config(payload["config"], arguments)

        data_dir = Path(str(arguments.get("input_data_dir") or payload["input_data_dir"])).resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"The patients artifact {data_dir} recorded by {model_dir} is missing; "
                "pass input_data_dir=<path> to point at a relocated copy"
            )
        _check_artifact(data_dir, ArtifactType.patients.value, PATIENTS_FORMAT)

        requested = requested_splits(arguments.get("splits"))
        device = resolve_device(config.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        extra: dict[str, Any] = {"config": config.to_dict()}
        with measured_artifact(
            output,
            artifact_type=self.artifact_type.value,
            command=self.name.value,
            producer="retain.commands.Predict",
            payload_format=PREDICTIONS_FORMAT,
            config_digest=config.digest(),
            inputs={
                "input_supervised_model_dir": {
                    "path": str(model_dir),
                    "payload_format": SUPERVISED_MODEL_FORMAT,
                    "config_digest": model_manifest["command"].get("config_digest"),
                },
                "input_data_dir": {"path": str(data_dir), "payload_format": PATIENTS_FORMAT},
                "external_labels_dir": {"path": str(labels_dir)},
                "splits": sorted(requested) if requested else None,
            },
            overwrite=bool(arguments.get("do_overwrite", False)),
            gpu_metrics=lambda: gpu_metrics(device),
            extra=extra,
        ) as staging:
            store = PatientStore(data_dir)
            labels = read_labels(labels_dir)
            assignments = store.split_of(labels.get_column("subject_id"))
            if requested is not None:
                keep = pl.Series([split in requested for split in assignments.to_list()])
                labels = labels.filter(keep)
                assignments = assignments.filter(keep)
            if not labels.height:
                observed = sorted({split for split in assignments.to_list() if split is not None})
                raise ValueError(
                    f"No label rows remain for splits {sorted(requested) if requested else 'all'}; "
                    f"observed splits: {observed}"
                )

            # Ground truth is dropped here; only this key frame reaches sequence construction.
            keys = prediction_keys(labels)
            checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
            model = Retain(checkpoint["vocabulary_size"], config).to(device)
            model.load_state_dict(checkpoint["state_dict"])

            batch = build_sequences(store, keys, config)
            probabilities = score(model, batch, config, device)

            # Associate by key, never by row order.
            scored = pl.DataFrame(
                {
                    "subject_id": pl.Series(batch.keys[:, 0]).cast(pl.Int64),
                    "prediction_time": pl.Series(batch.keys[:, 1]).cast(pl.Datetime("us")),
                    PREDICTED_PROBABILITY_FIELD: pl.Series(probabilities).cast(pl.Float32),
                }
            )
            predictions = labels.join(
                scored, on=["subject_id", "prediction_time"], how="left", validate="1:1"
            ).with_columns((pl.col(PREDICTED_PROBABILITY_FIELD) > 0.5).alias(PREDICTED_VALUE_FIELD))
            if predictions.get_column(PREDICTED_PROBABILITY_FIELD).null_count():
                raise ValueError(
                    "Some requested keys received no score; refusing to publish a partial artifact"
                )
            published = predictions.select(
                [column for column in _PREDICTION_COLUMNS if column in predictions.columns]
            )
            published.write_parquet(staging / "predictions.parquet")

            if config.interpretation_top_k > 0:
                extra["interpretation"] = {
                    "path": "interpretation.ndjson",
                    "top_k": config.interpretation_top_k,
                    "records": write_contributions(
                        staging / "interpretation.ndjson",
                        model=model,
                        batch=batch,
                        config=config,
                        device=device,
                        code_strings=store.code_strings(),
                        top_k=config.interpretation_top_k,
                    ),
                }

            extra["coverage"] = _coverage(assignments, published)
            extra["exclusions"] = {
                "policy": "none: every requested key is scored",
                "empty_history_keys": batch.n_empty,
            }
            extra["work_units"] = {
                "scored_keys": published.height,
                "unique_subjects": published.get_column("subject_id").n_unique(),
                "visits": int(batch.lengths.sum()),
            }
        return output


def _check_vocabulary_settings(config: RetainConfig, published: Mapping[str, Any]) -> None:
    """The vocabulary is fixed by `preprocess_data`; changing its filters later would be a lie."""
    for name in _PREPROCESS_ONLY:
        if published.get(name) != getattr(config, name):
            raise ValueError(
                f"{name}={getattr(config, name)!r} disagrees with the patients artifact, which was built "
                f"with {name}={published.get(name)!r}. It is a preprocess_data setting; rebuild the "
                "artifact to change it."
            )


def _prediction_config(published: Mapping[str, Any], arguments: Mapping[str, Any]) -> RetainConfig:
    """The trained model's own configuration, allowing only runtime overrides.

    Everything that shaped the sequences or the parameters has to match the fit, or the scores would be
    produced by a different model than the one that was trained.
    """
    supplied = {key: value for key, value in arguments.items() if key not in RESERVED_ARGUMENTS}
    forbidden = sorted(set(supplied) - _PREDICT_OVERRIDABLE)
    if forbidden:
        raise ValueError(
            f"predict takes its configuration from the trained model; {forbidden} cannot be changed here. "
            f"Overridable at prediction time: {sorted(_PREDICT_OVERRIDABLE)}."
        )
    return RetainConfig.from_arguments({**dict(published), **supplied})


def _training_rows(
    assignments: pl.Series, config: RetainConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map MEDS splits onto RETAIN's fit/validation roles.

    `train` fits the parameters and `tuning` selects the epoch, mirroring the training/validation roles of
    `train_RETAIN`. A dataset that declares no tuning split falls back to a seeded hold-out carved from
    the training rows, which is what the official `load_data_simple` does internally.
    """
    values = assignments.to_list()
    train_rows = np.array([i for i, split in enumerate(values) if split == TRAIN_SPLIT], dtype=np.int64)
    validation_rows = np.array(
        [i for i, split in enumerate(values) if split == TUNING_SPLIT], dtype=np.int64
    )
    report: dict[str, Any] = {
        "observed": sorted({str(split) for split in values if split is not None}),
        "unassigned_rows": int(sum(1 for split in values if split is None)),
        "train_split_rows": int(train_rows.size),
        "tuning_split_rows": int(validation_rows.size),
        "validation_source": TUNING_SPLIT,
    }
    if validation_rows.size == 0 and train_rows.size > 1 and config.validation_fraction > 0:
        generator = np.random.default_rng(config.seed)
        shuffled = generator.permutation(train_rows)
        n_validation = max(1, int(round(config.validation_fraction * shuffled.size)))
        n_validation = min(n_validation, shuffled.size - 1)
        validation_rows = np.sort(shuffled[:n_validation])
        train_rows = np.sort(shuffled[n_validation:])
        report["validation_source"] = f"seeded {config.validation_fraction:g} hold-out of {TRAIN_SPLIT}"
    report["fit_rows"] = int(train_rows.size)
    report["validation_rows"] = int(validation_rows.size)
    return train_rows, validation_rows, report


def _coverage(assignments: pl.Series, published: pl.DataFrame) -> dict[str, Any]:
    """Requested and written key counts per evaluated split.

    Diagnostic metadata only. SPEC section 5 is explicit that a self-reported count is not evidence of
    coverage, so the tests compare key *sets*; this exists so a consumer can spot a partial artifact.
    """
    counts: dict[str, int] = {}
    for split in assignments.to_list():
        name = str(split) if split is not None else "unassigned"
        counts[name] = counts.get(name, 0) + 1
    return {"requested_by_split": counts, "written_rows": published.height}


COMMANDS = {
    CommandName.preprocess_data: PreprocessData,
    CommandName.supervised_train: SupervisedTrain,
    CommandName.predict: Predict,
}
