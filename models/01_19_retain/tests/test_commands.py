"""Command-level behaviour: artifact compatibility, configuration discipline, and learning.

SPEC section 8 asks for artifact compatibility, atomic publication, absence of label leakage and
predicate provenance. This port consumes no predicates, so the provenance obligation is discharged by
refusing the argument outright rather than by recording a digest for an input it ignores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

torch = pytest.importorskip("torch")

from retain.commands import COMMANDS, IS_STUB  # noqa: E402
from retain.config import RetainConfig  # noqa: E402
from retain.model import Retain  # noqa: E402
from retain.training import _cross_entropy, collate, roc_auc, score, train  # noqa: E402
from retain.windows import SequenceBatch  # noqa: E402

from .fixtures import SMOKE_HYPERPARAMETERS, write_meds_dataset, write_task_labels  # noqa: E402


def test_the_port_is_not_a_stub():
    assert IS_STUB is False


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("retain-commands")
    meds_dir = write_meds_dataset(root / "meds")
    labels_dir = write_task_labels(meds_dir, root / "labels")
    data_dir = root / "data"
    COMMANDS["preprocess_data"]()(
        {"external_meds_dir": meds_dir, "output_data_dir": data_dir, "do_overwrite": True}
    )
    model_dir = root / "model"
    COMMANDS["supervised_train"]()(
        {
            "input_data_dir": data_dir,
            "external_labels_dir": labels_dir,
            "output_supervised_model_dir": model_dir,
            "do_overwrite": True,
            **SMOKE_HYPERPARAMETERS,
        }
    )
    return {"root": root, "meds": meds_dir, "labels": labels_dir, "data": data_dir, "model": model_dir}


def _manifest(directory: Path) -> dict:
    return yaml.safe_load((Path(directory) / "manifest.yaml").read_text())


def test_the_fixture_contains_a_key_with_no_history(dataset):
    """The degenerate cohort is present, so the coverage guarantee is actually being exercised."""
    extra = _manifest(dataset["model"])
    assert extra["work_units"]["empty_history_train_rows"] > 0


def test_every_requested_key_is_scored_including_empty_histories(dataset, tmp_path):
    predictions_dir = tmp_path / "predictions"
    COMMANDS["predict"]()(
        {
            "input_supervised_model_dir": dataset["model"],
            "external_labels_dir": dataset["labels"],
            "output_predictions_dir": predictions_dir,
            "do_overwrite": True,
        }
    )
    manifest = _manifest(predictions_dir)
    assert manifest["exclusions"]["policy"].startswith("none")
    assert manifest["exclusions"]["empty_history_keys"] > 0
    written = pl.read_parquet(predictions_dir / "predictions.parquet")
    requested = pl.read_parquet(sorted(Path(dataset["labels"]).rglob("*.parquet")))
    assert written.height == requested.height
    assert (
        written.select("subject_id", "prediction_time")
        .join(requested.select("subject_id", "prediction_time"), on=["subject_id", "prediction_time"], how="anti")
        .is_empty()
    )
    assert written.get_column("predicted_boolean_probability").is_between(0.0, 1.0).all()


def test_predictions_are_restricted_to_the_requested_split(dataset, tmp_path):
    predictions_dir = tmp_path / "held-out-only"
    COMMANDS["predict"]()(
        {
            "input_supervised_model_dir": dataset["model"],
            "external_labels_dir": dataset["labels"],
            "output_predictions_dir": predictions_dir,
            # MEDS-DEV expands its template to this non-JSON form.
            "splits": "[held_out]",
            "do_overwrite": True,
        }
    )
    written = pl.read_parquet(predictions_dir / "predictions.parquet")
    expected = pl.read_parquet(sorted((Path(dataset["labels"]) / "held_out").rglob("*.parquet")))
    assert written.height == expected.height
    assert _manifest(predictions_dir)["inputs"]["splits"] == ["held_out"]


def test_the_interpretation_pass_is_off_unless_asked_for(dataset, tmp_path):
    predictions_dir = tmp_path / "no-interpretation"
    COMMANDS["predict"]()(
        {
            "input_supervised_model_dir": dataset["model"],
            "external_labels_dir": dataset["labels"],
            "output_predictions_dir": predictions_dir,
            "interpretation_top_k": 0,
            "do_overwrite": True,
        }
    )
    assert not (predictions_dir / "interpretation.ndjson").exists()
    assert "interpretation" not in _manifest(predictions_dir)


def test_contributions_reproduce_the_score_they_explain(dataset, tmp_path):
    """Eq. (4) says the logit is the sum of every code's Eq. (5) contribution, plus the bias."""
    predictions_dir = tmp_path / "interpreted"
    COMMANDS["predict"]()(
        {
            "input_supervised_model_dir": dataset["model"],
            "external_labels_dir": dataset["labels"],
            "output_predictions_dir": predictions_dir,
            "interpretation_top_k": 10_000,
            "do_overwrite": True,
        }
    )
    records = [
        json.loads(line) for line in (predictions_dir / "interpretation.ndjson").read_text().splitlines()
    ]
    checkpoint = torch.load(Path(dataset["model"]) / "model.pt", weights_only=True)
    bias = float(checkpoint["state_dict"]["w_output.bias"])
    checked = 0
    for record in records:
        if not record["n_visits"]:
            continue
        total = sum(entry["contribution"] for entry in record["top_contributions"]) + bias
        assert 1.0 / (1.0 + np.exp(-total)) == pytest.approx(
            record["predicted_boolean_probability"], abs=1e-4
        )
        checked += 1
    assert checked > 0


def test_predict_rejects_a_patients_artifact_where_a_model_belongs(dataset, tmp_path):
    with pytest.raises(ValueError, match="is a 'patients' artifact"):
        COMMANDS["predict"]()(
            {
                "input_supervised_model_dir": dataset["data"],
                "external_labels_dir": dataset["labels"],
                "output_predictions_dir": tmp_path / "predictions",
            }
        )


def test_training_rejects_an_incompatible_payload_format(dataset, tmp_path):
    forged = tmp_path / "forged"
    forged.mkdir()
    manifest = _manifest(dataset["data"])
    manifest["artifact"]["payload_format"] = "retain-patients/v99"
    (forged / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="payload format 'retain-patients/v99'"):
        COMMANDS["supervised_train"]()(
            {
                "input_data_dir": forged,
                "external_labels_dir": dataset["labels"],
                "output_supervised_model_dir": tmp_path / "model",
            }
        )


@pytest.mark.parametrize("command", ["preprocess_data", "supervised_train", "predict"])
def test_every_command_refuses_a_predicates_file(dataset, tmp_path, command):
    predicates = tmp_path / "predicates.yaml"
    predicates.write_text("predicates: {}\n")
    with pytest.raises(ValueError, match="predicate-free"):
        COMMANDS[command]()(
            {
                "external_meds_dir": dataset["meds"],
                "input_data_dir": dataset["data"],
                "input_supervised_model_dir": dataset["model"],
                "external_labels_dir": dataset["labels"],
                "output_data_dir": tmp_path / "data",
                "output_supervised_model_dir": tmp_path / "model",
                "output_predictions_dir": tmp_path / "predictions",
                "external_predicates_file": predicates,
            }
        )


def test_a_mistyped_hyperparameter_is_an_error_not_a_silent_no_op(dataset, tmp_path):
    with pytest.raises(ValueError, match="Unknown configuration argument"):
        COMMANDS["supervised_train"]()(
            {
                "input_data_dir": dataset["data"],
                "external_labels_dir": dataset["labels"],
                "output_supervised_model_dir": tmp_path / "model",
                "embed_sizes": 32,
            }
        )


def test_vocabulary_settings_cannot_be_changed_after_preprocessing(dataset, tmp_path):
    with pytest.raises(ValueError, match="preprocess_data setting"):
        COMMANDS["supervised_train"]()(
            {
                "input_data_dir": dataset["data"],
                "external_labels_dir": dataset["labels"],
                "output_supervised_model_dir": tmp_path / "model",
                "min_code_count": 5,
            }
        )


def test_predict_refuses_to_re_specify_the_model_it_loads(dataset, tmp_path):
    with pytest.raises(ValueError, match="takes its configuration from the trained model"):
        COMMANDS["predict"]()(
            {
                "input_supervised_model_dir": dataset["model"],
                "external_labels_dir": dataset["labels"],
                "output_predictions_dir": tmp_path / "predictions",
                "embed_size": 4,
            }
        )


def test_a_failed_command_publishes_nothing(dataset, tmp_path):
    output = tmp_path / "never-published"
    with pytest.raises(FileNotFoundError):
        COMMANDS["preprocess_data"]()(
            {"external_meds_dir": tmp_path / "absent", "output_data_dir": output}
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".*staging*"))


def test_publishing_over_an_existing_artifact_needs_do_overwrite(dataset, tmp_path):
    arguments = {
        "external_meds_dir": dataset["meds"],
        "output_data_dir": tmp_path / "data",
        "max_vocab_size": 8,
    }
    COMMANDS["preprocess_data"]()(arguments)
    with pytest.raises(FileExistsError):
        COMMANDS["preprocess_data"]()(arguments)


def test_the_recorded_configuration_digest_tracks_the_configuration():
    assert RetainConfig().digest() == RetainConfig().digest()
    assert RetainConfig().digest() != RetainConfig(embed_size=64).digest()


def _toy_batch(n_rows, generator):
    """Sequences whose label is decided by whether code 1 ever occurs."""
    row_ptr = [0]
    visit_ptr = [0]
    codes: list[int] = []
    labels = []
    for _ in range(n_rows):
        positive = generator.random() < 0.5
        n_visits = generator.integers(1, 5)
        for visit in range(n_visits):
            visit_codes = [int(generator.integers(2, 8))]
            if positive and visit == n_visits - 1:
                visit_codes.append(1)
            codes.extend(visit_codes)
            visit_ptr.append(visit_ptr[-1] + len(visit_codes))
        row_ptr.append(row_ptr[-1] + int(n_visits))
        labels.append(float(positive))
    batch = SequenceBatch(
        keys=np.column_stack((np.arange(n_rows, dtype=np.int64), np.zeros(n_rows, dtype=np.int64))),
        row_ptr=np.array(row_ptr, dtype=np.int64),
        visit_ptr=np.array(visit_ptr, dtype=np.int64),
        visit_time=np.arange(len(visit_ptr) - 1, dtype=np.int64) * 86_400_000_000,
        codes=np.array(codes, dtype=np.int32),
    )
    return batch, np.array(labels, dtype=np.float32)


def test_every_trainable_parameter_receives_a_gradient():
    """A parameter left out of the graph would train silently to nothing."""
    config = RetainConfig(embed_size=8, alpha_hidden_dim_size=6, beta_hidden_dim_size=6, batch_size=4)
    generator = np.random.default_rng(0)
    batch, labels = _toy_batch(8, generator)
    model = Retain(8, config)
    tensors = collate(batch, np.arange(8), config, torch.device("cpu"))
    output = model(
        tensors.codes,
        tensors.offsets,
        tensors.slots,
        tensors.lengths,
        tensors.gaps,
        n_rows=tensors.n_rows,
        n_visits=tensors.n_visits,
    )
    loss = _cross_entropy(output.logits, torch.from_numpy(labels), config.log_eps)
    (loss + model.regularization()).backward()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_training_learns_a_separable_signal():
    """A synthetic sanity check that the optimiser moves the model, not a performance claim."""
    config = RetainConfig(
        embed_size=8, alpha_hidden_dim_size=8, beta_hidden_dim_size=8, batch_size=16, n_epochs=40
    )
    generator = np.random.default_rng(3)
    train_batch, train_labels = _toy_batch(160, generator)
    validation_batch, validation_labels = _toy_batch(60, generator)
    model, history = train(
        vocabulary_size=8,
        train_batch=train_batch,
        train_labels=train_labels,
        validation_batch=validation_batch,
        validation_labels=validation_labels,
        config=config,
        device=torch.device("cpu"),
    )
    assert history.train_losses[-1] < history.train_losses[0]
    assert roc_auc(validation_labels, score(model, validation_batch, config, torch.device("cpu"))) > 0.9
    assert history.selected_by == "validation_auc"
