import numpy as np
import pandas as pd
import pytest

from src.data import cv_folds, load_splits, make_splits, save_splits

SHA = "a" * 64


@pytest.fixture
def df():
    # 500 filas con proporciones parecidas al dataset real (50 / 32 / 18 %).
    target = ["Graduate"] * 250 + ["Dropout"] * 160 + ["Enrolled"] * 90
    return pd.DataFrame({"Admission grade": range(500), "Target": target})


@pytest.fixture
def splits(df):
    return make_splits(df, data_sha256=SHA)


def proportions(target):
    return target.value_counts(normalize=True)


def test_make_splits_is_reproducible(df):
    assert make_splits(df, data_sha256=SHA) == make_splits(df, data_sha256=SHA)


def test_train_and_test_are_disjoint_and_cover_all_rows(df, splits):
    train, test = set(splits["train"]), set(splits["test"])
    assert not train & test
    assert train | test == set(range(len(df)))
    assert len(test) == 100


def test_class_proportions_preserved_within_2_points(df, splits):
    total = proportions(df["Target"])
    for part in (splits["train"], splits["test"]):
        diff = (proportions(df["Target"].iloc[part]) - total).abs()
        assert diff.max() <= 0.02


def test_folds_are_defined_only_within_train(splits):
    assert len(splits["folds"]) == len(splits["train"])
    assert set(splits["folds"]) == set(range(5))


def test_cv_folds_partition_train_positions(tmp_path, splits):
    loaded = load_splits(save_splits(splits, tmp_path / "splits.json"), expected_sha256=SHA)
    validation = np.concatenate([val for _, val in cv_folds(loaded)])
    assert np.array_equal(np.sort(validation), np.arange(len(loaded["train"])))
    for fit, val in cv_folds(loaded):
        assert not np.intersect1d(fit, val).size


def test_load_splits_roundtrip(tmp_path, splits):
    path = save_splits(splits, tmp_path / "splits.json")
    loaded = load_splits(path, expected_sha256=SHA)
    assert loaded["train"].tolist() == splits["train"]
    assert loaded["test"].tolist() == splits["test"]


def test_load_splits_rejects_other_dataset_version(tmp_path, splits):
    path = save_splits(splits, tmp_path / "splits.json")
    with pytest.raises(ValueError, match="otra versión del dataset"):
        load_splits(path, expected_sha256="b" * 64)


def test_load_splits_rejects_overlap(tmp_path, splits):
    splits["test"] = splits["test"] + [splits["train"][0]]
    path = save_splits(splits, tmp_path / "splits.json")
    with pytest.raises(ValueError, match="solapan"):
        load_splits(path, expected_sha256=SHA)
