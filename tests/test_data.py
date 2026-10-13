import pandas as pd
import pytest

from src.data import build_target, download_dataset, file_sha256, load_raw


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "Admission grade": [120.0, 135.5, 110.0, 142.3],
            "Target": ["Dropout", "Graduate", "Enrolled", "Dropout"],
        }
    )


def test_build_target_binary_maps_dropout_to_one(df):
    y = build_target(df)
    assert y.tolist() == [1, 0, 0, 1]


def test_build_target_binary_keeps_enrolled_rows(df):
    y = build_target(df)
    assert len(y) == len(df)
    assert y[df["Target"] == "Enrolled"].tolist() == [0]


def test_build_target_multiclass_keeps_original_classes(df):
    y = build_target(df, binary=False)
    assert y.tolist() == ["Dropout", "Graduate", "Enrolled", "Dropout"]


def test_build_target_does_not_modify_input(df):
    original = df.copy()
    build_target(df)
    build_target(df, binary=False)
    pd.testing.assert_frame_equal(df, original)


def test_build_target_missing_column_raises(df):
    with pytest.raises(ValueError, match="Target"):
        build_target(df.drop(columns="Target"))


def test_build_target_unknown_class_raises(df):
    df.loc[0, "Target"] = "Transfer"
    with pytest.raises(ValueError, match="Transfer"):
        build_target(df)


# Descarga y carga: sin red, usando un CSV local servido con file://


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "origen" / "data.csv"
    path.parent.mkdir()
    path.write_text("Marital Status,Target\n1,Dropout\n2,Graduate\n", encoding="utf-8-sig")
    return path


def test_download_dataset_copies_and_verifies(source, tmp_path):
    dest = tmp_path / "raw" / "data.csv"
    result = download_dataset(dest, url=source.as_uri(), expected_sha256=file_sha256(source))
    assert result == dest
    assert file_sha256(dest) == file_sha256(source)


def test_download_dataset_skips_valid_existing_file(source, tmp_path):
    dest = tmp_path / "data.csv"
    dest.write_bytes(source.read_bytes())
    missing_url = (tmp_path / "no_existe.csv").as_uri()
    download_dataset(dest, url=missing_url, expected_sha256=file_sha256(source))
    assert file_sha256(dest) == file_sha256(source)


def test_download_dataset_replaces_corrupt_file(source, tmp_path):
    dest = tmp_path / "data.csv"
    dest.write_text("corrupto")
    download_dataset(dest, url=source.as_uri(), expected_sha256=file_sha256(source))
    assert file_sha256(dest) == file_sha256(source)


def test_download_dataset_wrong_hash_raises_and_leaves_nothing(source, tmp_path):
    dest = tmp_path / "raw" / "data.csv"
    with pytest.raises(ValueError, match="SHA-256"):
        download_dataset(dest, url=source.as_uri(), expected_sha256="0" * 64)
    assert list(dest.parent.iterdir()) == []


def test_load_raw_strips_bom(source):
    df = load_raw(source)
    assert df.columns[0] == "Marital Status"
    assert df.shape == (2, 2)


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="uv run python -m src.data"):
        load_raw(tmp_path / "no_existe.csv")
