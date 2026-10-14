import numpy as np
import pandas as pd
import pytest

from src.config import (
    DATA_SHA256,
    FEATURES_INGRESO,
    FEATURES_PROHIBIDAS,
    FEATURES_SENSIBILIDAD,
    RAW_DATA_PATH,
    TARGET_CLASSES,
    TARGET_COL,
    TEST_SIZE,
)
from src.data import (
    build_target,
    cv_folds,
    download_dataset,
    file_sha256,
    load_raw,
    load_splits,
    make_splits,
)
from src.features import select_features

ALL_FEATURES = list(FEATURES_INGRESO + FEATURES_SENSIBILIDAD + FEATURES_PROHIBIDAS)


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


# Pruebas de contrato y anti-fuga sobre los datos reales (#10, pruebas 1–3 de 8).
# Requieren data/raw/data.csv: uv run python -m src.data

DECIMAL_COLUMNS = {
    "Previous qualification (grade)",
    "Admission grade",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
}

BINARY_COLUMNS = (
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Gender",
    "International",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
)

# Rangos válidos según la escala de cada variable (reports/auditoria_datos.md, sección 2).
VALID_RANGES = {
    "Age at enrollment": (15, 100),
    "Admission grade": (0, 200),
    "Previous qualification (grade)": (0, 200),
    "Curricular units 1st sem (grade)": (0, 20),
    "Curricular units 2nd sem (grade)": (0, 20),
}


@pytest.fixture(scope="module")
def raw():
    if not RAW_DATA_PATH.exists():
        pytest.skip("Falta data/raw/data.csv: uv run python -m src.data")
    return load_raw()


def test_contrato_de_esquema(raw):
    """Prueba 1: el CSV real cumple el contrato de entrada del proyecto.

    Protege contra una versión distinta del dataset o una lectura incorrecta
    (separador, BOM, tipos): exige las 36 columnas de config.py más Target,
    tipos numéricos esperados, sin nulos ni duplicados, valores dentro de la
    escala de cada variable, binarias en {0, 1}, sin negativos (salvo
    inflación y PIB), solo las 3 clases del target
    y nunca más unidades aprobadas que inscritas.
    """
    assert file_sha256(RAW_DATA_PATH) == DATA_SHA256

    expected = set(ALL_FEATURES) | {TARGET_COL}
    assert set(raw.columns) == expected
    assert raw.shape == (4424, len(expected))

    for column in ALL_FEATURES:
        kind = "f" if column in DECIMAL_COLUMNS else "i"
        assert raw[column].dtype.kind == kind, column

    assert raw.isna().sum().sum() == 0
    assert not raw.duplicated().any()

    for column, (low, high) in VALID_RANGES.items():
        assert raw[column].between(low, high).all(), column
    for column in BINARY_COLUMNS:
        assert set(raw[column].unique()) <= {0, 1}, column
    # Inflation rate y GDP son variaciones anuales: pueden ser negativas.
    non_negative = [c for c in ALL_FEATURES if c not in ("Inflation rate", "GDP")]
    assert (raw[non_negative] >= 0).all().all()
    assert set(raw[TARGET_COL].unique()) == set(TARGET_CLASSES)

    for sem in ("1st", "2nd"):
        approved = raw[f"Curricular units {sem} sem (approved)"]
        enrolled = raw[f"Curricular units {sem} sem (enrolled)"]
        assert (approved <= enrolled).all(), sem


def test_anti_fuga_matriz_de_entrada(raw):
    """Prueba 2: ninguna variable posterior al ingreso llega al modelo.

    Protege la decisión 001: la matriz X que produce select_features() sobre
    los datos reales, tanto en el modelo principal como en el análisis de
    sensibilidad, no contiene FEATURES_PROHIBIDAS, ninguna columna de
    rendimiento semestral (comprobado por nombre, por si una lista de
    config.py se edita mal) ni el target. Además, ninguna columna de X
    reproduce el target por sí sola.
    """
    y = build_target(raw)
    for sensibilidad in (False, True):
        X = select_features(raw, sensibilidad=sensibilidad)
        columns = set(X.columns)
        assert not columns & set(FEATURES_PROHIBIDAS)
        assert not [c for c in columns if "Curricular units" in c]
        assert TARGET_COL not in columns
        assert len(X) == len(raw)
        for column in X.columns:
            assert not X[column].equals(y), column

    assert set(select_features(raw).columns) == set(FEATURES_INGRESO)


def test_particion_valida(raw):
    """Prueba 3: la partición versionada es válida y reproducible.

    Protege la decisión 003 sobre el archivo real data/processed/splits.json:
    corresponde al data.csv actual (SHA-256), train y test no se solapan y
    cubren todas las filas, se regenera idéntica con la semilla del proyecto
    y conserva la proporción de cada clase (±2 puntos) en train, test y cada
    fold de validación.
    """
    splits = load_splits()
    train, test = splits["train"], splits["test"]

    assert splits["n_rows"] == len(raw)
    assert not np.intersect1d(train, test).size
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(raw)))
    assert len(test) == round(len(raw) * TEST_SIZE)

    regenerated = make_splits(raw, data_sha256=DATA_SHA256)
    assert regenerated["train"] == train.tolist()
    assert regenerated["test"] == test.tolist()
    assert regenerated["folds"] == splits["folds"].tolist()

    target = raw[TARGET_COL]
    total = target.value_counts(normalize=True)
    train_target = target.iloc[train].reset_index(drop=True)
    parts = [target.iloc[train], target.iloc[test]]
    parts += [train_target[val] for _, val in cv_folds(splits)]
    for part in parts:
        diff = (part.value_counts(normalize=True) - total).abs()
        assert diff.max() <= 0.02
