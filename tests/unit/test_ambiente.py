import importlib

import pytest


@pytest.mark.parametrize(
    "modulo",
    [
        "pandas",
        "pyarrow",
        "duckdb",
        "pandera",
        "streamlit",
        "plotly",
        "sklearn",
        "scipy",
        "groq",
        "dotenv",
    ],
)
def test_dependencias_importam(modulo):
    importlib.import_module(modulo)
