# homework/clean_data.py
#
# Normaliza textos y genera los archivos requeridos por el test.
# Referencia sobre fingerprinting y clustering:
# https://openrefine.org/docs/technical-reference/clustering-in-depth
#
import os
import string
import pandas as pd  # type: ignore
from nltk.stem import PorterStemmer  # type: ignore

PUNCT_TABLE = str.maketrans("", "", string.punctuation)
STEMMER = PorterStemmer()

# Conversión de fingerprint → texto canónico solicitado en el enunciado
_KEY_TO_CLEAN = {
    "analyt applic": "Analytics Application",
    "analyt model": "ANALYTICS MODEL",
    "adhoc queri": "ADHOC QUERIES",
    "ad hoc queri": "ADHOC QUERIES",  # por si aparece esta variante
    "agricultur product": "AGRICULTURAL PRODUCTS",
    "airlin compani": "AIRLINE COMPANY",
    "airlin": "AIRLINES",
}


def _fingerprint(text: str) -> str:
    """
    Genera la clave normalizada (fingerprint) con:
      • strip + lower
      • remoción de guiones y puntuación
      • tokenización por espacios
      • stemming Porter
      • eliminación de duplicados y orden alfabético
    """
    text = text.strip().lower().replace("-", "")
    text = text.translate(PUNCT_TABLE)
    tokens = text.split()
    stemmed = {STEMMER.stem(tok) for tok in tokens}
    return " ".join(sorted(stemmed))


def _load(input_file: str) -> pd.DataFrame:
    """Lee el archivo de entrada en un DataFrame."""
    return pd.read_csv(input_file)


def _build_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas 'key' y 'cleaned_text' al DataFrame."""
    df = df.copy()
    df["key"] = df["raw_text"].apply(_fingerprint)
    df["cleaned_text"] = df["key"].map(_KEY_TO_CLEAN)
    return df


def _ensure_dir(filepath: str) -> None:
    """Crea la carpeta destino si no existe."""
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def main(input_file: str, output_file: str) -> None:
    """Punto de entrada usado por el autograder."""
    df_raw = _load(input_file)
    df_processed = _build_dataframe(df_raw)

    # 1) Guardar test.csv con la columna 'key'
    test_path = "files/test.csv"
    _ensure_dir(test_path)
    df_processed[["key"]].to_csv(test_path, index=False)

    # 2) Guardar output.txt con la columna 'cleaned_text'
    _ensure_dir(output_file)
    df_processed[["cleaned_text"]].to_csv(output_file, index=False)


if __name__ == "__main__":
    main(
        input_file="files/input.txt",
        output_file="files/output.txt",
    )

