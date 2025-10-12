import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler

COMMON_DT_COLS = ['datetime', 'date', 'time', 'fecha', 'timestamp']
COMMON_TEMP_COLS = ['temperature', 'temp', 'temperatura', 'temp_c', 't2m']

def load_weather_csv(path: str, datetime_col: Optional[str] = None) -> pd.DataFrame:
    """Carga el CSV y convierte la columna de fecha/hora si existe."""
    df = pd.read_csv(path)
    dt_col = datetime_col
    if dt_col is None:
        for c in df.columns:
            if any(k in c.lower() for k in COMMON_DT_COLS):
                dt_col = c
                break
    if dt_col is not None and dt_col in df.columns:
        df[dt_col] = pd.to_datetime(df[dt_col], dayfirst=True, errors='coerce')
        df = df.sort_values(dt_col).reset_index(drop=True)
        df = df.set_index(dt_col)
    else:
        df = df.reset_index(drop=True)
    return df


def pick_target_column(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """Detecta la columna de temperatura o usa la primera numérica."""
    if target_col and target_col in df.columns:
        return target_col
    for name in df.columns:
        if any(k == name.lower() for k in COMMON_TEMP_COLS):
            return name
    for name in df.columns:
        if any(k in name.lower() for k in COMMON_TEMP_COLS):
            return name
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No se encontraron columnas numéricas para usar como target.")
    return num_cols[0]


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Deja solo columnas numéricas y completa valores faltantes."""
    num = df.select_dtypes(include=[np.number]).copy()
    thresh = int(0.7 * len(num))
    num = num.dropna(axis=1, thresh=thresh)
    num = num.ffill().bfill()
    return num


def train_test_split_time(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset de manera temporal (sin mezclar)."""
    n = len(df)
    split = int((1 - test_ratio) * n)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def scale_features(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Escala features con StandardScaler (fit en train, transform en test)."""
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[cols] = scaler.fit_transform(train[cols])
    test_scaled[cols] = scaler.transform(test[cols])
    return train_scaled, test_scaled, scaler


def make_windows(X: np.ndarray, y: np.ndarray, window: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Crea ventanas deslizantes para una CNN 1D."""
    n = len(X)
    samples = n - window - horizon + 1
    if samples <= 0:
        raise ValueError("No hay suficientes datos para crear las ventanas.")
    Xw = np.stack([X[i:i + window] for i in range(samples)], axis=0)
    yw = y[window + (horizon - 1): window + (horizon - 1) + samples]
    return Xw, yw


def train_val_split_from_train(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1):
    """Divide el conjunto de entrenamiento en train y validación."""
    n = len(X)
    split = int((1 - val_ratio) * n)
    return (X[:split], y[:split]), (X[split:], y[split:])
