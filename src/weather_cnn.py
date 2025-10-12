import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, tensorflow as tf
np.random.seed(42); random.seed(42); tf.random.set_seed(42)
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from utils_timeseries import (
    load_weather_csv, pick_target_column, clean_numeric,
    train_test_split_time, scale_features, make_windows, train_val_split_from_train
)

CSV_PATH = Path('data/Argentina_weather_data.csv')  
DATETIME_COL = "Date"   
TARGET_COL = "Temp_Mean"   
WINDOW = 21           
HORIZON = 1           
TEST_RATIO = 0.2
VAL_RATIO = 0.1
EPOCHS = 30
BATCH_SIZE = 64
OUTPUTS_DIR = Path('outputs')
MODELS_DIR = Path('models/weather_cnn')
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(f'No se encontró el CSV en {CSV_PATH}. Colocá el archivo en data/')

df = load_weather_csv(str(CSV_PATH), datetime_col=DATETIME_COL)
numeric = clean_numeric(df)

numeric['Temp_Mean_lag1'] = numeric['Temp_Mean'].shift(1)
numeric = numeric.dropna() 

target_name = pick_target_column(numeric, target_col=TARGET_COL)
feature_cols = [c for c in numeric.columns if c != target_name]

print(f"Target: {target_name} | Ventana: {WINDOW} | Horizonte: {HORIZON}")
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Columnas numéricas totales: {list(numeric.columns)}")

if len(feature_cols) == 0:
    raise ValueError('No hay columnas numéricas adicionales al target.')

train_df, test_df = train_test_split_time(numeric, test_ratio=TEST_RATIO)

print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
print(f"Índices (train -> test): {train_df.index.min()} → {train_df.index.max()}  |  {test_df.index.min()} → {test_df.index.max()}")

train_df_scaled, test_df_scaled, feat_scaler = scale_features(train_df, test_df, feature_cols)
y_scaler = StandardScaler()
train_df_scaled[target_name] = y_scaler.fit_transform(train_df[[target_name]])
test_df_scaled[target_name] = y_scaler.transform(test_df[[target_name]])

X_train = train_df_scaled[feature_cols].to_numpy()
y_train = train_df_scaled[target_name].to_numpy()
X_test = test_df_scaled[feature_cols].to_numpy()
y_test = test_df_scaled[target_name].to_numpy()

Xw_train, yw_train = make_windows(X_train, y_train, window=WINDOW, horizon=HORIZON)
Xw_test, yw_test = make_windows(X_test, y_test, window=WINDOW, horizon=HORIZON)

(X_tr, y_tr), (X_val, y_val) = train_val_split_from_train(Xw_train, yw_train, val_ratio=VAL_RATIO)

n_features = X_tr.shape[-1]
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(WINDOW, n_features)),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=5e-4), loss='mae')
model.summary()

ckpt_path = (MODELS_DIR / 'best.keras')
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True)
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    shuffle=False
)

plt.figure(figsize=(7,4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Curva de entrenamiento")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.legend()
curve_path = OUTPUTS_DIR / "training_curve.png"
plt.savefig(str(curve_path), dpi=120)
print(f"Curva de entrenamiento guardada en: {curve_path}")

yhat_test_scaled = model.predict(Xw_test).reshape(-1)
yw_test_orig = y_scaler.inverse_transform(yw_test.reshape(-1, 1)).reshape(-1)
yhat_test_orig = y_scaler.inverse_transform(yhat_test_scaled.reshape(-1, 1)).reshape(-1)

y_test_orig_all = y_scaler.inverse_transform(test_df_scaled[[target_name]]).reshape(-1)
samples = len(yw_test_orig)
baseline_pred = y_test_orig_all[WINDOW-1 : WINDOW-1 + samples]

rmse = lambda a, b: mean_squared_error(a, b) ** 0.5 
mae_cnn = mean_absolute_error(yw_test_orig, yhat_test_orig)
rmse_cnn = rmse(yw_test_orig, yhat_test_orig)
mae_bl = mean_absolute_error(yw_test_orig, baseline_pred)
rmse_bl = rmse(yw_test_orig, baseline_pred)


print(f"""
Resultados (escala original):
CNN  -> MAE: {mae_cnn:.3f} | RMSE: {rmse_cnn:.3f}
Base -> MAE: {mae_bl:.3f} | RMSE: {rmse_bl:.3f}
""")

imp_mae = (mae_bl - mae_cnn) / mae_bl * 100
imp_rmse = (rmse_bl - rmse_cnn) / rmse_bl * 100
print(f"Mejora vs baseline -> MAE: {imp_mae:.2f}% | RMSE: {imp_rmse:.2f}%")

plt.figure(figsize=(10, 4))
N = min(500, len(yw_test_orig))
plt.plot(yw_test_orig[-N:], label='Real', linewidth=2)
plt.plot(yhat_test_orig[-N:], label='Predicción CNN', alpha=0.8)
plt.title('Comparación y_real vs y_predicha (Test)')
plt.xlabel('Tiempo (índice)')
plt.ylabel('Temperatura')
plt.legend()
plt.tight_layout()
plot_path = OUTPUTS_DIR / 'pred_vs_real_test.png'
plt.savefig(str(plot_path), dpi=120) 
print(f'Gráfico guardado en: {plot_path}')

final_model_path = MODELS_DIR / 'final.keras'
model.save(str(final_model_path))   
print(f'Modelo guardado en: {final_model_path}')
