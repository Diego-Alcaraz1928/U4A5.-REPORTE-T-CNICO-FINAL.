# ==========================================
# DETECTOR DE OSCILACIÓN PARA SEGUIDOR DE LÍNEA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CARGAR DATASET
# ==========================================

archivo = "C:/Users/diego/Desktop/robot.csv"

df = pd.read_csv(archivo)

print("\nPrimeras filas:")
print(df.head())

# ==========================================
# 2. CREAR DELTA ERROR
# ==========================================

df["error_anterior"] = df["error"].shift(1)

df["delta_error"] = (
    df["error"] - df["error_anterior"]
)

df = df.dropna()

# ==========================================
# 3. CREAR ETIQUETA OSCILANDO
# ==========================================

# Puedes modificar esta lógica

df["oscilando"] = df["estado"].apply(
    lambda x: 1
    if x in ["desviado", "recuperacion"]
    else 0
)

print("\nDistribución:")
print(df["oscilando"].value_counts())

# ==========================================
# 4. VARIABLES DE ENTRADA
# ==========================================

features = [
    "s0",
    "s1",
    "s2",
    "s3",
    "error",
    "corr",
    "error_anterior",
    "delta_error"
]

X = df[features]

y = df["oscilando"]

# ==========================================
# 5. DIVIDIR DATOS
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 6. ENTRENAR RANDOM FOREST
# ==========================================

modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

modelo.fit(X_train, y_train)

# ==========================================
# 7. EVALUACIÓN
# ==========================================

pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\n===================================")
print("RESULTADOS DEL MODELO")
print("===================================")

print(f"\nPrecisión: {accuracy*100:.2f}%")

print("\nReporte:")
print(classification_report(y_test, pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, pred))

# ==========================================
# 8. IMPORTANCIA DE VARIABLES
# ==========================================

importancias = pd.DataFrame({
    "Variable": features,
    "Importancia": modelo.feature_importances_
})

importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

print("\nImportancia de variables:")
print(importancias)

# ==========================================
# 9. ZONA MUERTA RECOMENDADA
# ==========================================

# Casos donde NO oscila

estables = df[df["oscilando"] == 0]

errores_estables = np.abs(estables["error"])

zona_muerta_80 = np.percentile(
    errores_estables,
    80
)

zona_muerta_90 = np.percentile(
    errores_estables,
    90
)

zona_muerta_95 = np.percentile(
    errores_estables,
    95
)

print("\n===================================")
print("ZONA MUERTA RECOMENDADA")
print("===================================")

print(
    f"Conservadora (80%): ±{zona_muerta_80:.3f}"
)

print(
    f"Recomendada (90%): ±{zona_muerta_90:.3f}"
)

print(
    f"Agresiva (95%): ±{zona_muerta_95:.3f}"
)

# ==========================================
# 10. GUARDAR MODELO
# ==========================================

import joblib

joblib.dump(
    modelo,
    "modelo_oscilacion.pkl"
)

print(
    "\nModelo guardado como:"
    " modelo_oscilacion.pkl"
)

# ==========================================
# 11. GRÁFICA IMPORTANCIA
# ==========================================

plt.figure(figsize=(10,5))

plt.bar(
    importancias["Variable"],
    importancias["Importancia"]
)

plt.title(
    "Importancia de Variables"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ==========================================
# 12. HISTOGRAMA ERROR
# ==========================================

plt.figure(figsize=(10,5))

plt.hist(
    errores_estables,
    bins=30
)

plt.axvline(
    zona_muerta_80,
    linestyle="--",
    label=f"80% = {zona_muerta_80:.3f}"
)

plt.axvline(
    zona_muerta_90,
    linestyle="--",
    label=f"90% = {zona_muerta_90:.3f}"
)

plt.axvline(
    zona_muerta_95,
    linestyle="--",
    label=f"95% = {zona_muerta_95:.3f}"
)

plt.legend()

plt.title(
    "Distribución del Error Estable"
)

plt.xlabel("abs(error)")
plt.ylabel("Frecuencia")

plt.show()

# ==========================================
# 13. RESUMEN FINAL
# ==========================================

print("\n===================================")
print("RESUMEN FINAL")
print("===================================")

print(
    f"Precisión del modelo: "
    f"{accuracy*100:.2f}%"
)

print(
    f"Zona muerta sugerida: "
    f"±{zona_muerta_90:.3f}"
)

print(
    "\nSi abs(error) es menor que la "
    "zona muerta, puedes probar:"
)

print(
    f"\nif(abs(error) < {zona_muerta_90:.3f})"
)

print("{")
print("    correccion = 0;")
print("}")