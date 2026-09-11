
# 💳 Detección de Fraude Financiero

Aplicación de Machine Learning que detecta transacciones fraudulentas usando el
dataset público **AIML / PaySim** (6,3 millones de transacciones). Incluye:

- Notebook de análisis y entrenamiento (`analysis_model.ipynb`)
- App interactiva en Streamlit (`fraud_detection.py`)
- Pipeline serializado con scikit-learn (`fraud_detection_pipeline.pkl`)

---

## 📊 Dataset

El dataset contiene transacciones financieras simuladas con 11 columnas:

| Columna | Descripción |
|---|---|
| `step` | Unidad temporal (hora) |
| `type` | Tipo: PAYMENT, TRANSFER, CASH_OUT, DEPOSIT, DEBIT |
| `amount` | Monto de la transacción |
| `nameOrig` | Cuenta origen |
| `oldbalanceOrg` / `newbalanceOrig` | Saldo antes/después en origen |
| `nameDest` | Cuenta destino |
| `oldbalanceDest` / `newbalanceDest` | Saldo antes/después en destino |
| `isFraud` | **Variable objetivo** (1 = fraude) |
| `isFlaggedFraud` | Bandera del sistema (descartada por *data leakage*) |

**Desbalance:** solo **0,13 %** de las transacciones son fraudulentas.

---

## 🔍 Hallazgos principales del EDA

- Solo los tipos **TRANSFER** y **CASH_OUT** presentan fraude.
- El fraude se concentra en ventanas temporales específicas.
- Las cuentas fraudulentas no se repiten (no hay reincidencia por `nameOrig`).
- `oldbalanceOrg` y `newbalanceOrig` están fuertemente correlacionadas (0,999).
- El patrón "cuenta origen queda en 0" ocurre en 1,18 M transacciones, pero
  no distingue por sí solo el fraude.

---

## 🧠 Modelo

Pipeline de scikit-learn:

```
ColumnTransformer
 ├─ num: StandardScaler   → amount, oldbalanceOrg, newbalanceOrig,
 │                          oldbalanceDest, newbalanceDest
 └─ cat: OneHotEncoder    → type
        ↓
LogisticRegression(class_weight="balanced", max_iter=1000)
```

### Métricas (conjunto de test, 30 %)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| No fraude | 1.00 | 0.95 | 0.97 |
| **Fraude** | **0.02** | **0.94** | **0.04** |

- **Recall de fraude 94 %** → detecta casi todos los fraudes.
- **Precision 2 %** → muchos falsos positivos.
- Accuracy global: **94,55 %** (métrica engañosa por el desbalance).

Mejoras posibles: ajustar el *threshold* de decisión, usar `RandomForest`,
`XGBoost` o `LightGBM`, y evaluar con PR-AUC.




## 🖥️ Ejecutar la app

```bash
streamlit run fraud_detection.py
```
 (https://github.com/Mar-Urzag/Portfolio-Fraud-detection/blob/main/Streamlit-09-11-2026_08_02_PM.png " APP ")


### Uso

1. Selecciona el **tipo** de transacción.
2. Introduce el **monto** y los **saldos** de origen y destino.
3. Pulsa **Predecir**.
4. La app devuelve si la transacción es **fraude** o **no fraude**.



---

---

