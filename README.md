# Proyecto de Machine Learning: Análisis de Consumo Eléctrico

Este proyecto utiliza algoritmos de Machine Learning para analizar y predecir patrones de consumo eléctrico. Se divide en tres tareas principales: clustering, clasificación y forecasting. A continuación, se describen los objetivos, estructura y pasos para ejecutar el proyecto.

---

## Estructura del Proyecto

```
proyecto/
├── all_data/
│   ├── electricity_consumption.parquet
│   ├── weather.parquet
│   └── socioeconomic.parquet
├── models/
│   ├── kmeans_model.pkl
│   ├── classification_model.pkl
│   └── forecasting_model.pkl
├── clustering.py
├── classification.py
├── forecasting.py
├── main.py
├── requirements.txt
└── README.md
```

### Descripción de Archivos
- **`all_data/`**: Directorio que contiene los datasets necesarios para el análisis.
- **`models/`**: Almacena los modelos entrenados.
- **`clustering.py`**: Implementación del clustering.
- **`classification.py`**: Implementación de la clasificación.
- **`forecasting.py`**: Implementación del forecasting.
- **`main.py`**: Script principal para ejecutar el proyecto.
- **`requirements.txt`**: Lista de bibliotecas necesarias.
- **`README.md`**: Archivo de información.

---

## Requisitos Previos

### Dependencias
Instalar las dependencias necesarias ejecutando:
```bash
pip install -r requirements.txt
```

### Datos
Asegúrate de que los archivos de datos estén en el directorio `all_data/` con los siguientes nombres:
- `electricity_consumption.parquet`
- `weather.parquet`
- `socioeconomic.parquet`

---

## Ejecución del Proyecto

### Pasos para Ejecutar
1. **Clonar el repositorio**:
   ```bash
   git clone <url_del_repositorio>
   cd proyecto
   ```
2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ejecutar el script principal**:
   ```bash
   python main.py
   ```

### Salidas Generadas
- Modelos entrenados se guardan en `models/`.

---

## Descripción de las Tareas

### 1. Clustering
**Objetivo**: Identificar patrones diarios de consumo eléctrico.
- Algoritmo: K-means.
- Salida: Modelo en `models/kmeans_model.pkl`.

### 2. Clasificación
**Objetivo**: Clasificar curvas de carga en clusters identificados.
- Algoritmo: Random Forest.
- Salida: Modelo en `models/classification_model.pkl`.

### 3. Forecasting
**Objetivo**: Predecir consumo eléctrico a corto plazo.
- Algoritmo: Gradient Boosting Regressor.
- Salida: Modelo en `models/forecasting_model.pkl`.

---


