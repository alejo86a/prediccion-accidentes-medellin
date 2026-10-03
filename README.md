# Medellín Traffic Accident Severity Prediction

[![lint](https://github.com/alejo86a/prediccion-accidentes-medellin/actions/workflows/lint.yml/badge.svg)](https://github.com/alejo86a/prediccion-accidentes-medellin/actions/workflows/lint.yml)

A machine learning project (CRISP-DM methodology) that predicts whether a traffic accident in Medellín, Colombia will require priority medical attention (ambulance dispatch), based on real accident report data.

## What this project does

- **Analysis & modeling** (`proyecto_integrador_crisp_dm_prediccion-severidad-accidentes.ipynb` / `proyecto_integrador_crisp_dm_v2.py`): explores and cleans the `accidentes_medellin.csv` dataset, engineers features from location, time, accident class, road design, and weather conditions, then trains a classification pipeline to predict accident severity/priority. The trained pipeline is serialized to `pipeline_accidentes_medellin.pkl`.
- **Interactive demo** (`app.py`): a [Streamlit](https://streamlit.io/) web app — styled as a mock "Secretaría de Movilidad de Medellín" tool — where you input the comuna (district), hour, accident class, road design, and weather, and get a real-time prediction of whether the incident likely requires priority medical attention.
- **Report**: `reporte_accidentes_medellin.html` contains an exploratory data analysis report, and `evidencia del proyecto corriendo en streamlit.png` shows the app running.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, joblib, Streamlit.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
proyecto_integrador_crisp_dm_prediccion-severidad-accidentes.ipynb   # Full CRISP-DM analysis & modeling notebook
proyecto_integrador_crisp_dm_v2.py                                   # Script version of the pipeline
app.py                                                                # Streamlit prediction demo
pipeline_accidentes_medellin.pkl                                      # Trained serialized model pipeline
accidentes_medellin.csv                                               # Source dataset
reporte_accidentes_medellin.html                                      # EDA report
requirements.txt
```

> Integrative/capstone project applying CRISP-DM to a real-world road-safety use case in Medellín.
