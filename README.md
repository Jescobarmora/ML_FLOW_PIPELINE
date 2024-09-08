# ML_FLOW_PIPELINE

## Descripción

`ML_FLOW_PIPELINE` es un proyecto diseñado para implementar y automatizar flujos de trabajo de machine learning (ML) utilizando PyCaret y otras herramientas de machine learning populares. Este proyecto abarca todo el proceso, desde la ingesta y preprocesamiento de datos hasta la modelización, la optimización de hiperparámetros y la implementación de modelos para realizar predicciones en nuevos datos.

## Características

- **Ingesta de Datos**: Carga automática de datos de entrenamiento y pruebas desde fuentes externas.
- **Preprocesamiento de Datos**: Aplicación de ingeniería de características, manejo de variables categóricas y escalado de características numéricas.
- **Modelización**: Utiliza PyCaret para configurar experimentos, comparar modelos y seleccionar los mejores algoritmos de machine learning para los datos dados.
- **Optimización de Hiperparámetros**: Optimización de modelos utilizando estrategias avanzadas como búsqueda bayesiana.
- **Implementación**: Guarda y carga modelos entrenados para su reutilización en predicciones.

## Estructura del Proyecto

- `src/`: Contiene el código fuente principal del proyecto, incluyendo scripts para la ingesta, preprocesamiento, modelización y evaluación.
- `models/`: Almacena los modelos entrenados en formato pickle para reutilización y predicción.
- `predictions/`: Almacena las predicciones de datasets de prueba en formato pickle para reutilización.
- `data/`: Carpeta dedicada a los datos de entrada, tanto de entrenamiento como de prueba.
- `notebooks/`: Jupyter Notebooks para experimentación y análisis exploratorio de datos.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

## Requisitos

Para ejecutar este proyecto, necesitarás las siguientes dependencias:

- Python 3.8 o superior
- PyCaret
- Scikit-learn
- Feature-engine
- Pandas
- Numpy
