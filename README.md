# ProyectoPLN

Este repositorio contiene los experimentos y la canalización de entrenamiento para clasificar textos históricos por década.

## Pipeline reproducible

El entrenamiento se refactorizó para evitar fugas de datos y para asegurar resultados reproducibles:

- El *split* de entrenamiento/validación se realiza **antes** de ajustar cualquier vectorizador o modelo.
- Todo el preprocesamiento (normalización, TF-IDF, Word2Vec y meta-features) vive dentro de un único [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) que utiliza un [`FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html).
- Cada componente se reentrena únicamente con `X_train` en cada `fold` gracias a la integración con `cross_val_score`/`StratifiedKFold`.
- Las meta-features se escalan con `StandardScaler(with_mean=False)` para armonizar sus magnitudes con las representaciones esparsas.
- Las representaciones Word2Vec se calculan como promedios ponderados por TF-IDF entrenados únicamente con el conjunto de entrenamiento.
- El clasificador final es una regresión logística con regularización elástica (`solver="saga"`, `penalty="elasticnet"`, `l1_ratio=0.5`, `C=0.5`, `class_weight="balanced"`).

Las piezas reutilizables residen en [`pln_pipeline.py`](pln_pipeline.py). Desde cualquier script o notebook se puede construir el pipeline mediante:

```python
from pln_pipeline import (
    build_text_classification_pipeline,
    save_pipeline_artifacts,
    load_pipeline_artifacts,
)

pipeline = build_text_classification_pipeline(random_state=42)
pipeline.fit(X_train_texts, y_train)
save_pipeline_artifacts(pipeline, "artifacts.joblib")

bundle = load_pipeline_artifacts("artifacts.joblib")
loaded_pipeline = bundle["pipeline"]
loaded_predictions = loaded_pipeline.predict(X_test_texts)
```

## Notebook

`ExploracionDatos.ipynb` ilustra el flujo completo:

1. Exploración y normalización de los textos.
2. Validación cruzada estratificada del pipeline.
3. Entrenamiento final y predicciones sobre el conjunto de evaluación usando el mismo objeto de pipeline.

## Requisitos

El proyecto utiliza Python 3.11+ y depende de bibliotecas como `scikit-learn`, `gensim`, `numpy`, `pandas` y `joblib`. Instala las dependencias necesarias antes de ejecutar los experimentos.
