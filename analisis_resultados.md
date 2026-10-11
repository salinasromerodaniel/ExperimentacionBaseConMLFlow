# Análisis de resultados

Se compararon seis configuraciones de regresión logística sobre la misma partición estratificada del conjunto Wine (142 registros de entrenamiento y 36 de prueba), con semilla 42. La mejor configuración fue `C=0.1`, penalización `l2`, sin ponderación de clases y `max_iter=5000`: obtuvo `accuracy=1.000`, `f1_macro=1.000` y `balanced_accuracy=1.000`. Una segunda ejecución con los mismos argumentos reprodujo exactamente las tres métricas.

`C` fue el hiperparámetro más influyente dentro del alcance del experimento. Manteniendo `l2` y `class_weight=none`, `C=0.01` redujo el F1 macro a 0.9162; `C=0.1` alcanzó 1.0000, y `C=1` y `C=10` obtuvieron 0.9710. En cambio, cambiar de `l2` a `l1` o activar `class_weight=balanced`, ambos con `C=1`, no modificó las métricas observadas.

La principal limitación es el tamaño del conjunto: una sola predicción cambia la exactitud en aproximadamente 2.78 puntos porcentuales. Además, la comparación usa una sola partición y el mismo conjunto de prueba para seleccionar configuración, por lo que el resultado perfecto puede ser optimista. Un estudio posterior debe comparar media y dispersión mediante validación cruzada y conservar un conjunto final sin consultar. Tampoco se evaluaron datos externos, deriva ni costo de inferencia; por ello, la mejor corrida es la mejor solo entre las configuraciones y datos evaluados.
