# Cuantización de precisión mixta con búsqueda evolutiva en una red de visión pequeña

**Pregunta:** ¿se puede reducir mucho una red de clasificación de imágenes (MobileNetV3-Small, CIFAR-10) para hardware
pequeño o para un circuito propio sin perder demasiada precisión?

**Respuesta resumida (en simulación):** una búsqueda evolutiva que decide, capa por capa, cuántos bits usar y dónde recortar
los valores de pesos y activaciones logró **entre un 17% y un 33% menos de operaciones** que la mejor alternativa no evolutiva
con la misma precisión, y la búsqueda aleatoria con el mismo esfuerzo **no mejoró nada**. Todo el trabajo es una simulación:
no se midió velocidad real en un dispositivo ni se construyó un circuito.

[![Abrir en Colab](https://colab.research.google.com/github/cflores23/cuantizacion-evolutiva-mobilenet/blob/main/notebooks/cuantizacion_evolutiva_mobilenet.ipynb)

---

## Hallazgos principales

1. **Exportar a ONNX Runtime** dejó la misma precisión (78.40%) y bajó el tiempo por imagen de unos 6 ms a 0.65 ms en una
   sesión de Colab (CPU, un hilo). Son motores distintos y los tiempos de Colab varían entre sesiones: falta medirlo en el dispositivo destino.
2. **La cuantización estándar a INT8 rompió el modelo:** 10.8% (dinámica) y 17.9% (estática) frente a 78.4%.
3. **Diagnóstico:** cuantizar solo los **pesos** a 8 bits casi no cambia la precisión (77.21% contra 77.42%). Lo que rompe la red son las
   **activaciones** (con rango MinMax a 8 bits: 18.6%).
4. **Evolución diferencial sobre los rangos de las activaciones (por capa):** con pesos de 8 bits y activaciones de 8 bits, la
   precisión pasó de 18.6% a 76.5%, y superó a la mejor regla fija (Percentil 99.9: 73.1%).
5. **Búsqueda conjunta multiobjetivo (NSGA-II):** bits de pesos, bits de activaciones, recorte de pesos y recorte de activaciones
   por capa, minimizando error, costo de operaciones y memoria. Resultados abajo.
6. **Ablación:** la asignación de bits por capa aporta el mayor ahorro; los recortes afinados aportan el último tramo de precisión.

---

## Resultados de la búsqueda conjunta (paso 5)

Media ± desviación entre 3 semillas del evolutivo y de la búsqueda aleatoria (mismas 1,850 evaluaciones). Los demás métodos son
una sola corrida. Precisión medida en **8,000 imágenes que ninguna búsqueda vio**. FP32 de referencia: 77.94%.

**Costo de operaciones (GBOPs) mínimo con la precisión indicada o más** (menor es mejor)

| Método | ≥ 72% | ≥ 70% | ≥ 65% |
|---|---|---|---|
| **Evolutivo (NSGA-II)** | **0.216 ± 0.009** | **0.186 ± 0.005** | **0.162 ± 0.009** |
| Búsqueda aleatoria | 0.352 | 0.264 | 0.264 |
| Uniforme (Percentil 99.9) | 0.352 | 0.264 | 0.264 |
| Sensibilidad + asignación voraz (Percentil 99.9) | 0.321 | 0.282 | 0.195 |
| Recetas de dos etapas (pasos 3 y 4) | 0.302 | 0.302 | 0.238 |

**Memoria de pesos (MB) mínima con la precisión indicada o más**

| Método | ≥ 72% | ≥ 70% | ≥ 65% |
|---|---|---|---|
| **Evolutivo (NSGA-II)** | **0.876 ± 0.060** | **0.773 ± 0.014** | 0.728 ± 0.025 |
| Búsqueda aleatoria | 1.552 | 1.174 | 1.174 |
| Uniforme (Percentil 99.9) | 1.552 | 1.174 | 1.174 |
| Sensibilidad + asignación voraz (Percentil 99.9) | 1.222 | 1.068 | 0.768 |
| Recetas de dos etapas (pasos 3 y 4) | 1.038 | 1.038 | 0.773 |

Con precisión de 75% o más, **solo llegan las recetas de dos etapas** (0.302 GBOPs, 1.04 MB): en el extremo de alta precisión la búsqueda
conjunta, con este presupuesto, no es la mejor. Con 65%, la diferencia en memoria contra la asignación voraz (0.728 contra 0.768 MB) no es concluyente.

Referencia: el uniforme de 8 bits con rangos Percentil 99.9 usa 0.352 GBOPs y 1.55 MB.

## Ablación (paso 6): ¿de dónde viene la ventaja?

Menor costo de operaciones (GBOPs), 3 semillas.

| Variante | ≥ 72% | ≥ 70% | ≥ 65% |
|---|---|---|---|
| Completa (bits por capa + recortes) | 0.216 ± 0.009 | 0.186 ± 0.005 | 0.162 ± 0.009 |
| Solo bits por capa (recortes fijos) | no alcanza | 0.206 ± 0.008 | 0.157 ± 0.007 |
| Solo recortes (bits iguales en todas las capas) | 0.264 ± 0.000 | 0.264 ± 0.000 | 0.264 ± 0.000 |

Lectura: sin bits por capa, el mejor resultado se queda en una sola configuración (pesos de 6 bits, activaciones de 8). Sin recortes
optimizados, la variante no llega a 72%. Ninguna variante gana en todo el rango.

---

## Trabajo relacionado y alcance de la comparación

La cuantización a 8 bits de modelos pequeños es práctica estándar en inferencia en dispositivos, y existen herramientas y técnicas
maduras para recuperar la precisión que la cuantización simple destruye (por ejemplo, Cross-Layer Equalization, Data-Free
Quantization y AdaRound en el kit abierto AIMET, además del entrenamiento consciente de la cuantización). También hay literatura
previa de precisión mixta con búsqueda evolutiva o NSGA-II.

**Este trabajo no pretende ser nuevo en ese sentido.** Es un estudio de caso reproducible que:
- documenta un fallo típico (activaciones con valores extremos que hunden la cuantización de 8 bits en MobileNetV3-Small),
- lo diagnostica (pesos bien, activaciones mal),
- y compara una búsqueda evolutiva contra alternativas **simples** (reglas fijas de recorte, calibración MSE, asignación voraz por sensibilidad,
  búsqueda aleatoria) con el mismo esfuerzo.

**No se compara contra herramientas industriales** (AIMET, AdaRound, entrenamiento consciente de la cuantización), que podrían igualar
o superar estos resultados. Esa comparación queda como trabajo futuro.

**Dónde puede tener sentido el enfoque:** en hardware propio (FPGA o circuitos a la medida) donde se pueden elegir anchos de bit
distintos por capa y el costo relevante es el de los multiplicadores (aquí, GBOPs), y no los formatos fijos de un procesador comercial.
Una advertencia relacionada: en microcontroladores, los beneficios por debajo de 8 bits pueden quedarse solo en simulación si no hay
instrucciones nativas, como ocurre en este trabajo.

---

## Cómo se mide el costo

- **GBOPs** = suma, sobre las capas, de (multiplicaciones-acumulaciones × bits de peso × bits de activación) ÷ 10⁹.
  Es un proxy del área y la energía de un multiplicador en un circuito propio. **No es velocidad medida.**
- **Memoria de pesos** = pesos a sus bits por capa + sesgos a 32 bits. No incluye las escalas por canal ni el empaquetado.

## Límites (léelos antes de citar los resultados)

- **Es simulación.** Los pesos se redondean y vuelven a decimales; no hay un motor de inferencia ni un circuito que ejecute enteros.
- **Solo se cuantizan las entradas de convoluciones y capas lineales y sus pesos.** Las sumas residuales, el bloque
  squeeze-excitation y las funciones de activación siguen en decimales.
- **No se midió velocidad ni consumo reales** (ni en una Raspberry Pi ni en otro dispositivo).
- **Una sola red y un solo conjunto de datos** (CIFAR-10 redimensionado a 64×64). No hay garantía de que se generalice.
- **Sin entrenamiento consciente de la cuantización (QAT)**, que podría mejorar los resultados de todos los métodos.
- **Tres semillas:** diferencias de unos pocos puntos porcentuales no son concluyentes.
- **Selección con imágenes finales:** el mejor miembro de cada frente se elige mirando las imágenes de evaluación finales; es un sesgo
  pequeño, pero un informe riguroso usaría un tercer conjunto solo para reportar.
- **Nota de transparencia:** las cifras de los pasos 1 a 4 (por ejemplo, FP32 de 77.42% en las imágenes finales) y las de los pasos 5 y 6
  (FP32 de 77.94%) provienen de dos entrenamientos distintos del modelo base. Compara solo dentro de cada grupo.
- El modelo parte de pesos preentrenados en ImageNet de `torchvision`.

---

## Estructura y cómo reproducir

```
notebooks/
  cuantizacion_evolutiva_mobilenet.ipynb    Todo el experimento en un solo cuaderno, con resultados
results/                                    Tablas (.csv), recetas (.json) y gráficas (.png)
models/                                     Modelo base entrenado (.pt)
```

El cuaderno recorre el experimento en este orden, compartiendo resultados entre secciones:

1. Línea base: modelo preentrenado ajustado a CIFAR-10 y medición de precisión, tamaño y latencia.
2. Exportación a ONNX y cuantización estándar (dinámica y estática), y diagnóstico de por qué falla INT8.
3. Cuantización simulada en PyTorch: pesos contra activaciones, sensibilidad por capa y evolución diferencial sobre los recortes.
4. Búsqueda evolutiva sobre los rangos de las activaciones.
5. Búsqueda conjunta multiobjetivo (NSGA-II) contra comparaciones: búsqueda aleatoria, asignación voraz por sensibilidad,
   uniformes con reglas fijas y calibración MSE.
6. Ablación (solo bits, solo recortes).

Ejecútalo en Google Colab con GPU, de arriba abajo. Las secciones posteriores usan archivos generados por las anteriores.
Tiempos aproximados con una GPU de Colab: las primeras secciones, minutos; la búsqueda conjunta completa y la ablación, algunas
decenas de minutos cada una.

## Qué sigue

- Medir el modelo (FP32 en ONNX y las versiones cuantizadas) en una Raspberry Pi.
- Construir en Verilog una unidad multiplicador-acumulador con los bits de una capa real y verificarla contra Python.
- Entrenamiento consciente de la cuantización y más redes y conjuntos de datos.

## Licencia

MIT (código). CIFAR-10 y los pesos preentrenados de `torchvision` tienen sus propias condiciones de uso.

## Contacto

Camilo Flores · (https://www.linkedin.com/in/camilo-flores-rodriguez/)
