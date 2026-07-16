# Componentes neuronales de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

Este paquete contiene bloques PyTorch agnósticos de la tarea. Cada implementación es un objeto en su
propio módulo; los `__init__.py` ofrecen imports públicos concisos.

## Contenidos

- [Modelos](#modelos)
- [Registro de componentes](#registro-de-componentes)
- [Activaciones y normalizaciones](#activaciones-y-normalizaciones)
- [Pooling](#pooling)
- [Distancias](#distancias)
- [Pérdidas](#pérdidas)
- [Añadir un componente](#añadir-un-componente)

## Modelos

| Clase | Contrato |
|---|---|
| `Model` | `forward` abstracto genérico; inferencia, conteo y freeze/unfreeze. |
| `MLP` | Pila densa configurable para tensores cuya última dimensión es `in_features`. |
| `CNN2D` | Pila convolucional NCHW; BatchNorm predeterminado correctamente bidimensional. |
| `BatchedKNN` | Construcción por lotes de índices vecinos desde coordenadas. |
| `ECMP` | Paso de mensajes condicionado por aristas con agregación configurable. |

`MLP` y `CNN2D` aceptan anchos/canales explícitos o un número de capas que interpola entre entrada y
salida. Activación, normalización, parámetros y dropout pueden compartirse o indicarse por capa. Las
listas deben coincidir exactamente y dimensiones/dropout inválidos fallan al construir. Las
conexiones residuales requieren formas iguales.

`Model.predict` entra en evaluación bajo `torch.inference_mode()` y restaura el modo anterior incluso
si `forward` falla. La clase base no impone un diccionario: el adaptador de tarea conecta entradas y
salidas arbitrarias.

## Registro de componentes

`ComponentRegistry` asocia alias aptos para YAML, sin distinguir mayúsculas, con clases de activación
y normalización. Ignora guiones y guiones bajos. Los modelos aceptan strings o clases compatibles:

```python
from lambdaforge.nn import ComponentRegistry
from lambdaforge.nn.models import MLP
from lambdaforge.nn.activations import GELU

first = MLP(32, 1, hidden=[64], activation="gelu")
second = MLP(32, 1, hidden=[64], activation=GELU)
ComponentRegistry.register_activation("project_gelu", GELU)
```

El registro es local al proceso. Con `spawn`, registra alias desde código importable de inicio o usa
la clase completa mediante una especificación YAML.

## Activaciones y normalizaciones

Las activaciones son objetos `torch.nn.Module` con nombre: `ELU`, `GELU`, `Identity`, `LeakyReLU`,
`ReLU`, `Sigmoid`, `SiLU` y `Tanh`.

Las normalizaciones comparten la base `Normalization`:

- `BatchNorm(num_features, dim=1|2|3)` selecciona el módulo PyTorch correspondiente;
- `LayerNorm(normalized_shape)` y `RMSNorm(normalized_shape)` actúan en dimensiones finales;
- `IdentityNorm` conserva la entrada y acepta parámetros no usados para construcción genérica.

En convoluciones NCHW usa BatchNorm con `dim: 2`. `CNN2D` lo aporta automáticamente si se elige el
`BatchNorm` incluido.

## Pooling

Los objetos de pooling de conjuntos suelen consumir `x` con forma `(batch, elementos, features)` y
una máscara booleana opcional `(batch, elementos)`. Las posiciones enmascaradas no contribuyen. El
docstring de cada clase describe parámetros y salidas especiales.

| Familia | Clases |
|---|---|
| Reducciones básicas | `SumPooling`, `MeanPooling`, `MaxPooling`, `MinPooling` |
| Reducciones suaves/aprendidas | `SoftmaxPooling`, `LogSumExpPooling`, `AutoPool` |
| Atención | `AttentionPooling`, `GatedAttentionPooling`, `MultiHeadGatedAttentionPooling` |
| Top-k | `TopKPooling`, `TopKMeanPooling`, `FractionalTopKMeanPooling` |
| Distribución/probabilidad | `MomentPooling`, `NoisyOrPooling`, `ProbabilityGeMPooling` |

Los operadores de probabilidad validan o presuponen entradas similares a probabilidades; no son
intercambiables con embeddings ilimitados. Evita máscaras sin elementos válidos salvo que la clase
documente un valor para conjunto vacío.

## Distancias

`EuclideanDistance` y `SquaredEuclideanDistance` implementan `Distance` y hacen broadcasting como
las operaciones PyTorch subyacentes. La forma cuadrada evita la raíz si solo importan distancias
relativas.

## Pérdidas

`Loss` es el contrato abstracto basado en mapas que usa `LightningTask`, con nombre estable y peso
escalar. El cálculo sensible incluido se ejecuta en float32 incluso con precisión mixta.

`BinaryCrossEntropyWithLogitsLoss` lee claves configurables y envuelve la formulación estable de
PyTorch. No apliques sigmoid antes de esta pérdida. AUROC puede consumir esos logits porque el orden
no cambia al aplicar sigmoid.

## Añadir un componente

1. Añade una clase en un `.py` dentro del paquete correspondiente.
2. Hereda del objeto abstracto o de un `torch.nn.Module` compatible.
3. Documenta módulo, clase, formas, parámetros, máscaras y errores.
4. Exporta la clase desde el `__init__.py` más cercano.
5. Registra un nombre corto solo si los modelos genéricos lo necesitan; los targets completos no.
6. Añade pruebas de formas, validación, gradientes y construcción YAML cuando corresponda.

Mantén fuera los supuestos de dominio. Un componente debe reutilizarse sin conocer un dataset,
molécula, colección de imágenes o vocabulario de etiquetas concreto.

