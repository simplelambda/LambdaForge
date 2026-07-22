# API de redes neuronales de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

`lambdaforge.nn` es la colección agnóstica de la tarea de modelos entrenables y componentes
tensoriales reutilizables. Cada implementación pública es un objeto, cada clase principal vive en
su propio módulo Python y la configuración reside en los constructores, no en bucles de
entrenamiento ocultos. Los componentes incluidos son módulos PyTorch ordinarios: se inicializan una
vez, se llama a `forward` y se componen con código del proyecto, `LightningTask` o un entrenador
externo.

Este documento separa el código implementado actualmente de las líneas de investigación. Nombres
como GradTree, GRANDE, NODE, GATv2, GCN relacional, PNA, GraphTransformer, EGNN, ResNet y
ConvNeXt describen arquitecturas nativas y componibles de
LambdaForge inspiradas en los trabajos citados; no implican paridad numérica con el repositorio de
referencia de sus autores ni reproducción de resultados publicados.

## Índice

- [Diseño y API pública](#diseño-y-api-pública)
- [Construcción desde Python y YAML](#construcción-desde-python-y-yaml)
- [Contratos de formas y enrutamiento](#contratos-de-formas-y-enrutamiento)
- [Modelos implementados](#modelos-implementados)
  - [Núcleo y utilidades geométricas](#núcleo-y-utilidades-geométricas)
  - [Modelos de grafos](#modelos-de-grafos)
    - [Contratos y configuración de grafos avanzados](#contratos-y-configuración-de-grafos-avanzados)
  - [Árboles diferenciables](#árboles-diferenciables)
  - [Modelos de secuencias](#modelos-de-secuencias)
  - [Modelos de conjuntos](#modelos-de-conjuntos)
  - [Modelos tabulares](#modelos-tabulares)
  - [Modelos de visión](#modelos-de-visión)
  - [Composición y modelos implícitos](#composición-y-modelos-implícitos)
  - [Modelos generativos, de incertidumbre y científicos](#modelos-generativos-de-incertidumbre-y-científicos)
  - [Conformidad arquitectónica](#conformidad-arquitectónica)
- [Componentes implementados](#componentes-implementados)
  - [Activaciones](#activaciones)
  - [Normalizaciones](#normalizaciones)
  - [Pooling denso y disperso](#pooling-denso-y-disperso)
  - [Distancias, similitudes y kernels](#distancias-similitudes-y-kernels)
  - [Pérdidas](#pérdidas)
  - [Codificaciones y regularización](#codificaciones-y-regularización)
- [Ejemplos YAML completos](#ejemplos-yaml-completos)
- [Coste, memoria y seguridad](#coste-memoria-y-seguridad)
- [Extender el catálogo](#extender-el-catálogo)
- [Roadmap adicional: no implementado](#roadmap-adicional-no-implementado)
- [Referencias primarias](#referencias-primarias)

## Diseño y API pública

El paquete sigue cuatro reglas:

1. `Model`, `Activation`, `Normalization`, `Pooling`, `SparsePooling`, `Distance`,
   `Similarity`, `Kernel`, `Loss`, `Encoding` y `Regularization` definen contratos acotados.
2. Un modelo no conoce el dataset, la tarea ni el entrenador. Sus entradas pueden ser un tensor,
   varios tensores o tensores con nombre.
3. Cada decisión arquitectónica expuesta por una implementación es un argumento del constructor y,
   por tanto, puede representarse en YAML.
4. Una clase principal por `.py` mantiene explícitos la propiedad, la documentación y los puntos de
   extensión.

Usa los paquetes por categoría como superficie de importación canónica:

```python
from lambdaforge.nn.activations import GELU, SiLU
from lambdaforge.nn.losses import CrossEntropyLoss
from lambdaforge.nn.models import GAT, GradTree, MLP
from lambdaforge.nn.pooling import SparseMeanPooling

model = MLP(
    in_features=32,
    out_features=4,
    hidden=[128, 64],
    activation=["gelu", "silu"],
    normalization=["layernorm", "rmsnorm"],
)
logits = model(features)
```

Subpaquetes como `lambdaforge.nn.models.graph` y `lambdaforge.nn.models.trees` también son públicos
cuando una importación más concreta mejora la claridad. Los módulos individuales de cada clase
siguen siendo importables, pero normalmente conviene usar las exportaciones de paquete. `Model`
añade `predict()`, `num_parameters()`, `freeze()`, `unfreeze()` y `parameter_groups()` con nombres,
sin imponer un esquema de entrada. Un modelo externo puede seguir siendo cualquier
`torch.nn.Module`.

`ComponentRegistry` resuelve alias de activaciones y normalizaciones sin distinguir mayúsculas y
minúsculas. Elimina guiones bajos y guiones, por lo que `layer-norm`, `Layer_Norm` y `layernorm`
resuelven igual. Los alias incluidos no se pueden sobrescribir por accidente; un reemplazo explícito
requiere `replace=True`. El registro no pretende ser un localizador universal de servicios: las
demás categorías se construyen como objetos normales, mediante rutas `target` o plugins instalados.

## Construcción desde Python y YAML

`ObjectFactory.build()` entiende recursivamente tres formas explícitas:

| Forma | Resultado |
|---|---|
| `{target: package.Class, params: {...}}` | Importa la clase, construye recursivamente sus parámetros y crea una instancia nueva. |
| `{ref: package.object}` | Importa y devuelve el propio objeto, útil para clases de optimizador o callables. |
| `{plugin: {kind: model, name: x}, params: {...}}` | Resuelve una clase de entry point instalada, valida su contrato y crea una instancia nueva. |

Los mappings, listas y tuplas ordinarios se recorren de forma recursiva. No hay inferencia
específica de la tarea ni una caché implícita de singletons.

```yaml
model:
  target: lambdaforge.nn.models.MLP
  params:
    in_features: 32
    out_features: 4
    hidden: [128, 64]
    activation: [gelu, silu]
    normalization: [layernorm, rmsnorm]
    dropout: [0.10, 0.05]
```

Cada argumento del constructor puede colocarse bajo `params`. Un módulo anidado usa otra
especificación `target` o `plugin`; una clase o función que no deba instanciarse usa `ref`. Estas
formas explícitas son preferibles a strings mágicos definidos por el proyecto.

Solo las activaciones que conservan la forma tienen alias cortos en `ComponentRegistry`. `GLU`,
`GEGLU`, `SwiGLU` y `ReGLU` dividen y reducen a la mitad la dimensión configurada, por lo que deben
usarse explícitamente en una arquitectura que produzca antes el doble de la anchura deseada. Son
deliberadamente inseguras como reemplazo directo en una capa oculta ordinaria de `MLP`.

## Contratos de formas y enrutamiento

Los símbolos usados a continuación son `B` (batch), `N/M` (elementos), `L` (longitud de secuencia),
`E` (aristas), `F` (características), `C` (canales), `H/W` (tamaño de imagen), `G` (grupos o
grafos), `T` (árboles), `D_tree` (dimensión de salida del árbol) y `D_coord` (dimensión de
coordenadas).

| Familia | Contrato de entrada | Contrato de salida |
|---|---|---|
| MLP y árboles tabulares | `(..., F_in)` | `(..., F_out)`; se conservan las dimensiones iniciales. |
| CNN/ResNet/ConvNeXt | `(B, C, H, W)` | Mapa denso en `CNN2D`; logits o embeddings agregados en los encoders de visión. |
| Encoders de grafo | `x=(N,F)`, `edge_index=(2,E)` entero | Una fila por nodo, `(N,F_out)`. |
| Encoders de grafo conscientes de aristas | Entradas de grafo más `edge_features=(E,F_edge)` opcional | Una fila por nodo, `(N,F_out)`. |
| Encoders de grafo relacionales | Entradas de grafo más `edge_types=(E,)` entero | Una fila por nodo, `(N,F_out)`. |
| Encoders de grafo equivariantes | Entradas de grafo, `coordinates=(N,D_coord)` y características de arista opcionales | Características `(N,F_out)` y, cuando se soliciten, coordenadas `(N,D_coord)`. |
| Readout de grafo | Filas de nodos más `group_index=(N,)` | Una fila por grupo/grafo, opcionalmente procesada por una cabeza. |
| Recurrentes/Transformer/TCN | `(B,L,F)` | `(B,L,F_out)` en modo `sequence` o `(B,F_out)` en un modo de reducción. |
| Modelos de conjuntos | `(B,N,F)` y máscara opcional de validez `(B,N)` | `(B,F_out)` o `(B,num_seeds,F_out)` si no se elimina la dimensión de semillas de Set Transformer. |
| Pooling denso | `(B,N,F)` y máscara opcional de validez `(B,N)` | Habitualmente `(B,F)`; las variantes concatenadas/estadísticas amplían la última dimensión. |
| Pooling disperso | `x=(N,F)`, `group_index=(N,)` | `(G,F)`. Los identificadores de grupo ausentes generan filas de salida vacías. |
| Distancia/similitud/kernel por pares | `(B,N,F)` y `(B,M,F)` | `(B,N,M)`. |
| BatchedKNN | Consulta `(B,N,F)` y fuente `(B,M,F)` | Índices locales y distancias, ambos `(B,N,K)`. |

Las máscaras de pooling denso y modelos de conjuntos usan `True` para entradas válidas. El
`padding_mask` de secuencias y las máscaras de padding de Transformer usan `True` para entradas de
relleno. Esta diferencia sigue la convención establecida de cada familia de PyTorch y se valida en
la frontera. Los modelos de secuencias, `SetTransformer` y `FTTransformer` mueven sus máscaras al
dispositivo de la entrada correspondiente tras validarlas. Los objetos de pooling independientes
esperan máscaras en un dispositivo compatible. Los índices de aristas deben tener ya dtype entero y
nunca se truncan desde coma flotante.

`LightningTask.model_input_keys` dirige modelos con varias entradas sin crear una tarea propia:

- Una secuencia como `[node_features, edge_index]` produce argumentos posicionales en ese orden.
- Un mapping como `{x: node_features, edge_index: graph_edges}` asocia nombres de argumentos del
  modelo con claves del batch.
- `model_input_key` sigue siendo la ruta compacta para un solo tensor y es mutuamente excluyente con
  `model_input_keys`.

Un resultado tensorial se publica bajo `model_output_key`. Un mapping del modelo, como el de
`MultiTaskModel` o `VariationalAutoEncoder`, se conserva sin cambios.

## Modelos implementados

### Núcleo y utilidades geométricas

| Objeto | Objetivo y controles principales |
|---|---|
| `MLP` | Pila totalmente conectada con anchuras exactas o número interpolado de capas; activación, normalización, kwargs y dropout por capa; residuos seguros respecto a la forma y bias configurable. |
| `CNN2D` | Análogo NCHW de `MLP` con canales, kernel, stride, padding y componentes por capa configurables. La capa final es una convolución simple. |
| `BatchedKNN` | Búsqueda por batch de vecinos próximos con `Distance` inyectable, exclusión propia opcional y procesamiento de consultas por bloques. |
| `ECMP` | Paso de mensajes nativo condicionado por aristas con estados origen/destino, atributos de arista y embeddings de relación opcionales, agregación y actualizaciones residuales configurables. |
| `Aggregation` | Políticas tipadas `sum`, `mean` y `max` usadas por los módulos de grafos. |
| `Scatter` | Objeto interno para reducciones indexadas y softmax por segmentos, incluidos scores multi-head. |

### Modelos de grafos

La pila de grafos se implementa con tensores PyTorch ordinarios y reducciones indexadas nativas. No
necesita un objeto contenedor de grafo. Cada encoder acepta una lista de aristas dirigidas
`source -> target`.

| Objeto | Comportamiento implementado |
|---|---|
| `GCN` / `GCNLayer` | Convolución dirigida normalizada con grado de salida del origen y grado de entrada del destino separados, un bias tras agregar, self-loops reemplazados sin duplicados, residuos y componentes por capa. |
| `GraphSAGE` / `GraphSAGELayer` | Agregación de vecindad inspirada en [GraphSAGE](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html), con pesos de raíz, proyección de vecinos opcional y normalización de salida. |
| `GAT` / `GATLayer` | Atención multi-head inspirada en [Graph Attention Network](https://arxiv.org/abs/1710.10903), con cabezas por capa, política de concatenación, dropout de características/atención, self-loops y residuos opcionales. `GATLayer.forward_with_attention()` también expone los pesos de atención. |
| `GATv2` / `GATv2Layer` | Atención dinámica origen/destino inspirada en [GATv2](https://arxiv.org/abs/2105.14491), con proyecciones de arista opcionales, pesos origen/destino compartidos o separados, características de self-loop alineadas y atención pre-dropout inspeccionable. |
| `RelationalGCN` / `RelationalGCNLayer` | Transformaciones de mensajes específicas de relación inspiradas en [R-GCN](https://arxiv.org/abs/1703.06103), con matrices exactas o descomposición en bases, relaciones de arista tipadas, agregación suma/media y transformaciones de raíz opcionales. |
| `PNA` / `PNALayer` | Paso de mensajes inspirado en [Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718), con MLP anterior/posterior que cruza reductores mean/min/max/std y escaladores de grado identity, amplification, attenuation, linear e inverse-linear. |
| `GraphTransformer` / `GraphTransformerLayer` | Atención dot-product dispersa local relacionada con el [Graph Transformer de UniMP](https://arxiv.org/abs/2009.03509): las características de arista modifican claves y valores; son configurables pre/post norm, anchura feed-forward, residuo y gating beta opcional. |
| `EGNN` / `EGNNLayer` | Paso de mensajes escalar [E(n)-equivariante](https://arxiv.org/abs/2102.09844), con distancias cuadradas como entrada y actualizaciones de coordenadas ponderadas por desplazamientos en cualquier dimensión de coordenadas. |
| `GIN` / `GINLayer` | Agregación por suma inspirada en [Graph Isomorphism Network](https://openreview.net/forum?id=ryGs6iA5Km), con épsilon fijo o entrenable y MLP internos configurables. |
| `GraphReadout` | Composición de cualquier encoder de nodos, cualquier `SparsePooling` y una cabeza opcional para predicción por grafo. Los argumentos con nombre adicionales se pasan al encoder. |

Son módulos diferenciables completos, no wrappers de operadores PyG/DGL. Por ello deben compararse
la semántica y el rendimiento antes de cargar pesos de una implementación externa.

#### Contratos y configuración de grafos avanzados

Cada modelo de grafo avanzado usa una lista dirigida de aristas `source -> destination` y crea todos
sus parámetros en el constructor. No necesita un objeto de grafo, adyacencia densa, runtime PyG/DGL
ni inferencia de parámetros en el primer forward.

| Argumento | Forma y validación |
|---|---|
| `x` | Características de nodo de coma flotante `(N,in_channels)`. |
| `edge_index` | Tensor entero `(2,E)`; la fila 0 es origen, la fila 1 destino y cada índice debe estar en `[0,N)`. Se rechazan índices de coma flotante y Booleanos. |
| `edge_features` | Valores numéricos reales `(E,edge_channels)`, convertidos al dispositivo/dtype de `x`. Son obligatorios si `edge_channels > 0`; cuando `edge_channels == 0` pueden omitirse y un tensor `(E,0)` explícito se acepta como ausencia de datos. |
| `edge_types` | Tensor entero exclusivo de R-GCN, `(E,)`, con identificadores en `[0,num_relations)`. |
| `coordinates` | Tensor flotante exclusivo de EGNN, `(N,D_coord)` y `D_coord >= 1`, exactamente en el mismo dispositivo y dtype que `x`. |

| Llamada al modelo | Resultado |
|---|---|
| `GATv2(x, edge_index, edge_features=None)` | Tensor `(N,out_channels)`. `GATv2Layer.forward_with_attention` devuelve además aristas enrutadas alineadas `(2,E_routed)` y pesos normalizados pre-dropout `(E_routed,heads)`. |
| `RelationalGCN(x, edge_index, edge_types)` | Tensor `(N,out_channels)`. |
| `PNA(x, edge_index, edge_features=None)` | Tensor `(N,out_channels)`. Las vecindades vacías y cada combinación de reductor/escalador permanecen finitas. |
| `GraphTransformer(x, edge_index, edge_features=None)` | Tensor `(N,out_channels)`. `forward_with_attention` devuelve además un tensor de aristas enrutadas y otro de pesos normalizados por bloque. |
| `EGNN(x, edge_index, coordinates, edge_features=None)` | Tensor `(N,out_channels)` en modo `features`, o mapping configurable con características y coordenadas actualizadas `(N,D_coord)` en modo `mapping`. `forward_with_coordinates` siempre devuelve el par. |

Los constructores de las pilas exponen toda la configuración siguiente. «Por capa» significa un
escalar difundido a todas las capas de grafo o una lista de exactamente
`L_graph = len(hidden_channels) + 1` entradas. Las listas solo ocultas tienen exactamente
`len(hidden_channels)` entradas.

| Objeto | Controles completos del constructor y ámbito de las listas |
|---|---|
| `GATv2` | Anchuras: `in_channels`, `out_channels`, `hidden_channels`; anchura de arista compartida: `edge_channels`; por capa: `heads`, `concatenate_heads`, `share_weights`, `feature_dropout`, `attention_dropout`, `negative_slope`, `add_self_loops`, `self_loop_fill` (`zero`, `mean` o número finito), `residual`, `bias`; solo ocultas: `activation`, `normalization`, `activation_kwargs`, `normalization_kwargs`. Las anchuras de salida con cabezas concatenadas deben ser divisibles por su cantidad de cabezas. |
| `GATv2Layer` | `in_channels`, `out_channels_per_head` por cabeza, `num_heads`, `concatenate_heads`, `share_weights`, `edge_channels`, `negative_slope`, `attention_dropout`, `add_self_loops`, `self_loop_fill` y `bias`. A diferencia de la pila, la capa llama `num_heads` a la cantidad de cabezas y su anchura de salida es por cabeza. |
| `RelationalGCN` | `in_channels`, `out_channels`, `num_relations`, `hidden_channels`; por capa: `num_bases` (`None` usa una matriz completa por relación), `aggregation` (`sum`/`mean`), `message_chunk_size` (entero positivo, `65536` predeterminado, o `None` para un grupo de relación sin límite), `dropout`, `residual`, `root_weight`, `bias`; solo ocultas: `activation`, `normalization` y ambos mappings de kwargs. |
| `RelationalGCNLayer` | `in_channels`, `out_channels`, `num_relations`, `num_bases`, `aggregation`, `root_weight`, `bias` y `message_chunk_size`. El dropout, los residuos y la activación/normalización oculta pertenecen a la pila. |
| `PNA` | `in_channels`, `out_channels`, `hidden_channels`, `aggregators` no vacío/sin duplicados (`mean`/`min`/`max`/`std`), `scalers` no vacío/sin duplicados (`identity`/`amplification`/`attenuation`/`linear`/`inverse_linear`), `edge_channels`; valores predeterminados compartidos por la pila: `message_channels`, `pre_mlp_hidden_channels`, `post_mlp_hidden_channels`, `average_degree`, `average_log_degree`, `epsilon`, `dropout`, `activation`, `activation_kwargs`, `bias`; solo ocultas: `normalization`, `normalization_kwargs`, `residual`; `layer_kwargs` es un mapping compartido o exactamente un mapping por capa de grafo y puede sobrescribir para esa capa agregadores, escaladores, anchuras de mensaje/MLP anterior/posterior, estadísticas, épsilon, dropout, activación, kwargs de activación y bias. `in_channels`, `out_channels` y `edge_channels` siguen perteneciendo a la pila. |
| `PNALayer` | `in_channels`, `out_channels`, `aggregators`, `scalers`, `edge_channels`, `message_channels`, ambas opciones de anchuras ocultas de MLP, ambas estadísticas de grado, `epsilon`, `dropout`, `activation`, `activation_kwargs` y `bias`. `None` da a cada MLP interno una capa oculta de la anchura de su salida; una lista vacía solicita una proyección lineal directa. |
| `GraphTransformer` | `in_channels`, `out_channels`, `hidden_channels`, `edge_channels`; por capa: `heads`, `concatenate_heads`, `feedforward_channels`, `activation`, `normalization`, `feature_dropout`, `attention_dropout`, `feedforward_dropout`, ambos mappings de kwargs, `add_self_loops`, `self_loop_edge_fill`, `pre_norm`, `residual`, `beta` y `bias`. `beta=true` exige ruta residual. |
| `GraphTransformerLayer` | `in_channels`, `out_channels`, `num_heads`, `concatenate_heads`, `edge_channels`, `feedforward_channels`, `activation`, `activation_kwargs`, `normalization`, `normalization_kwargs`, los tres dropouts, `add_self_loops`, `self_loop_edge_fill`, `pre_norm`, `residual`, `beta` y `bias`. A diferencia de la pila, la capa llama `num_heads` a la cantidad de cabezas. |
| `EGNN` | `in_channels`, `out_channels`, `hidden_channels`, `edge_channels`; por capa: `message_channels`, `feature_dropout`, `residual`, `bias` y `layer_kwargs`; solo ocultas: `activation`, `normalization` y ambos mappings de kwargs; salida: `output_mode` (`features`/`mapping`), `feature_output_key`, `coordinate_output_key`. La activación/normalización de pila actúa entre capas EGNN. |
| `EGNNLayer` mediante `layer_kwargs` | `message_hidden_channels`, `node_hidden_channels`, `coordinate_hidden_channels`, `activation`, `activation_kwargs`, `mlp_normalization`, `mlp_normalization_kwargs`, `message_aggregation`, `coordinate_aggregation`, `message_dropout`, `update_dropout`, `update_coordinates`, `normalize_displacements`, `distance_epsilon`, `distance_scale`, `coordinate_tanh`, `coordinate_scale` y `attention`. Allí no pueden sobrescribirse anchuras, `edge_channels`, `message_channels`, `residual` ni `bias` pertenecientes a la pila. La agregación de coordenadas es `sum` o `mean`. |

Los modelos de atención con self-loops los reemplazan por un único loop canónico por nodo en vez de
duplicarlos; el relleno configurado mantiene alineadas las filas de aristas. Las listas vacías de
aristas y los nodos aislados tienen salidas finitas definidas. Son núcleos arquitectónicos nativos
inspirados en los artículos enlazados, no réplicas de sus recetas completas de entrenamiento,
preprocesado, inicialización o benchmarks; LambdaForge no afirma paridad de checkpoints ni
benchmarks.

Las estadísticas de grado de PNA son estado del dataset, no parámetros aprendibles. Calcula
`average_degree = mean(in_degree)` y
`average_log_degree = mean(log(in_degree + 1))` **solo con el split/topología de entrenamiento**,
persístelas con la configuración del experimento y reutilízalas sin cambios en validación, test e
inferencia. No deben calcularse con todos los splits: filtraría topología reservada. Ambos valores
deben ser positivos y finitos; un grafo de entrenamiento sin aristas necesita una política de
referencia positiva explícita.

### Árboles diferenciables

LambdaForge ofrece **núcleos neuronales de árboles pensados para componerse dentro de redes
mayores**:

| Objeto | Comportamiento implementado |
|---|---|
| `GradTree` | Un árbol binario diferenciable con selectores de características, umbrales y valores de hoja aprendidos; selección y enrutamiento soft/hard straight-through configurables. Inspirado en [GradTree](https://arxiv.org/abs/2305.03515). |
| `GRANDE` | Ensemble vectorizado de árboles diferenciables con subconjuntos deterministas de características, pesos por estimador dependientes de la muestra, dropout de estimadores e inspección de estimadores individuales. Inspirado en [GRANDE](https://arxiv.org/abs/2309.17130). |
| `ObliviousDecisionTree` | Árboles oblivious vectorizados con un selector/umbral por profundidad, entmax/entmoid por defecto, salida aplanada opcional e inicialización explícita a partir de datos. |
| `NODE` | Pila densa de capas `ObliviousDecisionTree` con cantidades de árboles, profundidades, dimensiones, selectores, temperaturas y dropout por capa, y readout `mean`, `sum` o `linear`. Inspirado en [NODE](https://arxiv.org/abs/1909.06312). |

Todos aceptan `(..., in_features)` y exponen `route()` y `feature_importances()`. `GRANDE` añade
`estimator_weights()` y `forward_estimators()`; `NODE` añade `features()`. ODT y NODE proporcionan
`initialize_from_data()` como mutación explícita, nunca como efecto lateral del primer `forward`.

Estas clases **no son reproducciones completas de los estimadores oficiales GradTree, GRANDE o
NODE**. No afirman incluir el preprocesado, pipelines categóricos, recetas de optimización, early
stopping, protocolo de ensemble, paridad de benchmarks ni resultados publicados de sus autores. Su
objetivo es exponer el núcleo arquitectónico diferenciable mediante una interfaz estable
PyTorch/YAML.

Los parámetros de árboles se agrupan por significado. Según el modelo, los nombres incluyen
`selectors`, `thresholds`, `leaves`, `temperatures`, `routing`, `estimator_weights` y `head`. Esto
permite usar tasas de aprendizaje distintas sin acoplar el modelo a un entrenador.

### Modelos de secuencias

| Objeto | Objetivo y ejes configurables |
|---|---|
| `RNNModel` | RNN batch-first con tanh/ReLU, profundidad, bidireccionalidad, dropout, estado inicial y proyección de salida. |
| `LSTMModel` | LSTM con profundidad, bidireccionalidad, tamaño de proyección, estado oculto/celda inicial y proyección de salida. |
| `GRUModel` | GRU con profundidad, bidireccionalidad, dropout, estado inicial y proyección de salida. |
| `TransformerEncoderModel` | Proyección de entrada, profundidad/cabezas/anchura feed-forward configurables, activación, codificación posicional aprendida/sinusoidal/ninguna, token de clase opcional, máscaras causales/de atención y normalización final. |
| `TransformerDecoderModel` | Decoder batch-first con memoria proyectada, atención cruzada, máscaras de padding/atención, control causal y posiciones aprendidas/sinusoidales/ninguna. |
| `TransformerSeq2Seq` | Compone los contratos encoder y decoder para características continuas de origen/destino y expone `encode`/`decode`. |
| `ConformerModel` | Bloques feed-forward macaron, autoatención global y convolución depthwise local con políticas de salida secuencia/reducida. |
| `StateSpaceAdapter` | Frontera de layout/salida/estado alrededor de S4, Mamba u otro proveedor inyectado; LambdaForge no instala su dependencia. |
| `TemporalConvNet` / `TemporalBlock1D` | Pila de convoluciones 1D dilatadas con kernel, dilatación, activación, normalización y dropout por bloque; causal o no causal, residual y con normalización de pesos opcional. |
| `SequenceOutputMode` / `SequenceOutput` | Políticas tipadas `sequence`, `first`, `last`, `mean` y `max` con validación de máscaras/longitudes. |
| `PositionalEncodingType` | Políticas tipadas de posición `none`, `sinusoidal` y `learned` para Transformer. |

Las entradas son batch-first `(B,L,F)`. Las máscaras recurrentes deben describir padding a la
derecha cuando se requiera empaquetado; `lengths` y `padding_mask` deben coincidir si se proporcionan
ambos.

### Modelos de conjuntos

| Objeto | Objetivo |
|---|---|
| `DeepSets` | Encoder de elementos inspirado en [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), pooling denso inyectable y decoder, todos con estructura MLP configurable. |
| `SetTransformer` | Encoder de autoatención y pooling mediante consultas-semilla aprendidas inspirado en [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), con cantidad de semillas, cabezas, profundidad y anchura feed-forward configurables. |

Ambos aceptan `(B,N,F)` y una máscara de validez. Cada muestra debe contener al menos un elemento
válido.

### Modelos tabulares

| Objeto | Objetivo |
|---|---|
| `ResidualMLP` / `ResidualDenseBlock` | Baseline feed-forward residual con anchura, cantidad de bloques, expansión, normalización previa/posterior, dropout por bloque y layer scale opcional configurables. |
| `FTTransformer` | Tokenizador inspirado en [FT-Transformer](https://arxiv.org/abs/2106.11959) para características continuas y categóricas indexadas desde cero, tokens aprendidos para valores ausentes, token de clase, encoder Transformer y cabeza de predicción. |
| `TabNet` | Máscaras atencionales secuenciales sobre variables numéricas, transformaciones GLU y máscaras por paso inspeccionables. |
| `SAINT` | Alterna atención entre variables de una fila y atención entre filas sobre tokens continuos/categóricos. |
| `AutoInt` | Autoatención residual apilada para interacciones explícitas de orden alto entre campos continuos/categóricos. |
| `DeepFM` | Rutas conjuntas de primer orden, interacciones por máquina de factorización y ruta profunda no lineal. |

`FTTransformer` acepta por separado `continuous=(B,F_cont)` y `categorical=(B,F_cat)`, con máscaras
de igual forma. Cardinalidades, bias del tokenizador, anchura/dropout de embeddings, estructura
Transformer e inicialización son argumentos del constructor.
`CategoricalFeatureEncoder` es el objeto de preprocesado explícito correspondiente en
`lambdaforge.data`: el índice cero queda reservado para ausentes/desconocidos, el ajuste es
determinista, inferencia no amplía estado y `cardinalities` alimenta directamente estos
constructores. La atención entre filas de SAINT acopla deliberadamente los ejemplos de un batch;
define una semántica de batching estable al evaluar.

### Modelos de visión

| Objeto | Objetivo |
|---|---|
| `ResNet2D` / `ResidualBlock2D` | Stem y etapas configurables con estilo [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html): canales, cantidad de bloques, strides, dilatación, grupos, normalización, activación y dropout. |
| `ConvNeXt2D` / `ConvNeXtBlock2D` | Encoder jerárquico configurable con estilo [ConvNeXt](https://arxiv.org/abs/2201.03545), con anchuras/profundidades de etapas, kernel depthwise, expansión, layer scale, dropout y profundidad estocástica por bloque. |
| `MobileNetV2` / `InvertedResidualBlock2D` | Encoder jerárquico eficiente con anchuras, profundidades, strides, ratios de expansión, redondeo de canales, dropout y profundidad estocástica configurables. |
| `VisionTransformer2D` | Encoder por patches con rejilla posicional aprendida e interpolada, salida class/mean/token/mapa, cabeza opcional y política explícita de error o padding para restos. |
| `UNet2D` | Encoder/decoder simétrico de predicción densa con anchuras/profundidades, bottleneck, componentes y alineación exacta de skips impares configurables. |
| `FeaturePyramidNetwork2D` | Pirámide top-down de anchura uniforme sobre cualquier `HierarchicalBackbone2D`, con salidas fine-to-coarse y niveles gruesos adicionales opcionales. |

`ResNet2D`, `ConvNeXt2D` y `MobileNetV2` implementan `HierarchicalBackbone2D`: las tuplas de
`forward_feature_maps()` y la metadata `feature_channels` tienen exactamente el mismo orden de fino
a grueso. FPN solo depende de ese contrato, así que un backbone propio participa sin un adaptador
específico. Estos encoders devuelven una representación agregada con `out_features=None` y la salida
de la cabeza en otro caso.

ViT acepta resoluciones variables. `image_size` solo define la rejilla posicional aprendida de
referencia, interpolada a la rejilla real de patches. `remainder_policy: error` rechaza patches
parciales; `pad` añade ceros solo abajo/a la derecha. `output_mode` puede ser `class_token`, `mean`,
`tokens` o `feature_map`; una proyección `out_features` opcional conserva el rango y la disposición
espacial/de tokens. U-Net devuelve `(B,out_channels,H,W)` incluso con tamaños impares si la imagen es
suficientemente grande para la profundidad de pooling configurada.

Son implementaciones arquitectónicas, no checkpoints preentrenados ni presets exactos. ResNet,
MobileNet y U-Net adaptan BatchNorm/InstanceNorm y normalizaciones vectoriales/conscientes del canal
a NCHW; LayerNorm/RMSNorm simples se rechazan temprano porque su contrato de dimensión final no es
seguro con imágenes variables. Usa `ChannelLayerNorm` para normalización de capa por canal.

Un FPN se compone recursivamente en YAML sin nombrar módulos de implementación:

```yaml
model:
  target: lambdaforge.nn.models.FeaturePyramidNetwork2D
  params:
    out_channels: 128
    extra_levels: 1
    backbone:
      target: lambdaforge.nn.models.MobileNetV2
      params:
        in_channels: 3
        stage_channels: [16, 24, 32, 64]
        blocks_per_stage: [1, 2, 3, 2]
        stage_strides: [1, 2, 2, 2]
        expansion_ratios: [1, 6, 6, 6]
```

### Composición y modelos implícitos

| Objeto | Objetivo |
|---|---|
| `AutoEncoder` | Compone módulos encoder/decoder arbitrarios y transformaciones latente y de salida opcionales. |
| `VariationalAutoEncoder` | Compone encoder, cabezas de distribución y decoder; devuelve `reconstruction`, `mean`, `log_variance`, `latent` y `kl_divergence` por muestra. El muestreo y los límites de log-varianza son configurables. |
| `EnsembleModel` / `EnsembleReduction` | Combina modelos independientes mediante media, suma, mediana, mínimo, máximo, stack, concatenación o media ponderada fija/entrenable. |
| `MixtureOfExperts` | Gate denso o top-k straight-through sobre expertos arbitrarios, temperatura/ruido aprendibles opcionales y objetivo auxiliar `load_balance_loss()`. Top-k cambia los pesos, pero aún evalúa todos los expertos. |
| `MultiTaskModel` | Backbone compartido con cabezas nombradas, desconexión selectiva de gradiente del backbone y devolución opcional de características. |
| `SiameseModel` / `SiameseMerge` | Encoder compartido para dos entradas con diferencia, producto, concatenación, L1/L2 o coseno y comparator/cabeza opcionales. |
| `SIREN` | Red implícita de coordenadas inspirada en [SIREN](https://arxiv.org/abs/2006.09661), con frecuencias por etapa, límites de inicialización específicos, salida no lineal opcional e inspección de activaciones. |

Los objetos de composición validan que sus módulos hijos produzcan tensores compatibles. No
controlan el ponderado de pérdidas: los términos KL de VAE, balance de MoE y objetivos multitarea
siguen siendo objetos `Loss` explícitos o lógica de la tarea.

### Modelos generativos, de incertidumbre y científicos

| Objeto | Contrato |
|---|---|
| `VectorQuantizedAutoEncoder` | Encoder/decoder inyectables, codebook aprendido, códigos straight-through, índices, losses de commitment/codebook y perplejidad. La salida del encoder termina en `code_features`. |
| `DiffusionSchedule` / `GaussianDiffusion` | Buffers lineal/coseno, noising exacto, objetivos noise/sample y muestreo DDPM/DDIM completo alrededor de un denoiser `(sample,timestep[,conditioning])`. |
| `TemperatureScaler` | Calibración positiva y acotada de logits, ajustada explícitamente con datos reservados. |
| `ConformalPredictionInterval` | Bandas split-conformal por residuo absoluto con cuantiles corregidos para muestra finita; `fit` exige calibración reservada intercambiable. |
| `NeuralODE` / `ODEMethod` | Integración diferenciable Euler, midpoint o RK4 de paso fijo sobre un campo `(time,state)` inyectado; trayectoria o estado final. |
| `NeuralCDE` | Integración diferencial controlada lineal a trozos con campo `(B,H,C)` y módulos inicial/de salida opcionales. |
| `DeepONet` | Operador branch/trunk inyectable con consultas compartidas o por batch y salidas escalares/vectoriales. |
| `FourierNeuralOperator1D` | Convoluciones espectrales de modos bajos y rutas residuales punto a punto sobre campos 1D batch-first. |
| `TensorFieldNetwork` | Paso de mensajes E(3)-equivariante nativo para canales escalares invariantes (`l=0`) y vectoriales (`l=1`) en unidades cartesianas. |
| `EquivariantTensorAdapter` | Frontera validada para proveedores opcionales de orden superior estilo e3nn sin dependencia compilada base. |

`VariationalAutoEncoderLoss` ofrece reconstrucción MSE/L1/BCE, KL ponderado por beta y free bits
opcionales mediante el contrato `Loss` escalar basado en mappings. El muestreo de difusión recorre
todos los pasos; un schedule pequeño de smoke no es un schedule de investigación. Los integradores
ODE/CDE nativos priorizan transparencia y reproducibilidad, no rigidez ni pasos adaptativos. Inyecta
solvers externos o proveedores equivariantes de orden superior cuando necesites esas garantías y
registra proveedor/versión en la procedencia del entorno o plugin.

### Conformidad arquitectónica

`ArchitectureConformanceCase.capture()` fija inicialización determinista, número de parámetros,
salida esperada pequeña, tolerancias y URL de origen. `write_reference()` guarda atómicamente un
checkpoint weights-only; `from_reference()` lo carga con `weights_only=True`. Cada caso valida carga
estricta, forma/conteo, tolerancia numérica y checksums SHA-256.
`ArchitectureConformancePack.assert_conformant()` agrupa casos vinculados a paper/versión en una
frontera CI. Capturar y probar con el mismo build sólo es smoke: la paridad real requiere versionar
una referencia revisada producida desde la implementación/licencia externa declarada. LambdaForge
no redistribuye checkpoints de autores.

## Componentes implementados

### Activaciones

Todas heredan de `Activation`.

| Grupo | Clases |
|---|---|
| Estándar | `Identity`, `ReLU`, `ReLU6`, `LeakyReLU`, `PReLU`, `ELU`, `CELU`, `SELU`, `GELU`, `SiLU`, `Mish`, `Tanh`, `Sigmoid`, `Softplus`, `Softsign`, `Hardsigmoid`, `Hardswish`, `SquarePlus` |
| Periódicas | `Sine`, `Snake` |
| Enrutamiento disperso | `Entmax15`, `Entmoid15` |
| Gated, reducen la dimensión a la mitad | `GLU`, `GEGLU`, `SwiGLU`, `ReGLU` |

Los alias incluidos en el registro cubren las activaciones que conservan la forma y las de
enrutamiento. Cada capa recibe un objeto nuevo; por ello activaciones entrenables como `PReLU` y
`Snake` no comparten parámetros salvo que quien llama comparta explícitamente una instancia.

### Normalizaciones

Todas heredan de `Normalization`:

- `IdentityNorm`
- `BatchNorm` con implementación 1D/2D/3D seleccionable
- `InstanceNorm` con implementación 1D/2D/3D seleccionable
- `LayerNorm` y `ChannelLayerNorm` consciente del canal
- `RMSNorm`
- `GroupNorm`
- `L2Norm`
- `ScaleNorm`

La convención común de los modelos es `Normalization(features, **kwargs)`. Para entradas NCHW,
`CNN2D` selecciona automáticamente `dim=2` solo para `BatchNorm`; configura `InstanceNorm` con
`normalization_kwargs: {dim: 2}` y `ChannelLayerNorm` con
`normalization_kwargs: {channel_dim: 1}`. `ResidualBlock2D` y `ResNet2D` adaptan automáticamente
BatchNorm, InstanceNorm, ChannelLayerNorm, L2Norm y ScaleNorm a NCHW, y rechazan LayerNorm/RMSNorm
simples porque su contrato sobre la dimensión final no es seguro con tamaños espaciales variables.
`ConvNeXt2D` usa internamente normalización consciente del canal.

### Pooling denso y disperso

Los objetos `Pooling` densos consumen `(B,N,F)` y una máscara opcional de validez:

- Reducciones: `SumPooling`, `MeanPooling`, `MinPooling` y `MaxPooling`.
- Reducciones suaves/entrenables: `SoftmaxPooling`, `LogSumExpPooling`, `AutoPool`,
  `GeneralizedMeanPooling`, `ProbabilityGeMPooling` y `NoisyOrPooling`.
- Selección: `TopKMeanPooling`, `FractionalTopKMeanPooling` y `TopKPooling` aprendido.
- Atención: `AttentionPooling`, `GatedAttentionPooling`,
  `MultiHeadGatedAttentionPooling` y `MultiheadAttentionPooling`.
- Estadísticas ampliadas: `ConcatMeanMaxPooling`, `MomentPooling` y `StatisticsPooling`.

Los objetos `SparsePooling` consumen `x=(N,F)` y `group_index=(N,)`:
`SparseSumPooling`, `SparseMeanPooling`, `SparseMaxPooling` y `SparseAttentionPooling`.
`GraphReadout` compone directamente este contrato.

### Distancias, similitudes y kernels

Cada objeto por pares consume dos conjuntos de punto flotante con batch y produce `(B,N,M)`.

| Contrato | Implementaciones |
|---|---|
| `Distance` | `EuclideanDistance`, `SquaredEuclideanDistance`, `ManhattanDistance`, `MinkowskiDistance`, `ChebyshevDistance`, `CosineDistance`, `AngularDistance`, `MahalanobisDistance` |
| `Similarity` | `DotProductSimilarity`, `CosineSimilarity`, `BilinearSimilarity` |
| `Kernel` | `RBFKernel`, `LaplacianKernel`, `PolynomialKernel` |

La precisión de Mahalanobis y los pesos bilineales pueden ser entrenables. La escala de longitud
RBF/Laplacian y gamma/offset polinómicos también se pueden aprender. Las entradas deben compartir
tamaño de batch, anchura de características, dispositivo y dtype.

### Pérdidas

Cada `Loss` recibe `(outputs, batch, context)`, posee un nombre único y devuelve un tensor escalar.
`Reduction` solo acepta `mean` o `sum`; los objetivos sin reducir pertenecen a una clase de pérdida
específica de la tarea.

| Uso | Clases |
|---|---|
| Clasificación | `BinaryCrossEntropyWithLogitsLoss`, `CrossEntropyLoss`, `BinaryFocalLoss`, `MulticlassFocalLoss` |
| Regresión | `MeanSquaredErrorLoss`, `MeanAbsoluteErrorLoss`, `SmoothL1Loss`, `HuberLoss` |
| Segmentación/solapamiento | `DiceLoss`, `TverskyLoss` |
| Aprendizaje de representaciones | `ContrastiveLoss`, `TripletMarginLoss`, `InfoNCELoss` |
| Generativas | `VariationalAutoEncoderLoss` |

Las claves de salida y target, pesos de clases/positivos, índice ignorado, smoothing, márgenes,
temperaturas, normalización y reducción son argumentos de constructor. Las pérdidas declaran su
compatibilidad con precisión reducida y convierten a mayor precisión los cálculos inseguros cuando
lo exige el contrato base.

### Codificaciones y regularización

Implementaciones de `Encoding`:

- `SinusoidalPositionalEncoding`
- `LearnedPositionalEncoding`
- `RotaryPositionalEncoding`
- `FourierFeatureEncoding` con frecuencias fijas o aprendidas y semilla local de inicialización

Implementaciones de `Regularization`:

- `DropPath` para profundidad estocástica por muestra
- `FeatureDropout` con eje de características y dimensiones de máscara compartidas configurables
- `GaussianNoise` con escala absoluta/relativa y control de aplicación solo en entrenamiento

Son objetos independientes y pueden anidarse en modelos propios mediante `ObjectFactory`.

## Ejemplos YAML completos

### MLP por capas

Las longitudes de todas las listas por capa deben coincidir con la cantidad de capas ocultas:

```yaml
model:
  target: lambdaforge.nn.models.MLP
  params:
    in_features: 96
    out_features: 7
    hidden: [256, 128, 64]
    activation: [gelu, mish, silu]
    activation_kwargs:
      - {approximate: tanh}
      - {inplace: false}
      - {}
    normalization: [layernorm, rmsnorm, identity]
    normalization_kwargs:
      - {eps: 1.0e-5}
      - {eps: 1.0e-6}
      - {}
    dropout: [0.15, 0.10, 0.0]
    residual: true
    bias: true
```

La suma residual solo se realiza cuando coinciden las formas de entrada y salida.

### GAT con múltiples entradas

`model_input_keys` nombra los argumentos del modelo y las claves correspondientes del batch:

```yaml
model:
  target: lambdaforge.nn.models.GAT
  params:
    in_channels: 64
    out_channels: 8
    hidden_channels: [128, 64]
    heads: [8, 4, 1]
    concatenate_heads: [true, true, false]
    activation: [elu, gelu]
    normalization: [layernorm, layernorm]
    feature_dropout: [0.10, 0.10, 0.0]
    attention_dropout: [0.10, 0.05, 0.0]
    add_self_loops: true
    residual: true

task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_keys:
      x: node_features
      edge_index: edge_index
    model_output_key: logits
```

Hay una entrada de cabezas/concatenación/dropout por capa de grafo, mientras que activación y
normalización son políticas de las capas ocultas y tienen una entrada por capa oculta.

### PNA con estadísticas de grado solo de entrenamiento

Este ejemplo tiene dos capas de grafo. Las estadísticas de grado se calcularon una única vez con la
topología de entrenamiento y se reutilizan sin cambios en los demás splits:

```yaml
model:
  target: lambdaforge.nn.models.PNA
  params:
    in_channels: 48
    out_channels: 6
    hidden_channels: [96]
    edge_channels: 8
    aggregators: [mean, min, max, std]
    scalers: [identity, amplification, attenuation, linear, inverse_linear]
    message_channels: [64, 48]
    pre_mlp_hidden_channels: [96]
    post_mlp_hidden_channels: [128, 64]
    average_degree: [6.25, 6.25]
    average_log_degree: [1.72, 1.72]
    epsilon: [1.0e-8, 1.0e-8]
    dropout: [0.10, 0.0]
    activation: [gelu, relu]
    activation_kwargs: [{approximate: tanh}, {}]
    normalization: [layernorm]
    normalization_kwargs: [{eps: 1.0e-5}]
    residual: [false]
    bias: [true, true]
    layer_kwargs:
      - {}
      - {scalers: [identity, attenuation], post_mlp_hidden_channels: [64]}

task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_keys:
      x: node_features
      edge_index: edge_index
      edge_features: edge_features
    model_output_key: logits
```

La entrada final de `layer_kwargs` muestra un override local sin cambiar el contrato de aristas
compartido. Nunca derives `average_degree` ni `average_log_degree` de grafos de validación/test.

### Salida mapping de EGNN

El modo `mapping` permite que pérdidas y métricas seleccionen por separado las características de
nodo invariantes y las coordenadas equivariantes. `LightningTask` conserva el mapping en lugar de
envolverlo bajo `model_output_key`:

```yaml
model:
  target: lambdaforge.nn.models.EGNN
  params:
    in_channels: 32
    out_channels: 16
    hidden_channels: [64, 64]
    edge_channels: 4
    message_channels: [96, 96, 64]
    feature_dropout: [0.05, 0.05, 0.0]
    activation: [silu, silu]
    normalization: [layernorm, layernorm]
    residual: [true, true, true]
    bias: [true, true, true]
    layer_kwargs:
      - {message_aggregation: sum, coordinate_aggregation: mean}
      - {normalize_displacements: true, distance_epsilon: 1.0e-8}
      - {coordinate_tanh: true, coordinate_scale: 0.1}
    output_mode: mapping
    feature_output_key: node_features
    coordinate_output_key: updated_coordinates

task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_keys:
      x: node_features
      edge_index: edge_index
      coordinates: coordinates
      edge_features: edge_features
```

Configura cada pérdida con `output_key: node_features` u `output_key: updated_coordinates`, y cada
métrica con su clave de predicción/salida documentada (`pred_key` en la mayoría de las incluidas),
según corresponda. En modo `features` el mismo modelo devuelve un
único tensor y `model_output_key` se aplica con normalidad.

### Grupos de optimizador específicos de árboles

`LightningTask` aplica primero las opciones ordinarias del optimizador y después los overrides de
grupo:

```yaml
model:
  target: lambdaforge.nn.models.GradTree
  params:
    in_features: 48
    out_features: 3
    depth: 6
    feature_selector: entmax15
    split_function: softsign
    selector_temperature: 0.8
    split_temperature: 1.2
    hard_feature_selection: true
    hard_routing: true
    nan_policy: error
    max_leaves: 65536

optimizer:
  ref: torch.optim.AdamW
  params:
    lr: 0.001
    weight_decay: 0.0001

task:
  target: lambdaforge.training.LightningTask
  params:
    model_input_key: x
    optimizer_group_kwargs:
      selectors: {lr: 0.0003}
      thresholds: {lr: 0.001}
      leaves: {lr: 0.002, weight_decay: 0.0}
```

Los grupos desconocidos o duplicados fallan durante la validación. Los modelos ordinarios exponen
un grupo `default`; los árboles lo sustituyen por grupos semánticos. Los parámetros entrenables
propios de la tarea, si existen, forman `task`.

## Coste, memoria y seguridad

- Los árboles diferenciables asignan valores de hoja de forma exponencial con la profundidad.
  GradTree usa `2**depth` hojas; GRANDE usa `num_estimators * 2**depth`; ODT usa
  `num_trees * 2**depth * tree_dim`; NODE suma este coste entre capas. `max_leaves` y
  `max_total_leaves` limitan parámetros, mientras `max_route_elements_per_sample` (por defecto
  `262_144`) limita aparte el tensor dominante de rutas como hojas por profundidad y
  estimadores/árboles. El batch y autograd multiplican ese límite por muestra. No desactives estas
  protecciones sin un presupuesto de memoria medido.
- GAT, GATv2 y GraphTransformer dirigen la atención solo por las aristas suministradas: el trabajo y
  almacenamiento de atención dispersa crecen aproximadamente como `O(EH)` con las aristas y la
  anchura oculta/de cabezas, no como una matriz densa `N²` de nodos. Este GraphTransformer es local,
  no global. Transformer de secuencia, Set Transformer y el pooling multi-head denso siguen siendo
  cuadráticos en la longitud de secuencia/conjunto, salvo cuando el pooling por consultas reduce uno
  de los ejes.
- PNA multiplica su anchura agregada por
  `len(aggregators) * len(scalers)`. R-GCN asigna una transformación por relación salvo que
  `num_bases` active la descomposición en bases. R-GCN agrupa las aristas por relación y limita cada
  bloque de mensajes proyectados con `message_chunk_size`; `None` elimina ese límite, pero nunca
  crea adyacencia densa ni tensores de pesos por arista. EGNN almacena mensajes de arista y actualizaciones
  de coordenadas; su `feature_dropout` afecta la rama de mensajes/actualización mientras la
  proyección residual recibe el estado limpio.
- Las distancias, similitudes y kernels densos materializan `B*N*M` valores por pares. `BatchedKNN`
  limita su tensor temporal de distancias aproximadamente a `B*chunk_size*M`, pero la búsqueda
  sigue siendo exacta y puede ser costosa.
- La memoria de activaciones de CNN, ResNet y ConvNeXt depende de la resolución espacial, anchuras
  de etapas y activaciones guardadas durante el entrenamiento. La profundidad estocástica
  regulariza, pero no reduce la asignación del modelo.
- `top_k` de MoE usa gradiente straight-through del gate denso, también con `top_k=1`, pero se
  evalúan todos los expertos. Por ahora no reduce cómputo ni memoria de activaciones.
- `nan_policy="error"` es el valor seguro y rechaza NaN y ambas infinidades. `"zero"` sustituye
  explícitamente los tres casos no finitos; es una decisión de imputación, no aprendizaje de
  valores ausentes.
- Debe llamarse a `initialize_from_data()` antes de construir DDP, o inicializar en un rank y
  difundir el estado. Muta umbrales/temperaturas y nunca se oculta en `forward`.
- El enrutamiento/selección hard usa estimadores straight-through. Sigue siendo diferenciable en
  backward, pero su optimización no equivale a un solver de árboles discretos.
- La resolución de plugins importa código Python instalado y de confianza. No es un sandbox. Las
  importaciones `target` tienen la misma frontera de confianza.
- Las implementaciones usan la semántica estándar de dispositivo y dtype de PyTorch. Este catálogo
  no afirma disponibilidad CUDA, kernels propios ni una velocidad de benchmark. Deben probarse CPU/
  GPU, precisión, compilación y configuración distribuida de destino.
- Las implementaciones avanzadas de grafos son núcleos nativos con pocas dependencias, no suites de
  reproducción de artículos. No garantizan paridad de checkpoints publicados, puntuaciones de
  benchmark ni pesos de terceros. GraphTransformer atiende solo sobre `edge_index` y no añade
  atención global ni codificaciones posicionales. EGNN cubre escalares y coordenadas;
  `TensorFieldNetwork` añade características escalares/vectoriales `l=0/l=1` nativas. Las
  representaciones `l>=2` nativas siguen delegadas a un proveedor inyectado explícitamente mediante
  `EquivariantTensorAdapter`.

## Extender el catálogo

### Una clase local del proyecto

Crea una clase documentada en un módulo y hereda de la base más concreta:

```python
from torch import Tensor

from lambdaforge.nn.models import Model


class ProjectEncoder(Model):
    """Codifica características del proyecto sin depender de entrenador ni dataset."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # Construye aquí los módulos poseídos.

    def forward(self, x: Tensor) -> Tensor:
        # Conserva y documenta el contrato de formas.
        raise NotImplementedError
```

Después configura `target: my_project.models.ProjectEncoder`. No hay que editar ningún registro de
LambdaForge. Los modelos propios también pueden heredar directamente de `torch.nn.Module`. Una
`Loss` propia debe devolver un único escalar; los componentes de pooling denso/disperso y por pares
deben respetar sus máscaras y formas documentadas.

### Un alias corto de activación o normalización

```python
from lambdaforge.nn import ComponentRegistry
from my_project.activations import ProjectActivation

ComponentRegistry.register_activation("project_activation", ProjectActivation)
```

El registro valida la subclase y rechaza un alias existente salvo que `replace=True` sea explícito.
Este mecanismo es local al proceso y resulta apropiado para el código de arranque de una aplicación.

### Un plugin de entry point instalado

Las distribuciones externas pueden publicar entry points `model`, `metric`, `activation`,
`normalization`, `loss`, `distance`, `pooling`, `similarity`, `kernel`, `encoding` y
`regularization`. El YAML usa entonces:

```yaml
model:
  plugin:
    kind: model
    name: project_encoder
  params:
    in_features: 32
    out_features: 8
```

La resolución valida el contrato de la clase y crea una instancia nueva en cada construcción.
Consulta [descubrimiento de plugins](../plugins/README.es.md) para publicación, precedencia y
seguridad.

### Grupos de optimizador específicos del modelo

Sobrescribe `parameter_groups()` con un mapping de nombres semánticos estables a secuencias de
parámetros disjuntas. `LightningTask.optimizer_group_kwargs` valida duplicados, nombres desconocidos
y parámetros restantes de la tarea. Conviene conservar el grupo ordinario `default` salvo que
distintas políticas de optimización tengan un significado arquitectónico claro.

## Roadmap adicional: no implementado

La lista siguiente está separada deliberadamente del catálogo implementado. Es un posible roadmap
de investigación, no una promesa de API.

### Aprendizaje geométrico y en grafos

- Esquemas de grafos heterogéneos y almacenes tipados de nodos/aristas más allá de un tensor fijo de
  identificadores de relación.
- Muestreo/mini-batching de grafos, cachés de vecinos y kernels scatter dispersos compilados.
- Representaciones SE(3)/tensor-field `l>=2` nativas y primitivas de geometría molecular más ricas.
- Predicción de enlaces, autoencoders de grafos y codificaciones posicionales/estructurales por
  grafo.

### Árboles y aprendizaje tabular

- Adaptadores opcionales para los pipelines completos de los estimadores GradTree/GRANDE/NODE de
  sus autores, con pruebas de paridad sobre versiones fijadas en lugar de suposiciones por nombre.
- Preprocesado por cuantiles, cabezas calibradas específicas de tarea y recetas documentadas de
  entrenamiento/optimización.
- Deep & Cross, bosques diferenciables y ensembles tabulares eficientes modernos.
- Enrutamiento con menos memoria, poda de hojas, evaluación dispersa de expertos y bancos de
  benchmarks tabulares estandarizados.

### Secuencias, conjuntos y multimodalidad

- Cachés KV Transformer reutilizables y helpers de generación sobre tokens/vocabularios.
- WaveNet y kernels de espacio de estados nativos optimizados; S4/Mamba siguen disponibles por
  adaptador.
- Set2Set, bloques inducidos de Set Transformer, arrays latentes estilo Perceiver y objetos de
  fusión multimodal.

### Visión y predicción densa

- Cabezas de detección y adaptadores de pesos preentrenados con procedencia explícita.
- Swin, EfficientNet, familias convolucionales 1D/3D, modelos de vídeo y objetos de aumento.

### Modelos generativos y científicos

- Normalizing flows, backbones de difusión especializados y objetos de composición adversarial.
- Campos neuronales más allá de SIREN y redes Kolmogorov-Arnold.
- Cabezas probabilísticas de distribución y descomposición de incertidumbre por deep ensembles.

### Componentes e ingeniería

- Kernels Matérn/racional cuadrático; pérdidas Lovász/Jaccard/de contorno; distancias de transporte
  óptimo; variantes sparsemax; pooling adaptativo y jerárquico.
- Metadatos de esquemas de formas, resúmenes de modelos, estimaciones de FLOP/memoria de activaciones
  y linting de arquitecturas YAML antes de asignar memoria.
- Auditorías de precisión/`torch.compile`, pruebas de paridad distribuida, migración de checkpoints
  y benchmarks de referencia reproducibles para cada familia.

## Referencias primarias

- GradTree: [Marton et al., 2023](https://arxiv.org/abs/2305.03515)
- GRANDE: [Marton et al., 2023](https://arxiv.org/abs/2309.17130)
- NODE: [Popov et al., 2019](https://arxiv.org/abs/1909.06312)
- GAT: [Veličković et al., 2018](https://arxiv.org/abs/1710.10903)
- GraphSAGE: [Hamilton et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)
- GIN: [Xu et al., 2019](https://openreview.net/forum?id=ryGs6iA5Km)
- GATv2: [Brody et al., 2021](https://arxiv.org/abs/2105.14491)
- R-GCN: [Schlichtkrull et al., 2017](https://arxiv.org/abs/1703.06103)
- PNA: [Corso et al., 2020](https://arxiv.org/abs/2004.05718)
- Graph Transformer / UniMP: [Shi et al., 2020](https://arxiv.org/abs/2009.03509)
- EGNN: [Satorras et al., 2021](https://arxiv.org/abs/2102.09844)
- FT-Transformer: [Gorishniy et al., 2021](https://arxiv.org/abs/2106.11959)
- Deep Sets: [Zaheer et al., 2017](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)
- Set Transformer: [Lee et al., 2019](https://proceedings.mlr.press/v97/lee19d.html)
- ResNet: [He et al., 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- ConvNeXt: [Liu et al., 2022](https://arxiv.org/abs/2201.03545)
- SIREN: [Sitzmann et al., 2020](https://arxiv.org/abs/2006.09661)
