# LambdaForge neural-network API

[Repository guide](../../../README.md) · [Español](README.es.md)

`lambdaforge.nn` is the task-agnostic collection of trainable models and reusable tensor
components. Every public implementation is an object, every main class lives in its own Python
module, and configuration remains in constructors rather than hidden training loops. Built-ins are
ordinary PyTorch modules: instantiate once, call `forward`, and compose them with project code,
`LightningTask` or an external trainer.

This document distinguishes the code that is implemented now from research directions. Names such
as GradTree, GRANDE, NODE, GATv2, relational GCN, PNA, GraphTransformer, EGNN, ResNet and
ConvNeXt describe native, composable LambdaForge
architectures inspired by the cited work; they do not imply numerical parity with an authors'
reference repository or reproduction of published results.

## Contents

- [Start here](#start-here)
- [Design and public API](#design-and-public-api)
- [Construction from Python and YAML](#construction-from-python-and-yaml)
- [Shape and routing contracts](#shape-and-routing-contracts)
- [Implemented models](#implemented-models)
  - [Core and geometric utilities](#core-and-geometric-utilities)
  - [Graph models](#graph-models)
    - [Advanced graph contracts and configuration](#advanced-graph-contracts-and-configuration)
  - [Differentiable trees](#differentiable-trees)
  - [Sequence models](#sequence-models)
  - [Set models](#set-models)
  - [Tabular models](#tabular-models)
  - [Vision models](#vision-models)
  - [Composition and implicit models](#composition-and-implicit-models)
  - [Generative, uncertainty and scientific models](#generative-uncertainty-and-scientific-models)
  - [Architecture conformance](#architecture-conformance)
- [Implemented components](#implemented-components)
  - [Activations](#activations)
  - [Normalizations](#normalizations)
  - [Dense and sparse pooling](#dense-and-sparse-pooling)
  - [Distances, similarities and kernels](#distances-similarities-and-kernels)
  - [Losses](#losses)
  - [Encodings and regularization](#encodings-and-regularization)
- [Complete YAML examples](#complete-yaml-examples)
- [Cost, memory and safety](#cost-memory-and-safety)
- [Extending the catalogue](#extending-the-catalogue)
- [Further roadmap: not implemented](#further-roadmap-not-implemented)
- [Primary references](#primary-references)

## Start here

This package contains PyTorch building blocks; it does not choose an architecture from a dataset or
own the training loop. A **component** transforms tensors inside a network, a **model** maps model
inputs to predictions, and a **loss** turns predictions plus batch targets into a scalar for
optimization. Shapes and keys remain explicit because only the consumer knows their domain meaning.

For a first model, construct a small `MLP` in Python, run one synthetic batch and inspect its output
before putting the same constructor arguments under YAML `model.params`. Use a project-local
`nn.Module` when built-ins do not match the research design. Inherit `Model` only when its
prediction, freezing, parameter-count and optimizer-group conveniences are useful; ordinary
`nn.Module` classes remain valid.

## Design and public API

The package follows four rules:

1. `Model`, `Activation`, `Normalization`, `Pooling`, `SparsePooling`, `Distance`,
   `Similarity`, `Kernel`, `Loss`, `Encoding` and `Regularization` define narrow contracts.
2. A model does not know the dataset, task or trainer. Inputs may be one tensor, several tensors or
   named tensors.
3. Every architectural decision exposed by an implementation is a constructor argument and can
   therefore be represented in YAML.
4. One principal class per `.py` keeps ownership, documentation and extension points explicit.

Use the category packages as the canonical import surface:

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

Subpackages such as `lambdaforge.nn.models.graph` and `lambdaforge.nn.models.trees` are also public
when a narrower import is clearer. Individual class modules remain importable, but callers should
normally prefer package exports. `Model` adds `predict()`, `num_parameters()`, `freeze()`,
`unfreeze()` and named `parameter_groups()` without imposing an input schema. An external model may
still be any `torch.nn.Module`.

`ComponentRegistry` resolves case-insensitive activation and normalization aliases. It removes
underscores and hyphens, so `layer-norm`, `Layer_Norm` and `layernorm` resolve identically. Built-in
aliases cannot be overwritten accidentally; `replace=True` is required for an explicit replacement.
The registry is intentionally not a universal service locator: other component categories use
normal object construction, `target` paths or installed plugins.

## Construction from Python and YAML

`ObjectFactory.build()` recursively understands three explicit forms:

| Form | Result |
|---|---|
| `{target: package.Class, params: {...}}` | Import the class, recursively build its parameters and create a fresh instance. |
| `{ref: package.object}` | Import and return the object itself, useful for optimizer classes or callables. |
| `{plugin: {kind: model, name: x}, params: {...}}` | Resolve an installed entry-point class, validate its contract and create a fresh instance. |

Ordinary mappings, lists and tuples are traversed recursively. There is no task-specific inference
and no implicit singleton cache.

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

Every constructor parameter can be placed under `params`. A nested module uses another `target` or
`plugin` specification; a class or function that must not be instantiated uses `ref`. Prefer these
explicit forms over project-defined magic strings.

Only shape-preserving activations are short aliases in `ComponentRegistry`. `GLU`, `GEGLU`,
`SwiGLU` and `ReGLU` split and halve their configured dimension, so use them explicitly in an
architecture that first produces twice the desired width. They are deliberately unsafe as direct
drop-ins for an ordinary `MLP` hidden slot.

## Shape and routing contracts

Symbols used below are `B` (batch), `N/M` (items), `L` (sequence length), `E` (edges), `F`
(features), `C` (channels), `H/W` (image size), `G` (groups or graphs), `T` (trees),
`D_tree` (tree output dimension) and `D_coord` (coordinate dimension).

| Family | Input contract | Output contract |
|---|---|---|
| MLP and tabular trees | `(..., F_in)` | `(..., F_out)`; leading dimensions are preserved. |
| CNN/ResNet/ConvNeXt | `(B, C, H, W)` | Dense map for `CNN2D`; logits or pooled embeddings for vision encoders. |
| Graph encoders | `x=(N,F)`, integer `edge_index=(2,E)` | One row per node, `(N,F_out)`. |
| Edge-aware graph encoders | Graph inputs plus optional `edge_features=(E,F_edge)` | One row per node, `(N,F_out)`. |
| Relational graph encoders | Graph inputs plus integer `edge_types=(E,)` | One row per node, `(N,F_out)`. |
| Equivariant graph encoders | Graph inputs, `coordinates=(N,D_coord)` and optional edge features | Features `(N,F_out)` and, when requested, coordinates `(N,D_coord)`. |
| Graph readout | Node rows plus `group_index=(N,)` | One row per group/graph, optionally passed through a head. |
| Recurrent/Transformer/TCN | `(B,L,F)` | `(B,L,F_out)` for `sequence` mode or `(B,F_out)` for a reduction mode. |
| Set models | `(B,N,F)` and optional validity mask `(B,N)` | `(B,F_out)`, or `(B,num_seeds,F_out)` for unsqueezed Set Transformer seeds. |
| Dense pooling | `(B,N,F)` and optional validity mask `(B,N)` | Usually `(B,F)`; concatenating/statistical variants expand the last dimension. |
| Sparse pooling | `x=(N,F)`, `group_index=(N,)` | `(G,F)`. Missing group ids are represented by empty output rows. |
| Pairwise distance/similarity/kernel | `(B,N,F)` and `(B,M,F)` | `(B,N,M)`. |
| BatchedKNN | Query `(B,N,F)` and source `(B,M,F)` | Local indices and distances, both `(B,N,K)`. |

Dense pooling and set-model masks use `True` for valid entries. Sequence `padding_mask` and
Transformer padding masks use `True` for padded entries. This distinction follows each PyTorch
family's established convention and is validated at the boundary. Sequence models, `SetTransformer`
and `FTTransformer` move their masks to the corresponding input device after validation. Standalone
pooling objects expect masks already on a compatible device. Graph edge indices must already have an
integer dtype and are never truncated from floating point.

`LightningTask.model_input_keys` routes multi-input models without a custom task:

- A sequence such as `[node_features, edge_index]` produces positional arguments in that order.
- A mapping such as `{x: node_features, edge_index: graph_edges}` maps model argument names to batch
  keys.
- `model_input_key` remains the compact single-tensor route and is mutually exclusive with
  `model_input_keys`.

A tensor result is exposed under `model_output_key`. A model mapping, such as `MultiTaskModel` or
`VariationalAutoEncoder`, is preserved unchanged.

## Implemented models

### Core and geometric utilities

| Object | Purpose and principal controls |
|---|---|
| `MLP` | Fully connected stack with exact hidden widths or interpolated layer count; per-layer activation, normalization, kwargs and dropout; optional shape-safe residuals and configurable bias. |
| `CNN2D` | NCHW analogue of `MLP` with configurable channels, kernel, stride, padding and per-layer components. The final layer is a plain convolution. |
| `BatchedKNN` | Batched nearest-neighbour lookup with injectable `Distance`, optional self exclusion and query chunking. |
| `ECMP` | Native edge-conditioned message passing using source/destination states, optional edge attributes and relation embeddings, configurable aggregation and residual updates. |
| `Aggregation` | Type-safe `sum`, `mean` and `max` policies used by graph modules. |
| `Scatter` | Internal object for indexed reductions and segment softmax, including multi-head scores. |

### Graph models

The graph stack is implemented with ordinary PyTorch tensors and native indexed reductions. It does
not require a graph object wrapper. Every encoder accepts an edge list of directed `source -> target`
pairs.

| Object | Implemented behaviour |
|---|---|
| `GCN` / `GCNLayer` | Directed degree-normalized convolution using separate source out-degree and destination in-degree, one post-aggregation bias, replace-not-duplicate self-loops, residuals and per-layer components. |
| `GraphSAGE` / `GraphSAGELayer` | [GraphSAGE](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)-inspired neighbourhood aggregation with root weights, optional neighbour projection and output normalization. |
| `GAT` / `GATLayer` | [Graph Attention Network](https://arxiv.org/abs/1710.10903)-inspired multi-head attention with per-layer heads, concatenation policy, feature/attention dropout, self-loops and optional residuals. `GATLayer.forward_with_attention()` also exposes attention weights. |
| `GATv2` / `GATv2Layer` | Dynamic source/destination attention inspired by [GATv2](https://arxiv.org/abs/2105.14491), with optional edge projections, shared or separate source/destination weights, aligned self-loop features and inspectable pre-dropout attention. |
| `RelationalGCN` / `RelationalGCNLayer` | Relation-specific message transforms inspired by [R-GCN](https://arxiv.org/abs/1703.06103), with exact matrices or basis decomposition, typed edge relations, sum/mean aggregation and optional root transforms. |
| `PNA` / `PNALayer` | [Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718)-inspired pre/post MLP message passing that crosses mean/min/max/std reducers with identity, amplification, attenuation, linear and inverse-linear degree scalers. |
| `GraphTransformer` / `GraphTransformerLayer` | Local sparse dot-product attention related to [UniMP's graph Transformer](https://arxiv.org/abs/2009.03509): edge features modify keys and values; pre/post norm, feed-forward width, residual and optional beta gating are configurable. |
| `EGNN` / `EGNNLayer` | [E(n)-equivariant](https://arxiv.org/abs/2102.09844) scalar message passing with squared-distance inputs and displacement-weighted coordinate updates in arbitrary coordinate dimension. |
| `GIN` / `GINLayer` | [Graph Isomorphism Network](https://openreview.net/forum?id=ryGs6iA5Km)-inspired sum aggregation with fixed or trainable epsilon and configurable internal MLPs. |
| `GraphReadout` | Composition of any node encoder, any `SparsePooling` and an optional head for graph-level prediction. Extra keyword arguments are forwarded to the encoder. |

These are full differentiable modules, not wrappers over PyG/DGL operators. Consequently, compare
semantics and performance before loading weights from a third-party implementation.

#### Advanced graph contracts and configuration

Every advanced graph model uses a directed `source -> destination` edge list and creates all
parameters in its constructor. No graph object, dense adjacency, PyG/DGL runtime or first-forward
parameter inference is required.

| Argument | Shape and validation |
|---|---|
| `x` | Floating-point node features `(N,in_channels)`. |
| `edge_index` | Integer tensor `(2,E)`; row 0 is source, row 1 destination, and every index must be in `[0,N)`. Floating-point and Boolean indices are rejected. |
| `edge_features` | Real numeric `(E,edge_channels)`, converted to `x` device/dtype. It is required when `edge_channels > 0`; when `edge_channels == 0` it may be omitted, while an explicit `(E,0)` tensor is accepted as absent data. |
| `edge_types` | R-GCN-only integer tensor `(E,)` with identifiers in `[0,num_relations)`. |
| `coordinates` | EGNN-only floating tensor `(N,D_coord)`, `D_coord >= 1`, on exactly the same device and dtype as `x`. |

| Model call | Result |
|---|---|
| `GATv2(x, edge_index, edge_features=None)` | Tensor `(N,out_channels)`. `GATv2Layer.forward_with_attention` additionally returns aligned routed edges `(2,E_routed)` and normalized pre-dropout weights `(E_routed,heads)`. |
| `RelationalGCN(x, edge_index, edge_types)` | Tensor `(N,out_channels)`. |
| `PNA(x, edge_index, edge_features=None)` | Tensor `(N,out_channels)`. Empty neighborhoods and every reducer/scaler combination remain finite. |
| `GraphTransformer(x, edge_index, edge_features=None)` | Tensor `(N,out_channels)`. `forward_with_attention` additionally returns one routed-edge tensor and one normalized weight tensor per block. |
| `EGNN(x, edge_index, coordinates, edge_features=None)` | Tensor `(N,out_channels)` in `features` mode, or a configurable mapping containing features and updated `(N,D_coord)` coordinates in `mapping` mode. `forward_with_coordinates` always returns the pair. |

The stack constructors expose the complete configuration below. “Per layer” means either one scalar
broadcast to all graph layers or a list of exactly `L_graph = len(hidden_channels) + 1` entries.
Hidden-only lists have exactly `len(hidden_channels)` entries.

| Object | Complete constructor controls and list scope |
|---|---|
| `GATv2` | Widths: `in_channels`, `out_channels`, `hidden_channels`; shared edge width: `edge_channels`; per layer: `heads`, `concatenate_heads`, `share_weights`, `feature_dropout`, `attention_dropout`, `negative_slope`, `add_self_loops`, `self_loop_fill` (`zero`, `mean` or finite number), `residual`, `bias`; hidden-only: `activation`, `normalization`, `activation_kwargs`, `normalization_kwargs`. Concatenated output widths must be divisible by their head count. |
| `GATv2Layer` | `in_channels`, per-head `out_channels_per_head`, `num_heads`, `concatenate_heads`, `share_weights`, `edge_channels`, `negative_slope`, `attention_dropout`, `add_self_loops`, `self_loop_fill` and `bias`. Unlike the stack, the layer names the head count `num_heads` and its output width is per head. |
| `RelationalGCN` | `in_channels`, `out_channels`, `num_relations`, `hidden_channels`; per layer: `num_bases` (`None` means one full matrix per relation), `aggregation` (`sum`/`mean`), `message_chunk_size` (positive integer, default `65536`, or `None` for one unbounded relation group), `dropout`, `residual`, `root_weight`, `bias`; hidden-only: `activation`, `normalization` and both kwargs mappings. |
| `RelationalGCNLayer` | `in_channels`, `out_channels`, `num_relations`, `num_bases`, `aggregation`, `root_weight`, `bias` and `message_chunk_size`. Dropout, residuals and hidden activation/normalization belong to the stack. |
| `PNA` | `in_channels`, `out_channels`, `hidden_channels`, duplicate-free non-empty `aggregators` (`mean`/`min`/`max`/`std`), duplicate-free non-empty `scalers` (`identity`/`amplification`/`attenuation`/`linear`/`inverse_linear`), `edge_channels`; defaults shared by the stack: `message_channels`, `pre_mlp_hidden_channels`, `post_mlp_hidden_channels`, `average_degree`, `average_log_degree`, `epsilon`, `dropout`, `activation`, `activation_kwargs`, `bias`; hidden-only: `normalization`, `normalization_kwargs`, `residual`; `layer_kwargs` is one shared mapping or exactly one mapping per graph layer and may override the aggregators, scalers, message/pre/post MLP widths, statistics, epsilon, dropout, activation, activation kwargs and bias for that layer. `in_channels`, `out_channels` and `edge_channels` remain stack-owned. |
| `PNALayer` | `in_channels`, `out_channels`, `aggregators`, `scalers`, `edge_channels`, `message_channels`, both MLP hidden-width options, both degree statistics, `epsilon`, `dropout`, `activation`, `activation_kwargs` and `bias`. `None` gives each internal MLP one hidden layer at its output width; an empty list requests a direct linear projection. |
| `GraphTransformer` | `in_channels`, `out_channels`, `hidden_channels`, `edge_channels`; per layer: `heads`, `concatenate_heads`, `feedforward_channels`, `activation`, `normalization`, `feature_dropout`, `attention_dropout`, `feedforward_dropout`, both kwargs mappings, `add_self_loops`, `self_loop_edge_fill`, `pre_norm`, `residual`, `beta` and `bias`. `beta=true` requires a residual path. |
| `GraphTransformerLayer` | `in_channels`, `out_channels`, `num_heads`, `concatenate_heads`, `edge_channels`, `feedforward_channels`, `activation`, `activation_kwargs`, `normalization`, `normalization_kwargs`, all three dropouts, `add_self_loops`, `self_loop_edge_fill`, `pre_norm`, `residual`, `beta` and `bias`. Unlike the stack, the layer names the head count `num_heads`. |
| `EGNN` | `in_channels`, `out_channels`, `hidden_channels`, `edge_channels`; per layer: `message_channels`, `feature_dropout`, `residual`, `bias` and `layer_kwargs`; hidden-only: `activation`, `normalization` and both kwargs mappings; output: `output_mode` (`features`/`mapping`), `feature_output_key`, `coordinate_output_key`. Stack activation/normalization acts between EGNN layers. |
| `EGNNLayer` through `layer_kwargs` | `message_hidden_channels`, `node_hidden_channels`, `coordinate_hidden_channels`, `activation`, `activation_kwargs`, `mlp_normalization`, `mlp_normalization_kwargs`, `message_aggregation`, `coordinate_aggregation`, `message_dropout`, `update_dropout`, `update_coordinates`, `normalize_displacements`, `distance_epsilon`, `distance_scale`, `coordinate_tanh`, `coordinate_scale` and `attention`. Stack-owned widths, `edge_channels`, `message_channels`, `residual` and `bias` cannot be overridden there. Coordinate aggregation is `sum` or `mean`. |

Self-loop-enabled attention models replace existing loops with one canonical loop per node rather
than duplicating them; the configured fill keeps edge rows aligned. Empty edge lists and isolated
nodes have finite defined outputs. These are native architectural cores inspired by the linked
papers, not replicas of their complete training recipes, preprocessing, initialization or benchmark
pipelines, and LambdaForge does not claim checkpoint or benchmark parity.

PNA degree statistics are dataset state, not learnable parameters. Compute `average_degree =
mean(in_degree)` and `average_log_degree = mean(log(in_degree + 1))` **only from the training
split/topology**, persist them with the experiment configuration, and reuse them unchanged for
validation, test and inference. Never calculate them over all splits: that leaks held-out topology.
Both values must be positive and finite; a training graph with no edges needs an explicit positive
reference policy.

### Differentiable trees

LambdaForge supplies neural tree **cores intended for composition inside larger networks**:

| Object | Implemented behaviour |
|---|---|
| `GradTree` | One differentiable binary tree with learned feature selectors, thresholds and leaf values; configurable soft/hard straight-through selection and routing. Inspired by [GradTree](https://arxiv.org/abs/2305.03515). |
| `GRANDE` | Vectorized ensemble of differentiable trees with deterministic feature subsets, per-sample estimator weights, estimator dropout and inspection of individual estimators. Inspired by [GRANDE](https://arxiv.org/abs/2309.17130). |
| `ObliviousDecisionTree` | Vectorized oblivious trees with one selector/threshold per depth, entmax/entmoid defaults, optional flattened output and explicit data-driven initialization. |
| `NODE` | Dense stack of `ObliviousDecisionTree` layers with per-layer tree counts, depths, dimensions, selectors, temperatures, dropout and `mean`, `sum` or `linear` readout. Inspired by [NODE](https://arxiv.org/abs/1909.06312). |

All accept `(..., in_features)` and expose `route()` and `feature_importances()`. `GRANDE` additionally
provides `estimator_weights()` and `forward_estimators()`; `NODE` provides `features()`. ODT and NODE
offer `initialize_from_data()` as an explicit mutation, never as a first-forward side effect.

These classes **are not complete reproductions of the official GradTree, GRANDE or NODE
estimators**. They do not claim the authors' preprocessing, categorical pipelines, optimizer
recipes, early stopping, ensembling protocol, benchmark parity or published scores. Their purpose is
to expose the differentiable architectural core through a stable PyTorch/YAML interface.

Tree parameters are grouped by meaning. Depending on the model, names include `selectors`,
`thresholds`, `leaves`, `temperatures`, `routing`, `estimator_weights` and `head`. This permits
different learning rates without coupling the model to a trainer.

### Sequence models

| Object | Purpose and configurable axes |
|---|---|
| `RNNModel` | Batch-first vanilla RNN with tanh/ReLU, depth, bidirectionality, dropout, initial state and output projection. |
| `LSTMModel` | LSTM with depth, bidirectionality, projection size, initial hidden/cell state and output projection. |
| `GRUModel` | GRU with depth, bidirectionality, dropout, initial state and output projection. |
| `TransformerEncoderModel` | Input projection, configurable Transformer depth/heads/feed-forward width, activation, learned/sinusoidal/no positional encoding, optional class token, causal/attention masks and final normalization. |
| `TransformerDecoderModel` | Batch-first target decoder with projected memory, cross-attention, padding/attention masks, causal control and learned/sinusoidal/no positions. |
| `TransformerSeq2Seq` | Composes the encoder and decoder contracts for continuous source/target features and exposes separate `encode`/`decode` calls. |
| `ConformerModel` | Macaron feed-forward, global self-attention and local depthwise-convolution blocks with sequence/reduced output policies. |
| `StateSpaceAdapter` | Layout/output/state boundary around an injected S4, Mamba or other provider; no provider dependency is installed by LambdaForge. |
| `TemporalConvNet` / `TemporalBlock1D` | Dilated 1D convolution stack with per-block kernels, dilation, activation, normalization and dropout; causal or non-causal, residual and optional weight normalization. |
| `SequenceOutputMode` / `SequenceOutput` | Type-safe `sequence`, `first`, `last`, `mean` and `max` output policies with mask/length validation. |
| `PositionalEncodingType` | Type-safe `none`, `sinusoidal` and `learned` Transformer position policies. |

Inputs are batch-first `(B,L,F)`. Recurrent masks must describe right padding when packing is needed;
`lengths` and `padding_mask` must agree if both are supplied.

### Set models

| Object | Purpose |
|---|---|
| `DeepSets` | [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)-inspired element encoder, injectable dense pooling and decoder, all with configurable MLP structure. |
| `SetTransformer` | [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)-inspired self-attention encoder and learned seed-query pooling with configurable seed count, heads, depth and feed-forward width. |

Both accept `(B,N,F)` and a validity mask. Every sample must contain at least one valid item.

### Tabular models

| Object | Purpose |
|---|---|
| `ResidualMLP` / `ResidualDenseBlock` | Residual feed-forward baseline with configurable width, block count, expansion, pre/post normalization, per-block dropout and optional layer scale. |
| `FTTransformer` | [FT-Transformer](https://arxiv.org/abs/2106.11959)-inspired tokenizer for continuous and zero-based categorical features, learned missing-value tokens, class token, Transformer encoder and prediction head. |
| `TabNet` | Sequential attentive masks over numeric features, GLU feature transforms and inspectable per-step masks. |
| `SAINT` | Alternating within-row feature attention and across-row attention over continuous/categorical tokens. |
| `AutoInt` | Stacked residual self-attention for explicit high-order interactions between continuous/categorical fields. |
| `DeepFM` | Joint first-order, factorization-machine pairwise and deep nonlinear interaction paths. |

`FTTransformer` accepts `continuous=(B,F_cont)` and `categorical=(B,F_cat)` independently, with
same-shaped masks. Cardinalities, tokenizer bias, embedding width/dropout, Transformer structure and
initialization are constructor parameters.
`CategoricalFeatureEncoder` is the matching explicit preprocessing object in `lambdaforge.data`:
index zero is reserved for missing/unseen values, fitting is deterministic, inference never grows
state, and its `cardinalities` feed these constructors directly. SAINT's row-attention intentionally
couples examples in the same batch; define stable batching semantics for evaluation.

### Vision models

| Object | Purpose |
|---|---|
| `ResNet2D` / `ResidualBlock2D` | Configurable [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)-style stem and stages: channels, block counts, strides, dilation, groups, normalization, activation and dropout. |
| `ConvNeXt2D` / `ConvNeXtBlock2D` | Configurable [ConvNeXt](https://arxiv.org/abs/2201.03545)-style hierarchical encoder with stage widths/depths, depthwise kernel, expansion, layer scale, dropout and per-block stochastic depth. |
| `MobileNetV2` / `InvertedResidualBlock2D` | Efficient hierarchical encoder with configurable widths, depths, strides, expansion ratios, channel rounding, dropout and stochastic depth. |
| `VisionTransformer2D` | Patch encoder with an interpolated learned position grid, class/mean/token/feature-map output, optional projection head and explicit error-or-padding remainder policy. |
| `UNet2D` | Symmetric dense-prediction encoder/decoder with configurable stage widths/depths, bottleneck, components and exact odd-size skip alignment. |
| `FeaturePyramidNetwork2D` | Equal-width top-down pyramid over any `HierarchicalBackbone2D`, with fine-to-coarse outputs and optional extra coarse levels. |

`ResNet2D`, `ConvNeXt2D` and `MobileNetV2` implement `HierarchicalBackbone2D`: their
`forward_feature_maps()` tuples and `feature_channels` metadata have exactly the same fine-to-coarse
stage order. FPN depends only on this contract, so custom backbones can participate without an
architecture-specific adapter. These encoders return a pooled representation when
`out_features=None`, otherwise head outputs.

ViT accepts variable resolutions. `image_size` defines only the learned reference position grid,
which is interpolated to the actual patch grid. `remainder_policy: error` rejects partial patches;
`pad` adds zeros only on the right/bottom. Its `output_mode` is `class_token`, `mean`, `tokens` or
`feature_map`; an optional `out_features` projection preserves the selected rank and spatial/token
layout. U-Net returns `(B,out_channels,H,W)` even for odd dimensions, provided the image is large
enough for its configured pooling depth.

All are architecture implementations, not pretrained checkpoints or exact named size presets.
ResNet, MobileNet and U-Net adapt BatchNorm/InstanceNorm and vector/channel-aware normalizations to
NCHW; plain LayerNorm/RMSNorm are rejected early because their trailing-dimension contract is unsafe
for variable image sizes. Use `ChannelLayerNorm` for channel-wise layer normalization.

An FPN can be composed recursively in YAML without naming an implementation module:

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

### Composition and implicit models

| Object | Purpose |
|---|---|
| `AutoEncoder` | Compose arbitrary encoder/decoder modules plus optional latent and output transforms. |
| `VariationalAutoEncoder` | Compose an encoder, distribution heads and decoder; returns `reconstruction`, `mean`, `log_variance`, `latent` and per-sample `kl_divergence`. Sampling behaviour and log-variance bounds are configurable. |
| `EnsembleModel` / `EnsembleReduction` | Combine independent models by mean, sum, median, min, max, stack, concatenate or fixed/learned weighted mean. |
| `MixtureOfExperts` | Dense or straight-through top-k gate over arbitrary experts, optional learned temperature/noise and an explicit `load_balance_loss()` auxiliary objective. Top-k changes weights but still evaluates every expert. |
| `MultiTaskModel` | Shared backbone with named heads, selective backbone detachment and optional feature return. |
| `SiameseModel` / `SiameseMerge` | Shared encoder for two inputs with difference, product, concatenation, L1/L2 or cosine merge and optional comparator/head. |
| `SIREN` | [SIREN](https://arxiv.org/abs/2006.09661)-inspired implicit coordinate network with per-stage frequencies, SIREN-specific initialization bounds, optional nonlinear output and activation inspection. |

Composition objects validate that child modules return compatible tensors. They do not own loss
weighting: VAE KL terms, MoE balance terms and multi-task objectives remain explicit `Loss` objects
or task logic.

### Generative, uncertainty and scientific models

| Object | Contract |
|---|---|
| `VectorQuantizedAutoEncoder` | Injectable encoder/decoder, learned codebook, straight-through codes, indices, commitment/codebook losses and perplexity. Encoder outputs end in `code_features`. |
| `DiffusionSchedule` / `GaussianDiffusion` | Linear/cosine buffers, exact forward noising, noise/sample prediction targets and full DDPM/DDIM sampling around an injected `(sample,timestep[,conditioning])` denoiser. |
| `TemperatureScaler` | Positive bounded scalar logit calibration fitted explicitly on held-out data. |
| `ConformalPredictionInterval` | Split-conformal absolute-residual bands with finite-sample corrected quantiles; `fit` must use exchangeable held-out calibration data. |
| `NeuralODE` / `ODEMethod` | Differentiable Euler, midpoint or RK4 fixed-step integration of an injected `(time,state)` field; trajectories or final state. |
| `NeuralCDE` | Piecewise-linear controlled differential integration with an injected field returning `(B,H,C)` and optional initial/output modules. |
| `DeepONet` | Injectable branch/trunk operator with shared or batched query locations and scalar/vector outputs. |
| `FourierNeuralOperator1D` | Low-mode spectral convolutions plus pointwise residual paths on batch-first one-dimensional fields. |
| `TensorFieldNetwork` | Native E(3)-equivariant message passing for invariant scalar (`l=0`) and vector (`l=1`) channels in Cartesian units. |
| `EquivariantTensorAdapter` | Shape-validated injection boundary for optional e3nn-like higher-order providers without a base compiled dependency. |

`VariationalAutoEncoderLoss` supplies MSE/L1/BCE reconstruction plus beta-weighted KL and optional
free bits through the standard mapping-based scalar `Loss` contract. Diffusion sampling runs all
configured timesteps; small smoke schedules are not research-quality schedules. Native ODE/CDE
integrators target transparent reproducibility, not adaptive-solver stiffness. Use external solvers
or higher-order equivariant providers through injected modules when their numerical guarantees are
required, and record the provider/version in environment or plugin provenance.

### Architecture conformance

`ArchitectureConformanceCase.capture()` freezes a deterministic initialization, parameter count,
tiny expected output, tolerances and source URL. `write_reference()` stores a weights-only reference
checkpoint atomically; `from_reference()` loads it with `weights_only=True`. A case verifies strict
state loading, output shape/count, numerical tolerance and SHA-256 checksums.
`ArchitectureConformancePack.assert_conformant()` groups paper/version-linked cases into one CI
boundary. Capturing and testing against the same build is only a smoke test: meaningful parity
requires committing a reviewed reference produced from the declared external implementation and
licence/version. No author checkpoint is redistributed by LambdaForge.

## Implemented components

### Activations

All inherit `Activation`.

| Group | Classes |
|---|---|
| Standard | `Identity`, `ReLU`, `ReLU6`, `LeakyReLU`, `PReLU`, `ELU`, `CELU`, `SELU`, `GELU`, `SiLU`, `Mish`, `Tanh`, `Sigmoid`, `Softplus`, `Softsign`, `Hardsigmoid`, `Hardswish`, `SquarePlus` |
| Periodic | `Sine`, `Snake` |
| Sparse routing | `Entmax15`, `Entmoid15` |
| Gated, dimension-halving | `GLU`, `GEGLU`, `SwiGLU`, `ReGLU` |

Built-in registry aliases cover the shape-preserving and routing activations. Each layer receives a
fresh object; trainable activations such as `PReLU` and `Snake` therefore do not share parameters
unless the caller explicitly shares an instance.

### Normalizations

All inherit `Normalization`:

- `IdentityNorm`
- `BatchNorm` with selectable 1D/2D/3D implementation
- `InstanceNorm` with selectable 1D/2D/3D implementation
- `LayerNorm` and channel-aware `ChannelLayerNorm`
- `RMSNorm`
- `GroupNorm`
- `L2Norm`
- `ScaleNorm`

The common model convention is `Normalization(features, **kwargs)`. For NCHW inputs, `CNN2D`
automatically selects `dim=2` only for `BatchNorm`; configure `InstanceNorm` with
`normalization_kwargs: {dim: 2}` and `ChannelLayerNorm` with
`normalization_kwargs: {channel_dim: 1}`. `ResidualBlock2D` and `ResNet2D` adapt BatchNorm,
InstanceNorm, ChannelLayerNorm, L2Norm and ScaleNorm to NCHW automatically, and reject plain
LayerNorm/RMSNorm because their trailing-dimension contract is unsafe for variable spatial sizes.
`ConvNeXt2D` uses channel-aware normalization internally.

### Dense and sparse pooling

Dense `Pooling` objects consume `(B,N,F)` and an optional validity mask:

- Reductions: `SumPooling`, `MeanPooling`, `MinPooling`, `MaxPooling`.
- Smooth/learnable reductions: `SoftmaxPooling`, `LogSumExpPooling`, `AutoPool`,
  `GeneralizedMeanPooling`, `ProbabilityGeMPooling` and `NoisyOrPooling`.
- Selection: `TopKMeanPooling`, `FractionalTopKMeanPooling` and learned `TopKPooling`.
- Attention: `AttentionPooling`, `GatedAttentionPooling`,
  `MultiHeadGatedAttentionPooling` and `MultiheadAttentionPooling`.
- Expanded statistics: `ConcatMeanMaxPooling`, `MomentPooling` and `StatisticsPooling`.

Sparse `SparsePooling` objects consume `x=(N,F)` and `group_index=(N,)`:
`SparseSumPooling`, `SparseMeanPooling`, `SparseMaxPooling` and `SparseAttentionPooling`.
`GraphReadout` composes this contract directly.

### Distances, similarities and kernels

Every pairwise object consumes two batched floating-point sets and produces `(B,N,M)`.

| Contract | Implementations |
|---|---|
| `Distance` | `EuclideanDistance`, `SquaredEuclideanDistance`, `ManhattanDistance`, `MinkowskiDistance`, `ChebyshevDistance`, `CosineDistance`, `AngularDistance`, `MahalanobisDistance` |
| `Similarity` | `DotProductSimilarity`, `CosineSimilarity`, `BilinearSimilarity` |
| `Kernel` | `RBFKernel`, `LaplacianKernel`, `PolynomialKernel` |

Mahalanobis precision and bilinear weights may be trainable. RBF/Laplacian length scale and
polynomial gamma/offset may also be learned. Inputs must share batch size, feature width, device and
dtype.

### Losses

Every `Loss` receives `(outputs, batch, context)`, owns a unique name and returns a scalar tensor.
`Reduction` accepts only `mean` or `sum`; unreduced objectives belong in a task-specific loss class.

| Use | Classes |
|---|---|
| Classification | `BinaryCrossEntropyWithLogitsLoss`, `CrossEntropyLoss`, `BinaryFocalLoss`, `MulticlassFocalLoss` |
| Regression | `MeanSquaredErrorLoss`, `MeanAbsoluteErrorLoss`, `SmoothL1Loss`, `HuberLoss` |
| Segmentation/overlap | `DiceLoss`, `TverskyLoss` |
| Representation learning | `ContrastiveLoss`, `TripletMarginLoss`, `InfoNCELoss` |
| Generative | `VariationalAutoEncoderLoss` |

Output and target keys, class/positive weights, ignore index, smoothing, margins, temperatures,
normalization and reduction are constructor parameters. Losses declare reduced-precision support and
upcast unsafe computations when required by their base contract.

### Encodings and regularization

`Encoding` implementations:

- `SinusoidalPositionalEncoding`
- `LearnedPositionalEncoding`
- `RotaryPositionalEncoding`
- `FourierFeatureEncoding` with fixed or learned frequencies and a local initialization seed

`Regularization` implementations:

- `DropPath` for per-sample stochastic depth
- `FeatureDropout` with a configurable feature axis and shared mask dimensions
- `GaussianNoise` with absolute/relative scale and training-only control

These are standalone objects and can be nested inside custom models through `ObjectFactory`.

## Complete YAML examples

### Per-layer MLP

The lengths of every per-layer list must equal the number of hidden layers:

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

Residual addition occurs only when the incoming and outgoing tensor shapes match.

### Multi-input GAT

`model_input_keys` names the model arguments and the corresponding batch keys:

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

There is one head/concatenation/dropout entry per graph layer, but activation and normalization are
hidden-layer policies and therefore have one entry per hidden layer.

### PNA with training-only degree statistics

This example has two graph layers. The degree statistics were computed once from the training
topology and are reused unchanged for every other split:

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

The final `layer_kwargs` entry demonstrates a local override without changing the shared edge
contract. Never derive `average_degree` or `average_log_degree` from validation/test graphs.

### EGNN mapping output

`mapping` mode lets losses and metrics address invariant node features and equivariant coordinates
independently. `LightningTask` preserves the mapping instead of wrapping it under
`model_output_key`:

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

Configure each loss with `output_key: node_features` or `output_key: updated_coordinates`, and each
metric with its documented prediction/output key (`pred_key` for most built-ins), as appropriate. In
`features` mode the same model instead returns
one tensor and `model_output_key` applies normally.

### Tree-specific optimizer groups

`LightningTask` applies the ordinary optimizer options first, then group overrides:

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

Unknown or duplicated groups fail validation. Ordinary models expose a `default` group; tree models
override it with semantic groups. Task-owned trainable parameters, if any, form `task`.

## Cost, memory and safety

- Differentiable trees allocate leaf values exponentially in depth. GradTree uses `2**depth` leaves;
  GRANDE uses `num_estimators * 2**depth`; ODT uses
  `num_trees * 2**depth * tree_dim`; NODE sums this cost over layers. `max_leaves` and
  `max_total_leaves` bound parameters, while `max_route_elements_per_sample` (default `262_144`)
  separately bounds the dominant route tensor as leaves times depth and estimators/trees. Batch
  size and autograd multiply that per-sample bound. Do not disable these guards without a measured
  memory budget.
- GAT, GATv2 and GraphTransformer route attention only over supplied edges: sparse attention work
  and storage grow broadly as `O(EH)` with edges and hidden/head width, not as a dense `N²` node
  matrix. This GraphTransformer is local rather than global. Sequence Transformer, Set Transformer
  and dense multi-head attention pooling remain quadratic in sequence/set length unless the
  particular query pooling reduces one axis.
- PNA multiplies its aggregated width by
  `len(aggregators) * len(scalers)`. R-GCN allocates one transform per relation unless
  `num_bases` enables basis decomposition. R-GCN groups edges by relation and caps each projected
  message block with `message_chunk_size`; `None` removes that cap but never creates dense
  adjacency or per-edge weight tensors. EGNN stores edge messages plus coordinate updates; its
  `feature_dropout` affects the message/update branch while the residual projection receives the
  clean state.
- Dense distances, similarities and kernels materialize `B*N*M` pairwise values. `BatchedKNN`
  bounds its temporary distance tensor to approximately `B*chunk_size*M`, but its search remains
  exact and potentially expensive.
- CNN, ResNet and ConvNeXt activation memory depends on spatial resolution, stage widths and saved
  training activations. Stochastic depth reduces regularization, not model allocation.
- MoE `top_k` uses a straight-through dense gate gradient, including for `top_k=1`, but all experts
  are evaluated. It does not currently reduce expert compute or activation memory.
- `nan_policy="error"` is the safe tree default and rejects NaN and both infinities. `"zero"`
  explicitly replaces all three non-finite cases; it is an imputation choice, not missing-value
  learning.
- Call `initialize_from_data()` before constructing DDP, or initialize on one rank and broadcast the
  state. It mutates thresholds/temperatures and is intentionally never hidden in `forward`.
- Hard routing/selection uses straight-through estimators. It remains differentiable in backward,
  but its optimization behaviour is not equivalent to a discrete tree solver.
- Plugin discovery imports trusted installed Python code when a plugin is resolved. It is not a
  sandbox. `target` imports have the same trust boundary.
- The implementations use standard PyTorch device and dtype semantics. This catalogue makes no
  CUDA availability, custom-kernel or benchmark-speed claim. Test the intended CPU/GPU,
  precision, compilation and distributed configuration in the target environment.
- Advanced graph implementations are dependency-light native cores, not paper-reproduction suites.
  They have no published-checkpoint, benchmark-score or third-party weight parity guarantee.
  GraphTransformer attends only over `edge_index` and does not add global attention or positional
  encodings. EGNN covers scalar features and coordinate updates; `TensorFieldNetwork` adds native
  `l=0/l=1` scalar/vector features. Native `l>=2` representations remain delegated to an explicitly
  injected provider through `EquivariantTensorAdapter`.

## Extending the catalogue

### A project-local class

Create one documented class in one module and inherit the narrowest base:

```python
from torch import Tensor

from lambdaforge.nn.models import Model


class ProjectEncoder(Model):
    """Encode project features without depending on a trainer or dataset."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # Build owned modules here.

    def forward(self, x: Tensor) -> Tensor:
        # Preserve and document the shape contract.
        raise NotImplementedError
```

Then configure `target: my_project.models.ProjectEncoder`. No LambdaForge registry edit is required.
Custom models may also inherit `torch.nn.Module` directly. A custom `Loss` must return one scalar;
custom dense/sparse pooling and pairwise components must honour their documented masks and shapes.

### A short activation or normalization alias

```python
from lambdaforge.nn import ComponentRegistry
from my_project.activations import ProjectActivation

ComponentRegistry.register_activation("project_activation", ProjectActivation)
```

Registration validates the subclass and rejects an existing alias unless `replace=True` is explicit.
This mechanism is process-local and best suited to application bootstrap code.

### An installed entry-point plugin

External distributions can publish `model`, `metric`, `activation`, `normalization`, `loss`,
`distance`, `pooling`, `similarity`, `kernel`, `encoding` and `regularization` entry points. YAML then
uses:

```yaml
model:
  plugin:
    kind: model
    name: project_encoder
  params:
    in_features: 32
    out_features: 8
```

Plugin resolution validates the class contract and creates a fresh instance for every build. See
[plugin discovery](../plugins/README.md) for publication, precedence and security details.

### Model-specific optimizer groups

Override `parameter_groups()` with a mapping from stable semantic names to disjoint parameter
sequences. `LightningTask.optimizer_group_kwargs` validates duplicates, unknown names and remaining
task parameters. Keep the ordinary `default` group unless different optimizer policies have a clear
architectural meaning.

## Further roadmap: not implemented

The following list is intentionally separate from the implemented catalogue. It is a candidate
research roadmap, not an API promise.

### Graph and geometric learning

- Heterogeneous graph schemas and typed node/edge stores beyond a fixed relation-id tensor.
- Graph sampling/mini-batching, neighbour caches and compiled sparse scatter kernels.
- Native `l>=2` SE(3)/tensor-field representations and richer molecular geometry primitives.
- Link prediction, graph autoencoders and graph-level positional/structural encodings.

### Trees and tabular learning

- Optional adapters around the authors' complete GradTree/GRANDE/NODE estimator pipelines, with
  pinned-version parity tests rather than name-based assumptions.
- Quantile preprocessing, calibrated task-specific heads and documented optimizer/training recipes.
- Deep & Cross, differentiable forests and modern efficient tabular ensembles.
- Memory-reduced routing, leaf pruning, sparse expert evaluation and standardized tabular
  benchmark harnesses.

### Sequences, sets and multimodality

- Reusable Transformer KV caches and token/vocabulary generation helpers.
- WaveNet and native optimized state-space kernels; S4/Mamba remain available by adapter.
- Set2Set, induced Set Transformer blocks, Perceiver-style latent arrays and multimodal fusion
  objects.

### Vision and dense prediction

- Detection heads and pretrained-weight adapters with explicit provenance.
- Swin, EfficientNet, 1D/3D convolutional families, video models and augmentation objects.

### Generative and scientific models

- Normalizing flows, specialized diffusion backbones and adversarial composition objects.
- Neural fields beyond SIREN and Kolmogorov-Arnold networks.
- Probabilistic distribution heads and deep-ensemble uncertainty decomposition.

### Components and engineering

- Matérn/rational-quadratic kernels; Lovász/Jaccard/boundary losses; optimal-transport distances;
  sparsemax variants; adaptive and hierarchical pooling.
- Shape-schema metadata, model summaries, FLOP/activation-memory estimates and YAML architecture
  linting before allocation.
- Precision/`torch.compile` audits, distributed parity tests, checkpoint migration and reproducible
  reference benchmarks for every family.

## Primary references

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
