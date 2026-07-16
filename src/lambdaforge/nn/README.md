# LambdaForge neural components

[Repository guide](../../../README.md) · [Español](README.es.md)

This package contains task-neutral PyTorch building blocks. Every implementation is an object in its
own module; package `__init__.py` files provide concise public imports.

## Contents

- [Models](#models)
- [Component registry](#component-registry)
- [Activations and normalisations](#activations-and-normalisations)
- [Pooling](#pooling)
- [Distances](#distances)
- [Losses](#losses)
- [Adding a component](#adding-a-component)

## Models

| Class | Contract |
|---|---|
| `Model` | Generic abstract `forward`; inference helper, parameter count, freeze/unfreeze. |
| `MLP` | Configurable dense hidden stack for tensors whose last dimension is `in_features`. |
| `CNN2D` | Configurable NCHW convolution stack; default BatchNorm is correctly two-dimensional. |
| `BatchedKNN` | Batched neighbour-index construction from point coordinates. |
| `ECMP` | Edge-conditioned message-passing model with configurable aggregation. |

`MLP` and `CNN2D` accept either explicit hidden widths/channels or an integer count that interpolates
between input and output sizes. Activation, normalisation, keyword arguments and dropout can be one
shared value or one value per hidden layer. Per-layer lists must match exactly; invalid dimensions
and dropout probabilities fail during construction. Residual connections require matching shapes.

`Model.predict` enters evaluation mode under `torch.inference_mode()` and restores the previous mode
even when `forward` raises. The base class does not impose a dictionary schema: task adapters decide
how arbitrary model inputs and outputs are connected.

## Component registry

`ComponentRegistry` maps YAML-friendly, case-insensitive aliases to activation and normalisation
classes. Underscores and hyphens are ignored. Built-in models therefore accept both strings and
compatible Python classes:

```python
from lambdaforge.nn import ComponentRegistry
from lambdaforge.nn.models import MLP
from lambdaforge.nn.activations import GELU

first = MLP(32, 1, hidden=[64], activation="gelu")
second = MLP(32, 1, hidden=[64], activation=GELU)
ComponentRegistry.register_activation("project_gelu", GELU)
```

Registration is process-local. When using `spawn`, register aliases in importable startup code or use
the fully qualified class through a YAML object specification.

## Activations and normalisations

Activations are thin, named `torch.nn.Module` objects: `ELU`, `GELU`, `Identity`, `LeakyReLU`,
`ReLU`, `Sigmoid`, `SiLU` and `Tanh`.

Normalisations share the `Normalization` base:

- `BatchNorm(num_features, dim=1|2|3)` selects the corresponding PyTorch module;
- `LayerNorm(normalized_shape)` and `RMSNorm(normalized_shape)` operate on trailing dimensions;
- `IdentityNorm` preserves its input and accepts unused keyword arguments for generic construction.

For NCHW convolutions, use BatchNorm with `dim: 2`. `CNN2D` supplies this value automatically when
its selected normalisation class is the built-in `BatchNorm`.

## Pooling

Set/instance pooling objects generally consume `x` shaped `(batch, elements, features)` and an
optional boolean mask shaped `(batch, elements)`. Masked positions do not contribute. Individual
class docstrings document extra parameters and any specialised output.

| Family | Classes |
|---|---|
| Basic reductions | `SumPooling`, `MeanPooling`, `MaxPooling`, `MinPooling` |
| Smooth/learned reductions | `SoftmaxPooling`, `LogSumExpPooling`, `AutoPool` |
| Attention | `AttentionPooling`, `GatedAttentionPooling`, `MultiHeadGatedAttentionPooling` |
| Top-k | `TopKPooling`, `TopKMeanPooling`, `FractionalTopKMeanPooling` |
| Distribution/probability | `MomentPooling`, `NoisyOrPooling`, `ProbabilityGeMPooling` |

Probability-specific operators validate or mathematically assume probability-like inputs; they are
not interchangeable with arbitrary unbounded embeddings. Masks with no valid elements should be
avoided unless the selected class explicitly documents an empty-set value.

## Distances

`EuclideanDistance` and `SquaredEuclideanDistance` implement the `Distance` contract and broadcast
like the underlying PyTorch tensor operations. The squared form avoids the square root when only
relative distances are needed.

## Losses

`Loss` is the abstract mapping-based contract used by `LightningTask`. It provides a stable name and
a scalar weight. Numerically sensitive built-in computation runs in float32 even when the Trainer
uses mixed precision.

`BinaryCrossEntropyWithLogitsLoss` reads configurable output/target keys and wraps PyTorch's stable
logits formulation. Do not apply sigmoid before this loss. Metrics such as AUROC may consume the same
logits because ranking is invariant to sigmoid.

## Adding a component

1. Add one class in one `.py` file under the matching package.
2. Subclass the relevant abstract object or a compatible `torch.nn.Module`.
3. Give the module and class docstrings that state shapes, parameters, mask behaviour and errors.
4. Export the class from the nearest `__init__.py`.
5. Register a short name only when generic model constructors need it; fully qualified YAML targets
   do not require registry changes.
6. Add shape, validation, gradient and YAML-construction tests as appropriate.

Keep domain assumptions outside this package. A component should be reusable without knowledge of a
particular dataset, molecule, image collection or label vocabulary.

