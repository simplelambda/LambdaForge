"""Regression tests for the consolidated LambdaForge neural public API."""

from __future__ import annotations

import importlib

import lambdaforge.nn as neural
import lambdaforge.nn.models as models
from lambdaforge.experiments.ObjectFactory import ObjectFactory


class TestNeuralPublicAPI:
    """Keep every documented family reachable through stable package entry points."""

    COMPONENT_PACKAGES = (
        "lambdaforge.nn.activations",
        "lambdaforge.nn.distances",
        "lambdaforge.nn.encodings",
        "lambdaforge.nn.kernels",
        "lambdaforge.nn.losses",
        "lambdaforge.nn.normalizations",
        "lambdaforge.nn.pooling",
        "lambdaforge.nn.regularization",
        "lambdaforge.nn.similarities",
    )
    MODEL_PACKAGES = (
        "lambdaforge.nn.models.composition",
        "lambdaforge.nn.models.graph",
        "lambdaforge.nn.models.graph.attention",
        "lambdaforge.nn.models.graph.equivariant",
        "lambdaforge.nn.models.graph.message_passing",
        "lambdaforge.nn.models.implicit",
        "lambdaforge.nn.models.sequence",
        "lambdaforge.nn.models.sets",
        "lambdaforge.nn.models.tabular",
        "lambdaforge.nn.models.trees",
        "lambdaforge.nn.models.vision",
    )

    def test_all_lists_are_unique_and_resolvable(self) -> None:
        """Reject duplicate or dangling names in every neural package index."""
        package_names = (
            "lambdaforge.nn",
            "lambdaforge.nn.models",
            *self.COMPONENT_PACKAGES,
            *self.MODEL_PACKAGES,
        )
        for package_name in package_names:
            package = importlib.import_module(package_name)
            exported = tuple(package.__all__)
            assert len(exported) == len(set(exported)), package_name
            assert all(hasattr(package, name) for name in exported), package_name

    def test_root_api_contains_every_component_and_model(self) -> None:
        """Expose family symbols at the two documented convenience levels."""
        neural_names = set(neural.__all__)
        model_names = set(models.__all__)
        for package_name in self.COMPONENT_PACKAGES:
            package = importlib.import_module(package_name)
            assert set(package.__all__) <= neural_names
        for package_name in self.MODEL_PACKAGES:
            package = importlib.import_module(package_name)
            assert set(package.__all__) <= model_names
        assert model_names <= neural_names

    def test_object_factory_resolves_short_public_model_targets(self) -> None:
        """Keep package-level YAML targets independent from implementation files."""
        assert ObjectFactory.import_object("lambdaforge.nn.models.MLP") is models.MLP
        assert ObjectFactory.import_object("lambdaforge.nn.models.GAT") is models.GAT
        assert ObjectFactory.import_object("lambdaforge.nn.models.NODE") is models.NODE
        assert ObjectFactory.import_object("lambdaforge.nn.models.ConvNeXt2D") is models.ConvNeXt2D
        assert ObjectFactory.import_object("lambdaforge.nn.models.SIREN") is models.SIREN

    def test_object_factory_builds_advanced_graph_stacks_from_short_targets(self) -> None:
        """Construct every advanced graph family through the stable models API."""
        specifications = (
            ("EGNN", {"in_channels": 3, "out_channels": 2}),
            ("GATv2", {"in_channels": 3, "out_channels": 2}),
            ("GraphTransformer", {"in_channels": 3, "out_channels": 2}),
            ("PNA", {"in_channels": 3, "out_channels": 2}),
            (
                "RelationalGCN",
                {"in_channels": 3, "out_channels": 2, "num_relations": 4},
            ),
        )

        for name, parameters in specifications:
            model = ObjectFactory.build(
                {
                    "target": f"lambdaforge.nn.models.{name}",
                    "params": parameters,
                }
            )
            assert isinstance(model, getattr(models, name))
