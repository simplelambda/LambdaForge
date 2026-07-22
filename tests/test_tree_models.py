"""Tests for differentiable tree activations and neural model cores."""

import copy

import pytest
import torch

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.nn.activations import Entmax15, Entmoid15
from lambdaforge.nn.losses import BinaryCrossEntropyWithLogitsLoss
from lambdaforge.nn.models import GRANDE, NODE, GradTree, ObliviousDecisionTree
from lambdaforge.training import LightningTask


class TestTreeActivations:
    """Verify sparse simplex and binary entmax behavior."""

    def test_entmax_is_sparse_normalized_and_differentiable(self) -> None:
        logits = torch.tensor([[8.0, 0.0, -8.0], [0.2, 0.1, 0.0]], requires_grad=True)
        output = Entmax15()(logits)
        assert torch.allclose(output.sum(dim=-1), torch.ones(2), atol=1e-6)
        assert output[0, 2] == 0.0
        output.square().sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_entmoid_matches_two_class_entmax(self) -> None:
        logits = torch.linspace(-5.0, 5.0, 21)
        expected = Entmax15()(torch.stack([torch.zeros_like(logits), logits], dim=-1))[:, 1]
        assert torch.allclose(Entmoid15()(logits), expected, atol=1e-6)


class TestGradTree:
    """Verify hard routes, gradients and serializable tree state."""

    def test_known_axis_aligned_route(self) -> None:
        model = GradTree(
            in_features=2,
            out_features=1,
            depth=1,
            feature_selector="softmax",
            split_function="sigmoid",
            selector_init_std=0.0,
            threshold_init_std=0.0,
            leaf_init_std=0.0,
        )
        with torch.no_grad():
            model.feature_logits.copy_(torch.tensor([[10.0, -10.0]]))
            model.thresholds.zero_()
            model.leaf_values.copy_(torch.tensor([[-2.0], [3.0]]))

        x = torch.tensor([[-1.0, 100.0], [1.0, -100.0]])
        assert torch.equal(model(x), torch.tensor([[-2.0], [3.0]]))
        routes = model.route(x)
        assert torch.equal(routes, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def test_gradients_importances_and_state_round_trip(self) -> None:
        torch.manual_seed(17)
        model = GradTree(4, 2, depth=3, feature_selector="softmax")
        x = torch.randn(16, 4)
        model(x).square().mean().backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        assert torch.allclose(model.feature_importances().sum(), torch.tensor(1.0))

        clone = GradTree(4, 2, depth=3, feature_selector="softmax")
        clone.load_state_dict(copy.deepcopy(model.state_dict()))
        assert torch.equal(model(x), clone(x))

    def test_named_parameter_groups_integrate_with_training(self) -> None:
        model = GradTree(3, 1, depth=2)
        groups = model.parameter_groups()
        assert set(groups) == {"selectors", "thresholds", "leaves"}
        assert {id(value) for group in groups.values() for value in group} == {
            id(value) for value in model.parameters()
        }
        task = LightningTask(
            model=model,
            losses=BinaryCrossEntropyWithLogitsLoss(),
            optimizer_group_kwargs={
                "selectors": {"lr": 0.02},
                "thresholds": {"lr": 0.03},
            },
        )
        optimizer = task.configure_optimizers()
        learning_rates = {group["lr"] for group in optimizer.param_groups}
        assert {0.001, 0.02, 0.03} <= learning_rates


class TestGRANDE:
    """Verify instance weighting, local feature sampling and gradients."""

    def test_known_equal_weight_ensemble(self) -> None:
        model = GRANDE(
            in_features=1,
            out_features=1,
            depth=1,
            num_estimators=2,
            feature_selector="softmax",
            split_function="sigmoid",
            selector_init_std=0.0,
            threshold_init_std=0.0,
            leaf_init_std=0.0,
            weight_init_std=0.0,
        )
        with torch.no_grad():
            model.leaf_values[0].fill_(1.0)
            model.leaf_values[1].fill_(3.0)
            model.leaf_estimator_logits.zero_()

        x = torch.tensor([[-1.0], [1.0]])
        assert torch.equal(model(x), torch.full((2, 1), 2.0))
        assert torch.allclose(model.estimator_weights(x).sum(dim=-1), torch.ones(2))
        assert model.forward_estimators(x).shape == torch.Size([2, 2, 1])

    def test_feature_subsets_are_seeded_locally_and_receive_gradients(self) -> None:
        torch.manual_seed(1)
        first = GRANDE(8, 2, depth=2, num_estimators=4, selected_features=0.5, feature_seed=9)
        torch.manual_seed(999)
        second = GRANDE(8, 2, depth=2, num_estimators=4, selected_features=0.5, feature_seed=9)
        assert torch.equal(first.feature_indices, second.feature_indices)

        first(torch.randn(12, 8)).square().mean().backward()
        for parameter in first.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        assert torch.allclose(first.feature_importances().sum(), torch.tensor(1.0))


class TestObliviousDecisionTrees:
    """Verify NODE's elementary oblivious-tree layer."""

    def test_known_oblivious_route(self) -> None:
        model = ObliviousDecisionTree(
            in_features=2,
            num_trees=1,
            depth=1,
            tree_dim=1,
            flatten_output=False,
            feature_selector="softmax",
            bin_function="sigmoid",
            hard_feature_selection=True,
            hard_routing=True,
            selector_init_std=0.0,
            threshold_init_std=0.0,
            response_init_std=0.0,
        )
        with torch.no_grad():
            model.feature_logits[:, 0, 0].copy_(torch.tensor([10.0, -10.0]))
            model.responses.copy_(torch.tensor([[[-2.0, 3.0]]]))

        x = torch.tensor([[-1.0, 50.0], [1.0, -50.0]])
        assert torch.equal(model(x), torch.tensor([[[-2.0]], [[3.0]]]))
        assert torch.equal(
            model.route(x),
            torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        )

    def test_initialization_is_explicit(self) -> None:
        model = ObliviousDecisionTree(3, num_trees=2, depth=2)
        x = torch.randn(32, 3)
        before = model.thresholds.detach().clone()
        model(x)
        assert torch.equal(model.thresholds, before)
        model.initialize_from_data(x)
        assert not torch.equal(model.thresholds, before)
        assert torch.allclose(model.route(x).sum(dim=-1), torch.ones(32, 2), atol=1e-5)


class TestNODE:
    """Verify hierarchical composition, YAML construction and safety limits."""

    def test_dense_hierarchy_shapes_gradients_and_routes(self) -> None:
        model = NODE(
            in_features=4,
            out_features=3,
            num_layers=2,
            num_trees=[3, 2],
            depth=[2, 3],
            tree_dim=3,
        )
        x = torch.randn(2, 5, 4)
        assert model(x).shape == (2, 5, 3)
        assert model.features(x).shape == (2, 5, 15)
        routes = model.route(x)
        assert [route.shape for route in routes] == [(2, 5, 3, 4), (2, 5, 2, 8)]

        model(x).square().mean().backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    def test_linear_readout_and_yaml_factory(self) -> None:
        model = ObjectFactory.build(
            {
                "target": "lambdaforge.nn.models.NODE",
                "params": {
                    "in_features": 5,
                    "out_features": 2,
                    "num_layers": 2,
                    "num_trees": [2, 3],
                    "depth": [2, 2],
                    "tree_dim": [4, 3],
                    "readout": "linear",
                },
            }
        )
        assert isinstance(model, NODE)
        assert model(torch.randn(7, 5)).shape == (7, 2)

    @pytest.mark.parametrize(
        ("factory", "match"),
        [
            (lambda: GradTree(2, 1, depth=5, max_leaves=16), "max_leaves"),
            (
                lambda: GRANDE(
                    2,
                    1,
                    depth=4,
                    num_estimators=5,
                    max_total_leaves=64,
                ),
                "max_total_leaves",
            ),
            (
                lambda: NODE(
                    2,
                    1,
                    num_layers=2,
                    num_trees=5,
                    depth=4,
                    max_total_leaves=100,
                ),
                "max_total_leaves",
            ),
        ],
    )
    def test_exponential_cost_limits(self, factory: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            factory()  # type: ignore[operator]


class TestTreeSafety:
    """Verify route-memory guards, dense caps and non-finite input handling."""

    def test_route_work_limits_use_actual_intermediate_shapes(self) -> None:
        factories = (
            lambda: GradTree(
                2,
                1,
                depth=3,
                max_route_elements_per_sample=23,
            ),
            lambda: GRANDE(
                2,
                1,
                depth=2,
                num_estimators=3,
                max_route_elements_per_sample=23,
            ),
            lambda: ObliviousDecisionTree(
                2,
                num_trees=3,
                depth=2,
                max_route_elements_per_sample=23,
            ),
            lambda: NODE(
                2,
                1,
                num_layers=2,
                num_trees=[2, 3],
                depth=[2, 1],
                max_route_elements_per_sample=21,
            ),
        )
        for factory in factories:
            with pytest.raises(ValueError, match="max_route_elements_per_sample.*autograd"):
                factory()

        with pytest.raises(ValueError, match="positive integer"):
            GradTree(2, 1, max_route_elements_per_sample=0)

    def test_dense_feature_cap_retains_learned_connections(self) -> None:
        model = NODE(
            in_features=4,
            out_features=1,
            num_layers=2,
            num_trees=1,
            depth=1,
            tree_dim=1,
            dense_connections=True,
            max_features=2,
        )
        x = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [-0.4, -0.3, -0.2, -0.1]],
        )
        first = model.layers[0]
        assert isinstance(first, ObliviousDecisionTree)
        first_input = model._limit_features(x, x)
        first_output = first(first_input).flatten(start_dim=-2)
        dense_input = torch.cat([x, first_output], dim=-1)
        second_input = model._limit_features(dense_input, x)

        assert torch.equal(first_input, x[..., :2])
        assert torch.equal(second_input[..., :1], x[..., :1])
        assert torch.equal(second_input[..., 1:], first_output[..., -1:])

        second = model.layers[1]
        assert isinstance(second, ObliviousDecisionTree)
        with torch.no_grad():
            second.feature_logits.zero_()
            second.thresholds.zero_()
            second.responses.copy_(torch.tensor([[[-1.0, 1.0]]]))
        layer_outputs, _ = model._forward_layers(x, collect_routes=False)
        response_gradient = torch.autograd.grad(
            layer_outputs[1].sum(),
            first.responses,
        )[0]
        assert torch.count_nonzero(response_gradient) > 0

    def test_nan_policy_covers_all_non_finite_values(self) -> None:
        factories = (
            lambda policy: GradTree(2, 1, depth=1, nan_policy=policy),
            lambda policy: GRANDE(
                2,
                1,
                depth=1,
                num_estimators=2,
                nan_policy=policy,
            ),
            lambda policy: ObliviousDecisionTree(
                2,
                num_trees=2,
                depth=1,
                nan_policy=policy,
            ),
            lambda policy: NODE(
                2,
                1,
                num_layers=1,
                num_trees=2,
                depth=1,
                nan_policy=policy,
            ),
        )
        contaminated = torch.tensor(
            [[float("nan"), 1.0], [float("inf"), 2.0], [float("-inf"), 3.0]],
        )
        replaced = torch.nan_to_num(
            contaminated,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for factory in factories:
            with pytest.raises(ValueError, match="non-finite"):
                factory("error")(contaminated)
            model = factory("zero")
            output = model(contaminated)
            assert torch.isfinite(output).all()
            assert torch.equal(output, model(replaced))
