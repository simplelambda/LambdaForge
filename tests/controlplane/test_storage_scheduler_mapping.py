"""Site-specific scratch reservation must not be claimed by default."""

from lambdaforge.controlplane.SlurmResourceMapping import SlurmResourceMapping
from lambdaforge.execution.ResourceRequest import ResourceRequest


def test_slurm_storage_omission_is_explicit_and_configurable() -> None:
    requested = ResourceRequest.from_mapping({"storage": "100GiB"})
    directives, warnings = SlurmResourceMapping().render(requested)
    assert not any("--tmp" in value for value in directives)
    assert any("storage" in value and "not enforce" in value for value in warnings)
    configured = SlurmResourceMapping.from_mapping(
        {
            "storage": {"option": "tmp", "value": "{storage_mib}"},
        }
    )
    directives, warnings = configured.render(requested)
    assert "--tmp=102400" in directives and not warnings
