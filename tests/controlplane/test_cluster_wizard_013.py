"""The guided cluster UX remains a thin layer over native cluster commands."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import yaml

from lambdaforge.cli.ClusterWizard import ClusterWizard
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile


def test_setup_emits_one_secret_free_native_add_command() -> None:
    answers = StringIO(
        "\n".join(
            (
                "citius-ctgpgpu16",
                "ctgpgpu16.example",
                "researcher",
                "",  # port 22
                "",  # OpenSSH
                "",  # direct processes (safe default)
                "/home/researcher",
                "/home/researcher/WISDOM",
                "/data/researcher/datasets",
                "",  # managed environment
                "",  # python3
                "2",  # require CUDA
                "4",  # site command wrapper
                "",  # wrapper: gpu exec
                "1",  # configure automatic claim
                "",  # gpu claim --numgpus {gpu_count}
                "",  # gpu release
                "",  # user catalog
                "",  # save
                "",  # no advanced editor
            )
        )
        + "\n"
    )
    output = StringIO()
    commands: list[tuple[str, ...]] = []
    wizard = ClusterWizard(
        lambda command: commands.append(tuple(command)) or 0,
        input_stream=answers,
        output=output,
    )

    assert wizard.setup(offer_test=False) == 0
    assert len(commands) == 1
    command = commands[0]
    assert command[:2] == ("add", "citius-ctgpgpu16")
    prefix_index = command.index("--gpu-command-prefix")
    claim_index = command.index("--gpu-claim-command")
    release_index = command.index("--gpu-release-command")
    assert command[prefix_index + 1 : claim_index] == ("gpu", "exec")
    assert command[claim_index + 1 : release_index] == (
        "gpu",
        "claim",
        "--numgpus",
        "{gpu_count}",
    )
    assert command[release_index + 1 :] == ("gpu", "release")
    assert "--require-cuda" in command
    assert "password" not in " ".join(command).lower()
    assert "Nothing is written until you confirm" in output.getvalue()
    assert "not whether they use a GPU" in output.getvalue()


def test_setup_can_exit_from_any_prompt_without_writing() -> None:
    commands: list[tuple[str, ...]] = []
    output = StringIO()
    wizard = ClusterWizard(
        lambda command: commands.append(tuple(command)) or 0,
        input_stream=StringIO("example\nq\n"),
        output=output,
    )

    assert wizard.setup() == 130
    assert commands == []
    assert "no unconfirmed settings were written" in output.getvalue()
    assert "0/q: exit wizard" in output.getvalue()


def test_choice_fallback_explains_the_consequence_of_each_gpu_policy() -> None:
    output = StringIO()
    wizard = ClusterWizard(lambda _command: 0, input_stream=StringIO("1\n"), output=output)

    selected = wizard._choice(
        "GPU access",
        "Choose how a site grants accelerators.",
        (
            ("Exclusive", "exclusive", "Wait for an exclusive visible-device lease."),
            ("Shared", "shared", "Allow external use only when site policy permits it."),
        ),
    )

    assert selected == "exclusive"
    assert "Wait for an exclusive visible-device lease" in output.getvalue()
    assert "Allow external use only when site policy permits it" in output.getvalue()


def test_modify_uses_native_set_for_advanced_gpu_claim(tmp_path: Path) -> None:
    catalog_path = tmp_path / "clusters.yaml"
    ClusterCatalog.add(
        catalog_path,
        ClusterProfile(
            "atlas",
            transport="ssh",
            scheduler="local",
            host="atlas.invalid",
            workspace="/work",
        ),
    )
    answers = StringIO(
        "\n".join(
            (
                "4",  # GPU access section
                "4",  # command mode on a direct host
                "",  # gpu exec
                "1",  # persistent claim/release
                "",  # gpu claim --numgpus {gpu_count}
                "",  # gpu release
                "",  # save atomic policy
                "9",  # done
            )
        )
        + "\n"
    )
    commands: list[tuple[str, ...]] = []
    wizard = ClusterWizard(
        lambda command: commands.append(tuple(command)) or 0,
        catalog_path=catalog_path,
        input_stream=answers,
        output=StringIO(),
    )

    assert wizard.modify("atlas") == 0
    assert len(commands) == 1
    assert commands[0][:3] == ("set", "atlas", "gpu_access")
    assert yaml.safe_load(commands[0][3]) == {
        "mode": "command",
        "command_prefix": ["gpu", "exec"],
        "claim_command": ["gpu", "claim", "--numgpus", "{gpu_count}"],
        "release_command": ["gpu", "release"],
    }


def test_modify_can_exit_from_a_nested_prompt_without_applying_it(tmp_path: Path) -> None:
    catalog_path = tmp_path / "clusters.yaml"
    ClusterCatalog.add(
        catalog_path,
        ClusterProfile(
            "atlas",
            transport="ssh",
            scheduler="local",
            host="atlas.invalid",
            workspace="/work",
        ),
    )
    commands: list[tuple[str, ...]] = []
    wizard = ClusterWizard(
        lambda command: commands.append(tuple(command)) or 0,
        catalog_path=catalog_path,
        input_stream=StringIO("4\nq\n"),
        output=StringIO(),
    )

    assert wizard.modify("atlas") == 130
    assert commands == []
