"""Task help and shell completion derived from the sole argparse grammar."""

from __future__ import annotations

import argparse
import json
from typing import Any


def command_catalog(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    """Project implemented CLI help into discovery, not a third operation authority."""
    rows: list[dict[str, Any]] = []

    def visit(current: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        for action in current._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            help_text = {item.dest: item.help for item in action._get_subactions()}
            for name, child in action.choices.items():
                if name.startswith("_") or help_text.get(name) == argparse.SUPPRESS:
                    continue
                selected = (*path, name)
                rows.append(
                    {
                        "identifier": "lf " + " ".join(selected),
                        "category": path[0] if path else name,
                        "description": help_text.get(name)
                        or child.description
                        or "Inspect this command with --help.",
                        "usage": child.format_usage().strip(),
                        "service": "native CLI route; availability does not imply MCP permission",
                    }
                )
                visit(child, selected)

    visit(parser, ())
    return rows


def completion_candidates(parser: argparse.ArgumentParser, words: list[str]) -> list[str]:
    """Pure grammar expansion: no project imports, catalogue/secret reads or provider calls."""
    current = parser
    pending: argparse.Action | None = None
    tokens = words[1:] if words and words[0] == "--" else words
    prefix = tokens[-1] if tokens else ""
    for token in tokens[:-1]:
        if pending is not None:
            pending = None
            continue
        if token.startswith("-"):
            option = current._option_string_actions.get(token.split("=", 1)[0])
            if option is not None and option.nargs != 0 and "=" not in token:
                pending = option
            continue
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction) and token in action.choices:
                current = action.choices[token]
                break
    if prefix.startswith("--") and "=" in prefix:
        name, value = prefix.split("=", 1)
        option = current._option_string_actions.get(name)
        return (
            [
                name + "=" + str(choice)
                for choice in (option.choices or ())
                if str(choice).startswith(value)
            ]
            if option is not None
            else []
        )
    if pending is not None:
        return sorted(
            str(value) for value in (pending.choices or ()) if str(value).startswith(prefix)
        )
    values: set[str] = set()
    for action in current._actions:
        if action.help == argparse.SUPPRESS:
            continue
        values.update(action.option_strings)
        if isinstance(action, argparse._SubParsersAction):
            hidden = {
                item.dest for item in action._get_subactions() if item.help == argparse.SUPPRESS
            }
            values.update(
                name for name in action.choices if name not in hidden and not name.startswith("_")
            )
        elif not action.option_strings and action.choices is not None:
            values.update(str(value) for value in action.choices)
    return sorted(value for value in values if value.startswith(prefix))


def completion_script(shell: str) -> str:
    """Print opt-in activation scripts; never modify shell startup/configuration files."""
    if shell == "bash":
        return """# Activate: source <(lf completion bash)
_lf_complete() {
    local candidate
    COMPREPLY=()
    while IFS= read -r candidate; do COMPREPLY+=("$candidate"); done < <(
        "${COMP_WORDS[0]}" _complete -- "${COMP_WORDS[@]:1:COMP_CWORD}"
    )
}
complete -o default -F _lf_complete lf lambdaforge
"""
    if shell == "zsh":
        return """# Activate: source <(lf completion zsh)
_lf_complete() {
    local -a candidates
    candidates=("${(@f)$("${words[1]}" _complete -- "${(@)words[2,CURRENT]}")}")
    if (( ${#candidates} )); then compadd -- "${candidates[@]}"; else _files; fi
}
compdef _lf_complete lf lambdaforge
"""
    if shell == "fish":
        return """# Activate: lf completion fish | source
function __lf_complete
    set -l words (commandline -opc)
    set -l current (commandline -ct)
    command $words[1] _complete -- $words[2..-1] "$current"
end
complete -c lf -a '(__lf_complete)'
complete -c lambdaforge -a '(__lf_complete)'
"""
    raise ValueError("Completion shell must be bash, zsh or fish.")


def run_discovery_command(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if arguments.command == "completion":
        print(completion_script(arguments.shell), end="")
        return 0
    if arguments.command == "_complete":
        print("\n".join(completion_candidates(parser, arguments.words)))
        return 0
    if not 1 <= arguments.limit <= 100:
        raise ValueError("Task help --limit must be 1..100.")
    # Reuse the existing English/Spanish deterministic capability search vocabulary.
    from lambdaforge.agents.discovery import search_capabilities

    rows = search_capabilities(command_catalog(parser), arguments.search)[: arguments.limit]
    value = {
        "query": arguments.search,
        "items": rows,
        "source": "installed argparse grammar",
        "science_executed": False,
    }
    print(
        json.dumps(value, indent=2)
        if arguments.json
        else "\n\n".join(
            f"{row['identifier']} · {row['description']}\n  {row['usage']}" for row in rows
        )
        or "No matching command. Try shorter task terms or lf --help."
    )
    return 0
