"""Static authoring commands and native semantic inspection, separate from execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lambdaforge.configuration.ConfigurationGuidance import ConfigurationGuidance


def run_config_authoring(arguments: argparse.Namespace) -> int:
    if arguments.config_command == "schema":
        from lambdaforge.configuration.ConfigurationSchemas import ConfigurationSchemas

        value = ConfigurationSchemas.schema(getattr(arguments, "kind", "work"))
        text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        if arguments.output is not None:
            path = Path(arguments.output).expanduser().absolute()
            if path.resolve() != path:
                raise ValueError("Schema destination cannot contain symlinks.")
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            print(
                json.dumps({"path": str(path), "schema_id": value["$id"]})
                if arguments.json
                else f"Exported installed schema: {path}"
            )
        else:
            print(text, end="")
        return 0
    if arguments.config_command == "explain":
        value = ConfigurationGuidance.explain_field(arguments.config, arguments.field)
        human = (
            f"{value['field']} · {value['description']}\n"
            f"Type: {value['type']}\nConfigured: {value['configured']!r}\n"
            f"Normalized: {value['normalized']!r}\nDefault: {value['default']!r}\n"
            f"{value['normalization']}\nExamples: {value['examples']}\n"
            f"Related: {value['related']}\nCommon errors: {value['common_errors']}"
        )
    else:
        value = ConfigurationGuidance.diff(arguments.previous, arguments.proposed)
        human = (
            "\n".join(
                f"{row['category']} · {row['path']}: {row['previous']!r} -> {row['proposed']!r}"
                for row in value["changes"]
            )
            or "No normalized configuration differences."
        )
        human += "\nRecovery/extension compatibility and input byte equivalence are not evaluated."
    encoded = json.dumps(value, indent=2, allow_nan=False)
    if len(encoded.encode()) > 512 * 1024:
        raise ValueError("Authoring response exceeds 512 KiB; inspect a smaller configuration.")
    print(encoded if arguments.json else human)
    return 0
