"""Output formats supported by configuration migration previews."""

from enum import Enum


class MigrationPreviewFormat(str, Enum):
    """Select human diff, migrated YAML or machine-readable JSON output."""

    DIFF = "diff"
    YAML = "yaml"
    JSON = "json"
