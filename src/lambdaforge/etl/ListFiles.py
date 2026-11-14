"""Explicit file discovery below a declared root."""

from pathlib import Path

from lambdaforge.catalog.errors import ContractError


class ListFiles:
    """Deterministic paths only; does not read file contents or infer sample associations."""

    def __init__(self, pattern: str = "*", max_files: int = 10000) -> None:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts or max_files < 1:
            raise ContractError("UNSAFE_PATH", "Glob must remain below its explicit root.")
        self.pattern, self.max_files = pattern, max_files

    def __call__(self, root: str | Path) -> list[Path]:
        from lambdaforge.work.snapshot import validate_path

        base = Path(root).absolute()
        validate_path(base)
        result = []
        for path in base.glob(self.pattern):
            validate_path(path)
            if path.is_file():
                result.append(path)
            if len(result) > self.max_files:
                raise ContractError("READ_LIMIT", "File discovery exceeds max_files.")
        return sorted(result, key=lambda p: p.relative_to(base).as_posix().encode())
