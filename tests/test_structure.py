"""Architectural regression tests for LambdaForge's object-oriented layout."""

import ast
from pathlib import Path


class TestObjectOrientedStructure:
    """Keep the documented one-class-per-module convention enforceable."""

    SOURCE_ROOT = Path(__file__).parents[1] / "src" / "lambdaforge"

    @classmethod
    def source_modules(cls) -> list[Path]:
        """Return framework modules in deterministic order."""
        return sorted(cls.SOURCE_ROOT.rglob("*.py"))

    def test_modules_do_not_define_loose_functions(self) -> None:
        """Require behavior to live on objects except for imported callables."""
        offenders: list[str] = []
        for path in self.source_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if functions:
                offenders.append(f"{path.relative_to(self.SOURCE_ROOT)}: {functions}")
        assert not offenders, "Loose module functions found:\n" + "\n".join(offenders)

    def test_modules_define_at_most_one_top_level_class(self) -> None:
        """Keep each implementation class in its own Python module."""
        offenders: list[str] = []
        for path in self.source_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
            if len(classes) > 1:
                offenders.append(f"{path.relative_to(self.SOURCE_ROOT)}: {classes}")
        assert not offenders, "Multiple top-level classes found:\n" + "\n".join(offenders)

    def test_class_names_match_their_module_names(self) -> None:
        """Keep implementation modules discoverable with the Java-like convention."""
        offenders: list[str] = []
        for path in self.source_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
            if classes and classes[0] != path.stem:
                offenders.append(
                    f"{path.relative_to(self.SOURCE_ROOT)}: expected {path.stem}, got {classes[0]}"
                )
        assert not offenders, "Class/module name mismatches found:\n" + "\n".join(offenders)

    def test_every_source_module_has_documentation(self) -> None:
        """Require every source file to explain its module-level responsibility."""
        offenders: list[str] = []
        for path in self.source_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if ast.get_docstring(tree, clean=False) is None:
                offenders.append(str(path.relative_to(self.SOURCE_ROOT)))
        assert not offenders, "Undocumented modules found:\n" + "\n".join(offenders)

    def test_every_top_level_class_has_documentation(self) -> None:
        """Require a class docstring at the beginning of every class body."""
        offenders: list[str] = []
        for path in self.source_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and ast.get_docstring(node, clean=False) is None:
                    offenders.append(f"{path.relative_to(self.SOURCE_ROOT)}: {node.name}")
        assert not offenders, "Undocumented classes found:\n" + "\n".join(offenders)
