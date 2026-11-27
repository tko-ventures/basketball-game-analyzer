"""Fingerprint the source a backend runs, so cached outputs made by other code are not reused (#52).

A layer's directory hash (:meth:`bga.config.RunConfig.layer_hash`) covers its config entry and upstream hashes, not
code. Changing a backend's code while keeping its params therefore wrote to, and read from, the same directory.
:func:`source_hash` closes that gap. It follows a module's imports within ``bga`` and within the module's own
top-level package (top-level and inside functions), and hashes each module's syntax tree with docstrings removed.
Comments, docstrings, formatting and line numbers do not change the hash; any change to code does, committed or not.

Modules are found on disk under each package's directory, never imported, so hashing a CUDA backend does not load
``torch``. Third-party packages are not followed: a library upgrade is visible in ``uv.lock``, not here.

"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

#: Package whose modules every closure follows, besides the backend's own top-level package.
PACKAGE = "bga"

#: Modules left out of every closure: they map names to code and cannot change an output. Registering a new backend
#: edits ``bga.registry``, and must not invalidate every cached output.
IGNORED = frozenset({"bga.registry"})


def top_level_source(name: str) -> Path | None:
    """Where a top-level package or module lives, found without importing any of its submodules.

    Args:
        name: A top-level package or module name, e.g. ``"bga"``.

    Returns:
        The package directory, the ``.py`` file of a single-file module, or ``None`` when there is no source.

    Examples:
        >>> (top_level_source("bga") / "codehash.py").is_file()
        True
        >>> top_level_source("no_such_package_here") is None
        True

    """
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    return Path(spec.origin) if spec.origin and spec.origin.endswith(".py") else None


def _module_file(name: str, roots: dict[str, Path]) -> Path | None:
    """Source file of a module or package (its ``__init__.py``) under one of ``roots``, or ``None``."""
    top, *rest = name.split(".")
    root = roots.get(top)
    if root is None or (root.is_file() and rest):
        return None
    if root.is_file():  # a single-file top-level module
        return root
    base = root.joinpath(*rest)
    for path in (base.with_suffix(".py") if rest else None, base / "__init__.py"):
        if path is not None and path.is_file():
            return path
    return None


def _imported(tree: ast.AST, name: str, is_package: bool) -> set[str]:
    """Dotted names one parsed module imports, with relative imports resolved against ``name``.

    For ``from a import b`` both ``a`` and ``a.b`` are returned, since ``b`` may be a submodule.

    """
    found: set[str] = set()
    base = name if is_package else name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                parts = base.split(".")
                mod = ".".join(parts[: len(parts) - node.level + 1] + ([mod] if mod else []))
            found.add(mod)
            found |= {f"{mod}.{a.name}" for a in node.names}
    return found


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Drop the docstring of the module and of every class and function, in place."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def source_closure(module: str) -> dict[str, Path]:
    """Every module ``module`` imports from ``bga`` or its own top-level package or file, transitively, and itself.

    Parent packages of each module are included too, since importing ``a.b`` runs ``a/__init__.py``.

    Args:
        module: A dotted module name, e.g. ``"bga.ball.fused"``, or a single-file top-level module.

    Returns:
        Module name to source file, sorted by name.

    Raises:
        ModuleNotFoundError: When ``module`` has no Python source file.

    Examples:
        >>> closure = source_closure("bga.ball.fused")
        >>> {"bga.ball.fused", "bga.ball.base", "bga.stage", "bga"} <= set(closure)
        True
        >>> "numpy" in closure, "bga.registry" in closure
        (False, False)

    """
    tops = {PACKAGE, module.split(".")[0]}
    roots = {top: d for top in tops if (d := top_level_source(top)) is not None}
    if _module_file(module, roots) is None:
        raise ModuleNotFoundError(f"no Python source for {module!r}")
    seen: dict[str, Path] = {}
    todo = [module]
    while todo:
        name = todo.pop()
        path = None if name in seen or name in IGNORED else _module_file(name, roots)
        if path is None:  # already seen, ignored, outside the roots, or a name imported from a module
            continue
        seen[name] = path
        parents = [name.rsplit(".", i)[0] for i in range(1, name.count(".") + 1)]
        tree = ast.parse(path.read_text(), filename=str(path))
        todo += parents + sorted(_imported(tree, name, path.name == "__init__.py"))
    return dict(sorted(seen.items()))


def source_hash(module: str) -> str:
    """Hash of the code ``module`` runs: its import closure (:func:`source_closure`), ignoring docstrings and layout.

    Args:
        module: A dotted module name, e.g. ``"bga.ball.fused"``, or a single-file top-level module.

    Returns:
        The first 8 hex digits of a SHA-256 over each module's name and docstring-free syntax tree.

    Examples:
        >>> h = source_hash("bga.ball.fused")
        >>> len(h), h == source_hash("bga.ball.fused")
        (8, True)

    """
    digest = hashlib.sha256()
    for name, path in source_closure(module).items():
        tree = _strip_docstrings(ast.parse(path.read_text(), filename=str(path)))
        digest.update(f"{name}\n{ast.dump(tree)}\n".encode())
    return digest.hexdigest()[:8]
