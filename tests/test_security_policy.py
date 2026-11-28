"""Repository-wide security policy tests hedging against common AI-agent regressions."""

import ast
import re
import tokenize
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TOKENS = ("ultralytics", "yolo26", "sam3", "sam-3", "sam_3")
FORBIDDEN_RE = re.compile(r"\b(?:ultralytics|yolo26|sam[-_]?3)\b", re.IGNORECASE)
TROJAN_CODEPOINTS = frozenset({
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
    "\u200e",
    "\u200f",
    "\u061c",
})
NOQA_RE = re.compile(r"#\s*noqa\s*:\s*([^#]+)")
SEC_CODE_RE = re.compile(r"\bS\d+\b")
PIN_RE = re.compile(r"(?:==|@[0-9a-f]{40}\b|/archive/[0-9a-f]{40}\.)")
SCAN_DIRS = ("src", "tools", "scripts")
UNICODE_CHECK_TARGETS = (
    "src",
    "tools",
    "scripts",
    "tests",
    ".github",
    "configs",
    "docs",
    "AGENTS.md",
    "pyproject.toml",
    ".pre-commit-config.yaml",
)
BINARY_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".mp4", ".pyc", ".pth", ".safetensors", ".npy"})


def _iter_python_files(root_dir: str) -> list[Path]:
    base = ROOT / root_dir
    if not base.exists():
        return []
    excluded = {
        ROOT / "src" / "bga" / "track" / "_sv_bytetrack",
        ROOT / "src" / "bga" / "ingest" / "_transnetv2",
    }
    return sorted(p for p in base.rglob("*.py") if not any(p.is_relative_to(ex) for ex in excluded))


def _is_torch_load_call(node: ast.Call, imported_load: bool) -> bool:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == "load" and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
    return imported_load and isinstance(node.func, ast.Name) and node.func.id == "load"


def _torch_load_violations(root_dir: str) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(root_dir):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_load = any(
            isinstance(n, ast.ImportFrom) and n.module == "torch" and any(a.name == "load" for a in n.names)
            for n in ast.walk(tree)
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_torch_load_call(node, imported_load):
                continue
            weights_only = [kw for kw in node.keywords if kw.arg == "weights_only"]
            if (
                not weights_only
                or not isinstance(weights_only[0].value, ast.Constant)
                or weights_only[0].value.value is not True
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return violations


def _iter_yaml_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in _iter_yaml_strings(item)]
    if isinstance(value, dict):
        return [s for val in value.values() for s in _iter_yaml_strings(val)]
    return []


def _forbidden_hits_in_pyproject() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    opt = project.get("optional-dependencies", {})
    groups = data.get("dependency-groups", {})
    uv_cfg = data.get("tool", {}).get("uv", {})
    specs = [
        *project.get("dependencies", []),
        *(dep for deps in opt.values() for dep in deps),
        *(dep for deps in groups.values() for dep in deps),
        *uv_cfg.get("override-dependencies", []),
        *uv_cfg.get("sources", {}).keys(),
    ]
    return [spec for spec in specs if FORBIDDEN_RE.search(str(spec))]


def _forbidden_hits_in_uv_lock() -> list[str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return [pkg["name"] for pkg in lock.get("package", []) if FORBIDDEN_RE.search(pkg.get("name", ""))]


def _forbidden_hits_in_configs() -> list[str]:
    hits: list[str] = []
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        hits.extend(
            f"{path.relative_to(ROOT)}:{text}" for text in _iter_yaml_strings(loaded) if FORBIDDEN_RE.search(text)
        )
    return hits


def _resolve_constants(files: tuple[Path, ...]) -> dict[str, str]:
    constants: dict[str, str] = {}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                val = _eval_str_expr(stmt.value, constants)
                if val is not None:
                    constants[stmt.targets[0].id] = val
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                val = _eval_str_expr(stmt.value, constants)
                if val is not None:
                    constants[stmt.target.id] = val
    return constants


def _eval_str_expr(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for val in node.values:
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                parts.append(val.value)
            elif isinstance(val, ast.FormattedValue):
                inner = _eval_str_expr(val.value, constants)
                if inner is None:
                    return None
                parts.append(inner)
            else:
                return None
        return "".join(parts)
    return None


def _collect_remote_package_specs() -> list[tuple[str, str]]:
    dispatch = ROOT / "src" / "bga" / "train" / "dispatch.py"
    colab = ROOT / "src" / "bga" / "remote" / "colab_train.py"
    modal_runner = ROOT / "src" / "bga" / "remote" / "modal_runner.py"
    modal_train = ROOT / "src" / "bga" / "remote" / "modal_train.py"
    constants = _resolve_constants((dispatch, modal_runner, modal_train, colab))
    specs = [
        (f"dispatch.{name}", constants[name])
        for name in ("RFDETR_PACKAGE", "RFDETR_PLUS_PACKAGE", "REID_PACKAGE", "PARSEQ_PACKAGE")
    ]

    colab_tree = ast.parse(colab.read_text(encoding="utf-8"), filename=str(colab))
    for stmt in colab_tree.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "SETUP_PACKAGES":
            assert isinstance(stmt.value, ast.Dict)
            for key_node, tup_node in zip(stmt.value.keys, stmt.value.values, strict=True):
                key = _eval_str_expr(key_node, constants) if key_node is not None else "?"
                if isinstance(tup_node, ast.Tuple):
                    for item in tup_node.elts:
                        val = _eval_str_expr(item, constants)
                        specs.append((f"colab_train.SETUP_PACKAGES[{key}]", val or ast.dump(item)))

    for path, attr in ((modal_train, "pip_install"), (modal_runner, "uv_pip_install")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr:
                for arg in node.args:
                    val = _eval_str_expr(arg, constants)
                    specs.append((f"{path.name}:{node.lineno}", val or ast.dump(arg)))
    return specs


def _unpinned_remote_specs() -> list[str]:
    return [f"{loc}: {spec}" for loc, spec in _collect_remote_package_specs() if not PIN_RE.search(spec)]


def _forbidden_hits_in_remote_specs() -> list[str]:
    return [f"{loc}: {spec}" for loc, spec in _collect_remote_package_specs() if FORBIDDEN_RE.search(spec)]


def _iter_unicode_target_files() -> list[Path]:
    files: list[Path] = []
    for target in UNICODE_CHECK_TARGETS:
        path = ROOT / target
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix.lower() not in BINARY_SUFFIXES
                )
            )
    return files


def _trojan_source_hits() -> list[str]:
    hits: list[str] = []
    for path in _iter_unicode_target_files():
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8", errors="replace")
        hits.extend(
            f"{rel}:{idx}"
            for idx, line in enumerate(text.splitlines(), start=1)
            if any(ch in TROJAN_CODEPOINTS for ch in line)
        )
    return hits


def _has_noqa_reason(comment: str) -> bool:
    match = NOQA_RE.search(comment)
    if match is None:
        return True
    codes = match.group(1)
    if not SEC_CODE_RE.search(codes):
        return True
    tail = comment[match.end() :].strip()
    if tail.startswith("#") and len(tail.lstrip("#").strip()) >= 3:
        return True
    return "(" in codes and ")" in codes


def _unreasoned_security_noqa(root_dir: str) -> list[str]:
    hits: list[str] = []
    for path in _iter_python_files(root_dir):
        with path.open("rb") as handle:
            hits.extend(
                f"{path.relative_to(ROOT)}:{tok.start[0]}: {tok.string}"
                for tok in tokenize.tokenize(handle.readline)
                if tok.type == tokenize.COMMENT and not _has_noqa_reason(tok.string)
            )
    return hits


class TestTorchLoadSafety:
    @pytest.mark.parametrize("root_dir", [pytest.param(d, id=d) for d in SCAN_DIRS])
    def test_torch_load_requires_weights_only_true(self, root_dir: str):
        assert _torch_load_violations(root_dir) == []


class TestForbiddenPackagesAndModels:
    @pytest.mark.parametrize(
        ("label", "finder"),
        [
            pytest.param("pyproject", _forbidden_hits_in_pyproject, id="pyproject"),
            pytest.param("uv-lock", _forbidden_hits_in_uv_lock, id="uv-lock"),
            pytest.param("configs", _forbidden_hits_in_configs, id="configs"),
            pytest.param("remote-specs", _forbidden_hits_in_remote_specs, id="remote-specs"),
        ],
    )
    def test_no_forbidden_models_or_packages(self, label: str, finder):
        assert finder() == [], f"forbidden package/model reference in {label}"


class TestRemotePackagePins:
    def test_all_remote_and_trainer_packages_are_pinned(self):
        assert _unpinned_remote_specs() == []


class TestNoTrojanSourceUnicode:
    def test_no_bidi_or_invisible_control_characters(self):
        assert _trojan_source_hits() == []


class TestReasonedSecurityNoqa:
    @pytest.mark.parametrize("root_dir", [pytest.param(d, id=d) for d in SCAN_DIRS])
    def test_security_noqa_has_inline_rationale(self, root_dir: str):
        assert _unreasoned_security_noqa(root_dir) == []
