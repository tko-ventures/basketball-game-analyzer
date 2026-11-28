"""Repository-wide security policy tests hedging against common AI-agent regressions."""

import ast
import re
import shlex
import tokenize
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import InvalidRequirement, Requirement

ROOT = Path(__file__).resolve().parents[1]
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
NOQA_RE = re.compile(r"#\s*(?:(?:ruff|flake8)\s*:\s*)?noqa(?:\s*:\s*([^#]+)|\b(?![\w:]))", re.IGNORECASE)
SEC_CODE_RE = re.compile(r"\bS\d+\b")
PAREN_REASON_RE = re.compile(r"\(([^)]+)\)")
ARCHIVE_PIN_RE = re.compile(r"(?:@[0-9a-f]{40}\b|/archive/[0-9a-f]{40}\.(?:tar\.gz|zip)$)")
PIP_VALUE_FLAGS = frozenset({
    "-c",
    "--constraint",
    "-f",
    "--find-links",
    "-i",
    "--index-url",
    "--extra-index-url",
    "-r",
    "--requirement",
    "-t",
    "--target",
    "--prefix",
    "--root",
    "--cache-dir",
})
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


def _collect_torch_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    torch_aliases: set[str] = set()
    load_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            torch_aliases.update(a.asname or "torch" for a in node.names if a.name == "torch")
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            load_aliases.update(a.asname or "load" for a in node.names if a.name == "load")
    return torch_aliases, load_aliases


def _is_torch_load_call(node: ast.Call, torch_aliases: set[str], load_aliases: set[str]) -> bool:
    if isinstance(node.func, ast.Attribute):
        return (
            node.func.attr == "load" and isinstance(node.func.value, ast.Name) and node.func.value.id in torch_aliases
        )
    return isinstance(node.func, ast.Name) and node.func.id in load_aliases


def _torch_load_violations_in_source(source: str, label: str) -> list[str]:
    tree = ast.parse(source, filename=label)
    torch_aliases, load_aliases = _collect_torch_aliases(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_torch_load_call(node, torch_aliases, load_aliases):
            continue
        weights_only = [kw for kw in node.keywords if kw.arg == "weights_only"]
        if (
            not weights_only
            or not isinstance(weights_only[0].value, ast.Constant)
            or weights_only[0].value.value is not True
        ):
            violations.append(f"{label}:{node.lineno}")
    return violations


def _torch_load_violations(root_dir: str) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(root_dir):
        violations.extend(
            _torch_load_violations_in_source(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
        )
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


def _find_pip_install_start(tokens: list[str], idx: int) -> int | None:
    if tokens[idx] in {"pip", "pip3"} and idx + 1 < len(tokens) and tokens[idx + 1] == "install":
        return idx + 2
    if tokens[idx : idx + 3] == ["uv", "pip", "install"]:
        return idx + 3
    if tokens[idx : idx + 4] == ["python", "-m", "pip", "install"]:
        return idx + 4
    return None


def _extract_pip_install_specs(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    specs: list[str] = []
    i = 0
    while i < len(tokens):
        start = _find_pip_install_start(tokens, i)
        if start is None:
            i += 1
            continue
        i = start
        while i < len(tokens) and tokens[i] not in {"&&", "||", ";", "|"}:
            tok = tokens[i]
            if tok in PIP_VALUE_FLAGS:
                i += 2
                continue
            if not tok.startswith("-"):
                specs.append(tok)
            i += 1
    return specs


def _extract_colab_packages(colab: Path, constants: dict[str, str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    tree = ast.parse(colab.read_text(encoding="utf-8"), filename=str(colab))
    for stmt in tree.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        if stmt.target.id == "SETUP_PACKAGES" and isinstance(stmt.value, ast.Dict):
            for key_node, tup_node in zip(stmt.value.keys, stmt.value.values, strict=True):
                key = _eval_str_expr(key_node, constants) if key_node is not None else "?"
                if isinstance(tup_node, ast.Tuple):
                    specs.extend(
                        (f"colab_train.SETUP_PACKAGES[{key}]", _eval_str_expr(item, constants) or ast.dump(item))
                        for item in tup_node.elts
                    )
        elif stmt.target.id == "BASE_PACKAGES" and isinstance(stmt.value, ast.Tuple):
            for pair in stmt.value.elts:
                if isinstance(pair, ast.Tuple) and len(pair.elts) == 2:
                    val = _eval_str_expr(pair.elts[1], constants)
                    specs.append(("colab_train.BASE_PACKAGES", val or ast.dump(pair.elts[1])))
    return specs


def _extract_remote_file_installs(path: Path, constants: dict[str, str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"pip_install", "uv_pip_install"}:
            specs.extend(
                (f"{path.name}:{node.lineno}", _eval_str_expr(arg, constants) or ast.dump(arg)) for arg in node.args
            )
        elif node.func.attr == "run_commands":
            for arg in node.args:
                cmd = _eval_str_expr(arg, constants)
                if cmd is not None:
                    specs.extend((f"{path.name}:{node.lineno}", spec) for spec in _extract_pip_install_specs(cmd))
    return specs


def _collect_remote_package_specs() -> list[tuple[str, str]]:
    dispatch = ROOT / "src" / "bga" / "train" / "dispatch.py"
    remote_files = tuple(sorted((ROOT / "src" / "bga" / "remote").glob("*.py")))
    constants = _resolve_constants((dispatch, *remote_files))
    specs = [
        (f"dispatch.{name}", constants[name])
        for name in ("RFDETR_PACKAGE", "RFDETR_PLUS_PACKAGE", "REID_PACKAGE", "PARSEQ_PACKAGE")
    ]
    specs.extend(_extract_colab_packages(ROOT / "src" / "bga" / "remote" / "colab_train.py", constants))
    for path in remote_files:
        file_constants = _resolve_constants((dispatch, path))
        specs.extend(_extract_remote_file_installs(path, file_constants))
    return specs


def _is_pinned_requirement(spec: str) -> bool:
    if spec.startswith(("http://", "https://")):
        return bool(ARCHIVE_PIN_RE.search(spec))
    try:
        req = Requirement(spec)
    except InvalidRequirement:
        return False
    if req.url is not None:
        return bool(ARCHIVE_PIN_RE.search(req.url))
    return any(sp.operator in {"==", "==="} and "*" not in sp.version for sp in req.specifier)


def _unpinned_remote_specs() -> list[str]:
    return [f"{loc}: {spec}" for loc, spec in _collect_remote_package_specs() if not _is_pinned_requirement(spec)]


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
    codes_group = match.group(1)
    if codes_group is not None:
        codes_part = codes_group.split("(", 1)[0]
        if not SEC_CODE_RE.search(codes_part):
            return True
    tail = comment[match.end() :].strip()
    if tail.startswith("#") and len(tail.lstrip("#").strip()) >= 3:
        return True
    if tail.startswith("("):
        paren = PAREN_REASON_RE.match(tail)
        if paren is not None and len(paren.group(1).strip()) >= 3:
            return True
    paren_in_codes = PAREN_REASON_RE.search(codes_group or "")
    return paren_in_codes is not None and len(paren_in_codes.group(1).strip()) >= 3


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

    @pytest.mark.parametrize(
        ("source", "expected_violation"),
        [
            pytest.param("import torch\ntorch.load('a.pt')\n", True, id="direct-unsafe"),
            pytest.param("import torch as t\nt.load('a.pt')\n", True, id="module-alias-unsafe"),
            pytest.param("from torch import load as torch_load\ntorch_load('a.pt')\n", True, id="func-alias-unsafe"),
            pytest.param("import torch as t\nt.load('a.pt', weights_only=True)\n", False, id="module-alias-safe"),
            pytest.param(
                "from torch import load as torch_load\ntorch_load('a.pt', weights_only=True)\n",
                False,
                id="func-alias-safe",
            ),
        ],
    )
    def test_detects_aliased_torch_load_calls(self, source: str, expected_violation: bool):
        assert bool(_torch_load_violations_in_source(source, "snippet.py")) is expected_violation


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

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            pytest.param("torch==2.15.0", True, id="exact-pin"),
            pytest.param("rfdetr[train]==1.12.1", True, id="extra-exact-pin"),
            pytest.param(
                "strhub @ git+https://github.com/baudm/parseq@1902db043c029a7e03a3818c616c06600af574be",
                True,
                id="git-sha-pin",
            ),
            pytest.param(
                "https://github.com/facebookresearch/sam2/archive/2b90b9f5ceec907a1c18123530e92e794ad901a4.tar.gz",
                True,
                id="archive-sha-pin",
            ),
            pytest.param("torch==2.*", False, id="wildcard-pin-rejected"),
            pytest.param("numpy<2.3", False, id="upper-bound-rejected"),
            pytest.param("huggingface_hub", False, id="unpinned-name-rejected"),
            pytest.param("pkg @ git+https://github.com/org/repo@main", False, id="branch-ref-rejected"),
        ],
    )
    def test_pin_validator_rejects_wildcards_and_unpinned_specs(self, spec: str, expected: bool):
        assert _is_pinned_requirement(spec) is expected


class TestNoTrojanSourceUnicode:
    def test_no_bidi_or_invisible_control_characters(self):
        assert _trojan_source_hits() == []


class TestReasonedSecurityNoqa:
    @pytest.mark.parametrize("root_dir", [pytest.param(d, id=d) for d in SCAN_DIRS])
    def test_security_noqa_has_inline_rationale(self, root_dir: str):
        assert _unreasoned_security_noqa(root_dir) == []

    @pytest.mark.parametrize(
        ("comment", "expected_valid"),
        [
            pytest.param("# noqa", False, id="blanket-unreasoned"),
            pytest.param("# ruff: noqa", False, id="file-blanket-unreasoned"),
            pytest.param("# ruff: noqa: S310", False, id="file-security-unreasoned"),
            pytest.param("# ruff: noqa: S310  # scheme validated", True, id="file-security-reasoned"),
            pytest.param("# ruff: noqa: E501  (sentences kept whole)", True, id="non-security-file-noqa"),
            pytest.param("# noqa: S603  # fixed argv", True, id="inline-security-reasoned"),
            pytest.param("# noqa: S603 ()", False, id="empty-parens-rejected"),
        ],
    )
    def test_comment_reason_parser_handles_blanket_and_file_noqa(self, comment: str, expected_valid: bool):
        assert _has_noqa_reason(comment) is expected_valid
