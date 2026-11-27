"""Source fingerprints: a code change moves the hash, a docstring, comment or layout change does not (#52)."""

import pytest

from bga import codehash

MODULE = '''"""Module docstring."""

from fakecode import helper


def f(x):
    """Docstring."""
    return helper.g(x) + 1
'''

HELPER = """def g(x):
    return x * 2
"""


@pytest.fixture
def package(tmp_path, monkeypatch):
    """A throwaway top-level package ``fakecode`` with a module importing a helper, importable from ``tmp_path``."""
    pkg = tmp_path / "fakecode"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(MODULE)
    (pkg / "helper.py").write_text(HELPER)
    (pkg / "unused.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return pkg


class TestSourceClosure:
    def test_follows_imports_within_the_package(self, package):
        assert list(codehash.source_closure("fakecode.mod")) == ["fakecode", "fakecode.helper", "fakecode.mod"]

    def test_missing_module_is_an_error(self, package):
        with pytest.raises(ModuleNotFoundError, match="no Python source"):
            codehash.source_closure("fakecode.missing")


class TestSourceHash:
    @pytest.mark.parametrize(
        ("file", "text"),
        [
            pytest.param("mod.py", MODULE.replace('"""Docstring."""', '"""Another docstring."""'), id="docstring"),
            pytest.param("mod.py", MODULE.replace("return", "# a comment\n    return"), id="comment"),
            pytest.param("mod.py", MODULE.replace("helper.g(x) + 1", "helper.g( x )  +  1"), id="layout"),
            pytest.param("unused.py", "X = 2\n", id="module-not-imported"),
        ],
    )
    def test_unchanged_by_non_code_edits(self, package, file, text):
        before = codehash.source_hash("fakecode.mod")
        (package / file).write_text(text)
        assert codehash.source_hash("fakecode.mod") == before

    @pytest.mark.parametrize(
        ("file", "text"),
        [
            pytest.param("mod.py", MODULE.replace("+ 1", "+ 2"), id="own-module"),
            pytest.param("helper.py", HELPER.replace("* 2", "* 3"), id="imported-helper"),
        ],
    )
    def test_changed_by_code_edits(self, package, file, text):
        before = codehash.source_hash("fakecode.mod")
        (package / file).write_text(text)
        assert codehash.source_hash("fakecode.mod") != before
