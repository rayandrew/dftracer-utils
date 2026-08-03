# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import ast
import os
import inspect
import importlib
import subprocess
import sys
from pathlib import Path

# conf.py's own directory holds _ext_stub; ensure it is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ext_stub import install_extension_stub  # noqa: E402

# Auto-generate Mermaid class diagrams from Doxygen XML before building
_docs_dir = Path(__file__).parent.parent  # docs/
_script = _docs_dir / "scripts" / "generate_class_diagrams.py"
_xml_dir = _docs_dir / "doxygen" / "xml"
_gen_dir = _docs_dir / "source" / "_generated"
if _script.exists() and _xml_dir.exists():
    print("Generating Mermaid class diagrams from Doxygen XML...")
    subprocess.run(
        [
            sys.executable,
            str(_script),
            "--xml-dir",
            str(_xml_dir),
            "--output-dir",
            str(_gen_dir),
        ],
        check=False,
    )

# Auto-generate C++ API reference pages from Doxygen XML
_api_script = _docs_dir / "scripts" / "generate_api_index.py"
_api_out = _docs_dir / "source" / "cpp_api" / "api"
if _api_script.exists() and _xml_dir.exists():
    print("Generating C++ API reference pages from Doxygen XML...")
    subprocess.run(
        [
            sys.executable,
            str(_api_script),
            "--xml-dir",
            str(_xml_dir),
            "--output-dir",
            str(_api_out),
        ],
        check=False,
    )

ON_READTHEDOCS = os.environ.get("READTHEDOCS", "").lower() == "true"
PYTHON_SOURCE_DIR = _docs_dir.parent / "python"
autodoc_mock_imports = []




def _repo_url() -> str:
    """Return the GitHub repository URL used for source links."""
    repo = os.environ.get("READTHEDOCS_GIT_REPOSITORY")
    if repo:
        repo = repo.removesuffix(".git")
        if repo.startswith("git@github.com:"):
            repo = repo.replace("git@github.com:", "https://github.com/", 1)
        elif repo.startswith("https://github.com/"):
            return repo
        if repo.startswith("github.com/"):
            return f"https://{repo}"

    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return f"https://github.com/{repo}"

    try:
        remote = (
            subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                cwd=_docs_dir.parent,
                text=True,
            )
            .strip()
            .removesuffix(".git")
        )
        if remote.startswith("git@github.com:"):
            return remote.replace("git@github.com:", "https://github.com/", 1)
        if remote.startswith("https://github.com/"):
            return remote
    except Exception:
        pass

    return "https://github.com/LLNL/dftracer-utils"


def _source_ref() -> str:
    """Return the git ref used for source links."""
    for env_name in ("READTHEDOCS_GIT_COMMIT_HASH", "GITHUB_SHA"):
        value = os.environ.get(env_name)
        if value:
            return value
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_docs_dir.parent,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "develop"

REPO_URL = _repo_url()
SOURCE_REF = _source_ref()


def _pyi_target_for_extension(fullname: str) -> tuple[Path, list[str]] | None:
    """Map extension-exported objects to their public type-stub file."""
    top = fullname.split(".", 1)[0]
    utility_map = {
        "AggregatorUtility": "python/dftracer/utils/utilities/_aggregator.pyi",
        "ComparatorUtility": "python/dftracer/utils/utilities/_comparator.pyi",
        "MetadataCollectorUtility": (
            "python/dftracer/utils/utilities/_metadata_collector.pyi"
        ),
        "StatisticsQueryUtility": (
            "python/dftracer/utils/utilities/_statistics_query.pyi"
        ),
        "StatisticsAggregatorUtility": (
            "python/dftracer/utils/utilities/_statistics_aggregator.pyi"
        ),
        "ReorganizationPlannerUtility": (
            "python/dftracer/utils/utilities/_reorganization_planner.pyi"
        ),
        "ReconstructionPlannerUtility": (
            "python/dftracer/utils/utilities/_reconstruction_planner.pyi"
        ),
    }
    rel_path = utility_map.get(top, "python/dftracer/utils/dftracer_utils_ext.pyi")
    return (_docs_dir.parent / rel_path, fullname.split("."))


def _find_symbol_lines(path: Path, parts: list[str]) -> tuple[int, int] | None:
    """Find source lines for a class/function/method in a Python source or stub file."""
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return None

    node = tree
    current_body = tree.body
    for part in parts:
        match = None
        for child in current_body:
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name == part:
                    match = child
                    break
        if match is None:
            return None
        node = match
        current_body = getattr(match, "body", [])

    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", start)
    if start is None:
        return None
    return (start, end or start)


def _github_url(path: Path, lines: tuple[int, int] | None) -> str | None:
    """Build a GitHub blob URL for a repo-relative path and optional lines."""
    try:
        rel = path.resolve().relative_to(_docs_dir.parent.resolve()).as_posix()
    except Exception:
        return None
    url = f"{REPO_URL}/blob/{SOURCE_REF}/{rel}"
    if lines is not None:
        start, end = lines
        url += f"#L{start}"
        if end != start:
            url += f"-L{end}"
    return url


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Resolve Python objects to GitHub source links."""
    if domain != "py":
        return None

    module_name = info.get("module")
    fullname = info.get("fullname")
    if not module_name or not fullname:
        return None

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None

    obj = module
    for part in fullname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    obj_module = getattr(obj, "__module__", module_name)
    if obj_module == "dftracer.utils.dftracer_utils_ext":
        target = _pyi_target_for_extension(fullname)
        if target is None:
            return None
        path, parts = target
        lines = _find_symbol_lines(path, parts)
        return _github_url(path, lines)

    try:
        source_file = Path(inspect.getsourcefile(obj) or inspect.getfile(obj))
        _, start = inspect.getsourcelines(obj)
        end = start + max(len(inspect.getsource(obj).splitlines()) - 1, 0)
        return _github_url(source_file, (start, end))
    except Exception:
        return None


if ON_READTHEDOCS:
    sys.path.insert(0, str(PYTHON_SOURCE_DIR))
    install_extension_stub()
    # Mock the heavy optional deps. pandas hard-imports pyarrow (reads
    # pa.__version__), so pandas/numpy must be mocked whenever pyarrow is,
    # or importing dftracer.utils.dfanalyzer for autodoc fails.
    autodoc_mock_imports = [
        "numpy",
        "pandas",
        "pyarrow",
        "dask",
        "dask.distributed",
    ]

try:
    import dftracer.utils

    print("✓ dftracer.utils package found and imported successfully.")
except (ImportError, ModuleNotFoundError) as e:
    if not ON_READTHEDOCS and PYTHON_SOURCE_DIR.exists():
        print(f"Warning: installed dftracer.utils package not found: {e}")
        print("Falling back to source package with RTD extension stubs.")
        sys.path.insert(0, str(PYTHON_SOURCE_DIR))
        install_extension_stub()
        autodoc_mock_imports = [
            "numpy",
            "pandas",
            "pyarrow",
            "dask",
            "dask.distributed",
        ]
        import dftracer.utils
    else:
        print(f"Warning: dftracer.utils package not found: {e}")
        print("API documentation will have limited information.")
        print("To generate full API docs, install the package: pip install -e .")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "dftracer-utils"
copyright = "%Y, Ray Andrew Sinurat, Hariharan Devarajan"
author = "Ray Andrew Sinurat, Hariharan Devarajan"

# The version info for the project. Resolved automatically (git tags via
# setuptools_scm, then the generated _version.py, then installed metadata) so
# the docs never drift from a hardcoded number. The full ``release`` keeps the
# local segment, so dev builds show the distance since the last tag and the
# commit hash (e.g. "0.1.dev535+g926fc871"); ``version`` is the short "X.Y".
def _resolve_release() -> str:
    # 1. Git tags via setuptools_scm - same provider as pyproject.toml, but
    #    keeping the local (+g<hash>) segment so dev docs show the exact commit.
    try:
        from setuptools_scm import get_version

        return get_version(
            root=str(_docs_dir.parent),
            relative_to=__file__,
            version_scheme="post-release",
        )
    except Exception:
        pass
    # 2. The setuptools_scm-generated _version.py, reattaching the commit id.
    try:
        import importlib.util

        vf = _docs_dir.parent / "python" / "dftracer" / "utils" / "_version.py"
        if vf.exists():
            spec = importlib.util.spec_from_file_location("_dftracer_version", vf)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            commit = getattr(mod, "commit_id", None)
            return f"{mod.version}+{commit}" if commit else mod.version
    except Exception:
        pass
    # 3. Installed package metadata (wheel / editable install).
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("dftracer-utils")
    except Exception:
        pass
    return "0.0.0"


release = _resolve_release()

# Collapse the setuptools_scm local segment to just "+g<7-char hash>",
# dropping the dirty-tree date marker (".dYYYYMMDD") so the version stays short.
import re

release = re.sub(r"\+g([0-9a-fA-F]+).*$", lambda m: "+g" + m.group(1)[:7], release)
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",  # Disabled: manual docs in api/reader.rst and api/indexer.rst
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    # sphinx_autodoc_typehints disabled: it strips types from signatures
    # and loses C extension __text_signature__. Sphinx's built-in autodoc
    # handles both Python type hints and C extension __text_signature__.
    "myst_parser",  # For Markdown support
    "breathe",  # Always enable breathe
    "sphinx.ext.ifconfig",  # For conditional inclusion
    "sphinxcontrib.mermaid",  # Mermaid diagrams
]

# Mermaid configuration
mermaid_version = "11"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme:'neutral'});"
mermaid_d3_zoom = True

# Check if Doxygen XML output exists and set up Breathe config
doxygen_xml_path = Path(__file__).parent.parent / "doxygen" / "xml"
if doxygen_xml_path.exists():
    cpp_api_enabled = True
    # Breathe configuration for C++ documentation
    breathe_projects = {"dftracer-utils": str(doxygen_xml_path)}
    breathe_default_project = "dftracer-utils"
else:
    cpp_api_enabled = False
    print("Warning: Doxygen XML output not found. C++ API documentation will be skipped.")
    print(f"Expected path: {doxygen_xml_path}")
    print("Run 'doxygen Doxyfile' in the docs directory to generate C++ documentation.")

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add mappings for intersphinx - link to main DFTracer docs and Python docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "dftracer": ("https://dftracer.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["api/_autosummary"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Search configuration
html_search_language = "en"

# Theme options
html_theme_options = {
    "navigation_with_keys": True,
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
}

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Type annotations in both signature and description
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented"

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for autosummary -------------------------------------------------
autosummary_generate = False
