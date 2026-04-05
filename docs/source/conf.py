import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import torchfoo

project = "torchfoo"
author = "Vinam Arora"
copyright = f"2026, {author}"
version = torchfoo.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autosummary_generate = ["generated/api"]
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

html_title = "torchfoo docs"
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
_DOCS_BASE_URL = "https://torchfoo.readthedocs.io"
if os.environ.get("READTHEDOCS"):
    _SWITCHER_JSON_URL = f"{_DOCS_BASE_URL}/en/latest/switcher.json"
else:
    _SWITCHER_JSON_URL = "switcher.json"
html_theme_options = {
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": _SWITCHER_JSON_URL,
        "version_match": os.environ.get("READTHEDOCS_VERSION", version),
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/vinamarora8/torchfoo",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/torchfoo/",
            "icon": "fa-brands fa-python",
        },
    ],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# Modules to include in API reference.
# Each entry: (module_dotted_name, section_title, members_to_exclude)
_API_MODULES = [
    ("torchfoo", "torchfoo", ["distributed", "dist", "module"]),
    ("torchfoo.distributed", "torchfoo.distributed", []),
    ("torchfoo.module", "torchfoo.module", []),
]


def _build_api_rst(app):
    import importlib
    import pathlib

    src = pathlib.Path(app.srcdir)
    generated = src / "generated"
    generated.mkdir(exist_ok=True)
    lines = ["API Reference", "=============", ""]
    for mod_name, title, exclude in _API_MODULES:
        mod = importlib.import_module(mod_name)
        members = [m for m in (getattr(mod, "__all__", []) or []) if m not in exclude]
        lines += [
            title,
            "-" * len(title),
            "",
            ".. autosummary::",
            "   :toctree: .",
            "   :nosignatures:",
            f"   :caption: {title}",
            "",
        ]
        for m in members:
            lines.append(f"   ~{mod_name}.{m}")
        lines.append("")
    generated.joinpath("api.rst").write_text("\n".join(lines))


def _build_switcher_json(app, exception):
    """Generate switcher.json from semver git tags."""
    if exception:
        return

    import json
    import pathlib
    import re
    import subprocess

    tags = subprocess.check_output(
        ["git", "tag"], text=True
    ).splitlines()
    semver_re = re.compile(r"^(v?\d+\.\d+\.\d+.*)$")
    versions = []
    for tag in tags:
        m = semver_re.match(tag.strip())
        if m:
            versions.append(m.group(1))

    versions.sort(key=lambda v: [int(x) for x in re.findall(r"\d+", v)], reverse=True)

    entries = [
        {
            "name": "latest (main)",
            "version": "latest",
            "url": f"{_DOCS_BASE_URL}/en/latest/",
        },
    ]
    for v in versions:
        entries.append(
            {
                "name": v,
                "version": v,
                "url": f"{_DOCS_BASE_URL}/en/{v}/",
            }
        )

    out = pathlib.Path(app.outdir) / "switcher.json"
    out.write_text(json.dumps(entries, indent=2) + "\n")


def setup(app):
    app.connect("builder-inited", _build_api_rst, priority=100)
    app.connect("build-finished", _build_switcher_json)
