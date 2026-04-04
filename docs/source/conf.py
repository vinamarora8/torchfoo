import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../.."))

# Mock torch so RTD doesn't need to install it
for _mod in [
    "torch",
    "torch.nn",
    "torch.nn.parallel",
    "torch.distributed",
    "torch.cuda",
    "torch.autograd",
    "torch.multiprocessing",
]:
    sys.modules[_mod] = MagicMock()

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

autosummary_generate = ["api"]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_theme_options = {
    "show_toc_level": 2,
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
    ("torchfoo", "torchfoo", ["distributed", "dist", "ddp"]),
    ("torchfoo.distributed", "torchfoo.distributed", []),
    ("torchfoo.ddp", "torchfoo.ddp", []),
]


def _build_api_rst(app):
    import importlib
    import pathlib

    lines = ["API Reference", "=============", ""]
    for mod_name, title, exclude in _API_MODULES:
        mod = importlib.import_module(mod_name)
        members = [m for m in (getattr(mod, "__all__", []) or []) if m not in exclude]
        lines += [
            title,
            "-" * len(title),
            "",
            ".. autosummary::",
            "   :toctree: generated",
            "   :nosignatures:",
            "",
        ]
        for m in members:
            lines.append(f"   ~{mod_name}.{m}")
        lines.append("")
    pathlib.Path(app.srcdir).joinpath("api.rst").write_text("\n".join(lines))


def setup(app):
    app.connect("builder-inited", _build_api_rst, priority=100)
