import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "torchfoo"
author = "Vinam Arora"
copyright = f"2025, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_toc_level": 2,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
