"""Sphinx configuration for auto-generating documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "Lobium: A Real-Time Predictive System for Market Microstructure Forecasting"
copyright = "2025, D. Papangelis"
author = "D. Papangelis"
release = "0.0.1"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Templates
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Napoleon settings (Google/NumPy docstring style)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
