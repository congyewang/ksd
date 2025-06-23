import os
import sys

# Specify the Path of the Source Code
sys.path.insert(0, os.path.abspath("../../src/"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ksd"
copyright = "2025, Congye Wang"
author = "Congye Wang"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Automatically generates documents from docstring
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstring
    "sphinx.ext.autosummary",  # Automatic generation of overviews
    "sphinx_autodoc_typehints",  # Support for type annotations
    "sphinx.ext.viewcode",  # Add links to source code
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for autodoc -----------------------------------------------------
autodoc_inherit_docstrings = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Options for HTML output -------------------------------------------------
html_theme_options = {"navigation_depth": 4, "titles_only": False}

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for linkcheck extension ----------------------------------------
linkcheck_ignore = [
    r"https://ntfy\.greenlimes\.top/.*",
    r"https://alist\.greenlimes\.top/.*",
]

# -- Options for reference handling ----------------------------------------
default_role = "any"
