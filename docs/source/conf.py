# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root directory to the system path
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Parametric Matrix Models"
copyright = "2025, Patrick Cook"
author = "Patrick Cook"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_title = "Parametric Matrix Models"

templates_path = ["_templates"]
exclude_patterns = []

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.githubpages",
]

autodoc_typehints = "signature"
napoleon_use_rtype = False

autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "private-members": True,
    "special-members": "__init__",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "dill": ("https://dill.readthedocs.io/en/latest/", None),
}

add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_context = {"default_mode": "auto"}
html_static_path = ["_static"]

html_show_copyright = False
html_show_sphinx = False

html_favicon = "_static/favicon.ico"

html_theme_options = {
    "logo": {
        "text": "PMM",
        "image_light": "_static/pmmlogo.svg",
        "image_dark": "_static/pmmlogo.svg",
    },
    "collapse_navigation": False,
    "show_version_warning_banner": True,
    "github_url": "https://github.com/Parametric-Matrix-Models/pyPMM",
    "show_toc_level": 2,
    "navigation_depth": 4,
    "footer_start": [],
    "footer_end": [],
}
