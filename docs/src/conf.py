# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import pathlib
import sys

basedir = str(pathlib.Path(__file__).parent.parent.parent.resolve())

sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------

project = "ome_arrow"
# is used here due to sphinx decision-making: https://github.com/sphinx-doc/sphinx/issues/8132
copyright = "2024, Way Science Community"  # noqa: A001
author = "Way Science Community"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "pydata_sphinx_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []  # type: ignore

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wayscience/ome-arrow",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {"text": "ome_arrow"},
    "use_edit_page_button": False,
    "show_toc_level": 1,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "secondary_sidebar_items": {
        "**/*": ["page-toc", "edit-this-page", "sourcelink"],
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# set option to avoid rendering default variables
autodoc_preserve_defaults = True

# enable anchor creation
myst_heading_anchors = 3

# disable notebook execution
nb_execution_mode = "off"
