# This file is used to configure the documentation of the package
# It is used by the Sphinx documentation generator to generate the documentation
# The documentation is generated in the docs folder of the package
# The documentation is generated in HTML and LaTeX format

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# 0. Import the necessary modules

import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

# 1. Set the project informations

def read_version():
    version_file = os.path.join(os.path.dirname(__file__), '..', '..', 'pyzernike', '__version__.py')
    with open(version_file, "r") as file:
        exec(file.read()) 
    return locals()["__version__"]

import datetime
project = "pyzernike"
project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
doc_source_directory = os.path.abspath(os.path.dirname(__file__))
copyright = f"2025-{datetime.datetime.now().year}, Artezaru"
author = "Artezaru"
release = read_version()

# 2. General configuration

import pydata_sphinx_theme
html_theme = "pydata_sphinx_theme"

extensions = [
    "sphinx.ext.autodoc", # Automatically document the code
    "sphinx.ext.viewcode", # Add links to the code
    "sphinx.ext.napoleon", # Support for Google and Numpy docstring formats
    "sphinx.ext.githubpages", # Publish the documentation on GitHub
    "sphinx.ext.autosummary", # Generate summaries of the modules
    "sphinx_design",  # Add design elements to the documentation
    "sphinx_gallery.gen_gallery",  # Generate a gallery of examples
    "sphinx.ext.mathjax",  # Render math in the documentation
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

sphinx_gallery_conf = {
    "examples_dirs": [
        os.path.join(project_directory, "gallery"),
    ],
    "gallery_dirs": [
        os.path.join(doc_source_directory, "_gallery"),
    ],
    "backreferences_dir": os.path.join(doc_source_directory, "_gallery_backreferences"),
    "capture_repr": ["__repr__", "__str__", "_repr_html_"],
    "image_scrapers": ("matplotlib",),
    "filename_pattern": r"\.py$",
}
