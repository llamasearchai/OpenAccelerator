# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import os
import sys
import typing

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "OpenAccelerator"
copyright = "2024, Nik Jois"
author = "Nik Jois"
version = "1.0.2"
release = "1.0.2"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.doctest",
    # Third-party extensions (carefully selected to avoid conflicts)
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
    "sphinx_click",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for autodoc ----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": True,
}

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "transformers",
    "openai",
    "dspy",
    "pydicom",
    "SimpleITK",
    "nibabel",
    "vtk",
    "itk",
    "cv2",
    "medpy",
    "dicom2nifti",
    "pynetdicom",
    "gdcm",
    "deid",
    "anonymizedf",
    "uvicorn",
    "fastapi",
    "starlette",
    "websockets",
    "httpx",
    "tiktoken",
    "dspy-ai",
    "scikit-learn",
    "typer",
    "rich",
    "click",
    "pydantic",
    "pydantic-settings",
    "pillow",
    "h5py",
    "zarr",
    "structlog",
    "prometheus-client",
    "python-json-logger",
    "cryptography",
    "bcrypt",
    "python-jose",
    "passlib",
    "line-profiler",
    "memory-profiler",
    "py-spy",
    "jupyter",
    "ipywidgets",
    "jupyter-book",
]

# Generate autosummary pages
autosummary_generate = True
autosummary_generate_overwrite = True

# -- Options for Napoleon (Google/NumPy docstring support) ------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
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

# -- Options for intersphinx ------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- Options for MyST parser ------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_url_schemes = ("http", "https", "mailto")
myst_heading_anchors = 3

# -- Options for HTML output ------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_context = {
    "display_github": True,
    "github_user": "nikjois",
    "github_repo": "OpenAccelerator",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    "**": [
        "versions.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "OpenAcceleratordoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": "",
    # Latex figure (float) alignment
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "OpenAccelerator.tex",
        "OpenAccelerator Documentation",
        "Nik Jois",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "openaccelerator",
        "OpenAccelerator Documentation",
        [author],
        1,
    )
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "OpenAccelerator",
        "OpenAccelerator Documentation",
        author,
        "OpenAccelerator",
        "Advanced ML Accelerator Simulator for Medical AI Applications",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
epub_identifier = "https://github.com/llamasearch/OpenAccelerator"

# A unique identification for the text.
epub_uid = "OpenAccelerator"

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for coverage extension ------------------------------------------

coverage_show_missing_items = True
# -- Custom Variables --------------------------------------------------------

# Define a variable to be used in the documentation
rst_epilog = """
.. |author| replace:: LlamaSearch AI Research
"""

# -- Coverage Settings -------------------------------------------------------

# Used by the `sphinx.ext.coverage` extension
coverage_write_headline: bool = True
coverage_skip_undoc_in_module: bool = False
coverage_ignore_modules: typing.List[str] = []
coverage_ignore_functions: typing.List[str] = []
coverage_ignore_classes: typing.List[str] = []

# -- Options for sphinx_click extension --------------------------------------

click_module_name = "open_accelerator.cli"
click_command_name = "main"

# -- Options for sphinx_copybutton extension --------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# -- Options for sphinx_design extension ------------------------------------

sd_fontawesome_latex = True
