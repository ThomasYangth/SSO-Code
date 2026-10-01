"""Filesystem locations.

Two roots, both overridable by environment variables:

``SSO_OUTPUT``  where compute drivers write raw results (large NPZ files,
                eigenvectors, per-seed samples).  Default: ``<repo>/output``.
                On a cluster point this at scratch storage.
``SSO_DATA``    the slim, version-controlled figure data shipped with the
                repository.  Default: ``<repo>/data``.

Figures are written to ``<repo>/figures`` (override with ``SSO_FIGURES``).
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def output_dir(*sub):
    """Raw compute output directory ``$SSO_OUTPUT/<sub...>`` (created)."""
    base = os.environ.get("SSO_OUTPUT", os.path.join(REPO_ROOT, "output"))
    return _ensure(os.path.join(base, *sub))


def data_dir(*sub):
    """Shipped figure-data directory ``$SSO_DATA/<sub...>``."""
    base = os.environ.get("SSO_DATA", os.path.join(REPO_ROOT, "data"))
    return os.path.join(base, *sub)


def figure_path(name):
    """Output path for a rendered figure (directory created)."""
    base = os.environ.get("SSO_FIGURES", os.path.join(REPO_ROOT, "figures"))
    return os.path.join(_ensure(base), name)
