"""Minimal ``--key=value`` command-line parsing used by all driver scripts."""

import sys


def parse_argv(argv=None):
    """Parse ``--key=value`` / ``--flag`` tokens into a dict of strings/True."""
    argv = sys.argv if argv is None else argv
    args = {}
    for arg in argv[1:]:
        if arg.startswith("--"):
            if "=" in arg:
                k, v = arg[2:].split("=", 1)
                args[k] = v
            else:
                args[arg[2:]] = True
    return args


def find_value(d, name, type_cast=str, default=None):
    """Fetch ``d[name]`` cast to ``type_cast`` (bool accepts true/1/yes)."""
    if name not in d:
        return default
    v = d[name]
    if type_cast is bool:
        return v.lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v)
    return type_cast(v)


def fprint(*args):
    """print(..., flush=True) — keeps slurm logs current."""
    print(*args, flush=True)
