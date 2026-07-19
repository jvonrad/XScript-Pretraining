"""YAML loading with a corrected float resolver.

PyYAML's default implicit float resolver requires a signed exponent, so plain
scientific notation like `1.0e15` or `2.0e9` is loaded as a *string*. Our
configs are full of token budgets in that form, so we install the fixed
resolver everywhere configs are read.
"""
import re

import yaml


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:
        [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
       |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
       |[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)
       |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
       |[-+]?\.(?:inf|Inf|INF)
       |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def load(path) -> dict:
    from pathlib import Path
    return yaml.load(Path(path).read_text(), Loader=_Loader)


def loads(text: str) -> dict:
    return yaml.load(text, Loader=_Loader)
