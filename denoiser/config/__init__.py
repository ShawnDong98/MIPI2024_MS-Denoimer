# Copyright (c) Facebook, Inc. and its affiliates.

from .config import CfgNode, get_cfg
from .lazy import LazyCall, LazyConfig
from .instantiate import instantiate


__all__ = [
    "CfgNode",
    "get_cfg",
    "LazyCall",
    "LazyConfig",
    "instantiate",
]