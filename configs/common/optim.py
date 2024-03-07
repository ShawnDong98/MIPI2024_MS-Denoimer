from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

from denoiser.config import LazyCall as L

import torch

SGD = L(torch.optim.SGD)(
    params=[],
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)


AdamW = L(torch.optim.AdamW)(
    params=[],
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)

Adam = L(torch.optim.Adam)(
    params=[],
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)