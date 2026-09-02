from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from .train import train

# Loaded on first access so importing inference does not pull in the training stack.
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={"train": ["train"]},
)
