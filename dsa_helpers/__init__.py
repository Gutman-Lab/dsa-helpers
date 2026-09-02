from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from .imread import imread
    from .imwrite import imwrite

# Version of the dsa-helpers package
__version__ = "3.2.2"

# Loaded on first access so importing other submodules does not pull in OpenCV.
__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={"imread": ["imread"], "imwrite": ["imwrite"]},
)

# To avoid slow downs, do not allow from dsa_helpers import * to import anything.
__all__ = []
