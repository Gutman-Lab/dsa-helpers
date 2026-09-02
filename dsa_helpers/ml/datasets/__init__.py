from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from .SegFormerSegmentationDataset import SegFormerSegmentationDataset

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "SegFormerSegmentationDataset": ["SegFormerSegmentationDataset"],
    },
)

__all__ = []
