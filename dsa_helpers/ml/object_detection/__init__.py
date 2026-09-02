from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from .get_pearce_roi_images import get_pearce_roi_images
    from .tile_image import tile_image

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "get_pearce_roi_images": ["get_pearce_roi_images"],
        "tile_image": ["tile_image"],
    },
)

__all__ = []
