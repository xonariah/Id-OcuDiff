import numpy as np
from PIL import Image

SEGMENTATION_RGB = np.array([
    [31, 119, 180],
    [214, 39, 40],
    [227, 119, 194],
    [23, 190, 207],
], dtype=np.uint8)


def tab10_label_image(
    label_mask: np.ndarray,
) -> Image.Image:

    label_mask = np.asarray(
        label_mask,
        dtype=np.uint8,
    )

    unique_values = np.unique(
        label_mask
    )

    if (
        len(unique_values) > 0
        and unique_values.max() > 3
    ):
        raise ValueError(
            "Segmentacijska maska mora vsebovati "
            f"razrede 0..3, dobila pa je {unique_values}."
        )

    rgb = SEGMENTATION_RGB[
        label_mask
    ]

    return Image.fromarray(
        rgb,
        mode="RGB",
    )
