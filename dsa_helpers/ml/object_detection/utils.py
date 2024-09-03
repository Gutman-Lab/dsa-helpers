import numpy as np
from shapely.geometry import Polygon


def convert_box_type(box):
    """Convert a box type from YOLO format (x-center, y-center, box-width, box-height) to (x1, y1, x2, y2) where point 1 is the
    top left corner of box and point 2 is the bottom right corner

    INPUT
    -----
    box : array
        [N, 4], each row a point and the format being (x-center, y-center, box-width, box-height)

    RETURN
    ------
    new_box : array
        [N, 4] each row a point and the format x1, y1, x2, y2

    """
    # get half the box height and width
    half_bw = box[:, 2] / 2
    half_bh = box[:, 3] / 2

    new_box = np.zeros(box.shape, dtype=box.dtype)
    new_box[:, 0] = box[:, 0] - half_bw
    new_box[:, 1] = box[:, 1] - half_bh
    new_box[:, 2] = box[:, 0] + half_bw
    new_box[:, 3] = box[:, 1] + half_bh

    return new_box


def read_yolo_label(filepath, im_shape=None, shift=None, convert=False):
    """Read a yolo label text file. It may contain a confidence value for the labels or not, will handle both cases

    INPUTS
    ------
    filepath : str
        the path of the text file
    im_shape : tuple or int (default: None)
        image width and height corresponding to the label, if an int it is assumed both are the same. Will scale coordinates
        to int values instead of normalized if given
    shift : tuple or int (default: None)
        shift value in the x and y direction, if int it is assumed to be the same in both. These values will be subtracted and applied
        after scaling if needed
    convert : bool (default: False)
        If True, convert the output boxes from yolo format (label, x-center, y-center, width, height, conf) to (label, x1, y1, x2, y2, conf)
        where point 1 is the top left corner of box and point 2 is the bottom corner of box

    RETURN
    ------
    coords : array
        coordinates array, [N, 4 or 5] depending if confidence was in input file

    """
    coords = []

    with open(filepath, "r") as fh:
        for line in fh.readlines():
            if len(line):
                coords.append([float(ln) for ln in line.strip().split(" ")])

    coords = np.array(coords)

    # scale coords if needed
    if im_shape is not None:
        if isinstance(im_shape, int):
            w, h = im_shape, im_shape
        else:
            w, h = im_shape[:2]

        coords[:, 1] *= w
        coords[:, 3] *= w
        coords[:, 2] *= h
        coords[:, 4] *= h

    # shift coords
    if shift is not None:
        if isinstance(shift, int):
            x_shift, y_shift = shift, shift
        else:
            x_shift, y_shift = shift[:2]

        coords[:, 1] -= x_shift
        coords[:, 2] -= y_shift

    if convert:
        coords[:, 1:5] = convert_box_type(coords[:, 1:5])

    return coords


def corners_to_polygon(x1: int, y1: int, x2: int, y2: int) -> Polygon:
    """Return a Polygon from shapely with the box coordinates given the top left and bottom right corners of a
    rectangle (can be rotated).

    Args:
        x1, y1, x2, y2: Coordinates of the top left corner (point 1) and the bottom right corner (point 2) of a box.

    Returns:
        Shapely polygon object of the box.

    """
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
