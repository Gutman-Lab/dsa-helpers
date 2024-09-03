from girder_client import GirderClient
import pandas as pd
from pathlib import Path

from ...girder_utils import get_region, get_item_large_image_metadata
from ... import imwrite


def get_pearce_roi_images(
    gc: GirderClient,
    item_id: str,
    class_map: dict[str, int],
    save_dir: str = ".",
    mag: float | None = None,
) -> pd.DataFrame:
    """Download the smallestl bounding box around an ROI annotations and its
    accompanying object detection annotations. The format used to determine ROIs
    and boxes is based on the convention used by Dr. Thomas Pearce in his
    OpenSeadragon applications. Labels are saved as text files in the YOLO format.

    Args:
        gc (girder_client.GirderClient): The Girder client to use.
        item_id (str): The item ID of the WSI.
        class_map (dict[str, int]): A dictionary mapping class names to integers.
        save_dir (str, optional): The directory to save the images and labels.
            Defaults to ".".
        mag (float | None, optional): The magnification to download the ROI at.
            If None, the scan magnification is used. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing metadata for each ROI

    """
    # Create directories to save images and labels.
    img_dir = Path(save_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    label_dir = Path(save_dir) / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    # Get all the annotation documents with name starting with "ROI".
    ann_docs_ids = []

    for ann_doc in gc.get(
        f"annotation?itemId={item_id}&limit=0&offset=0&sort=lowerName&sortdir=1"
    ):
        if ann_doc.get("annotation", {}).get("name", "").startswith("ROI"):
            ann_docs_ids.append(ann_doc["_id"])

    if not len(ann_docs_ids):
        # No annotation documents, return None.
        return

    # Track metadata for each ROI.
    roi_metadata = []

    # Loop through each document.
    for ann_doc_id in ann_docs_ids:
        # Get the full annotation document.
        ann_doc = gc.get(f"annotation/{ann_doc_id}")

        # Separate the ROI element and the other elements.
        roi_el = None
        other_els = []

        for element in ann_doc.get("annotation", {}).get("elements", []):
            if element.get("user", {}).get("role") == "ROI":
                roi_el = element
            else:
                other_els.append(element)

        if roi_el is None:
            raise Exception("No ROI element found in annotation document.")

        # Download the ROI image.
        xc, yc = roi_el["center"][:2]
        roi_w, roi_h = int(roi_el["width"]), int(roi_el["height"])

        # Get the top left coordinate of the ROI.
        roi_left = int(xc - roi_w / 2)
        roi_top = int(yc - roi_h / 2)

        # Get scale factor to go from scan magnification to mag.
        scan_mag = get_item_large_image_metadata(gc, item_id)["magnification"]

        if mag is None:
            mag = scan_mag

        roi_img = get_region(
            gc, item_id, left=roi_left, top=roi_top, width=roi_w, height=roi_h, mag=mag
        )

        fn = f"itemId-{item_id}-x{roi_left}y{roi_top}x{roi_left+roi_w}y{roi_top+roi_h}"

        img_fp = img_dir / f"{fn}.png"
        imwrite(img_fp, roi_img)

        # Convert the boxes to YOLO format.
        labels = ""

        for element in other_els:
            el_class = element["user"]["class"]

            if el_class not in class_map:
                continue

            xc, yc = element["center"][:2]
            xc, yc = int(xc), int(yc)

            # Subtract the left and top of the ROI to get the correct coordinates.
            xc = xc - roi_left
            yc = yc - roi_top

            # Normalize by the ROI width and height.
            xc /= roi_w
            yc /= roi_h

            w = element["width"] / roi_w
            h = element["height"] / roi_h

            labels += f"{class_map[el_class]} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}\n"

        # Save the label to a text file.
        label_fp = label_dir / f"{fn}.txt"

        with open(label_fp, "w") as fh:
            fh.write(labels.strip())

        # Append all the metadata to the list.
        roi_metadata.append(
            [item_id, img_fp, roi_left, roi_top, roi_w, roi_h, mag, scan_mag]
        )

    return pd.DataFrame(
        roi_metadata, columns=["item_id", "fp", "x", "y", "w", "h", "mag", "scan_mag"]
    )
