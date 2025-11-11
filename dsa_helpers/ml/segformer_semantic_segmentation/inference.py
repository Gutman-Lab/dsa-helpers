import torch
import geopandas as gpd
import histomicstk as htk
import numpy as np
import large_image_source_openslide
from PIL import Image
from time import perf_counter
from tqdm import tqdm
from multiprocessing import Pool
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from shapely import make_valid
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ...image_utils import label_mask_to_polygons
from ...gpd_utils import rdp_by_fraction_of_max_dimension, make_multi_polygons

stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

# specify stains of input image
stains = [
    "hematoxylin",  # nuclei stain
    "eosin",  # cytoplasm stain
    "null",
]  # set to null if input contains only two stains

# create stain matrix
W = np.array([stain_color_map[st] for st in stains]).T

# The standard values are taken from Emory's 40x scanned images.
STANDARD_MM_PER_PX = 0.0002519
STANDARD_MAG = 40


def inference(
    model: str | torch.nn.Module,
    wsi_fp: str,
    label_ranks: list[int] | None = None,
    batch_size: int = 64,
    tile_size: int = 512,
    mag: float | None = 10.0,
    mm_px: float | None = None,
    workers: int = 8,
    chunk_mult: int = 2,
    prefetch: int = 2,
    device: str | None = None,
    small_hole_thr: int = 50000,
    buffer: int = 10,
    fraction: float = 0.001,
    nproc: int = 20,
    interior_max_area: int = 100000,
    hematoxylin_channel: bool = False,
) -> tuple[gpd.GeoDataFrame, float, float]:
    """Inference using SegFormer semantic segmentation model on a WSI.

    Args:
        model (str | torch.nn.Module): Path to the model checkpoint or
            a pre-loaded model.
        wsi_fp (str): File path to the WSI.
        label_ranks (list[int], optional): List of int labels (as
            outputed by the model) ordered by rank with index 0 being
            the lowest rank. If None, The labels will be ranked by
            their int value.
        batch_size (int, optional): Batch size for inference. Defaults
            to 64.
        tile_size (int, optional): Tile size for inference. Defaults to
            512.
        mag (float, optional): Magnification to grab the tiles at for
            inference. This is calculated from the mm_px from the WSI
            metadata, converted to the standard mm_px to magnification
            ratio at Emory. Defaults to 10.0.
        mm_px (float, optional): Micrometers per pixel to grab the tiles
            at for inference. If not provided, will be inferred from the
            mag parameter, if that isn't provided it will use the scan
            resolution of the WSI. Defaults to None.
        workers (int, optional): Number of workers for inference.
            Defaults to 8.
        chunk_mult (int, optional): Chunk multiplier for inference.
            Defaults to 2.
        prefetch (int, optional): Number of prefetch for inference.
            Defaults to 2.
        device (str, optional): Device for inference. Default is None,
            will use "gpu" if available, otherwise "cpu".
        small_hole_thr (int, optional): Threshold in area to identify
            small objects. Defaults to 50000.
        buffer (int, optional): Buffer to add to polygons before
            dissolving. Defaults to 10.
        fraction (float, optional): Fraction of the maximum dimension
            to use for RDP. Defaults to 0.001.
        nproc (int, optional): Number of processes to use for parallel
            RDP. Defaults to 20.
        interior_max_area (int, optional): Maximum area of a hole to fill.
            Used when filling gaps created by RDP. Defaults to 100000.
        hematoxylin_channel (bool, optional): Whether to use the
            hematoxylin channel when predicting the segmentation mask.
            Defaults to False.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the predicted
            polygons and labels.
        mag (float): The magnification used for inference.
        mm_px (float): The micrometers per pixel used for inference.

    """
    # Get the tile source.
    ts = large_image_source_openslide.open(wsi_fp)

    ts_metadata = ts.getMetadata()

    # Calculate the mm per pixel to use.
    if mm_px is None and mag is None:
        print(
            "Using scan resolution, please note we use mm_x and assume mm_y is"
            "the same."
        )
        mm_px = ts_metadata["mm_x"]

        # Calculate what the standard magnification.
        mag = STANDARD_MAG * STANDARD_MM_PER_PX / mm_px
    elif mm_px is None:
        # Calculate the mm per pixel.
        mm_px = STANDARD_MAG * STANDARD_MM_PER_PX / mag
    else:
        # Calculate the magnification.
        mag = STANDARD_MAG * STANDARD_MM_PER_PX / mm_px

    # Scale factor, multiply to go from scan magnification to desired mag.
    sf_x = ts_metadata["mm_x"] / mm_px
    sf_y = ts_metadata["mm_y"] / mm_px

    # Create eager iterator.
    iterator = ts.eagerIterator(
        scale={"mm_x": mm_px, "mm_y": mm_px},
        tile_size={"width": tile_size, "height": tile_size},
        chunk_mult=chunk_mult,
        batch=batch_size,
        prefetch=prefetch,
        workers=workers,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    device = torch.device(device)

    # Load the model.
    if isinstance(model, str):
        model = SegformerForSemanticSegmentation.from_pretrained(
            model, local_files_only=True, device_map=device
        )

    model.eval()

    if label_ranks is None:
        id2label = model.config.id2label
        max_label = max([int(v) for v in id2label.keys()])
        label_ranks = list(range(max_label + 1))

    # Iterate through batches.
    batch_n = 0

    # Image processor for images.
    processor = SegformerImageProcessor()

    # Track all predicted polygons.
    wsi_polygons = []

    for batch in iterator:
        tiles = batch["tile"].view()  # images are in BCHW format

        if hematoxylin_channel:
            # Deconvolve to get images of the hematoxylin channel.
            img_list = []

            for img in imgs:
                img = (
                    htk.preprocessing.color_deconvolution.color_deconvolution(
                        img, W
                    ).Stains[:, :, 0]
                )
                img = np.stack([img, img, img], axis=-1)
                img_list.append(img)

            tiles = img_list

        # Convert the numpy arrays to PIL images.
        imgs = [Image.fromarray(img) for img in tiles]

        # Pass the images through the processor.
        inputs = processor(imgs, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Predict on the batch.
        with torch.no_grad():
            output = model(inputs["pixel_values"])
            logits = output.logits

            # Get the logits out, resizing them to the original tile size.
            logits = torch.nn.functional.interpolate(
                logits,
                size=tile_size,
                mode="bilinear",
            )

            # Get predicted class labels for each pixel.
            masks = torch.argmax(logits, dim=1).detach().cpu().numpy()

        # Top left corner of each tile, at scan magnification.
        tile_x_coords = batch["gx"]
        tile_y_coords = batch["gy"]

        for i, mask in enumerate(masks):
            x, y = tile_x_coords[i], tile_y_coords[i]

            # Convert top left point to the desired magnification.
            x_scaled = int(x * sf_x)
            y_scaled = int(y * sf_y)

            polygon_and_labels = label_mask_to_polygons(
                mask,
                x_offset=x_scaled,
                y_offset=y_scaled,
            )

            for polygon_and_label in polygon_and_labels:
                polygon, label = polygon_and_label
                label = int(label)

                # Do something with the polygon and label.
                wsi_polygons.append([polygon, label])

        batch_n += 1
        print(f"\r    Processed batch {batch_n}.    ", end="")
    print()

    # Convert polygons and labels to a GeoDataFrame.
    gdf = gpd.GeoDataFrame(wsi_polygons, columns=["geometry", "label"])

    # Add a small buffer to the polygons to make polygons from adjacent tiles
    # touch, this allows merging adjacent tile polygons when dissolving.
    gdf["geometry"] = gdf["geometry"].buffer(buffer)

    gdf = gdf.dissolve(by="label", as_index=False)

    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # Scale the geometries.
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: scale(geom, xfact=1 / sf_x, yfact=1 / sf_y, origin=(0, 0))
    )

    cleanup_pipe = SegFormerSSInferenceCleanup(
        gdf,
        label_ranks,
        small_hole_thr=small_hole_thr,
        fraction=fraction,
        nproc=nproc,
        interior_max_area=interior_max_area,
    )

    gdf = cleanup_pipe.cleanup()

    return gdf, mag, mm_px


class SegFormerSSInferenceCleanup:
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        label_ranks: list[int],
        small_hole_thr: int = 50000,
        buffer: int = 1,
        fraction: float = 0.001,
        nproc: int = 20,
        interior_max_area: int = 100000,
    ):
        """Initiate the class for cleaning up the inference output.

        Args:
            gdf (geopandas.GeoDataFrame): Input inference output.
            label_ranks (list[int]): List of labels, ordered by rank
                with index 0 being the lowest rank.
            small_hole_thr (int, optional): Threshold in area to
                identify small objects. Defaults to 50000.
            buffer (int, optional): Buffer to add to polygons before
                dissolving. Defaults to 1.
            fraction (float, optional): Fraction of the maximum
                dimension to use for RDP. Defaults to 0.001.
            nproc (int, optional): Number of processes to use for
                parallel RDP. Defaults to 20.
            interior_max_area (int, optional): Maximum area of a hole
                to fill. Used when filling gaps created by RDP.
                Defaults to 100000.

        """
        for i, r in gdf.iterrows():
            label = r["label"]

            if label not in label_ranks:
                raise ValueError(f"Label {label} not in label_ranks.")
            gdf.loc[i, "rank"] = label_ranks.index(r["label"])

        self.__version__ = "1.0.1"
        self.input_gdf = gdf
        self.small_hole_thr = small_hole_thr
        self.output_gdf = None
        self.fraction = fraction
        self.nproc = nproc
        self.interior_max_area = interior_max_area
        self.time = {}

    def _make_gpd_valid(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # Make the geometries valid in the gdf, keep only polygons.
        gdf["geometry"] = gdf["geometry"].apply(make_valid)
        gdf = gdf.explode(index_parts=False)
        gdf = gdf[
            (gdf["geometry"].geom_type == "Polygon")
            & (gdf["geometry"].is_valid)
        ]

        return gdf.reset_index(drop=True)

    def _remove_intersections(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # Remove intersections between polygons.
        gdf = gdf.reset_index(drop=True)
        n = len(gdf)

        if n in (0, 1):
            return gdf

        # Loop until the second to last row.
        for i in tqdm(
            range(n - 1), total=n - 1, desc="Removing intersections"
        ):
            r1 = gdf.iloc[i]

            # Subtract the r1 geometry from all others.
            for j in range(i + 1, n):
                r2 = gdf.iloc[j]

                # Subtract r1 from r2.
                geom = r2["geometry"].difference(r1["geometry"])

                gdf.loc[j, "geometry"] = geom

        return self._make_gpd_valid(gdf)

    def _remove_small_holes(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.reset_index(drop=True)

        n = len(gdf)

        for i in tqdm(range(n), total=n, desc="Removing small holes"):
            geom = gdf.iloc[i]["geometry"]

            exterior = geom.exterior
            interiors = geom.interiors

            new_interiors = []
            for interior in interiors:
                if Polygon(interior).area > self.small_hole_thr:
                    new_interiors.append(interior)

            geom = Polygon(exterior, new_interiors)

            gdf.loc[i, "geometry"] = geom

        return gdf

    def _remove_small_contained_polygons(
        self, gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        n = len(gdf)

        i_removed = []

        for i in tqdm(
            range(n), total=n, desc="Removing small contained polygons"
        ):
            exterior = gdf.iloc[i]["geometry"].exterior

            geom = Polygon(exterior)

            if geom.area < self.small_hole_thr:
                # Check if this is contained in another polygon.
                contained = gdf[
                    (gdf["geometry"].contains(geom)) & (gdf.index != i)
                ]

                if len(contained):
                    i_removed.append(i)

        gdf = gdf.drop(i_removed)

        return gdf

    def _remove_small_polygons_not_contained(
        self, gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        gdf = gdf.reset_index(drop=True)
        n = len(gdf)

        i_removed = []

        for i in tqdm(
            range(n), total=n, desc="Removing small polygons not contained"
        ):
            geom = gdf.iloc[i]["geometry"]

            if geom.geom_type == "MultiPolygon":
                area = geom.area
            else:
                area = Polygon(geom.exterior).area

            if area < self.small_hole_thr:
                # Check for any touching polygons.
                touching = gdf[
                    (~gdf.index.isin(i_removed + [i]))
                    & (gdf["geometry"].touches(geom))
                ].copy()
                if len(touching):
                    touching["intersection_length"] = (
                        touching["geometry"].intersection(geom).length
                    )

                    touching = touching.sort_values(
                        by="intersection_length", ascending=False
                    )

                    r = touching.iloc[0]

                    touching_geom = r["geometry"]

                    # Merge the polygons.
                    geom = geom.union(touching_geom)

                    gdf.loc[r.name, "geometry"] = geom
                    i_removed.append(i)
                else:
                    # Remove this polygon.
                    i_removed.append(i)

        gdf = gdf.drop(i_removed)
        gdf = self._make_gpd_valid(gdf)
        return gdf

    def _rdp_polygon(self, geom, idx, fraction):
        geom = rdp_by_fraction_of_max_dimension(geom, fraction=fraction)
        return geom, idx

    def _fill_rdp_gaps(
        self, gdf: gpd.GeoDataFrame, interior_max_area: int = 100000
    ) -> gpd.GeoDataFrame:
        # Fill the gaps between polygons created by RDP.
        gdf["area"] = gdf["geometry"].area

        gdf_union = gdf["geometry"].union_all()

        # Collect all the holes / interiors.
        interiors = []

        for geom in gdf_union.geoms:
            for interior in geom.interiors:
                interior = Polygon(interior)

                if interior.area < interior_max_area:
                    interiors.append(interior)

        # Loop through each hole that was small.
        for interior in tqdm(interiors, desc="Filling holes between polygons"):
            # Check polygons that are touching this interior.
            touching = gdf[gdf["geometry"].distance(interior) == 0]

            unique_ranks = touching["rank"].unique()

            if len(unique_ranks) > 1:
                # Sort by rank then area.
                touching = touching.sort_values(
                    by=["rank", "area"], ascending=False
                )

                # Get the first row.
                r = touching.iloc[0]

                # Merge the hole with the polygon.
                geom = unary_union([r["geometry"], interior])

                if geom.geom_type == "MultiPolygon":
                    # Buff the interior a bit.
                    interior_buffed = interior.buffer(1)
                    geom = unary_union([interior_buffed, geom])

                    if geom.geom_type == "MultiPolygon":
                        print(
                            "MultiPolygon after buffering and union, discarding hole."
                        )
                        continue

                gdf.loc[r.name, "geometry"] = geom
                gdf.loc[r.name, "area"] = geom.area

        return gdf

    def cleanup(self):
        """Pipeline for cleaning up the inference output."""
        print("Running inference cleanup:\n")
        time = self.time
        gdf = self.input_gdf.copy()
        gdf = self._make_gpd_valid(gdf)

        print("[1/7] Removing intersections...")
        start_time = perf_counter()
        gdf = make_multi_polygons(gdf, "label")
        gdf = self._remove_intersections(gdf)
        time["remove-intersections"] = perf_counter() - start_time

        print("[2/7] Removing small holes...")
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        gdf = self._make_gpd_valid(gdf)
        start_time = perf_counter()
        gdf = self._remove_small_holes(gdf)
        time["remove-small-holes"] = perf_counter() - start_time

        print("[3/7] Removing small polygons contained in other polygons...")
        start_time = perf_counter()
        gdf = self._remove_small_contained_polygons(gdf)
        time["remove-small-contained-polygons"] = perf_counter() - start_time

        print("[4/7] Removing small polygons not contained...")
        start_time = perf_counter()
        gdf = self._remove_small_polygons_not_contained(gdf)
        time["remove-small-polygons-not-contained"] = (
            perf_counter() - start_time
        )

        # Parallel RDP.
        print("[5/7] Reducing points in polygons via RDP...")
        start_time = perf_counter()
        with Pool(processes=self.nproc) as pool:
            jobs = [
                pool.apply_async(
                    func=self._rdp_polygon,
                    args=(r["geometry"], i, self.fraction),
                )
                for i, r in gdf.iterrows()
            ]

            n = len(gdf)

            for job in tqdm(jobs, total=n, desc="Reducing points in polygons"):
                geom, idx = job.get()
                gdf.loc[idx, "geometry"] = geom

        time["rdp"] = perf_counter() - start_time

        gdf = self._make_gpd_valid(gdf)

        print("[6/7] Removing intersections again...")
        start_time = perf_counter()
        gdf = make_multi_polygons(gdf, "label")
        gdf = self._remove_intersections(gdf)
        time["remove-intersections-2"] = perf_counter() - start_time

        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        gdf = self._make_gpd_valid(gdf)

        print("[7/7] Filling RDP gaps...")
        start_time = perf_counter()
        gdf = self._fill_rdp_gaps(gdf, self.interior_max_area)
        time["fill-rdp-gaps"] = perf_counter() - start_time
        gdf = self._make_gpd_valid(gdf)

        self.output_gdf = gdf
        self.time = time

        total_time = sum(time.values())
        print(f"\nTotal time: {total_time:.2f} seconds")
        return gdf
