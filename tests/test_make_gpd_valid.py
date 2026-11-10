import geopandas as gpd
from shapely.geometry import Polygon

from dsa_helpers import gpd_utils
from dsa_helpers.ml.segformer_semantic_segmentation.inference import (
    SegFormerSSInferenceCleanup,
)

PINCHED_WITH_EDGE_HOLE = Polygon(  # make_valid -> GC[MultiPolygon, MultiLineString]
    [(0, 0), (100, 0), (100, 100), (50, 100), (50, 110), (60, 120),
     (40, 120), (50, 110), (50, 100), (0, 100)],
    [[(0, 40), (10, 40), (10, 60), (0, 60)]],
)
EDGE_HOLE = Polygon(  # make_valid -> GC[Polygon, LineString]
    [(0, 0), (100, 0), (100, 100), (0, 100)],
    [[(0, 40), (10, 40), (10, 60), (0, 60)]],
)
COLLAPSED = Polygon([(0, 0), (10, 0), (10, 0), (0, 0)])  # no polygon area
VALID = Polygon([(200, 0), (300, 0), (300, 100), (200, 100)])


def _run(geoms, labels):
    gdf = gpd.GeoDataFrame({"label": labels}, geometry=geoms)
    cleanup = SegFormerSSInferenceCleanup(gdf.copy(), sorted(set(labels)))
    return cleanup._make_gpd_valid(gdf.copy())


def test_keeps_multipolygon_nested_in_geometry_collection():
    out = _run([PINCHED_WITH_EDGE_HOLE], [1])
    assert len(out) == 2
    assert out.area.sum() == 10100.0
    assert set(out.geom_type) == {"Polygon"}
    assert out.is_valid.all()


def test_mixed_rows():
    out = _run([PINCHED_WITH_EDGE_HOLE, EDGE_HOLE, COLLAPSED, VALID], [1, 2, 3, 4])
    areas = out.groupby("label").geometry.apply(lambda s: s.area.sum()).to_dict()
    assert areas == {1: 10100.0, 2: 10000.0, 4: 10000.0}
    assert set(out.geom_type) == {"Polygon"}


def test_does_not_mutate_input():
    gdf = gpd.GeoDataFrame({"label": [1]}, geometry=[PINCHED_WITH_EDGE_HOLE])
    cleanup = SegFormerSSInferenceCleanup(gdf.copy(), [1])
    cleanup._make_gpd_valid(gdf)
    assert gdf.geometry.iloc[0].equals(PINCHED_WITH_EDGE_HOLE)


def test_gpd_utils_make_gpd_valid_mixed_rows():
    gdf = gpd.GeoDataFrame(
        {"label": [1, 2, 3, 4]},
        geometry=[PINCHED_WITH_EDGE_HOLE, EDGE_HOLE, COLLAPSED, VALID],
    )
    out = gpd_utils.make_gpd_valid(gdf)
    areas = out.groupby("label").geometry.apply(lambda s: s.area.sum()).to_dict()
    assert areas == {1: 10100.0, 2: 10000.0, 4: 10000.0}
    assert set(out.geom_type) == {"Polygon"}
    assert out.is_valid.all()
    assert gdf.geometry.iloc[0].equals(PINCHED_WITH_EDGE_HOLE)
