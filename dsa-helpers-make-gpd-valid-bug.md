# Bug: `SegFormerSSInferenceCleanup._make_gpd_valid` drops whole MultiPolygons

- **Package:** `dsa_helpers` ([Gutman-Lab/dsa-python-utils](https://github.com/Gutman-Lab/dsa-python-utils))
- **Affected version:** 3.2.4 (cleanup class `__version__ = "1.0.2"`)
- **File:** `dsa_helpers/ml/segformer_semantic_segmentation/inference.py`
- **Function:** `SegFormerSSInferenceCleanup._make_gpd_valid`
- **Impact:** SegFormer inference cleanup can silently delete large tissue
  regions. It raises no error and prints no warning.

## Symptom

On `case10_Syn_frontal-cortex.svs` with the `5x-rgb` split 1 model, most
of the gray matter is missing from the posted annotation. The model
predicts it correctly. The cleanup deletes it.

Gray matter area (mm²) after each cleanup step:

| step | gray matter |
|---|---|
| cleanup input | 283.77 |
| 1 remove intersections | 283.77 |
| 2 remove small holes | 283.78 |
| 3 remove small contained polygons | 283.77 |
| 4 remove small polygons not contained | 283.77 |
| 5a RDP | 283.81 |
| **5b `_make_gpd_valid` after RDP** | **44.14** |
| 7 final output | 44.16 |

The other labels change by less than 2%.

## Root cause

The current implementation:

```python
def _make_gpd_valid(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Make the geometries valid in the gdf, keep only polygons.
    gdf["geometry"] = gdf["geometry"].apply(shapely.make_valid)
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[
        (gdf["geometry"].geom_type == "Polygon")
        & (gdf["geometry"].is_valid)
    ]

    return gdf.reset_index(drop=True)
```

`shapely.make_valid` can return a `GeometryCollection` that contains a
`MultiPolygon`, for example `GeometryCollection[MultiPolygon, LineString]`.
`GeoDataFrame.explode` only splits one level, so the result is one
`MultiPolygon` row and one `LineString` row. The `geom_type == "Polygon"`
filter then drops **both rows**, including every polygon in the
`MultiPolygon`.

How the invalid polygon comes about on the real slide:

1. After step 4, the main gray matter polygon (about 240 mm², 23 holes)
   is valid. One tiny hole (152 px²) touches the exterior at a single
   pinch point, which is allowed.
2. Step 5 (`gpd_utils.rdp_by_fraction_of_max_dimension`) simplifies the
   exterior and each hole separately, each with its own epsilon
   (`fraction` × the ring's largest dimension). The exterior's epsilon
   is large because it comes from the size of the whole cortex ribbon, so
   RDP removes the exterior's small notch toward the hole. The simplified
   exterior then cuts straight through the hole, and the polygon becomes
   invalid (`Self-intersection`).
3. `make_valid` returns `GeometryCollection[MultiPolygon, LineString]`.
   The whole gray matter MultiPolygon is dropped.

`_make_gpd_valid` is called 8 times in `cleanup()`, plus once at the end
of `_remove_intersections` and once at the end of
`_remove_small_polygons_not_contained`. Any of these calls can hit the
bug. The call right after RDP is the most likely to, because RDP is the
step that creates invalid geometries.

`dsa_helpers.gpd_utils.make_gpd_valid`, the public function, already
handles this case: it keeps `Polygon` and `MultiPolygon` rows after the
first explode and then explodes again. Only the private copy in the
cleanup class has the bug.

## Minimal reproduction

```python
import geopandas as gpd
import shapely
from shapely.geometry import Polygon

from dsa_helpers.ml.segformer_semantic_segmentation.inference import (
    SegFormerSSInferenceCleanup,
)

# Exterior pinches into two lobes; a hole lies on the exterior edge.
p = Polygon(
    [(0, 0), (100, 0), (100, 100), (50, 100), (50, 110), (60, 120),
     (40, 120), (50, 110), (50, 100), (0, 100)],
    [[(0, 40), (10, 40), (10, 60), (0, 60)]],
)
v = shapely.make_valid(p)
print(v.geom_type, [g.geom_type for g in v.geoms])
# GeometryCollection ['MultiPolygon', 'MultiLineString']
print(v.area)  # 10100.0

gdf = gpd.GeoDataFrame({"label": [1]}, geometry=[p])
cleanup = SegFormerSSInferenceCleanup(gdf.copy(), [1])
out = cleanup._make_gpd_valid(gdf.copy())
print(len(out), out.area.sum())  # 0 0.0   <-- all 10100 area lost
```

## Fix

Pull out every `Polygon` from the `make_valid` output, at any nesting
depth, before filtering. Suggested implementation, which I tested and
which raises no geopandas warnings:

```python
def _polygon_parts(geom) -> list:
    """Return every Polygon in a geometry, at any nesting depth."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _polygon_parts(g)]
    return []


class SegFormerSSInferenceCleanup:
    ...

    def _make_gpd_valid(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # make_valid can return a GeometryCollection holding a MultiPolygon,
        # which a single explode does not reach.
        gdf = gdf.copy()
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: shapely.geometry.MultiPolygon(
                _polygon_parts(shapely.make_valid(geom))
            )
        )
        gdf = gdf.explode(index_parts=False)
        gdf = gdf[
            gdf["geometry"].notna()
            & ~gdf["geometry"].is_empty
            & (gdf["geometry"].geom_type == "Polygon")
            & gdf["geometry"].is_valid
        ]

        return gdf.reset_index(drop=True)
```

Notes for the implementer:

- `_polygon_parts` could live in `dsa_helpers/gpd_utils.py` and be used
  by both `SegFormerSSInferenceCleanup._make_gpd_valid` and
  `gpd_utils.make_gpd_valid`. The second one currently explodes exactly
  twice and raises `ValueError` if a `MultiPolygon` remains. Using
  `_polygon_parts` there would make both behave the same and handle any
  depth.
- `gdf.copy()` is new. The current method writes the new geometry column
  into the caller's GeoDataFrame before exploding. No caller relies on
  that, but keep the copy so the method has no side effects.
- Other columns, including `label` and `rank`, are carried through
  `explode` unchanged.
- Do **not** change RDP to fix this. RDP producing invalid geometry is
  expected; `make_valid` is there to repair it. The bug is only in which
  repaired parts are kept.

## Regression tests

```python
import geopandas as gpd
from shapely.geometry import Polygon

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
```

All three tests fail with the current code and pass with the fix above.
The first two fail because label 1 comes back with no rows. The third
fails because the current method overwrites the caller's geometry
column.

## Verification on real data

Results from the `ihc-tissue-detection` project, model `5x-rgb` split 1,
running the real `inference()`:

- On `case10_Syn_frontal-cortex.svs`, gray matter is 44.16 mm² with the
  current code and 283.94 mm² with the fix (283.77 mm² goes into
  cleanup). All output geometries are valid `Polygon`s. With the fix,
  superficial and artifact shrink slightly (17.02 to 16.83 mm² and 13.97
  to 13.93 mm²), because the intersection-removal and gap-filling steps
  now assign their shared borders to gray matter.
- I compared the current and fixed cleanup on all 40 Emory IHC slides
  (TDP-43, aBeta, Syn, Tau). Only `case10_Syn_frontal-cortex` changes by
  more than 0.1%. On the other 39 slides, `_make_gpd_valid` drops less
  than 1 px² of polygon area at every call.

The cached cleanup input for case10 is
`/nashome/jvizcar/data/ihc-tissue-detection/debug-missing-gm/case10_syn_frontal_5x-rgb_split1.pkl`.
It is a pickle with keys `cleanup_input` (a GeoDataFrame), `cleanup_kwargs`,
and `final_gdf`, and can be used as an end-to-end check:

```python
import pickle

from dsa_helpers.ml.segformer_semantic_segmentation.inference import (
    SegFormerSSInferenceCleanup,
)

LABEL_RANKS = [4, 3, 2, 1, 5]
d = pickle.load(open(".../case10_syn_frontal_5x-rgb_split1.pkl", "rb"))
out = SegFormerSSInferenceCleanup(
    d["cleanup_input"].copy(), LABEL_RANKS, **d["cleanup_kwargs"]
).cleanup()
gm = out[out["label"] == 1].area.sum()
assert gm > 4.4e9, gm  # base px²; 6.96e8 with the bug, 4.47e9 fixed
```

A full walkthrough with figures is in
`ihc-tissue-detection/notebooks/debug-missing-gray-matter.ipynb`.

## Release checklist

1. Apply the fix to `SegFormerSSInferenceCleanup._make_gpd_valid`.
   Optionally move `_polygon_parts` into `gpd_utils` and use it in
   `gpd_utils.make_gpd_valid` as well.
2. Add the regression tests above.
3. Bump the cleanup class version in `__init__`:
   `self.__version__ = "1.0.2"` → `"1.0.3"`.
4. Bump the package version (`dsa_helpers/__init__.py`
   `__version__ = "3.2.4"` and the version in `pyproject.toml`) to 3.2.5.
5. Add a changelog entry, for example: "Fix inference cleanup silently
   dropping MultiPolygons nested in GeometryCollections returned by
   `make_valid`, which could delete large tissue regions."

## Downstream follow-up (not part of `dsa_helpers`)

- Annotations posted to DSA with the buggy cleanup need to be re-run.
  At least `5x-rgb_1` on `case10_Syn_frontal-cortex.svs` is affected.
  Other model and split combinations have not been checked. To check one
  in `ihc-tissue-detection`, run
  `debug-missing-gm/scan.py <model> <split> <shard> <n_shards>` once per
  GPU (each shard uses the GPU with its shard number), then
  `debug-missing-gm/compare.py <model>_<split>`.
- `ihc-tissue-detection/utils/tiling.py` (around lines 543–546) has the
  same pattern on ground-truth annotations (`make_valid` → `explode` →
  `geom_type == "Polygon"`) and needs the same fix.
