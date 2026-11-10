# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.5] - 2026-11-10

### Fixed

- `SegFormerSSInferenceCleanup` no longer silently drops MultiPolygons
  nested in GeometryCollections returned by `shapely.make_valid`, which
  could delete large tissue regions from inference output. Annotations
  produced with earlier versions may be missing regions and should be
  re-run. The cleanup class version is now `1.0.3`.

### Changed

- `gpd_utils.make_gpd_valid` no longer modifies the GeoDataFrame passed
  to it, handles polygons nested at any depth, and no longer raises
  `ValueError` when MultiPolygons remain after exploding.
