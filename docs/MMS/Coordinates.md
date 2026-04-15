## Coordinate System & Tile Export

### Overview

The offline viewer supports georeferenced PLY export in multiple coordinate systems, with optional tile-based splitting compatible with major public LiDAR datasets. All coordinate features require a valid `gnss_datum.json` in the dump — controls are greyed out without it.

---

### Coordinates Menu

#### Export Coordinate System

Selects the projected coordinate system for PLY export. Three options:

**UTM WGS84** — Universal Transverse Mercator on the WGS84 ellipsoid. Zone is auto-detected from the datum longitude and locked for the session. When "Consider zones on export" is enabled, each point is individually checked and reprojected to its correct zone if it differs from the datum zone. Coordinates exported as `property double x/y/z` (absolute UTM, no precision loss).

**JGD2011** — Japan Plane Rectangular Coordinate System (日本測地系2011 平面直角座標系), 19 zones (I–XIX). Zone is auto-detected by point-in-polygon test of the datum against prefecture boundaries (loaded from `EGM_tables/japan_prefectures.geojson`). Prefecture and zone are shown in the menu (e.g. "Shizuoka — Zone VIII"). Manual override available via the Prefecture submenu. Uses Approach A reprojection: per-point inverse UTM → lat/lon → JGD2011 TM forward, giving 0.02 mm precision. WGS84 and JGD2011 share the GRS80 ellipsoid so no datum shift is needed.

**Custom** — Reserved for future implementation.

#### Consider zones on export ✓

When enabled (default: on), applies per-point zone correction during export:
- **UTM WGS84**: each point's longitude determines its correct UTM zone. Points in a different zone than the datum are reprojected via inverse UTM → lat/lon → forward UTM in the correct zone. A `utm_zone` integer property is added to the PLY.
- **JGD2011 with Trim by tile**: each tile's centroid is tested against prefecture boundaries to determine the correct JGD2011 zone for that tile. All points in the tile are projected using that zone. This handles long trajectories crossing prefecture boundaries (e.g. highway mapping).

#### Tile Presets

Selects a predefined tile grid for export splitting. Four options:

**PNOA Spain (1×1 km, 3ª Cobertura 2022–2025)** — 1×1 km tiles aligned to the UTM ETRS89 grid. Naming convention: `PNOA_MMS_EEE_NNNN.ply` where EEE = SW corner easting in km, NNNN = SW corner northing in km (e.g. `PNOA_MMS_422_4594.ply`). Compatible with the IGN/CNIG PNOA LiDAR 3rd coverage distribution.

**ICGC Cataluña (1×1 km)** — 1×1 km tiles aligned to the UTM ETRS89 grid. Naming convention: `EEENNN.ply` where EEE = easting in km, NNN = northing in km minus 4,000,000 (e.g. easting 422000, northing 4594000 → `422594.ply`). Compatible with the ICGC LiDAR territorial dataset distribution.

**Japan (JGD2011)** — 300 m (N–S) × 400 m (E–W) tiles using the nationwide 国土基本図図郭500 standard (map information level 500), defined in 公共測量標準図式 by the Geospatial Information Authority of Japan. This grid is used by all Japanese prefectures publishing open LiDAR data. Naming convention: 8-character mesh code `PPXXYYZZ.ply`:
- `PP`: JGD2011 zone number zero-padded (e.g. `08` = Zone VIII)
- `XX`: Level 50000 block (30 km × 40 km) — 2 letters A–T (row, south→north) and A–H (column, west→east)
- `YY`: Level 5000 subdivision (3 km × 4 km) — 2 digits 00–99
- `ZZ`: Level 500 subdivision (300 m × 400 m) — 2 digits 00–99

Example: `08NC3558.ply` — Zone VIII (Shizuoka), compatible with Virtual Shizuoka MMS dataset filenames. With "Consider zones on export" enabled, tiles crossing prefecture boundaries automatically get the correct zone prefix.

**Custom tile grid (SHP in target coords)** — Reserved for future implementation.

#### Settings

- **Default tile size** — tile size in km used when no preset is active (default: 2.0 km).
- **Reset to defaults** — resets tile size to 2.0 km.

---

### Trim by tile (Save dialog)

When checked, the PLY export is split into one file per tile. Only tiles containing at least one point are written. Output goes to a selected folder rather than a single file. A summary log shows total tiles exported and total point count.

Without this option, a single PLY file is exported with all points in the selected coordinate system.

---

### Geoid correction (Save dialog)

Corrects the ellipsoidal altitude from the GPS datum (WGS84 height) to orthometric height (above geoid / mean sea level), which is the vertical reference used by public LiDAR datasets (PNOA, ICGC, Virtual Shizuoka).

Three modes:
- **None** — no correction, exports ellipsoidal altitude
- **Manual offset** — user enters the geoid undulation N in metres (H_orthometric = h_ellipsoidal − N)
- **Auto EGM2008** — reads geoid undulation from lookup tables in `EGM_tables/`. Tables are scanned in filename order; the first file covering the datum lat/lon is used. Currently includes `01_Spain.geoid` and `02_Japan.geoid`.

---

### Acknowledgements & Data Sources

**Coordinate system specifications:**

- UTM WGS84 projection formulas: Snyder, J.P. (1987). *Map Projections — A Working Manual*. USGS Professional Paper 1395. U.S. Government Printing Office.
- JGD2011 zone parameters: EPSG Geodetic Registry, EPSG:6669–6687. https://epsg.io
- 国土基本図図郭500 mesh code algorithm: Geospatial Information Authority of Japan (国土地理院). *公共測量作業規程の準則 付録7 公共測量標準図式*. https://www.gsi.go.jp

**Prefecture boundary data:**

- `japan_prefectures.geojson` — derived from 国土数値情報 行政区域データ (N03-20230101), Ministry of Land, Infrastructure, Transport and Tourism of Japan (国土交通省). Dissolved to prefecture level (N03_001 field). License: free use with attribution.  
  Source: https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-v3_1.html

**Geoid undulation tables:**

- EGM2008 geoid model: National Geospatial-Intelligence Agency (NGA). Implemented as regional lookup grids with bilinear interpolation.

**Tile naming conventions:**

- PNOA Spain: IGN/CNIG PNOA-LiDAR 3ª Cobertura (2022–2025), 1×1 km blocks. Reference: https://pnoa.ign.es/pnoa-lidar/productos-a-descarga
- ICGC Cataluña: Institut Cartogràfic i Geològic de Catalunya, LiDAR Territorial 3ª Cobertura (2021–2023), 1×1 km blocks. Reference: https://www.icgc.cat/es/Geoinformacion-y-mapas/Datos-y-productos/Bessons-digitals-Elevacions/LiDAR-Territorial
- Japan (JGD2011): 国土基本図図郭500, nationwide standard. Reference implementation verified against Virtual Shizuoka MMS dataset mesh list. Source: https://www.geospatial.jp/ckan/dataset/virtual-shizuoka-mw

**Reference datasets used for validation:**

- PNOA LiDAR 2ª/3ª Cobertura — Instituto Geográfico Nacional (IGN) / Centro Nacional de Información Geográfica (CNIG). ETRS89 UTM, orthometric heights. https://centrodedescargas.cnig.es
- ICGC LiDAR Territorial — Institut Cartogràfic i Geològic de Catalunya. ETRS89 UTM, orthometric heights. https://www.icgc.cat
- Virtual Shizuoka (バーチャル静岡) MMS dataset — Shizuoka Prefecture. JGD2011 / Japan Plane Rectangular CS VIII. CC BY 4.0 / ODbL. https://www.geospatial.jp/ckan/dataset/virtual-shizuoka-mw

---

### Architectural Changes

To support georeferenced export in the offline viewer, several structural changes were made to the glim/glim_ext codebase:

**geodetic.hpp/cpp moved to glim core**

Previously located in `glim_ext/modules/mapping/gnss_global/`, geodetic math is now part of `libglim.so`:
- `include/glim/util/geodetic.hpp`
- `src/glim/util/geodetic.cpp`

Extended with: generic parameterized Transverse Mercator forward projection (`tm_forward()`), inverse UTM (`utm_inverse()`), JGD2011 19-zone parameter table, and the 国土基本図図郭500 mesh code algorithm (`xy_to_zukaku500()`). Namespace typo (`gir` → `glim`) fixed during migration.

**gnss_global module moved to glim core**

The GNSS global constraints module was moved from `glim_ext` to `glim`, compiled as a separate ROS2-gated shared library (`libgnss_global.so`) following the same pattern as `libinteractive_viewer.so`. Rationale: for MMS pipelines, GNSS is not an optional extension but a fundamental component — GPS is the primary georeferencing source, with LiDAR odometry as fallback.

Files moved:
- `include/glim/mapping/gnss_global_module.hpp`
- `src/glim/mapping/gnss_global_module_ros2.cpp`
- `include/glim/util/nmea_parser.hpp`
- `config/config_gnss_global.json`
- `EGM_tables/` (geoid undulation lookup tables)

Config system updated from `GlobalConfigExt::get_config_path` to `GlobalConfig::get_config_path`. The `config_gnss_global` entry moved from `config_ext.json` to `config.json`.

**glim_ext cleaned**

All GNSS-related code removed from glim_ext: `ENABLE_GNSS` option, `gnss_global/` directory, `EGM_tables/`, and the `config_gnss_global` entry in `config_ext.json`. glim_ext retains its other extension modules unchanged.

**Japan prefecture boundaries**

`EGM_tables/japan_prefectures.geojson` added — 47 prefecture polygons dissolved from the official 国土数値情報 行政区域データ (N03-20230101), used at export time for JGD2011 zone auto-detection via point-in-polygon. Loaded lazily on first JGD2011 export, cached for the session.
