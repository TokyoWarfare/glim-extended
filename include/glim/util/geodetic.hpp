#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim {

// ---------------------------------------------------------------------------
// Ellipsoid conversions (ECEF / WGS84)
// ---------------------------------------------------------------------------

Eigen::Vector3d ecef_to_wgs84(const Eigen::Vector3d& xyz);

Eigen::Vector3d wgs84_to_ecef(double lat, double lon, double alt);

Eigen::Isometry3d calc_T_ecef_nwz(const Eigen::Vector3d& ecef, double radius = 6378137);

double harversine(const Eigen::Vector2d& latlon1, const Eigen::Vector2d& latlon2);

// ---------------------------------------------------------------------------
// UTM projection (WGS84 Transverse Mercator, Snyder 1987)
// ---------------------------------------------------------------------------

/// Returns the UTM zone number (1-60) for the given longitude in degrees.
int ecef_to_utm_zone(double lat, double lon);

/// Projects (lat, lon) in degrees to UTM (easting, northing) in metres.
/// Auto-detects the zone from the longitude.
Eigen::Vector2d wgs84_to_utm_xy(double lat, double lon);

/// Same as wgs84_to_utm_xy but forces a specific UTM zone (for zone-crossing safety).
Eigen::Vector2d wgs84_to_utm_xy(double lat, double lon, int forced_zone);

/// Converts an ENU point (metres from datum) to UTM easting / northing / altitude.
Eigen::Vector3d enu_to_utm(const Eigen::Vector3d& enu, double datum_lat, double datum_lon, double datum_alt);

/// Inverse UTM projection: (easting, northing) in metres -> (lat, lon) in degrees.
/// @param south_hemisphere  true for UTM zones in the southern hemisphere.
Eigen::Vector2d utm_inverse(double easting, double northing, int zone, bool south_hemisphere = false);

// ---------------------------------------------------------------------------
// Generic Transverse Mercator projection
// ---------------------------------------------------------------------------

/// Parameters for a generic Transverse Mercator projection on the WGS84/GRS80 ellipsoid.
struct TMProjectionParams {
  double lat_0;           // origin latitude (degrees)
  double lon_0;           // central meridian (degrees)
  double k0;              // scale factor on central meridian
  double false_easting;   // metres added to easting
  double false_northing;  // metres added to northing
};

/// Generic Transverse Mercator forward projection (Snyder 1987).
/// Projects (lat, lon) in degrees to (easting, northing) in metres.
Eigen::Vector2d tm_forward(double lat, double lon, const TMProjectionParams& params);

// ---------------------------------------------------------------------------
// JGD2011 Plane Rectangular Coordinate System (Japan, 19 zones)
// ---------------------------------------------------------------------------

/// Returns TM projection parameters for the given JGD2011 zone (1-19).
/// All zones use k0=0.9999, false_easting=0, false_northing=0, GRS80 ellipsoid.
/// Returns zone VIII params if zone is out of range.
TMProjectionParams jgd2011_zone_params(int zone);

/// Returns the Roman numeral name for a JGD2011 zone (1-19).
/// Returns "?" for out-of-range zones.
const char* jgd2011_zone_name(int zone);

/// Auto-detect JGD2011 zone from lat/lon.
/// STUB: returns zone VIII (Shizuoka/Kanto). Will be replaced with
/// prefecture-boundary lookup when japan_prefectures.geojson is available.
int jgd2011_auto_zone(double lat, double lon);

}  // namespace glim
