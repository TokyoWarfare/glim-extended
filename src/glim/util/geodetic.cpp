#include <glim/util/geodetic.hpp>

#include <cmath>

namespace glim {

/**
 * Code originally from ethz-asl/geodetic_utils
 * (BSD3 https://github.com/ethz-asl/geodetic_utils/blob/master/geodetic_utils/LICENSE)
 *
 * Extended with generic TM projection, inverse UTM, and JGD2011 zone table.
 */

// WGS84 / GRS80 ellipsoid parameters (effectively identical for our purposes:
// WGS84 1/f = 298.257223563, GRS80 1/f = 298.257222101 — difference ~0.1 mm over 10 000 km).
static const double kSemimajorAxis = 6378137;
static const double kSemiminorAxis = 6356752.3142;
static const double kFirstEccentricitySquared = 6.69437999014 * 0.001;
static const double kSecondEccentricitySquared = 6.73949674228 * 0.001;
static const double kFlattening = 1 / 298.257223563;

// ---------------------------------------------------------------------------
// ECEF / WGS84 conversions
// ---------------------------------------------------------------------------

Eigen::Vector3d ecef_to_wgs84(const Eigen::Vector3d& xyz) {
  // J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates
  // to geodetic coordinates," IEEE Transactions on Aerospace and
  // Electronic Systems, vol. 30, pp. 957-961, 1994.

  const double x = xyz.x();
  const double y = xyz.y();
  const double z = xyz.z();

  double r = sqrt(x * x + y * y);
  double Esq = kSemimajorAxis * kSemimajorAxis - kSemiminorAxis * kSemiminorAxis;
  double F = 54 * kSemiminorAxis * kSemiminorAxis * z * z;
  double G = r * r + (1 - kFirstEccentricitySquared) * z * z - kFirstEccentricitySquared * Esq;
  double C = (kFirstEccentricitySquared * kFirstEccentricitySquared * F * r * r) / pow(G, 3);
  double S = cbrt(1 + C + sqrt(C * C + 2 * C));
  double P = F / (3 * pow((S + 1 / S + 1), 2) * G * G);
  double Q = sqrt(1 + 2 * kFirstEccentricitySquared * kFirstEccentricitySquared * P);
  double r_0 = -(P * kFirstEccentricitySquared * r) / (1 + Q) +
               sqrt(0.5 * kSemimajorAxis * kSemimajorAxis * (1 + 1.0 / Q) - P * (1 - kFirstEccentricitySquared) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r);
  double U = sqrt(pow((r - kFirstEccentricitySquared * r_0), 2) + z * z);
  double V = sqrt(pow((r - kFirstEccentricitySquared * r_0), 2) + (1 - kFirstEccentricitySquared) * z * z);
  double Z_0 = kSemiminorAxis * kSemiminorAxis * z / (kSemimajorAxis * V);

  const double alt = U * (1 - kSemiminorAxis * kSemiminorAxis / (kSemimajorAxis * V));
  const double lat = atan((z + kSecondEccentricitySquared * Z_0) / r) * 180.0 / M_PI;
  const double lon = atan2(y, x) * 180.0 / M_PI;

  return {lat, lon, alt};
}

Eigen::Vector3d wgs84_to_ecef(double lat, double lon, double alt) {
  double lat_rad = lat * M_PI / 180.0;
  double lon_rad = lon * M_PI / 180.0;
  double xi = sqrt(1 - kFirstEccentricitySquared * sin(lat_rad) * sin(lat_rad));
  const double x = (kSemimajorAxis / xi + alt) * cos(lat_rad) * cos(lon_rad);
  const double y = (kSemimajorAxis / xi + alt) * cos(lat_rad) * sin(lon_rad);
  const double z = (kSemimajorAxis / xi * (1 - kFirstEccentricitySquared) + alt) * sin(lat_rad);

  return {x, y, z};
}

Eigen::Isometry3d calc_T_ecef_nwz(const Eigen::Vector3d& ecef, double radius) {
  const Eigen::Vector3d z = ecef.normalized();
  const Eigen::Vector3d to_north = (Eigen::Vector3d::UnitZ() * radius - ecef).normalized();
  const Eigen::Vector3d x = (to_north - to_north.dot(z) * z).normalized();
  const Eigen::Vector3d y = z.cross(x);

  Eigen::Isometry3d T_ecef_nwz = Eigen::Isometry3d::Identity();
  T_ecef_nwz.linear().col(0) = x;
  T_ecef_nwz.linear().col(1) = y;
  T_ecef_nwz.linear().col(2) = z;
  T_ecef_nwz.translation() = ecef;

  return T_ecef_nwz;
}

double harversine(const Eigen::Vector2d& latlon1, const Eigen::Vector2d& latlon2) {
  const double lat1 = latlon1[0];
  const double lon1 = latlon1[1];
  const double lat2 = latlon2[0];
  const double lon2 = latlon2[1];

  const double dlat = lat2 - lat1;
  const double dlon = lon2 - lon1;

  const double a = std::pow(sin(dlat / 2), 2) + cos(lat1) * cos(lat2) * std::pow(sin(dlon / 2), 2);
  const double c = 2 * atan2(sqrt(a), sqrt(1 - a));

  return kSemimajorAxis * c;
}

// ---------------------------------------------------------------------------
// Meridional arc helper (used by TM forward, inverse UTM)
// ---------------------------------------------------------------------------

/// Meridional arc length from equator to latitude phi (radians).
static double meridional_arc(double phi) {
  const double e2 = kFirstEccentricitySquared;
  const double e4 = e2 * e2;
  const double e6 = e4 * e2;
  return kSemimajorAxis * (
      (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0)  * phi
    - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * std::sin(2.0 * phi)
    + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0)                 * std::sin(4.0 * phi)
    - (35.0 * e6 / 3072.0)                                       * std::sin(6.0 * phi));
}

// ---------------------------------------------------------------------------
// UTM projection (WGS84 Transverse Mercator, Snyder 1987 section 8)
// ---------------------------------------------------------------------------

static constexpr double kUTMk0       = 0.9996;
static constexpr double kUTMFalseE   = 500000.0;
static constexpr double kUTMFalseN_S = 10000000.0;

int ecef_to_utm_zone(double /*lat*/, double lon) {
  return static_cast<int>(std::floor((lon + 180.0) / 6.0)) % 60 + 1;
}

Eigen::Vector2d wgs84_to_utm_xy(double lat, double lon) {
  return wgs84_to_utm_xy(lat, lon, ecef_to_utm_zone(lat, lon));
}

Eigen::Vector2d wgs84_to_utm_xy(double lat, double lon, int forced_zone) {
  const int zone      = forced_zone;
  const double lon0   = ((zone - 1) * 6 - 180 + 3) * M_PI / 180.0;

  const double phi    = lat * M_PI / 180.0;
  const double lam    = lon * M_PI / 180.0;
  const double dL     = lam - lon0;

  const double sinphi = std::sin(phi);
  const double cosphi = std::cos(phi);
  const double tanphi = std::tan(phi);

  const double N_pv   = kSemimajorAxis / std::sqrt(1.0 - kFirstEccentricitySquared * sinphi * sinphi);

  const double T  = tanphi * tanphi;
  const double e2 = kFirstEccentricitySquared;
  const double ep2 = e2 / (1.0 - e2);
  const double C  = ep2 * cosphi * cosphi;
  const double A  = cosphi * dL;
  const double A2 = A * A;
  const double A3 = A2 * A;
  const double A4 = A3 * A;
  const double A5 = A4 * A;
  const double A6 = A5 * A;

  const double M = meridional_arc(phi);

  const double easting = kUTMk0 * N_pv * (
      A
    + (1.0 - T + C)                                      * A3 / 6.0
    + (5.0 - 18.0 * T + T * T + 72.0 * C - 58.0 * ep2)  * A5 / 120.0)
    + kUTMFalseE;

  const double northing = kUTMk0 * (
      M
    + N_pv * tanphi * (
        A2 / 2.0
      + (5.0 - T + 9.0 * C + 4.0 * C * C)                              * A4 / 24.0
      + (61.0 - 58.0 * T + T * T + 600.0 * C - 330.0 * ep2)            * A6 / 720.0))
    + (lat < 0.0 ? kUTMFalseN_S : 0.0);

  return {easting, northing};
}

Eigen::Vector3d enu_to_utm(const Eigen::Vector3d& enu, double datum_lat, double datum_lon, double datum_alt) {
  const Eigen::Vector2d origin = wgs84_to_utm_xy(datum_lat, datum_lon);
  return {origin.x() + enu.x(), origin.y() + enu.y(), datum_alt + enu.z()};
}

// ---------------------------------------------------------------------------
// Inverse UTM (Snyder 1987 section 8, inverse formulas)
// ---------------------------------------------------------------------------

Eigen::Vector2d utm_inverse(double easting, double northing, int zone, bool south_hemisphere) {
  const double k0   = kUTMk0;
  const double lon0 = ((zone - 1) * 6 - 180 + 3) * M_PI / 180.0;
  const double false_n = south_hemisphere ? kUTMFalseN_S : 0.0;

  const double e2  = kFirstEccentricitySquared;
  const double ep2 = e2 / (1.0 - e2);

  // Footpoint latitude from meridional arc.
  const double M1  = (northing - false_n) / k0;
  const double e4  = e2 * e2;
  const double e6  = e4 * e2;
  const double mu1 = M1 / (kSemimajorAxis * (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0));

  const double e1  = (1.0 - std::sqrt(1.0 - e2)) / (1.0 + std::sqrt(1.0 - e2));
  const double e12 = e1 * e1;
  const double e13 = e12 * e1;
  const double e14 = e13 * e1;

  const double phi1 = mu1
    + (3.0 * e1 / 2.0 - 27.0 * e13 / 32.0)    * std::sin(2.0 * mu1)
    + (21.0 * e12 / 16.0 - 55.0 * e14 / 32.0)  * std::sin(4.0 * mu1)
    + (151.0 * e13 / 96.0)                      * std::sin(6.0 * mu1)
    + (1097.0 * e14 / 512.0)                    * std::sin(8.0 * mu1);

  const double sinphi1 = std::sin(phi1);
  const double cosphi1 = std::cos(phi1);
  const double tanphi1 = std::tan(phi1);

  const double N1 = kSemimajorAxis / std::sqrt(1.0 - e2 * sinphi1 * sinphi1);
  const double T1 = tanphi1 * tanphi1;
  const double C1 = ep2 * cosphi1 * cosphi1;
  const double R1 = kSemimajorAxis * (1.0 - e2) / std::pow(1.0 - e2 * sinphi1 * sinphi1, 1.5);
  const double D  = (easting - kUTMFalseE) / (N1 * k0);
  const double D2 = D * D;
  const double D3 = D2 * D;
  const double D4 = D3 * D;
  const double D5 = D4 * D;
  const double D6 = D5 * D;

  const double lat_rad = phi1 - (N1 * tanphi1 / R1) * (
      D2 / 2.0
    - (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1 * C1 - 9.0 * ep2)                         * D4 / 24.0
    + (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1 * T1 - 252.0 * ep2 - 3.0 * C1 * C1)   * D6 / 720.0);

  const double lon_rad = lon0 + (
      D
    - (1.0 + 2.0 * T1 + C1)                                                              * D3 / 6.0
    + (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1 * C1 + 8.0 * ep2 + 24.0 * T1 * T1)        * D5 / 120.0) / cosphi1;

  return {lat_rad * 180.0 / M_PI, lon_rad * 180.0 / M_PI};
}

// ---------------------------------------------------------------------------
// Generic Transverse Mercator forward projection (Snyder 1987)
// ---------------------------------------------------------------------------

Eigen::Vector2d tm_forward(double lat, double lon, const TMProjectionParams& params) {
  const double phi  = lat * M_PI / 180.0;
  const double lam  = lon * M_PI / 180.0;
  const double phi0 = params.lat_0 * M_PI / 180.0;
  const double lam0 = params.lon_0 * M_PI / 180.0;
  const double k0   = params.k0;

  const double dL     = lam - lam0;
  const double sinphi = std::sin(phi);
  const double cosphi = std::cos(phi);
  const double tanphi = std::tan(phi);

  const double N_pv = kSemimajorAxis / std::sqrt(1.0 - kFirstEccentricitySquared * sinphi * sinphi);

  const double T   = tanphi * tanphi;
  const double e2  = kFirstEccentricitySquared;
  const double ep2 = e2 / (1.0 - e2);
  const double C   = ep2 * cosphi * cosphi;
  const double A   = cosphi * dL;
  const double A2  = A * A;
  const double A3  = A2 * A;
  const double A4  = A3 * A;
  const double A5  = A4 * A;
  const double A6  = A5 * A;

  const double M  = meridional_arc(phi);
  const double M0 = meridional_arc(phi0);

  const double easting = k0 * N_pv * (
      A
    + (1.0 - T + C)                                      * A3 / 6.0
    + (5.0 - 18.0 * T + T * T + 72.0 * C - 58.0 * ep2)  * A5 / 120.0)
    + params.false_easting;

  const double northing_val = k0 * (
      M - M0
    + N_pv * tanphi * (
        A2 / 2.0
      + (5.0 - T + 9.0 * C + 4.0 * C * C)                              * A4 / 24.0
      + (61.0 - 58.0 * T + T * T + 600.0 * C - 330.0 * ep2)            * A6 / 720.0))
    + params.false_northing;

  return {easting, northing_val};
}

// ---------------------------------------------------------------------------
// JGD2011 Plane Rectangular Coordinate System (EPSG:6669-6687)
// ---------------------------------------------------------------------------

// 19-zone table: {lat_0, lon_0} from JIS X 0410 / EPSG definitions.
// All zones: k0 = 0.9999, false_easting = 0, false_northing = 0.
static const double kJGD2011Zones[19][2] = {
  { 33.0, 129.5    },  // Zone I    (EPSG:6669) — Nagasaki, parts of Saga
  { 33.0, 131.0    },  // Zone II   (EPSG:6670) — Fukuoka, Saga, Kumamoto, Oita
  { 36.0, 132.1667 },  // Zone III  (EPSG:6671) — Yamaguchi, Shimane, Hiroshima
  { 33.0, 133.5    },  // Zone IV   (EPSG:6672) — Tokushima, Kagawa, Ehime, Kochi
  { 36.0, 134.3333 },  // Zone V    (EPSG:6673) — Hyogo, Tottori, Okayama
  { 36.0, 136.0    },  // Zone VI   (EPSG:6674) — Kyoto, Osaka, Fukui, Shiga, Mie, ...
  { 36.0, 137.1667 },  // Zone VII  (EPSG:6675) — Ishikawa, Toyama, Gifu, Nagano
  { 36.0, 138.5    },  // Zone VIII (EPSG:6676) — Niigata, Nagano, Gunma, Saitama, Tokyo, Chiba, Kanagawa, Yamanashi, Shizuoka
  { 36.0, 139.8333 },  // Zone IX   (EPSG:6677) — Tokyo, Fukushima, Tochigi, Ibaraki, Saitama, Chiba
  { 40.0, 140.8333 },  // Zone X    (EPSG:6678) — Aomori, Akita, Yamagata, Iwate, Miyagi
  { 44.0, 140.25   },  // Zone XI   (EPSG:6679) — Hokkaido (west)
  { 44.0, 142.25   },  // Zone XII  (EPSG:6680) — Hokkaido (central)
  { 44.0, 144.25   },  // Zone XIII (EPSG:6681) — Hokkaido (east)
  { 26.0, 142.0    },  // Zone XIV  (EPSG:6682) — Tokyo islands (Ogasawara north)
  { 26.0, 127.5    },  // Zone XV   (EPSG:6683) — Okinawa (main)
  { 26.0, 124.0    },  // Zone XVI  (EPSG:6684) — Sakishima islands
  { 26.0, 131.0    },  // Zone XVII (EPSG:6685) — Daito islands
  { 20.0, 136.0    },  // Zone XVIII(EPSG:6686) — Ogasawara (south)
  { 26.0, 154.0    },  // Zone XIX  (EPSG:6687) — Minami-Torishima
};

static const char* kJGD2011ZoneNames[19] = {
  "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
  "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX"
};

TMProjectionParams jgd2011_zone_params(int zone) {
  const int idx = (zone >= 1 && zone <= 19) ? zone - 1 : 7;  // default to zone VIII
  return {kJGD2011Zones[idx][0], kJGD2011Zones[idx][1], 0.9999, 0.0, 0.0};
}

const char* jgd2011_zone_name(int zone) {
  if (zone < 1 || zone > 19) return "?";
  return kJGD2011ZoneNames[zone - 1];
}

int jgd2011_auto_zone(double /*lat*/, double /*lon*/) {
  // STUB: prefecture-boundary lookup not yet implemented.
  // Returns zone VIII (Shizuoka / Kanto region) as default.
  // Will be replaced with geojson-based point-in-polygon when
  // japan_prefectures.geojson is available.
  return 8;
}

}  // namespace glim
