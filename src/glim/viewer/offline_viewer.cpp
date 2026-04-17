#include <glim/viewer/offline_viewer.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
#include <gtsam_points/config.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>
#include <glim/util/config.hpp>
#include <glim/util/geodetic.hpp>
#include <glim/util/post_processing.hpp>
#include <glim/util/map_cleaner.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam_points/factors/integrated_matching_cost_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>

#include <spdlog/spdlog.h>
#include <portable-file-dialogs.h>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <GL/gl3w.h>
#include <glk/thin_lines.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <guik/camera/fps_camera_control.hpp>
#include <glk/io/ply_io.hpp>
#include <guik/recent_files.hpp>
#include <guik/progress_modal.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace glim {

// ---------------------------------------------------------------------------
// Geoid undulation lookup — EGM2008 table files
// ---------------------------------------------------------------------------

namespace {

// Intensity to blue-cyan-white color ramp (for intensity blend visualization)
inline Eigen::Vector3f intensity_to_color(float t) {
  // t in [0,1]: 0=dark blue, 0.5=cyan, 1.0=white
  if (t < 0.5f) {
    const float s = t * 2.0f;
    return Eigen::Vector3f(0.05f, 0.1f + 0.7f * s, 0.3f + 0.7f * s);  // dark blue → cyan
  } else {
    const float s = (t - 0.5f) * 2.0f;
    return Eigen::Vector3f(s, 0.8f + 0.2f * s, 1.0f);  // cyan → white
  }
}

// Parsed contents of one .geoid file.
struct GeoidTable {
  std::string path;
  double lat_min, lat_max, lon_min, lon_max;
  double lat_step, lon_step;
  int nrows, ncols;
  std::vector<float> data;  // row-major: row 0 = lat_min, col 0 = lon_min

  float at(int row, int col) const { return data[row * ncols + col]; }

  bool covers(double lat, double lon) const {
    return lat >= lat_min && lat <= lat_max &&
           lon >= lon_min && lon <= lon_max;
  }
};

// Parse a GLIM_GEOID_V1 text file.  Returns false on any format error.
bool load_geoid_table(const std::string& path, GeoidTable& out) {
  std::ifstream f(path);
  if (!f) return false;

  std::string line;
  if (!std::getline(f, line) || line != "GLIM_GEOID_V1") return false;

  out.path = path;
  out.lat_min = out.lat_max = out.lon_min = out.lon_max = 0.0;
  out.lat_step = out.lon_step = 1.0;
  out.nrows = out.ncols = 0;
  bool have_lat_min = false, have_lat_max = false;
  bool have_lon_min = false, have_lon_max = false;
  bool have_lat_step = false, have_lon_step = false;

  // Read header key=value lines and data rows.
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;

    // Key=value header?
    const auto eq = line.find('=');
    if (eq != std::string::npos) {
      const std::string key = line.substr(0, eq);
      const double val = std::stod(line.substr(eq + 1));
      if      (key == "lat_min")  { out.lat_min  = val; have_lat_min  = true; }
      else if (key == "lat_max")  { out.lat_max  = val; have_lat_max  = true; }
      else if (key == "lon_min")  { out.lon_min  = val; have_lon_min  = true; }
      else if (key == "lon_max")  { out.lon_max  = val; have_lon_max  = true; }
      else if (key == "lat_step") { out.lat_step = val; have_lat_step = true; }
      else if (key == "lon_step") { out.lon_step = val; have_lon_step = true; }
      continue;
    }

    // Must be a data row.
    if (!have_lat_min || !have_lat_max || !have_lon_min ||
        !have_lon_max || !have_lat_step || !have_lon_step) {
      return false;  // data before all headers — malformed
    }

    // Derive expected dimensions on first data row encounter.
    if (out.nrows == 0) {
      out.nrows = static_cast<int>(
        std::round((out.lat_max - out.lat_min) / out.lat_step) + 1);
      out.ncols = static_cast<int>(
        std::round((out.lon_max - out.lon_min) / out.lon_step) + 1);
      out.data.reserve(out.nrows * out.ncols);
    }

    std::istringstream ss(line);
    float v;
    while (ss >> v) out.data.push_back(v);
  }

  if (out.nrows == 0 || out.ncols == 0) return false;
  if (static_cast<int>(out.data.size()) != out.nrows * out.ncols) return false;
  return true;
}

// Scan for EGM_tables directories.  Search order:
//   1. Each prefix in AMENT_PREFIX_PATH: <prefix>/share/glim/EGM_tables
//   2. <config_dir>/EGM_tables  (allows local per-map overrides)
std::vector<std::string> find_egm_table_files() {
  std::vector<std::string> dirs;

  // 1. AMENT_PREFIX_PATH entries
  const char* ament = std::getenv("AMENT_PREFIX_PATH");
  if (ament) {
    std::istringstream ss(ament);
    std::string prefix;
    while (std::getline(ss, prefix, ':')) {
      if (!prefix.empty()) {
        dirs.push_back(prefix + "/share/glim/EGM_tables");
      }
    }
  }

  // 2. Local config-dir override
  const std::string config_dir =
    GlobalConfig::instance()->param<std::string>("global", "config_path", ".");
  dirs.push_back(config_dir + "/EGM_tables");

  // Collect all *.geoid files from the first directory that exists.
  std::vector<std::string> files;
  for (const auto& dir : dirs) {
    if (!boost::filesystem::is_directory(dir)) continue;
    for (boost::filesystem::directory_iterator it(dir), end; it != end; ++it) {
      const auto& p = it->path();
      if (p.extension() == ".geoid") {
        files.push_back(p.string());
      }
    }
    if (!files.empty()) break;  // stop at first dir that has files
  }

  std::sort(files.begin(), files.end());  // prefix 01_, 02_ controls priority
  return files;
}

// ---------------------------------------------------------------------------
// Tile naming helpers
// ---------------------------------------------------------------------------

/// Compute 国土基本図図郭500 (1:500 map sheet) mesh code from JGD2011 Plane
/// Rectangular coordinates.  X = northing (m), Y = easting (m).
/// Returns an 8-character string like "08MD8614".
std::string xy_to_zukaku500(int zone, double X, double Y) {
  // Level 50000: 30 km (N-S) x 40 km (E-W) sections, 2-letter code
  int row_50k = static_cast<int>(std::floor((300000.0 - X) / 30000.0));
  int col_50k = static_cast<int>(std::floor((Y + 160000.0) / 40000.0));
  row_50k = std::max(0, std::min(19, row_50k));
  col_50k = std::max(0, std::min(7, col_50k));

  const char letter1 = 'A' + row_50k;
  const char letter2 = 'A' + col_50k;

  // Level 5000: 3 km x 4 km cells, 2 digits (row, col within section)
  const double section_north = 300000.0 - row_50k * 30000.0;
  const double section_west  = -160000.0 + col_50k * 40000.0;
  int row_5k = static_cast<int>(std::floor((section_north - X) / 3000.0));
  int col_5k = static_cast<int>(std::floor((Y - section_west) / 4000.0));
  row_5k = std::max(0, std::min(9, row_5k));
  col_5k = std::max(0, std::min(9, col_5k));

  // Level 500: 300 m x 400 m cells, 2 digits (row, col within level-5000 cell)
  const double cell5k_north = section_north - row_5k * 3000.0;
  const double cell5k_west  = section_west  + col_5k * 4000.0;
  int row_500 = static_cast<int>(std::floor((cell5k_north - X) / 300.0));
  int col_500 = static_cast<int>(std::floor((Y - cell5k_west) / 400.0));
  row_500 = std::max(0, std::min(9, row_500));
  col_500 = std::max(0, std::min(9, col_500));

  char buf[9];
  std::snprintf(buf, sizeof(buf), "%02d%c%c%d%d%d%d",
                zone, letter1, letter2, row_5k, col_5k, row_500, col_500);
  return std::string(buf);
}

/// Generate tile filename stem for a point at (x, y) in projected coordinates.
/// x = easting (m), y = northing (m) for UTM / JGD2011.
/// Tile name encodes the SW corner (min easting, min northing).
std::string tile_name_for_point(
  double x, double y, int preset, double tile_size_m, int jgd_zone)
{
  char buf[64];
  if (preset == 1) {
    // PNOA Spain 1x1 km: PNOA_MMS_EEE_NNNN
    // SW corner = floor at 1000 m multiples, expressed in km
    const int tile_col = static_cast<int>(std::floor(x / 1000.0));
    const int tile_row = static_cast<int>(std::floor(y / 1000.0));
    const int sw_e_km  = tile_col;       // easting of SW corner in km
    const int sw_n_km  = tile_row;       // northing of SW corner in km
    std::snprintf(buf, sizeof(buf), "PNOA_MMS_%d_%d", sw_e_km, sw_n_km);
  } else if (preset == 2) {
    // ICGC Cataluna 1x1 km: EEENNN (northing offset -4 000 000)
    // SW corner = floor at 1000 m multiples
    const int tile_col    = static_cast<int>(std::floor(x / 1000.0));
    const int tile_row    = static_cast<int>(std::floor(y / 1000.0));
    const int sw_e_km     = tile_col;             // easting in km
    const int sw_n_adj_km = tile_row - 4000;      // northing - 4 000 000, in km
    std::snprintf(buf, sizeof(buf), "%03d%03d", sw_e_km, sw_n_adj_km);
  } else if (preset == 3) {
    // Japan (JGD2011): kokudo kihonzu zukaku 500 mesh code (300m N-S x 400m E-W)
    // JGD2011 convention: X = northing, Y = easting
    // Our coords: x = easting (tm_forward[0]), y = northing (tm_forward[1])
    // Zone prefix (first 2 chars) comes from jgd_zone parameter
    return xy_to_zukaku500(jgd_zone, /*X=northing*/y, /*Y=easting*/x);
  } else {
    // Default: TILE_EEEEEEE_NNNNNNN (SW corner in metres)
    // SW corner = floor at tile_size_m multiples
    const double tile_col = std::floor(x / tile_size_m);
    const double tile_row = std::floor(y / tile_size_m);
    const long sw_e_m = static_cast<long>(tile_col * tile_size_m);
    const long sw_n_m = static_cast<long>(tile_row * tile_size_m);
    std::snprintf(buf, sizeof(buf), "TILE_%07ld_%07ld", sw_e_m, sw_n_m);
  }
  return std::string(buf);
}

// ---------------------------------------------------------------------------
// JGD2011 prefecture → zone mapping
// ---------------------------------------------------------------------------

struct PrefZoneEntry { const char* jp; const char* en; int zone; };
static const PrefZoneEntry kPrefZoneTable[] = {
  {"北海道",   "Hokkaido",   12}, {"青森県",   "Aomori",     10},
  {"岩手県",   "Iwate",      10}, {"宮城県",   "Miyagi",     10},
  {"秋田県",   "Akita",      10}, {"山形県",   "Yamagata",   10},
  {"福島県",   "Fukushima",   9}, {"茨城県",   "Ibaraki",     9},
  {"栃木県",   "Tochigi",     9}, {"群馬県",   "Gunma",       9},
  {"埼玉県",   "Saitama",     9}, {"千葉県",   "Chiba",       9},
  {"東京都",   "Tokyo",       9}, {"神奈川県", "Kanagawa",    9},
  {"新潟県",   "Niigata",     8}, {"富山県",   "Toyama",      7},
  {"石川県",   "Ishikawa",    7}, {"福井県",   "Fukui",       6},
  {"山梨県",   "Yamanashi",   8}, {"長野県",   "Nagano",      8},
  {"岐阜県",   "Gifu",        7}, {"静岡県",   "Shizuoka",    8},
  {"愛知県",   "Aichi",       7}, {"三重県",   "Mie",         6},
  {"滋賀県",   "Shiga",       6}, {"京都府",   "Kyoto",       6},
  {"大阪府",   "Osaka",       6}, {"兵庫県",   "Hyogo",       5},
  {"奈良県",   "Nara",        6}, {"和歌山県", "Wakayama",    6},
  {"鳥取県",   "Tottori",     5}, {"島根県",   "Shimane",     3},
  {"岡山県",   "Okayama",     5}, {"広島県",   "Hiroshima",   3},
  {"山口県",   "Yamaguchi",   3}, {"徳島県",   "Tokushima",   4},
  {"香川県",   "Kagawa",      4}, {"愛媛県",   "Ehime",       4},
  {"高知県",   "Kochi",       4}, {"福岡県",   "Fukuoka",     2},
  {"佐賀県",   "Saga",        2}, {"長崎県",   "Nagasaki",    1},
  {"熊本県",   "Kumamoto",    2}, {"大分県",   "Oita",        2},
  {"宮崎県",   "Miyazaki",    2}, {"鹿児島県", "Kagoshima",   2},
  {"沖縄県",   "Okinawa",    15},
};
static constexpr int kPrefZoneTableSize = sizeof(kPrefZoneTable) / sizeof(kPrefZoneTable[0]);

/// Look up JGD2011 zone for a prefecture name (N03_001 field). Returns 0 if not found.
int prefecture_to_zone(const std::string& name_jp) {
  for (int i = 0; i < kPrefZoneTableSize; i++) {
    if (name_jp == kPrefZoneTable[i].jp) return kPrefZoneTable[i].zone;
  }
  return 0;
}

/// Look up English name for a prefecture. Returns "" if not found.
const char* prefecture_english(const std::string& name_jp) {
  for (int i = 0; i < kPrefZoneTableSize; i++) {
    if (name_jp == kPrefZoneTable[i].jp) return kPrefZoneTable[i].en;
  }
  return "";
}

// ---------------------------------------------------------------------------
// Point-in-polygon (ray casting)
// ---------------------------------------------------------------------------

/// Test whether (px, py) is inside a closed ring of (x, y) vertices.
bool point_in_ring(double px, double py, const std::vector<Eigen::Vector2d>& ring) {
  bool inside = false;
  const int n = static_cast<int>(ring.size());
  for (int i = 0, j = n - 1; i < n; j = i++) {
    if ((ring[i].y() > py) != (ring[j].y() > py) &&
        px < (ring[j].x() - ring[i].x()) * (py - ring[i].y()) /
             (ring[j].y() - ring[i].y()) + ring[i].x()) {
      inside = !inside;
    }
  }
  return inside;
}

// ---------------------------------------------------------------------------
// GeoJSON loader for japan_prefectures.geojson
// ---------------------------------------------------------------------------

/// Search AMENT_PREFIX_PATH for japan_prefectures.geojson.
std::string find_prefecture_geojson() {
  const char* ament = std::getenv("AMENT_PREFIX_PATH");
  if (!ament) return "";
  std::istringstream ss(ament);
  std::string prefix;
  while (std::getline(ss, prefix, ':')) {
    if (prefix.empty()) continue;
    const std::string path = prefix + "/share/glim/EGM_tables/japan_prefectures.geojson";
    if (boost::filesystem::exists(path)) return path;
  }
  // Also check config dir
  const std::string config_dir =
    GlobalConfig::instance()->param<std::string>("global", "config_path", ".");
  const std::string local = config_dir + "/EGM_tables/japan_prefectures.geojson";
  if (boost::filesystem::exists(local)) return local;
  return "";
}

/// Parse a GeoJSON ring (array of [lon, lat] arrays) into Vector2d(lon, lat).
std::vector<Eigen::Vector2d> parse_ring(const nlohmann::json& coords) {
  std::vector<Eigen::Vector2d> ring;
  ring.reserve(coords.size());
  for (const auto& pt : coords) {
    ring.emplace_back(pt[0].get<double>(), pt[1].get<double>());
  }
  return ring;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------

OfflineViewer::OfflineViewer(const std::string& init_map_path) : init_map_path(init_map_path) {}

OfflineViewer::~OfflineViewer() {}

void OfflineViewer::load_gnss_datum() {
  gnss_datum_available = false;

  const std::string datum_path = GlobalConfig::get_config_path("gnss_datum");
  if (!boost::filesystem::exists(datum_path)) {
    logger->debug("gnss_datum.json not found at {}", datum_path);
    return;
  }

  std::ifstream ifs(datum_path);
  const auto j = nlohmann::json::parse(ifs, nullptr, /*exceptions=*/false);
  if (j.is_discarded()) {
    logger->warn("Failed to parse {}", datum_path);
    return;
  }

  gnss_utm_zone              = j.value("utm_zone", 0);
  gnss_utm_easting_origin    = j.value("utm_easting_origin", 0.0);
  gnss_utm_northing_origin   = j.value("utm_northing_origin", 0.0);
  gnss_datum_alt             = j.value("altitude", 0.0);
  gnss_datum_lat             = j.value("latitude",  0.0);
  gnss_datum_lon             = j.value("longitude", 0.0);

  gnss_datum_available = true;
  logger->info(
    "GNSS datum loaded: UTM zone {} E={:.3f} N={:.3f} alt={:.3f} lat={:.9f} lon={:.9f} from {}",
    gnss_utm_zone, gnss_utm_easting_origin, gnss_utm_northing_origin, gnss_datum_alt,
    gnss_datum_lat, gnss_datum_lon, datum_path);
}

double OfflineViewer::lookup_geoid_undulation(double lat, double lon) const {
  const std::vector<std::string> files = find_egm_table_files();

  if (files.empty()) {
    logger->warn(
      "[Geoid] No EGM table files found.  Place *.geoid files in "
      "<ament_prefix>/share/glim/EGM_tables/ or <config_dir>/EGM_tables/. "
      "Falling back to no geoid correction.");
    return 0.0;
  }

  for (const auto& file : files) {
    GeoidTable table;
    if (!load_geoid_table(file, table)) {
      logger->warn("[Geoid] Failed to parse table file: {}", file);
      continue;
    }
    if (!table.covers(lat, lon)) continue;

    // Bilinear interpolation.
    const double row_f = (lat - table.lat_min) / table.lat_step;
    const double col_f = (lon - table.lon_min) / table.lon_step;
    const int r0 = static_cast<int>(std::floor(row_f));
    const int c0 = static_cast<int>(std::floor(col_f));
    const int r1 = std::min(r0 + 1, table.nrows - 1);
    const int c1 = std::min(c0 + 1, table.ncols - 1);
    const double dr = row_f - r0;
    const double dc = col_f - c0;

    const double N =
      (1 - dr) * (1 - dc) * table.at(r0, c0) +
      (1 - dr) *      dc  * table.at(r0, c1) +
           dr  * (1 - dc) * table.at(r1, c0) +
           dr  *      dc  * table.at(r1, c1);

    logger->info("[Geoid] Using table: {}  N({:.4f}, {:.4f}) = {:.3f} m",
                 boost::filesystem::path(file).filename().string(), lat, lon, N);
    return N;
  }

  // No table covered the datum location.
  logger->warn(
    "[Geoid] No EGM table covers datum location (lat={:.4f}, lon={:.4f}). "
    "Currently only Spain (01_Spain.geoid) and Japan (02_Japan.geoid) are bundled. "
    "To correct your data, obtain EGM2008 undulation values for your region "
    "(e.g. from geographiclib or the NGA EGM2008 online calculator) and add a "
    "numbered .geoid file to the EGM_tables directory. "
    "Falling back to no geoid correction for this export.",
    lat, lon);
  return 0.0;
}

void OfflineViewer::build_trajectory() {
  trajectory_data.clear();
  double cumul = 0.0;
  Eigen::Vector3d prev_pos = Eigen::Vector3d::Zero();
  bool first = true;
  for (const auto& submap : submaps) {
    if (!submap) continue;
    if (hidden_sessions.count(submap->session_id)) continue;
    const Eigen::Isometry3d T_ep = submap->T_world_origin * submap->T_origin_endpoint_L;
    const Eigen::Isometry3d T_odom0 = submap->frames.front()->T_world_imu;
    for (const auto& frame : submap->frames) {
      const Eigen::Isometry3d T_world_imu = T_ep * T_odom0.inverse() * frame->T_world_imu;
      const Eigen::Isometry3d T_world_lidar = T_world_imu * frame->T_lidar_imu.inverse();
      const Eigen::Vector3d pos = T_world_lidar.translation();
      if (!first) cumul += (pos - prev_pos).norm();
      prev_pos = pos;
      first = false;
      trajectory_data.push_back({T_world_lidar, cumul, frame->stamp, submap->session_id, frame->id});
    }
  }
  trajectory_total_dist = cumul;
  follow_total_dist = cumul;
  trajectory_built = true;
  logger->info("[Trajectory] Built: {} points, {:.0f} m", trajectory_data.size(), trajectory_total_dist);
}
void OfflineViewer::ensure_prefectures_loaded() {
  if (prefectures_loaded) return;
  prefectures_loaded = true;  // mark even on failure to avoid retrying

  const std::string path = find_prefecture_geojson();
  if (path.empty()) {
    logger->error("[JGD2011] japan_prefectures.geojson not found. Auto-detect unavailable.");
    return;
  }

  logger->info("[JGD2011] Loading prefecture boundaries from {}", path);
  std::ifstream ifs(path);
  const auto geojson = nlohmann::json::parse(ifs, nullptr, /*exceptions=*/false);
  if (geojson.is_discarded() || !geojson.contains("features")) {
    logger->error("[JGD2011] Failed to parse {}", path);
    return;
  }

  for (const auto& feature : geojson["features"]) {
    const std::string name_jp = feature.value("/properties/N03_001"_json_pointer, std::string());
    if (name_jp.empty()) continue;

    const int zone = prefecture_to_zone(name_jp);
    if (zone == 0) {
      logger->warn("[JGD2011] Unknown prefecture: {}", name_jp);
      continue;
    }

    PrefectureEntry entry;
    entry.name_jp = name_jp;
    entry.name_en = prefecture_english(name_jp);
    entry.jgd_zone = zone;

    const auto& geom = feature["geometry"];
    const std::string geom_type = geom.value("type", "");

    if (geom_type == "Polygon") {
      // First ring is exterior
      entry.rings.push_back(parse_ring(geom["coordinates"][0]));
    } else if (geom_type == "MultiPolygon") {
      for (const auto& polygon : geom["coordinates"]) {
        entry.rings.push_back(parse_ring(polygon[0]));
      }
    }

    prefectures.push_back(std::move(entry));
  }
  logger->info("[JGD2011] Loaded {} prefectures", prefectures.size());

  // Auto-detect from datum if available
  if (gnss_datum_available) {
    for (const auto& pref : prefectures) {
      for (const auto& ring : pref.rings) {
        if (point_in_ring(gnss_datum_lon, gnss_datum_lat, ring)) {
          detected_pref_jp = pref.name_jp;
          detected_pref_en = pref.name_en;
          detected_jgd_zone = pref.jgd_zone;
          logger->info("[JGD2011] Datum in {} ({}) — Zone {} ({})",
                       detected_pref_jp, detected_pref_en,
                       jgd2011_zone_name(detected_jgd_zone), detected_jgd_zone);
          goto detection_done;
        }
      }
    }
    logger->warn("[JGD2011] Datum ({:.4f}, {:.4f}) not inside any prefecture",
                 gnss_datum_lat, gnss_datum_lon);
    detection_done:;
  }
}

void OfflineViewer::setup_ui() {
  auto viewer = guik::LightViewer::instance();
  viewer->register_ui_callback("main_menu", [this] { main_menu(); });

  // Session visibility filter — hides submaps and spheres for disabled sessions
  viewer->register_drawable_filter("session_filter", [this](const std::string& name) {
    if (sessions.size() <= 1) return true;  // no filtering for single session

    // Extract submap ID from drawable name
    for (const char* prefix : {"submap_", "sphere_", "coord_", "bbox_"}) {
      const std::string pfx(prefix);
      if (name.size() > pfx.size() && name.compare(0, pfx.size(), pfx) == 0) {
        const int submap_id = std::stoi(name.substr(pfx.size()));
        // Find the submap's session_id
        for (const auto& submap : submaps) {
          if (submap && submap->id == submap_id) {
            for (const auto& sess : sessions) {
              if (sess.id == submap->session_id) {
                return sess.visible;
              }
            }
            break;
          }
        }
      }
    }
    return true;
  });

  // Orbit W-S dolly: move camera toward/away from orbit center via scroll simulation
  viewer->register_ui_callback("orbit_dolly", [this] {
    if (camera_mode_sel != 0) return;
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    auto vw = guik::LightViewer::instance();
    auto cam = vw->get_camera_control();
    if (!cam) return;

    const float speed = ImGui::GetIO().KeyShift ? 5.0f : 1.0f;
    if (ImGui::IsKeyDown(ImGuiKey_W)) {
      cam->scroll(Eigen::Vector2f(0.0f, speed));  // zoom in
    }
    if (ImGui::IsKeyDown(ImGuiKey_S)) {
      cam->scroll(Eigen::Vector2f(0.0f, -speed));  // zoom out
    }
  });

  // FPV controls: shift-speed + position smoothing
  viewer->register_ui_callback("fpv_controls", [this] {
    if (camera_mode_sel != 1) return;
    auto fps = std::dynamic_pointer_cast<guik::FPSCameraControl>(
      guik::LightViewer::instance()->get_camera_control());
    if (!fps) return;
    fps->set_translation_speed(ImGui::GetIO().KeyShift ? fpv_speed * fpv_speed_mult : fpv_speed);

    // Smooth FPV position
    const Eigen::Matrix4f vm = fps->view_matrix();
    const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));
    // Smooth position only — rotation stays crisp (Iridescence handles it natively)
    const Eigen::Vector3f fwd = -vm.block<1, 3>(2, 0).transpose();
    const float yaw = std::atan2(fwd.y(), fwd.x()) * 180.0f / M_PI;
    const float pitch = std::asin(std::clamp(fwd.z(), -1.0f, 1.0f)) * 180.0f / M_PI;

    if (!fpv_smooth_init) {
      fpv_smooth_pos = cam_pos;
      fpv_smooth_init = true;
    } else {
      fpv_smooth_pos += (cam_pos - fpv_smooth_pos) * fpv_smoothness;
      fps->set_pose(fpv_smooth_pos, yaw, pitch);
    }
  });

  // Follow Trajectory mode: camera follows path with playback controls
  viewer->register_ui_callback("trajectory_dataectory", [this] {
    if (camera_mode_sel != 2 || trajectory_data.empty()) return;

    const double now = ImGui::GetTime();
    const double dt = now - follow_last_time;
    follow_last_time = now;

    // W-S speed control (works while playing or paused, Shift = 5x acceleration)
    const float accel = ImGui::GetIO().KeyShift ? 100.0f : 20.0f;
    if (ImGui::IsKeyDown(ImGuiKey_W)) {
      if (!follow_playing) { follow_playing = true; follow_speed_kmh = 0.0f; }
      follow_speed_kmh = std::min(follow_speed_kmh + accel * static_cast<float>(dt), 500.0f);
    }
    if (ImGui::IsKeyDown(ImGuiKey_S)) {
      if (!follow_playing) { follow_playing = true; follow_speed_kmh = 0.0f; }
      follow_speed_kmh = std::max(follow_speed_kmh - accel * static_cast<float>(dt), -500.0f);
    }

    // Space toggles play/pause — pause sets speed to 0, unpause recovers last speed
    static float follow_saved_speed = 30.0f;
    if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
      if (follow_playing) {
        follow_saved_speed = follow_speed_kmh;
        follow_speed_kmh = 0.0f;
        follow_playing = false;
      } else {
        follow_speed_kmh = follow_saved_speed;
        follow_playing = true;
      }
    }

    // Advance along trajectory (supports negative speed = reverse)
    if (follow_playing && follow_total_dist > 0.0) {
      const double speed_ms = follow_speed_kmh / 3.6;  // km/h to m/s
      const double advance = speed_ms * dt;
      const double current_dist = follow_progress * follow_total_dist;
      const double new_dist = std::clamp(current_dist + advance, 0.0, follow_total_dist);
      follow_progress = static_cast<float>(new_dist / follow_total_dist);
      if (follow_progress >= 1.0f || follow_progress <= 0.0f) {
        follow_speed_kmh = 0.0f;
        follow_playing = false;
      }
    }

    // Mouse drag for turret rotation (right-click drag — smooth, no UI conflict)
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Right, 1.0f)) {
      const ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right, 1.0f);
      follow_yaw_offset -= delta.x * 0.3f;
      follow_pitch_offset -= delta.y * 0.3f;
      follow_pitch_offset = std::clamp(follow_pitch_offset, -80.0f, 80.0f);
      ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
    }
    // Return to forward only when right mouse is fully released
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      follow_yaw_offset *= 0.95f;
      follow_pitch_offset *= 0.95f;
      if (std::abs(follow_yaw_offset) < 0.1f) follow_yaw_offset = 0.0f;
      if (std::abs(follow_pitch_offset) < 0.1f) follow_pitch_offset = 0.0f;
    }

    // Interpolate pose at current progress using Catmull-Rom spline
    const double target_dist = follow_progress * follow_total_dist;
    size_t idx = 0;
    for (size_t k = 1; k < trajectory_data.size(); k++) {
      if (trajectory_data[k].cumulative_dist >= target_dist) { idx = k - 1; break; }
      if (k == trajectory_data.size() - 1) idx = k - 1;
    }
    const size_t next = std::min(idx + 1, trajectory_data.size() - 1);

    // Catmull-Rom: use 4 control points (p0, p1, p2, p3)
    const size_t i0 = (idx > 0) ? idx - 1 : 0;
    const size_t i3 = std::min(next + 1, trajectory_data.size() - 1);
    const Eigen::Vector3d p0 = trajectory_data[i0].pose.translation();
    const Eigen::Vector3d p1 = trajectory_data[idx].pose.translation();
    const Eigen::Vector3d p2 = trajectory_data[next].pose.translation();
    const Eigen::Vector3d p3 = trajectory_data[i3].pose.translation();

    const double seg_len = trajectory_data[next].cumulative_dist - trajectory_data[idx].cumulative_dist;
    const double t = (seg_len > 0.001) ? (target_dist - trajectory_data[idx].cumulative_dist) / seg_len : 0.0;
    const double t2 = t * t, t3 = t2 * t;

    // Catmull-Rom spline interpolation (standard tau=0.5)
    const Eigen::Vector3d pos = 0.5 * (
      (2.0 * p1) +
      (-p0 + p2) * t +
      (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
      (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);

    // Heading from spline tangent (derivative of Catmull-Rom)
    Eigen::Vector3d heading = 0.5 * (
      (-p0 + p2) +
      (4.0 * p0 - 10.0 * p1 + 8.0 * p2 - 2.0 * p3) * t +
      (-3.0 * p0 + 9.0 * p1 - 9.0 * p2 + 3.0 * p3) * t2);
    if (heading.norm() < 0.001) heading = (p2 - p1).normalized();
    else heading.normalize();

    // Base yaw/pitch from spline tangent
    float base_yaw = std::atan2(heading.y(), heading.x()) * 180.0f / M_PI;
    float base_pitch = std::asin(std::clamp(heading.z(), -1.0, 1.0)) * 180.0f / M_PI;

    // Apply turret offset
    float target_yaw = base_yaw + follow_yaw_offset;
    float target_pitch = std::clamp(base_pitch + follow_pitch_offset, -89.0f, 89.0f);

    // Exponential smoothing for suspension-like feel
    const double smooth_pos = static_cast<double>(follow_smoothness);
    const double smooth_rot = static_cast<double>(follow_smoothness * 1.25f);
    if (!follow_smooth_init) {
      follow_smooth_pos = pos;
      follow_smooth_yaw = target_yaw;
      follow_smooth_pitch = target_pitch;
      follow_smooth_init = true;
    } else {
      follow_smooth_pos += (pos - follow_smooth_pos) * smooth_pos;
      // Smooth yaw with wrap-around handling
      float yaw_diff = target_yaw - follow_smooth_yaw;
      if (yaw_diff > 180.0f) yaw_diff -= 360.0f;
      if (yaw_diff < -180.0f) yaw_diff += 360.0f;
      follow_smooth_yaw += yaw_diff * static_cast<float>(smooth_rot);
      follow_smooth_pitch += (target_pitch - follow_smooth_pitch) * static_cast<float>(smooth_rot);
    }

    // Measure actual camera speed from smoothed position change
    static Eigen::Vector3d prev_smooth_pos = follow_smooth_pos;
    if (dt > 0.001) {
      follow_actual_speed_ms = (follow_smooth_pos - prev_smooth_pos).norm() / dt;
    }
    prev_smooth_pos = follow_smooth_pos;

    // Update FPS camera with smoothed values
    auto fps = std::dynamic_pointer_cast<guik::FPSCameraControl>(
      guik::LightViewer::instance()->get_camera_control());
    if (fps) {
      fps->set_pose(follow_smooth_pos.cast<float>(), follow_smooth_yaw, follow_smooth_pitch);
    }

    // Overlay HUD
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 95), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(500, 82), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.6f);
    if (ImGui::Begin("Follow Trajectory", nullptr,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings)) {
      float pct = follow_progress * 100.0f;
      ImGui::SetNextItemWidth(-1);
      if (ImGui::SliderFloat("##progress", &pct, 0.0f, 100.0f, "%.1f%%")) {
        follow_progress = pct / 100.0f;
      }
      const double actual_kmh = follow_actual_speed_ms * 3.6;
      ImGui::Text("%.0f km/h (actual %.0f)  |  %.0f / %.0f m  |  %s",
                   follow_speed_kmh, actual_kmh,
                   follow_progress * follow_total_dist,
                   follow_total_dist,
                   follow_playing ? "Playing" : "Paused");
      ImGui::TextDisabled("Space=play/pause  W/S=speed  RMB=look around");
    }
    ImGui::End();
  });

  // PatchWork++ config window
  viewer->register_ui_callback("pw_config_window", [this] {
    if (!show_pw_config) return;
    ImGui::SetNextWindowSize(ImVec2(300, 500), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("PatchWork++ Config", &show_pw_config)) {
      auto& p = glim::MapCleanerFilter::getPatchWorkParams();
      ImGui::Checkbox("Use intensity (RNR)", &p.enable_RNR);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reflected Noise Removal using intensity.\nRequires intensities.bin in HD frames.");
      ImGui::Checkbox("Enable RVPF", &p.enable_RVPF);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Region-wise Vertical Plane Fitting.");
      ImGui::Checkbox("Enable TGR", &p.enable_TGR);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Temporal Ground Revert.");
      ImGui::Separator();
      float sh = static_cast<float>(p.sensor_height);
      if (ImGui::DragFloat("Sensor height (m)", &sh, 0.01f, 0.5f, 5.0f, "%.3f")) p.sensor_height = sh;
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Height of LiDAR sensor above ground.");
      float mr = static_cast<float>(p.max_range);
      if (ImGui::DragFloat("Max range (m)", &mr, 1.0f, 10.0f, 200.0f, "%.0f")) p.max_range = mr;
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Points beyond this range are ignored.");
      float mnr = static_cast<float>(p.min_range);
      if (ImGui::DragFloat("Min range (m)", &mnr, 0.1f, 0.5f, 10.0f, "%.1f")) p.min_range = mnr;
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Points closer than this range are ignored.");
      ImGui::Separator();
      float ts = static_cast<float>(p.th_seeds);
      if (ImGui::DragFloat("Seed threshold", &ts, 0.01f, 0.01f, 1.0f, "%.3f")) p.th_seeds = ts;
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Threshold for initial seed selection.");
      float td = static_cast<float>(p.th_dist);
      if (ImGui::DragFloat("Ground thickness", &td, 0.01f, 0.01f, 1.0f, "%.3f")) p.th_dist = td;
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max distance from plane to count as ground.");
      float ut = static_cast<float>(p.uprightness_thr);
      if (ImGui::DragFloat("Uprightness thr", &ut, 0.01f, 0.3f, 1.0f, "%.3f")) p.uprightness_thr = ut;
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("How upright the surface must be (0.707 = 45 deg).");
      ImGui::DragInt("Num iterations", &p.num_iter, 1, 1, 10);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Ground plane fitting iterations per patch.");
      ImGui::DragInt("Num LPR", &p.num_lpr, 1, 5, 50);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max lowest points for seed selection.");
      ImGui::DragInt("Min points/patch", &p.num_min_pts, 1, 1, 50);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Min points to estimate ground in a patch.");
      ImGui::Separator();
      if (ImGui::Button("Reset defaults##pw")) {
        p.enable_RNR = false;
        p.enable_RVPF = true;
        p.enable_TGR = true;
        p.sensor_height = 1.723;
        p.max_range = 80.0;
        p.min_range = 2.0;
        p.th_seeds = 0.125;
        p.th_dist = 0.125;
        p.uprightness_thr = 0.707;
        p.num_iter = 3;
        p.num_lpr = 20;
        p.num_min_pts = 10;
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Restore all PatchWork++ parameters to defaults.");
      ImGui::Separator();
      ImGui::Checkbox("Frame accumulation", &pw_accumulate);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Accumulate neighboring frames before running PatchWork++.\nGives much denser input for better ground classification.\nPoints from neighbors are transformed to current frame's sensor-local coords.");
      if (pw_accumulate) {
        ImGui::DragInt("Prior/next frames", &pw_accumulate_count, 1, 1, 50);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of neighboring frames to include.\nAt the start of the dataset, uses next frames instead.");
      }
    }
    ImGui::End();
  });

  // Trail refinement config window
  // Voxelize HD tool window
  viewer->register_ui_callback("voxelize_tool", [this] {
    if (!show_voxelize_tool) return;
    ImGui::SetNextWindowSize(ImVec2(300, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Voxelize HD Data", &show_voxelize_tool)) {
      ImGui::DragFloat("Voxel size (m)", &vox_size, 0.005f, 0.005f, 0.5f, "%.3f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Size of each voxel cell.\n0.01-0.03m for 3DGS, 0.05-0.10m for visualization.");
      ImGui::Combo("Placement", &vox_mode, "Voxel center\0Weighted\0XY grid + Z weighted\0");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Voxel center: regular 3D grid (staircase on slopes).\nWeighted: centroid of points (smooth, irregular).\nXY grid + Z weighted: regular XY, smooth Z (best for 3DGS).");
      vox_use_center = (vox_mode == 0);

      // Ground-only mode: check if aux_ground.bin exists for at least one frame
      bool has_ground_bin = false;
      if (!hd_frames_path.empty()) {
        // Quick check: test first frame dir that exists
        for (const auto& submap : submaps) {
          if (!submap || submap->frames.empty()) continue;
          std::string shd = hd_frames_path;
          for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
          char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", submap->frames.front()->id);
          if (boost::filesystem::exists(shd + "/" + dn + "/aux_ground.bin")) { has_ground_bin = true; break; }
        }
      }
      if (!has_ground_bin) {
        ImGui::BeginDisabled();
        vox_ground_only = false;
      }
      ImGui::Checkbox("Ground only (1 pt/XY)", &vox_ground_only);
      if (!has_ground_bin) {
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
          ImGui::SetTooltip("Requires aux_ground.bin per frame.\nGenerate with Data Filter > Dynamic > Classify ground to scalar.");
      } else {
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("Keep only ground-classified points.\nOne point per XY column — removes ground noise.\nRequires aux_ground.bin from Dynamic filter.");
      }
      if (vox_ground_only) {
        ImGui::SameLine();
        ImGui::TextDisabled("(forces XY+Z weighted)");
      }

      ImGui::DragFloat("Chunk size (m)", &vox_chunk_size, 5.0f, 20.0f, 200.0f, "%.0f");
      ImGui::DragFloat("Chunk spacing (m)", &vox_chunk_spacing, 5.0f, 10.0f, 100.0f, "%.0f");
      ImGui::Separator();

      if (vox_processing) {
        ImGui::Text("%s", vox_status.c_str());
      } else {
        // Preview: voxelize one chunk at camera position
        if (ImGui::Button("Preview")) {
          vox_processing = true;
          vox_status = "Loading...";
          std::thread([this] {
            auto vw = guik::LightViewer::instance();
            const Eigen::Matrix4f vm = vw->view_matrix();
            const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));

            if (!trajectory_built) build_trajectory();
            // Find nearest trajectory point
            double min_d = 1e9;
            size_t best_idx = 0;
            for (size_t k = 0; k < trajectory_data.size(); k++) {
              const double d = (trajectory_data[k].pose.translation().cast<float>() - cam_pos).cast<double>().norm();
              if (d < min_d) { min_d = d; best_idx = k; }
            }
            // Build one chunk
            const Eigen::Vector3d c = trajectory_data[best_idx].pose.translation();
            const size_t next = std::min(best_idx + 1, trajectory_data.size() - 1);
            Eigen::Vector3d fwd = trajectory_data[next].pose.translation() - c;
            fwd.z() = 0; if (fwd.norm() < 0.01) fwd = Eigen::Vector3d::UnitX(); else fwd.normalize();
            const Eigen::Vector3d up = Eigen::Vector3d::UnitZ(), right = fwd.cross(up).normalized();
            Eigen::Matrix3d R; R.col(0) = fwd; R.col(1) = right; R.col(2) = up;
            glim::Chunk chunk{c, R, R.transpose(), vox_chunk_size * 0.5, 50.0};
            const auto chunk_aabb = chunk.world_aabb();

            // Load all frames overlapping chunk
            vox_status = "Loading frames...";
            const float inv_vox = 1.0f / vox_size;
            std::unordered_map<uint64_t, std::vector<std::pair<Eigen::Vector3f, float>>> voxel_data;  // key → (pos, intensity)
            int total_input = 0;

            const bool ground_only = vox_ground_only;
            for (const auto& submap : submaps) {
              if (!submap) continue;
              if (hidden_sessions.count(submap->session_id)) continue;
              std::string shd = hd_frames_path;
              for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
              const Eigen::Isometry3d T0 = submap->frames.front()->T_world_imu;
              for (const auto& fr : submap->frames) {
                char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                const std::string fd = shd + "/" + dn;
                auto fi = glim::frame_info_from_meta(fd,
                  glim::compute_frame_world_pose(submap->T_world_origin, submap->T_origin_endpoint_L, T0, fr->T_world_imu, fr->T_lidar_imu));
                if (fi.num_points == 0 || !chunk_aabb.intersects(fi.world_bbox)) continue;
                std::vector<Eigen::Vector3f> pts; std::vector<float> rng, ints(fi.num_points, 0.0f);
                if (!glim::load_bin(fd + "/points.bin", pts, fi.num_points)) continue;
                glim::load_bin(fd + "/range.bin", rng, fi.num_points);
                glim::load_bin(fd + "/intensities.bin", ints, fi.num_points);
                // Load ground mask if ground-only mode
                std::vector<float> ground;
                if (ground_only) glim::load_bin(fd + "/aux_ground.bin", ground, fi.num_points);
                const Eigen::Matrix3f Rf = fi.T_world_lidar.rotation().cast<float>();
                const Eigen::Vector3f t = fi.T_world_lidar.translation().cast<float>();
                for (int i = 0; i < fi.num_points; i++) {
                  if (!rng.empty() && rng[i] < 1.5f) continue;
                  if (ground_only && (ground.empty() || ground[i] < 0.5f)) continue;
                  const Eigen::Vector3f wp = Rf * pts[i] + t;
                  if (!chunk.contains(wp)) continue;
                  // Ground-only: XY-only key (one point per column); normal: full 3D key
                  const uint64_t key = ground_only ? glim::voxel_key(
                    static_cast<int>(std::floor(wp.x() * inv_vox)),
                    static_cast<int>(std::floor(wp.y() * inv_vox)), 0)
                    : glim::voxel_key(wp, inv_vox);
                  voxel_data[key].push_back({wp, ints[i]});
                  total_input++;
                }
              }
            }

            logger->info("[Voxelize preview] {} input points → {} voxels (size={:.3f}m, ground_only={})", total_input, voxel_data.size(), vox_size, ground_only);
            vox_status = "Voxelizing " + std::to_string(voxel_data.size()) + " voxels from " + std::to_string(total_input) + " points...";

            // Build voxelized output
            std::vector<Eigen::Vector3f> out_pts;
            std::vector<float> out_ints;
            for (const auto& [key, pts_in_voxel] : voxel_data) {
              Eigen::Vector3f pos;
              if (ground_only) {
                // Ground-only: XY grid center + Z weighted (always, regardless of vox_mode)
                const int vx = static_cast<int>((key >> 42) & 0x1FFFFF) - 1048576;
                const int vy = static_cast<int>((key >> 21) & 0x1FFFFF) - 1048576;
                float avg_z = 0.0f;
                for (const auto& p : pts_in_voxel) avg_z += p.first.z();
                avg_z /= static_cast<float>(pts_in_voxel.size());
                pos = Eigen::Vector3f((vx + 0.5f) * vox_size, (vy + 0.5f) * vox_size, avg_z);
              } else if (vox_mode == 0) {
                // Full voxel center
                const int vx = static_cast<int>((key >> 42) & 0x1FFFFF) - 1048576;
                const int vy = static_cast<int>((key >> 21) & 0x1FFFFF) - 1048576;
                const int vz = static_cast<int>(key & 0x1FFFFF) - 1048576;
                pos = Eigen::Vector3f((vx + 0.5f) * vox_size, (vy + 0.5f) * vox_size, (vz + 0.5f) * vox_size);
              } else if (vox_mode == 1) {
                // Full weighted centroid
                pos = Eigen::Vector3f::Zero();
                for (const auto& p : pts_in_voxel) pos += p.first;
                pos /= static_cast<float>(pts_in_voxel.size());
              } else {
                // XY grid center + Z weighted
                const int vx = static_cast<int>((key >> 42) & 0x1FFFFF) - 1048576;
                const int vy = static_cast<int>((key >> 21) & 0x1FFFFF) - 1048576;
                float avg_z = 0.0f;
                for (const auto& p : pts_in_voxel) avg_z += p.first.z();
                avg_z /= static_cast<float>(pts_in_voxel.size());
                pos = Eigen::Vector3f((vx + 0.5f) * vox_size, (vy + 0.5f) * vox_size, avg_z);
              }
              // Average intensity
              float avg_int = 0.0f;
              for (const auto& p : pts_in_voxel) avg_int += p.second;
              avg_int /= static_cast<float>(pts_in_voxel.size());
              out_pts.push_back(pos);
              out_ints.push_back(avg_int);
            }

            // Render preview
            vw->invoke([this, out_pts, out_ints] {
              auto v = guik::LightViewer::instance();
              lod_hide_all_submaps = true;
              v->remove_drawable("rf_preview_kept");
              v->remove_drawable("rf_preview_removed");
              if (!out_pts.empty()) {
                const int n = out_pts.size();
                std::vector<Eigen::Vector4d> p4(n);
                for (int i = 0; i < n; i++) p4[i] = Eigen::Vector4d(out_pts[i].x(), out_pts[i].y(), out_pts[i].z(), 1.0);
                auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), n);
                cb->add_buffer("intensity", out_ints);
                cb->set_colormap_buffer("intensity");
                v->update_drawable("rf_preview_kept", cb, guik::FlatColor(0.0f, 0.8f, 0.2f, 1.0f));
              }
            });

            char buf[256];
            std::snprintf(buf, sizeof(buf), "Preview: %zu voxels from %d input points (%.1fx reduction)",
              out_pts.size(), total_input, total_input > 0 ? static_cast<double>(total_input) / out_pts.size() : 0.0);
            vox_status = buf;
            logger->info("[Voxelize] {}", vox_status);
            vox_processing = false;
          }).detach();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Preview voxelization at current camera position.\nShows one chunk of voxelized data.");

        ImGui::SameLine();
        if (ImGui::Button("Intensity##vox")) {
          auto vw = guik::LightViewer::instance();
          auto drawable = vw->find_drawable("rf_preview_kept");
          if (drawable.first) {
            static bool vox_intensity_mode = false;
            vox_intensity_mode = !vox_intensity_mode;
            if (vox_intensity_mode) {
              drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLORMAP);
            } else {
              drawable.first->set_color_mode(guik::ColorMode::FLAT_COLOR);
            }
          }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle intensity colormap on preview.");
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
          auto v = guik::LightViewer::instance();
          v->remove_drawable("rf_preview_kept");
          v->remove_drawable("rf_preview_removed");
          lod_hide_all_submaps = false;
          vox_status.clear();
        }

        ImGui::Separator();

        // Apply to full dataset
        if (ImGui::Button("Apply to all HD")) {
          vox_processing = true;
          vox_status = "Starting voxelization...";
          std::thread([this] {
            if (!trajectory_built) build_trajectory();
            const auto start_time = std::chrono::steady_clock::now();
            const bool ground_only = vox_ground_only;
            auto chunks = glim::build_chunks(trajectory_data, trajectory_total_dist, vox_chunk_spacing, vox_chunk_size * 0.5);
            logger->info("[Voxelize] {} chunks along {:.0f}m trajectory (ground_only={})", chunks.size(), trajectory_total_dist, ground_only);

            // Index all frames
            std::vector<glim::FrameInfo> all_frames;
            for (const auto& submap : submaps) {
              if (!submap) continue;
              if (hidden_sessions.count(submap->session_id)) continue;
              std::string shd = hd_frames_path;
              for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
              const Eigen::Isometry3d T0 = submap->frames.front()->T_world_imu;
              for (const auto& fr : submap->frames) {
                char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                auto fi = glim::frame_info_from_meta(shd + "/" + dn,
                  glim::compute_frame_world_pose(submap->T_world_origin, submap->T_origin_endpoint_L, T0, fr->T_world_imu, fr->T_lidar_imu),
                  submap->id, submap->session_id);
                if (fi.num_points > 0) all_frames.push_back(std::move(fi));
              }
            }

            // Create output directory
            const std::string vox_dir = hd_frames_path + (ground_only ? "_ground" : "_voxelized");
            boost::filesystem::create_directories(vox_dir);

            // Per-frame output buffers: accumulate voxelized points assigned to each frame
            struct FrameOutput {
              std::vector<Eigen::Vector3f> points;
              std::vector<float> intensities;
              std::vector<float> ranges;
            };
            std::unordered_map<std::string, FrameOutput> frame_outputs;  // frame_dir → output
            // Initialize empty outputs for all frames
            for (const auto& fi : all_frames) frame_outputs[fi.dir] = {};

            const float inv_vox = 1.0f / vox_size;
            size_t total_voxels = 0;

            // Frame cache: loaded world-space points per frame (sliding window)
            struct CachedFrame {
              std::vector<Eigen::Vector3f> world_pts;
              std::vector<float> intensities;
              std::vector<float> ranges;
              std::string dir;
            };
            std::unordered_map<std::string, std::shared_ptr<CachedFrame>> frame_cache;

            for (size_t ci = 0; ci < chunks.size(); ci++) {
              const auto& chunk = chunks[ci];
              const auto chunk_aabb = chunk.world_aabb();
              glim::Chunk core = chunk;
              core.half_size = vox_chunk_size * 0.5;

              if (ci % 5 == 0) {
                char buf[256]; std::snprintf(buf, sizeof(buf), "Chunk %zu/%zu (cache: %zu frames)...", ci + 1, chunks.size(), frame_cache.size());
                vox_status = buf;
              }

              // Determine which frames overlap this chunk
              std::vector<const glim::FrameInfo*> chunk_frame_infos;
              std::unordered_set<std::string> needed_dirs;
              for (const auto& fi : all_frames) {
                if (fi.num_points == 0 || !chunk_aabb.intersects(fi.world_bbox)) continue;
                chunk_frame_infos.push_back(&fi);
                needed_dirs.insert(fi.dir);
              }

              // Evict frames no longer needed
              std::vector<std::string> evict_keys;
              for (const auto& [dir, _] : frame_cache) {
                if (!needed_dirs.count(dir)) evict_keys.push_back(dir);
              }
              for (const auto& k : evict_keys) frame_cache.erase(k);

              // Load missing frames into cache
              for (const auto* fi : chunk_frame_infos) {
                if (frame_cache.count(fi->dir)) continue;
                std::vector<Eigen::Vector3f> pts; std::vector<float> rng, ints(fi->num_points, 0.0f);
                if (!glim::load_bin(fi->dir + "/points.bin", pts, fi->num_points)) continue;
                glim::load_bin(fi->dir + "/range.bin", rng, fi->num_points);
                glim::load_bin(fi->dir + "/intensities.bin", ints, fi->num_points);
                // Load ground mask if ground-only mode
                std::vector<float> ground;
                if (ground_only) glim::load_bin(fi->dir + "/aux_ground.bin", ground, fi->num_points);
                const Eigen::Matrix3f R = fi->T_world_lidar.rotation().cast<float>();
                const Eigen::Vector3f t = fi->T_world_lidar.translation().cast<float>();
                auto cf = std::make_shared<CachedFrame>();
                cf->dir = fi->dir;
                for (int i = 0; i < fi->num_points; i++) {
                  if (!rng.empty() && rng[i] < 1.5f) continue;
                  if (ground_only && (ground.empty() || ground[i] < 0.5f)) continue;
                  cf->world_pts.push_back(R * pts[i] + t);
                  cf->intensities.push_back(ints[i]);
                  cf->ranges.push_back(rng.empty() ? 0.0f : rng[i]);
                }
                frame_cache[fi->dir] = cf;
              }

              // Build voxel grid from cached frames
              struct VoxPt { Eigen::Vector3f wp; float intensity; float range; std::string dir; };
              std::unordered_map<uint64_t, std::vector<VoxPt>> voxels;
              for (const auto* fi : chunk_frame_infos) {
                auto it = frame_cache.find(fi->dir);
                if (it == frame_cache.end()) continue;
                const auto& cf = it->second;
                for (size_t i = 0; i < cf->world_pts.size(); i++) {
                  if (!chunk.contains(cf->world_pts[i])) continue;
                  const uint64_t key = ground_only ? glim::voxel_key(
                    static_cast<int>(std::floor(cf->world_pts[i].x() * inv_vox)),
                    static_cast<int>(std::floor(cf->world_pts[i].y() * inv_vox)), 0)
                    : glim::voxel_key(cf->world_pts[i], inv_vox);
                  voxels[key].push_back({cf->world_pts[i], cf->intensities[i], cf->ranges[i], cf->dir});
                }
              }

              // Process only core area voxels — assign to frames round-robin
              int voxel_idx = 0;
              // Collect contributing frame dirs for round-robin
              std::vector<std::string> contributing_dirs;
              for (const auto& fi : all_frames) {
                if (fi.num_points > 0 && chunk_aabb.intersects(fi.world_bbox)) {
                  contributing_dirs.push_back(fi.dir);
                }
              }

              for (const auto& [key, pts_in_voxel] : voxels) {
                // Compute voxel position
                Eigen::Vector3f pos;
                if (ground_only) {
                  // Ground-only: XY grid center + Z weighted average
                  const int vx = static_cast<int>((key >> 42) & 0x1FFFFF) - 1048576;
                  const int vy = static_cast<int>((key >> 21) & 0x1FFFFF) - 1048576;
                  float avg_z = 0.0f;
                  for (const auto& p : pts_in_voxel) avg_z += p.wp.z();
                  avg_z /= static_cast<float>(pts_in_voxel.size());
                  pos = Eigen::Vector3f((vx + 0.5f) * vox_size, (vy + 0.5f) * vox_size, avg_z);
                } else if (vox_mode == 0) {
                  const int vx = static_cast<int>((key >> 42) & 0x1FFFFF) - 1048576;
                  const int vy = static_cast<int>((key >> 21) & 0x1FFFFF) - 1048576;
                  const int vz = static_cast<int>(key & 0x1FFFFF) - 1048576;
                  pos = Eigen::Vector3f((vx + 0.5f) * vox_size, (vy + 0.5f) * vox_size, (vz + 0.5f) * vox_size);
                } else if (vox_mode == 1) {
                  pos = Eigen::Vector3f::Zero();
                  for (const auto& p : pts_in_voxel) pos += p.wp;
                  pos /= static_cast<float>(pts_in_voxel.size());
                } else {
                  const int vx = static_cast<int>((key >> 42) & 0x1FFFFF) - 1048576;
                  const int vy = static_cast<int>((key >> 21) & 0x1FFFFF) - 1048576;
                  float avg_z = 0.0f;
                  for (const auto& p : pts_in_voxel) avg_z += p.wp.z();
                  avg_z /= static_cast<float>(pts_in_voxel.size());
                  pos = Eigen::Vector3f((vx + 0.5f) * vox_size, (vy + 0.5f) * vox_size, avg_z);
                }

                // Only include core area voxels
                if (!core.contains(pos)) continue;

                // Average attributes
                float avg_int = 0.0f, avg_rng = 0.0f;
                for (const auto& p : pts_in_voxel) { avg_int += p.intensity; avg_rng += p.range; }
                avg_int /= pts_in_voxel.size(); avg_rng /= pts_in_voxel.size();

                // Assign to a frame — round-robin across contributing frames
                const std::string& target_dir = contributing_dirs[voxel_idx % contributing_dirs.size()];
                frame_outputs[target_dir].points.push_back(pos);
                frame_outputs[target_dir].intensities.push_back(avg_int);
                frame_outputs[target_dir].ranges.push_back(avg_rng);
                voxel_idx++;
                total_voxels++;
              }
            }

            // Write output frames
            vox_status = "Writing voxelized frames...";
            int frames_written = 0;
            for (const auto& [src_dir, output] : frame_outputs) {
              if (output.points.empty()) continue;
              // Derive output dir from source dir
              const std::string dirname = boost::filesystem::path(src_dir).filename().string();
              const std::string out_dir = vox_dir + "/" + dirname;
              boost::filesystem::create_directories(out_dir);

              const int n = output.points.size();
              // Write points as sensor-local (identity transform — points are already in world space)
              // For the frame structure, store world-space points directly
              { std::ofstream f(out_dir + "/points.bin", std::ios::binary);
                f.write(reinterpret_cast<const char*>(output.points.data()), sizeof(Eigen::Vector3f) * n); }
              { std::ofstream f(out_dir + "/range.bin", std::ios::binary);
                f.write(reinterpret_cast<const char*>(output.ranges.data()), sizeof(float) * n); }
              { std::ofstream f(out_dir + "/intensities.bin", std::ios::binary);
                f.write(reinterpret_cast<const char*>(output.intensities.data()), sizeof(float) * n); }

              // Write frame_meta.json with identity transform (points are world-space)
              { std::ofstream ofs(out_dir + "/frame_meta.json");
                ofs << std::setprecision(15) << std::fixed;
                ofs << "{\n  \"num_points\": " << n << ",\n";
                ofs << "  \"stamp\": 0.0,\n";
                ofs << "  \"scan_end_time\": 0.0,\n";
                // Identity T_world_lidar (points already in world frame)
                ofs << "  \"T_world_lidar\": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],\n";
                // Compute bbox
                Eigen::Vector3f bmin = output.points[0], bmax = output.points[0];
                for (const auto& p : output.points) { bmin = bmin.cwiseMin(p); bmax = bmax.cwiseMax(p); }
                ofs << "  \"bbox_world_min\": [" << bmin.x() << "," << bmin.y() << "," << bmin.z() << "],\n";
                ofs << "  \"bbox_world_max\": [" << bmax.x() << "," << bmax.y() << "," << bmax.z() << "]\n";
                ofs << "}\n";
              }
              frames_written++;
            }

            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            char buf[256];
            std::snprintf(buf, sizeof(buf), "Done: %zu voxels, %d frames written to %s (%.1f sec)",
              total_voxels, frames_written, vox_dir.c_str(), elapsed);
            vox_status = buf;
            logger->info("[Voxelize] {}", vox_status);
            vox_processing = false;
          }).detach();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Voxelize all HD frames along trajectory.\nWrites to hd_frames_voxelized/ folder.\nOriginal HD data is preserved.");

        if (!vox_status.empty()) ImGui::TextWrapped("%s", vox_status.c_str());
      }
    }
    ImGui::End();
  });

  viewer->register_ui_callback("trail_config_window", [this] {
    if (!show_trail_config) return;
    ImGui::SetNextWindowSize(ImVec2(250, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Trail Refinement Config", &show_trail_config)) {
      ImGui::DragFloat("Refine voxel (m)", &df_refine_voxel, 0.05f, 0.1f, 5.0f, "%.2f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Voxel size for clustering candidates.");
      ImGui::DragFloat("Min length (m)", &df_trail_min_length, 1.0f, 2.0f, 100.0f, "%.0f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum trail extent in longest axis.");
      ImGui::DragFloat("Min aspect ratio", &df_trail_min_aspect, 0.5f, 1.0f, 20.0f, "%.1f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum longest/shortest axis ratio.\nTrails are elongated (>3).");
      ImGui::DragFloat("Min density (pts/m³)", &df_trail_min_density, 1.0f, 1.0f, 500.0f, "%.0f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum point density in occupied voxels.");
    }
    ImGui::End();
  });

  // Colorize context menu items (injected into base class popup via extension point)
  extra_context_menu_items = [this] {
    const PickType type = static_cast<PickType>(right_clicked_info[0]);

    // Camera right-click
    if (type == PickType::CAMERA) {
      const int src_idx = right_clicked_info[1];
      const int frame_idx = right_clicked_info[3];
      if (src_idx >= 0 && src_idx < static_cast<int>(image_sources.size()) &&
          frame_idx >= 0 && frame_idx < static_cast<int>(image_sources[src_idx].frames.size())) {
        const auto& frame = image_sources[src_idx].frames[frame_idx];
        const std::string fname = boost::filesystem::path(frame.filepath).filename().string();
        ImGui::TextUnformatted(fname.c_str());
        ImGui::TextDisabled("Source: %s", image_sources[src_idx].name.c_str());
        if (frame.timestamp > 0.0) {
          char ts_buf[64]; std::snprintf(ts_buf, sizeof(ts_buf), "Time: %.3f", frame.timestamp);
          ImGui::TextDisabled("%s", ts_buf);
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Open image (in-app)")) {
          auto img = cv::imread(frame.filepath);
          if (!img.empty()) {
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            image_original_w = img.cols; image_original_h = img.rows;
            if (img.cols > 1920) { const double s = 1920.0 / img.cols; cv::resize(img, img, cv::Size(), s, s); }
            if (image_viewer_texture) glDeleteTextures(1, &image_viewer_texture);
            glGenTextures(1, &image_viewer_texture);
            glBindTexture(GL_TEXTURE_2D, image_viewer_texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);
            glBindTexture(GL_TEXTURE_2D, 0);
            image_viewer_w = img.cols; image_viewer_h = img.rows;
            image_viewer_title = fname;
            show_image_viewer = true;
          }
        }
        if (ImGui::MenuItem("Check alignment")) {
          align_cam_src = src_idx;
          align_cam_idx = frame_idx;
          align_loaded_path.clear();  // force reload if window was already open with another image
          align_last_submap_id = -1;  // force submap point cache refresh
          align_show = true;
        }
        if (ImGui::MenuItem("Colorize from this camera")) {
          colorize_last_cam_src = src_idx; colorize_last_cam_idx = frame_idx; colorize_last_submap = -1;
          // Highlight this camera in yellow
          if (frame.located) {
            auto vw = guik::LightViewer::instance();
            const Eigen::Vector3f hp = frame.T_world_cam.translation().cast<float>();
            Eigen::Affine3f hbtf = Eigen::Affine3f::Identity(); hbtf.translate(hp);
            hbtf.linear() = frame.T_world_cam.rotation().cast<float>();
            hbtf = hbtf * Eigen::Scaling(Eigen::Vector3f(0.12f, 0.18f, 0.12f));
            vw->update_drawable("cam_" + std::to_string(src_idx) + "_" + std::to_string(frame_idx),
              glk::Primitives::cube(), guik::FlatColor(1.0f, 1.0f, 0.0f, 0.9f, hbtf).add("info_values",
                Eigen::Vector4i(static_cast<int>(PickType::CAMERA), src_idx, 0, frame_idx)));
          }
          if (frame.located && frame.timestamp > 0.0) {
            // Find submap by timestamp — the submap whose frames bracket the camera's time
            const double cam_time = frame.timestamp + image_sources[src_idx].time_shift;
            int best_sm = -1; double best_dt = 1e9;
            for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
              if (!submaps[si] || submaps[si]->frames.empty()) continue;
              const double t_first = submaps[si]->frames.front()->stamp;
              const double t_last = submaps[si]->frames.back()->stamp;
              // Camera time within submap's time range → perfect match
              if (cam_time >= t_first && cam_time <= t_last) { best_sm = si; break; }
              // Otherwise find closest boundary
              const double dt = std::min(std::abs(cam_time - t_first), std::abs(cam_time - t_last));
              if (dt < best_dt) { best_dt = dt; best_sm = si; }
            }
            if (best_sm >= 0) {
              logger->info("[Colorize] Projecting from camera {} onto submap {}", fname, best_sm);
              // Ensure mask is loaded
              if (colorize_mask.empty() && !image_sources.empty()) {
                for (const auto& ms : image_sources) {
                  if (ms.path.empty()) continue;
                  const std::string mp = ms.path + "/mask.png";
                  if (boost::filesystem::exists(mp)) { colorize_mask = cv::imread(mp, cv::IMREAD_UNCHANGED); break; }
                }
              }
              // Load only 1-2 nearest HD frames (not the full submap)
              const auto& sm = submaps[best_sm];
              const auto hd_it = session_hd_paths.find(sm->session_id);
              const Eigen::Isometry3d T_ep = sm->T_world_origin * sm->T_origin_endpoint_L;
              const Eigen::Isometry3d T_odom0 = sm->frames.front()->T_world_imu;
              std::vector<Eigen::Vector3f> world_pts;
              std::vector<float> ints;
              if (hd_it != session_hd_paths.end()) {
                // Find the 2 frames closest in time to this camera
                std::vector<std::pair<double, size_t>> frame_dists;
                for (size_t fi2 = 0; fi2 < sm->frames.size(); fi2++) {
                  frame_dists.push_back({std::abs(sm->frames[fi2]->stamp - cam_time), fi2});
                }
                std::sort(frame_dists.begin(), frame_dists.end());
                const int max_frames = std::min(2, static_cast<int>(frame_dists.size()));
                for (int nf = 0; nf < max_frames; nf++) {
                  const auto& fr = sm->frames[frame_dists[nf].second];
                  char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                  const std::string fd = hd_it->second + "/" + dn;
                  std::vector<Eigen::Vector3f> pts; std::vector<float> rng, fi_ints;
                  auto fi_info = glim::frame_info_from_meta(fd,
                    glim::compute_frame_world_pose(sm->T_world_origin, sm->T_origin_endpoint_L, T_odom0, fr->T_world_imu, fr->T_lidar_imu));
                  if (fi_info.num_points == 0) continue;
                  if (!glim::load_bin(fd + "/points.bin", pts, fi_info.num_points)) continue;
                  glim::load_bin(fd + "/range.bin", rng, fi_info.num_points);
                  glim::load_bin(fd + "/intensities.bin", fi_ints, fi_info.num_points);
                  const Eigen::Matrix3f R = fi_info.T_world_lidar.rotation().cast<float>();
                  const Eigen::Vector3f t = fi_info.T_world_lidar.translation().cast<float>();
                  for (int pi = 0; pi < fi_info.num_points; pi++) {
                    const float r = (!rng.empty()) ? rng[pi] : pts[pi].norm();
                    if (r < 1.5f) continue;
                    world_pts.push_back(R * pts[pi] + t);
                    ints.push_back(pi < static_cast<int>(fi_ints.size()) ? fi_ints[pi] : 0.0f);
                  }
                }
                logger->info("[Colorize] Loaded {} points from {} nearby frames", world_pts.size(), max_frames);
              }
              if (!world_pts.empty()) {
                std::vector<CameraFrame> cams = {frame};
                auto cr = Colorizer::project_colors(cams, image_sources[src_idx].intrinsics, world_pts, ints, colorize_max_range, colorize_blend, colorize_min_range, colorize_mask);
                logger->info("[Colorize] {} / {} points colored", cr.colored, cr.total);
                colorize_last_result = cr;
                { auto vw = guik::LightViewer::instance(); lod_hide_all_submaps = true;
                  const size_t n = cr.points.size();
                  float imin = std::numeric_limits<float>::max(), imax = std::numeric_limits<float>::lowest();
                  for (size_t i = 0; i < n && i < cr.intensities.size(); i++) { imin = std::min(imin, cr.intensities[i]); imax = std::max(imax, cr.intensities[i]); }
                  if (imin >= imax) { imin = 0; imax = 255; }
                  std::vector<Eigen::Vector4d> p4(n); std::vector<Eigen::Vector4f> c4(n);
                  for (size_t i = 0; i < n; i++) {
                    p4[i] = Eigen::Vector4d(cr.points[i].x(), cr.points[i].y(), cr.points[i].z(), 1.0);
                    Eigen::Vector3f rgb = cr.colors[i];
                    if (colorize_intensity_blend && i < cr.intensities.size()) {
                      float inv = (cr.intensities[i] - imin) / (imax - imin);
                      if (colorize_nonlinear_int) inv = std::sqrt(inv);
                      rgb = rgb * (1.0f - colorize_intensity_mix) + intensity_to_color(inv) * colorize_intensity_mix;
                    }
                    c4[i] = Eigen::Vector4f(rgb.x(), rgb.y(), rgb.z(), 1.0f);
                  }
                  auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), p4.size()); cb->add_color(c4);
                  vw->update_drawable("colorize_preview", cb, guik::Rainbow().set_color_mode(guik::ColorMode::VERTEX_COLOR)); }
              }
            }
          }
        }
        if (ImGui::MenuItem("Calibrate from this camera")) {
          // Open image and enter calibration mode
          auto img = cv::imread(frame.filepath);
          if (!img.empty()) {
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            image_original_w = img.cols; image_original_h = img.rows;
            if (img.cols > 1920) { const double s = 1920.0 / img.cols; cv::resize(img, img, cv::Size(), s, s); }
            if (image_viewer_texture) glDeleteTextures(1, &image_viewer_texture);
            glGenTextures(1, &image_viewer_texture);
            glBindTexture(GL_TEXTURE_2D, image_viewer_texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);
            glBindTexture(GL_TEXTURE_2D, 0);
            image_viewer_w = img.cols; image_viewer_h = img.rows;
            image_viewer_title = "Calibrate: " + fname;
            show_image_viewer = true;
            calib_active = true;
            calib_cam_src = src_idx;
            calib_cam_idx = frame_idx;
            calib_waiting_3d = true;
            calib_pairs.clear();
            calib_status = "Click a 3D point in the viewer (short click). Hold Ctrl for super zoom.";
            // Move 3D view to camera position and orientation using FPS camera
            auto vw = guik::LightViewer::instance();
            {
              auto fps_cam = vw->use_fps_camera_control(60.0);
              // Extract yaw/pitch from camera forward direction
              const Eigen::Vector3f cam_fwd_dir = frame.T_world_cam.rotation().col(0).cast<float>();
              const float yaw = std::atan2(cam_fwd_dir.y(), cam_fwd_dir.x()) * 180.0f / M_PI;
              const float pitch = std::asin(std::clamp(cam_fwd_dir.z(), -1.0f, 1.0f)) * 180.0f / M_PI;
              fps_cam->set_pose(frame.T_world_cam.translation().cast<float>(), yaw, pitch);
            }
            // Highlight calibration camera in yellow
            const auto& cT = frame.T_world_cam;
            const Eigen::Vector3f cpos = cT.translation().cast<float>();
            const Eigen::Matrix3f cR = cT.rotation().cast<float>();
            const Eigen::Vector3f cfwd = cR.col(0).normalized(), cright = cR.col(1).normalized(), cup = cR.col(2).normalized();
            const float fl = 0.6f, fw = 0.3f, fh = 0.2f;
            const Eigen::Vector3f cbc = cpos + cfwd * fl;
            std::vector<Eigen::Vector3f> cverts = {
              cpos, cbc+cright*fw+cup*fh, cpos, cbc-cright*fw+cup*fh, cpos, cbc-cright*fw-cup*fh, cpos, cbc+cright*fw-cup*fh,
              cbc+cright*fw+cup*fh, cbc-cright*fw+cup*fh, cbc-cright*fw+cup*fh, cbc-cright*fw-cup*fh,
              cbc-cright*fw-cup*fh, cbc+cright*fw-cup*fh, cbc+cright*fw-cup*fh, cbc+cright*fw+cup*fh
            };
            vw->update_drawable("cam_fov_" + std::to_string(src_idx) + "_" + std::to_string(frame_idx),
              std::make_shared<glk::ThinLines>(cverts.data(), static_cast<int>(cverts.size()), false),
              guik::FlatColor(1.0f, 1.0f, 0.0f, 1.0f));
            Eigen::Affine3f cbtf = Eigen::Affine3f::Identity(); cbtf.translate(cpos); cbtf.linear() = cR;
            cbtf = cbtf * Eigen::Scaling(Eigen::Vector3f(0.12f, 0.18f, 0.12f));
            vw->update_drawable("cam_" + std::to_string(src_idx) + "_" + std::to_string(frame_idx),
              glk::Primitives::cube(),
              guik::FlatColor(1.0f, 1.0f, 0.0f, 0.9f, cbtf).add("info_values",
                Eigen::Vector4i(static_cast<int>(PickType::CAMERA), src_idx, 0, frame_idx)));
          }
        }
      }
    }

    // Submap right-click — add colorize option
    if (type == PickType::FRAME && !image_sources.empty()) {
      const int submap_id = right_clicked_info[3];
      if (submap_id >= 0 && submap_id < static_cast<int>(submaps.size()) && submaps[submap_id]) {
        ImGui::Separator();
        if (ImGui::MenuItem("Colorize submap")) {
          colorize_last_submap = submap_id; colorize_last_cam_src = -1; colorize_last_cam_idx = -1;
          const auto& sm = submaps[submap_id];
          // Select cameras by timestamp — those within the submap's time range + margin
          const double t_first = sm->frames.front()->stamp;
          const double t_last = sm->frames.back()->stamp;
          const double t_margin = 1.0;  // 1 second before/after submap time range
          std::vector<CameraFrame> nearby_cams;
          for (auto& src : image_sources) {
            for (auto& cam : src.frames) {
              if (!cam.located || cam.timestamp <= 0.0) continue;
              const double cam_t = cam.timestamp + src.time_shift;
              if (cam_t >= t_first - t_margin && cam_t <= t_last + t_margin) nearby_cams.push_back(cam);
            }
          }
          logger->info("[Colorize] Submap {}: {} cameras (t={:.1f}-{:.1f}s, margin={:.1f}s)",
            submap_id, nearby_cams.size(), t_first, t_last, t_margin);
          if (!nearby_cams.empty()) {
            // Ensure mask is loaded
            if (colorize_mask.empty() && !image_sources.empty()) {
              for (const auto& ms : image_sources) {
                if (ms.path.empty()) continue;
                const std::string mp = ms.path + "/mask.png";
                if (boost::filesystem::exists(mp)) { colorize_mask = cv::imread(mp, cv::IMREAD_UNCHANGED); break; }
              }
            }
            auto hd = load_hd_for_submap(submap_id, false);
            if (hd && hd->size() > 0) {
              const Eigen::Isometry3d T_wo = sm->T_world_origin;
              std::vector<Eigen::Vector3f> world_pts(hd->size());
              std::vector<float> ints(hd->size(), 0.0f);
              for (size_t i = 0; i < hd->size(); i++) {
                world_pts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
              }
              logger->info("[Colorize] Mask status: empty={}, size={}x{}", colorize_mask.empty(), colorize_mask.cols, colorize_mask.rows);
              auto cr = Colorizer::project_colors(nearby_cams, image_sources[colorize_source_idx].intrinsics, world_pts, ints, colorize_max_range, colorize_blend, colorize_min_range, colorize_mask);
              logger->info("[Colorize] {} / {} points colored from {} cameras", cr.colored, cr.total, nearby_cams.size());
              colorize_last_result = cr;
              { auto vw = guik::LightViewer::instance(); lod_hide_all_submaps = true;
                const size_t n = cr.points.size();
                float imin = std::numeric_limits<float>::max(), imax = std::numeric_limits<float>::lowest();
                for (size_t i = 0; i < n && i < cr.intensities.size(); i++) { imin = std::min(imin, cr.intensities[i]); imax = std::max(imax, cr.intensities[i]); }
                if (imin >= imax) { imin = 0; imax = 255; }
                std::vector<Eigen::Vector4d> p4(n); std::vector<Eigen::Vector4f> c4(n);
                for (size_t i = 0; i < n; i++) {
                  p4[i] = Eigen::Vector4d(cr.points[i].x(), cr.points[i].y(), cr.points[i].z(), 1.0);
                  Eigen::Vector3f rgb = cr.colors[i];
                  if (colorize_intensity_blend && i < cr.intensities.size()) {
                    const float inv = (cr.intensities[i] - imin) / (imax - imin);
                    rgb = rgb * (1.0f - colorize_intensity_mix) + Eigen::Vector3f(inv, inv, inv) * colorize_intensity_mix;
                  }
                  c4[i] = Eigen::Vector4f(rgb.x(), rgb.y(), rgb.z(), 1.0f);
                }
                auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), p4.size()); cb->add_color(c4);
                vw->update_drawable("colorize_preview", cb, guik::Rainbow().set_color_mode(guik::ColorMode::VERTEX_COLOR)); }
            }
          }
        }
      }
    }
  };

  // In-app image viewer
  viewer->register_ui_callback("image_viewer", [this] {
    if (!show_image_viewer || !image_viewer_texture) return;
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(image_viewer_w) * 0.75f + 16, static_cast<float>(image_viewer_h) * 0.75f + 120), ImGuiCond_Appearing);
    if (ImGui::Begin(image_viewer_title.c_str(), &show_image_viewer)) {
      const ImVec2 avail = ImGui::GetContentRegionAvail();
      float panel_h = calib_active ? 160.0f : 0.0f;
      float img_avail_h = avail.y - panel_h;
      // Maintain aspect ratio
      float disp_w = avail.x, disp_h = avail.x * image_viewer_h / image_viewer_w;
      if (disp_h > img_avail_h) { disp_h = img_avail_h; disp_w = img_avail_h * image_viewer_w / image_viewer_h; }

      // Get image position for pixel coordinate computation
      const ImVec2 img_pos = ImGui::GetCursorScreenPos();
      ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(image_viewer_texture)), ImVec2(disp_w, disp_h));

      // Draw calibration point markers on image
      if (calib_active) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        const float scale_x = disp_w / image_viewer_w;
        const float scale_y = disp_h / image_viewer_h;
        const float marker_sx = disp_w / (image_original_w > 0 ? image_original_w : image_viewer_w);
        const float marker_sy = disp_h / (image_original_h > 0 ? image_original_h : image_viewer_h);
        for (size_t i = 0; i < calib_pairs.size(); i++) {
          const float px = img_pos.x + static_cast<float>(calib_pairs[i].pt_2d.x()) * marker_sx;
          const float py = img_pos.y + static_cast<float>(calib_pairs[i].pt_2d.y()) * marker_sy;
          draw_list->AddCircleFilled(ImVec2(px, py), 6.0f, IM_COL32(0, 255, 0, 255));
          char label[8]; std::snprintf(label, sizeof(label), "%zu", i + 1);
          draw_list->AddText(ImVec2(px + 8, py - 8), IM_COL32(255, 255, 255, 255), label);
        }

        // Virtual cursor in ORIGINAL image resolution for precision picking
        static float vpx_orig = 0.0f, vpy_orig = 0.0f;  // in original-res pixels
        static float prev_raw_px = 0.0f, prev_raw_py = 0.0f;
        static bool vpx_init = false;
        // Scale from display to original resolution
        const float orig_sx = (image_original_w > 0) ? static_cast<float>(image_original_w) / image_viewer_w : 1.0f;
        const float orig_sy = (image_original_h > 0) ? static_cast<float>(image_original_h) / image_viewer_h : 1.0f;
        // vpx/vpy in display coords for zoom window rendering
        float vpx = 0.0f, vpy = 0.0f;

        // Zoomed crosshair preview (top-left corner of image window)
        if (ImGui::IsItemHovered()) {
          const ImVec2 mouse = ImGui::GetMousePos();
          const float raw_px = (mouse.x - img_pos.x) / scale_x;  // display-res
          const float raw_py = (mouse.y - img_pos.y) / scale_y;
          const bool ctrl_held_pre = ImGui::GetIO().KeyCtrl;
          if (!ctrl_held_pre || !vpx_init) {
            vpx_orig = raw_px * orig_sx; vpy_orig = raw_py * orig_sy;
            prev_raw_px = raw_px; prev_raw_py = raw_py;
            vpx_init = true;
          } else {
            // Move virtual cursor at 1/zoom_factor speed in original-res space
            const float zoom_factor = 16.0f;
            const float dx = (raw_px - prev_raw_px) * orig_sx / zoom_factor;
            const float dy = (raw_py - prev_raw_py) * orig_sy / zoom_factor;
            vpx_orig += dx; vpy_orig += dy;
            vpx_orig = std::clamp(vpx_orig, 0.0f, static_cast<float>(image_original_w - 1));
            vpy_orig = std::clamp(vpy_orig, 0.0f, static_cast<float>(image_original_h - 1));
            prev_raw_px = raw_px; prev_raw_py = raw_py;
          }
          // Convert back to display coords for rendering
          vpx = vpx_orig / orig_sx; vpy = vpy_orig / orig_sy;
          const float px = vpx;
          const float py = vpy;
          if (px >= 0 && px < image_viewer_w && py >= 0 && py < image_viewer_h) {
            // Draw zoom window in top-left corner (Ctrl = higher zoom)
            const bool ctrl_held = ImGui::GetIO().KeyCtrl;
            const float zoom = ctrl_held ? 16.0f : 4.0f;
            const float zoom_size = ctrl_held ? 200.0f : 120.0f;
            const float half_src = zoom_size / (2.0f * zoom * scale_x);  // source region half-size in image pixels
            // UV coordinates for the zoomed region
            const float u0 = std::max(0.0f, (px - half_src) / image_viewer_w);
            const float v0 = std::max(0.0f, (py - half_src) / image_viewer_h);
            const float u1 = std::min(1.0f, (px + half_src) / image_viewer_w);
            const float v1 = std::min(1.0f, (py + half_src) / image_viewer_h);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 zp0(img_pos.x + 4, img_pos.y + 4);
            const ImVec2 zp1(zp0.x + zoom_size, zp0.y + zoom_size);
            dl->AddImage(reinterpret_cast<void*>(static_cast<intptr_t>(image_viewer_texture)),
              zp0, zp1, ImVec2(u0, v0), ImVec2(u1, v1));
            dl->AddRect(zp0, zp1, IM_COL32(255, 255, 255, 200));
            // Red crosshair
            const float cx = (zp0.x + zp1.x) * 0.5f, cy = (zp0.y + zp1.y) * 0.5f;
            dl->AddLine(ImVec2(cx - 10, cy), ImVec2(cx + 10, cy), IM_COL32(255, 0, 0, 255), 1.0f);
            dl->AddLine(ImVec2(cx, cy - 10), ImVec2(cx, cy + 10), IM_COL32(255, 0, 0, 255), 1.0f);
            // Pixel coords text
            char coord_buf[48]; std::snprintf(coord_buf, sizeof(coord_buf), "%.1f, %.1f (orig: %.0f, %.0f)", px, py, vpx_orig, vpy_orig);
            dl->AddText(ImVec2(zp0.x, zp1.y + 2), IM_COL32(255, 255, 255, 255), coord_buf);
          }
        }

        // Handle 2D click (when waiting for 2D point)
        if (!calib_waiting_3d && ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
          logger->info("[Calibrate] 2D click: vpx_orig={:.1f}, vpy_orig={:.1f}, orig_w={}, orig_h={}", vpx_orig, vpy_orig, image_original_w, image_original_h);
          if (vpx_orig >= 0 && vpx_orig < image_original_w && vpy_orig >= 0 && vpy_orig < image_original_h) {
            calib_pairs.back().pt_2d = Eigen::Vector2d(vpx_orig, vpy_orig);
            calib_waiting_3d = true;
            char buf[128]; std::snprintf(buf, sizeof(buf), "Pair %zu added. Click next 3D point%s", calib_pairs.size(), calib_pairs.size() >= 6 ? " (or hit Solve)" : " (need 6+)");
            calib_status = buf;
            logger->info("[Calibrate] Pair {}: 2D=({:.0f}, {:.0f})", calib_pairs.size(), vpx_orig, vpy_orig);
          }
        }

        // Calibration panel
        ImGui::Separator();

        // Camera navigation: << < [ID] > >>
        auto switch_calib_cam = [this](int new_idx) {
          auto& src = image_sources[calib_cam_src];
          if (new_idx < 0 || new_idx >= static_cast<int>(src.frames.size())) return;
          if (!src.frames[new_idx].located) return;
          // Revert old camera to white
          auto vw = guik::LightViewer::instance();
          if (calib_cam_idx >= 0 && calib_cam_idx < static_cast<int>(src.frames.size()) && src.frames[calib_cam_idx].located) {
            const Eigen::Vector3f op = src.frames[calib_cam_idx].T_world_cam.translation().cast<float>();
            Eigen::Affine3f obtf = Eigen::Affine3f::Identity(); obtf.translate(op);
            obtf.linear() = src.frames[calib_cam_idx].T_world_cam.rotation().cast<float>();
            obtf = obtf * Eigen::Scaling(Eigen::Vector3f(0.12f, 0.18f, 0.12f));
            vw->update_drawable("cam_" + std::to_string(calib_cam_src) + "_" + std::to_string(calib_cam_idx),
              glk::Primitives::cube(), guik::FlatColor(1.0f, 1.0f, 1.0f, 0.9f, obtf).add("info_values",
                Eigen::Vector4i(static_cast<int>(PickType::CAMERA), calib_cam_src, 0, calib_cam_idx)));
          }
          // Auto-save current pairs before switching
          if (!calib_pairs.empty() && !loaded_map_path.empty()) {
            nlohmann::json j;
            j["cam_src"] = calib_cam_src; j["cam_idx"] = calib_cam_idx;
            j["time_shift"] = src.time_shift;
            j["pairs"] = nlohmann::json::array();
            for (const auto& p : calib_pairs) {
              j["pairs"].push_back({{"pt_3d", {p.pt_3d.x(), p.pt_3d.y(), p.pt_3d.z()}}, {"pt_2d", {p.pt_2d.x(), p.pt_2d.y()}}});
            }
            const std::string sp = loaded_map_path + "/calib_pairs_cam" + std::to_string(calib_cam_idx) + ".json";
            std::ofstream ofs(sp); ofs << std::setprecision(10) << j.dump(2);
            logger->info("[Calibrate] Auto-saved {} pairs for cam {}", calib_pairs.size(), calib_cam_idx);
          }
          // Clean up old 3D markers
          for (size_t i = 0; i < calib_pairs.size(); i++) vw->remove_drawable("calib_pt_" + std::to_string(i));
          calib_pairs.clear();
          calib_cam_idx = new_idx;
          calib_waiting_3d = true;
          // Load new image
          const auto& nf = src.frames[new_idx];
          auto img = cv::imread(nf.filepath);
          if (!img.empty()) {
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            image_original_w = img.cols; image_original_h = img.rows;
            if (img.cols > 1920) { const double s = 1920.0 / img.cols; cv::resize(img, img, cv::Size(), s, s); }
            if (image_viewer_texture) glDeleteTextures(1, &image_viewer_texture);
            glGenTextures(1, &image_viewer_texture);
            glBindTexture(GL_TEXTURE_2D, image_viewer_texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);
            glBindTexture(GL_TEXTURE_2D, 0);
            image_viewer_w = img.cols; image_viewer_h = img.rows;
            image_viewer_title = "Calibrate: " + boost::filesystem::path(nf.filepath).filename().string();
          }
          // Move 3D view
          auto fps_cam = vw->use_fps_camera_control(60.0);
          const Eigen::Vector3f cfwd = nf.T_world_cam.rotation().col(0).cast<float>();
          fps_cam->set_pose(nf.T_world_cam.translation().cast<float>(),
            std::atan2(cfwd.y(), cfwd.x()) * 180.0f / M_PI,
            std::asin(std::clamp(cfwd.z(), -1.0f, 1.0f)) * 180.0f / M_PI);
          // Highlight new camera in yellow
          const Eigen::Vector3f np = nf.T_world_cam.translation().cast<float>();
          Eigen::Affine3f nbtf = Eigen::Affine3f::Identity(); nbtf.translate(np);
          nbtf.linear() = nf.T_world_cam.rotation().cast<float>();
          nbtf = nbtf * Eigen::Scaling(Eigen::Vector3f(0.12f, 0.18f, 0.12f));
          vw->update_drawable("cam_" + std::to_string(calib_cam_src) + "_" + std::to_string(new_idx),
            glk::Primitives::cube(), guik::FlatColor(1.0f, 1.0f, 0.0f, 0.9f, nbtf).add("info_values",
              Eigen::Vector4i(static_cast<int>(PickType::CAMERA), calib_cam_src, 0, new_idx)));
          // Auto-load pairs for new camera if they exist
          if (!loaded_map_path.empty()) {
            const std::string lp = loaded_map_path + "/calib_pairs_cam" + std::to_string(new_idx) + ".json";
            std::ifstream ifs(lp);
            if (ifs) {
              auto lj = nlohmann::json::parse(ifs, nullptr, false);
              if (!lj.is_discarded() && lj.contains("pairs")) {
                for (const auto& jp : lj["pairs"]) {
                  CalibPair p;
                  p.pt_3d = Eigen::Vector3d(jp["pt_3d"][0], jp["pt_3d"][1], jp["pt_3d"][2]);
                  p.pt_2d = Eigen::Vector2d(jp["pt_2d"][0], jp["pt_2d"][1]);
                  calib_pairs.push_back(p);
                  Eigen::Affine3f mtf = Eigen::Affine3f::Identity();
                  mtf.translate(p.pt_3d.cast<float>()); mtf.scale(calib_sphere_size);
                  vw->update_drawable("calib_pt_" + std::to_string(calib_pairs.size() - 1),
                    glk::Primitives::sphere(), guik::FlatColor(0.0f, 1.0f, 0.0f, 0.5f, mtf).make_transparent());
                }
                logger->info("[Calibrate] Auto-loaded {} pairs for cam {}", calib_pairs.size(), new_idx);
              }
            }
          }
          calib_status = calib_pairs.empty() ? "Click a 3D point in the viewer" :
            std::to_string(calib_pairs.size()) + " pairs loaded. Add more or Solve.";
        };

        if (ImGui::Button("<<")) { switch_calib_cam(0); }
        ImGui::SameLine();
        if (ImGui::Button("<")) {
          // Find previous located camera
          for (int ci = calib_cam_idx - 1; ci >= 0; ci--) {
            if (image_sources[calib_cam_src].frames[ci].located) { switch_calib_cam(ci); break; }
          }
        }
        ImGui::SameLine();
        char cam_label[64]; std::snprintf(cam_label, sizeof(cam_label), "Camera %d / %zu",
          calib_cam_idx, image_sources[calib_cam_src].frames.size());
        ImGui::Text("%s", cam_label);
        ImGui::SameLine();
        if (ImGui::Button(">")) {
          for (int ci = calib_cam_idx + 1; ci < static_cast<int>(image_sources[calib_cam_src].frames.size()); ci++) {
            if (image_sources[calib_cam_src].frames[ci].located) { switch_calib_cam(ci); break; }
          }
        }
        ImGui::SameLine();
        if (ImGui::Button(">>")) {
          for (int ci = static_cast<int>(image_sources[calib_cam_src].frames.size()) - 1; ci >= 0; ci--) {
            if (image_sources[calib_cam_src].frames[ci].located) { switch_calib_cam(ci); break; }
          }
        }

        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%s", calib_status.c_str());
        ImGui::Text("Pairs: %zu", calib_pairs.size());
        if (calib_pairs.size() > 0) {
          ImGui::SameLine();
          if (ImGui::Button("Undo last")) {
            auto vw = guik::LightViewer::instance();
            vw->remove_drawable("calib_pt_" + std::to_string(calib_pairs.size() - 1));
            calib_pairs.pop_back();
            calib_waiting_3d = true;
            calib_status = "Click a 3D point in the viewer";
          }
          // Per-pair list with remove buttons
          int remove_idx = -1;
          for (size_t i = 0; i < calib_pairs.size(); i++) {
            char pair_label[128];
            std::snprintf(pair_label, sizeof(pair_label), "%zu: 3D(%.1f,%.1f,%.1f) 2D(%.0f,%.0f)",
              i + 1, calib_pairs[i].pt_3d.x(), calib_pairs[i].pt_3d.y(), calib_pairs[i].pt_3d.z(),
              calib_pairs[i].pt_2d.x(), calib_pairs[i].pt_2d.y());
            ImGui::TextDisabled("%s", pair_label);
            ImGui::SameLine();
            char btn_label[16]; std::snprintf(btn_label, sizeof(btn_label), "X##rm%zu", i);
            if (ImGui::SmallButton(btn_label)) remove_idx = static_cast<int>(i);
          }
          if (remove_idx >= 0) {
            auto vw = guik::LightViewer::instance();
            // Remove all markers and rebuild (indices shift)
            for (size_t i = 0; i < calib_pairs.size(); i++) vw->remove_drawable("calib_pt_" + std::to_string(i));
            calib_pairs.erase(calib_pairs.begin() + remove_idx);
            for (size_t i = 0; i < calib_pairs.size(); i++) {
              Eigen::Affine3f mtf = Eigen::Affine3f::Identity();
              mtf.translate(calib_pairs[i].pt_3d.cast<float>()); mtf.scale(calib_sphere_size);
              vw->update_drawable("calib_pt_" + std::to_string(i),
                glk::Primitives::sphere(), guik::FlatColor(0.0f, 1.0f, 0.0f, 0.5f, mtf).make_transparent());
            }
            calib_waiting_3d = true;
          }
        }
        if (calib_pairs.size() >= 6) {
          ImGui::SameLine();
          if (ImGui::Button("Solve")) {
            // Collect correspondences
            std::vector<Eigen::Vector3d> pts_3d;
            std::vector<Eigen::Vector2d> pts_2d;
            for (const auto& p : calib_pairs) { pts_3d.push_back(p.pt_3d); pts_2d.push_back(p.pt_2d); }

            // Get camera's world-space lidar pose (T_world_lidar at this camera's time)
            auto& src = image_sources[calib_cam_src];
            const auto& cam = src.frames[calib_cam_idx];
            // Get T_world_lidar from trajectory (not from T_world_cam which includes current extrinsic)
            if (!trajectory_built) build_trajectory();
            std::vector<TimedPose> timed_traj(trajectory_data.size());
            for (size_t ti = 0; ti < trajectory_data.size(); ti++) timed_traj[ti] = {trajectory_data[ti].stamp, trajectory_data[ti].pose};
            const Eigen::Isometry3d T_world_lidar = Colorizer::interpolate_pose(timed_traj, cam.timestamp + src.time_shift);

            auto T_lidar_cam_new = Colorizer::solve_extrinsic(pts_3d, pts_2d, src.intrinsics, T_world_lidar);

            // Extract lever arm + RPY from result
            src.lever_arm = T_lidar_cam_new.translation();
            // Extract RPY using atan2 (consistent with solvePnP log)
            const Eigen::Matrix3d R_ext = T_lidar_cam_new.rotation();
            const double yaw = std::atan2(R_ext(1, 0), R_ext(0, 0)) * 180.0 / M_PI;
            const double pitch = std::asin(-std::clamp(R_ext(2, 0), -1.0, 1.0)) * 180.0 / M_PI;
            const double roll = std::atan2(R_ext(2, 1), R_ext(2, 2)) * 180.0 / M_PI;
            src.rotation_rpy = Eigen::Vector3d(roll, pitch, yaw);

            char buf[256]; std::snprintf(buf, sizeof(buf), "Solved! Lever=[%.3f, %.3f, %.3f] RPY=[%.2f, %.2f, %.2f] deg",
              src.lever_arm.x(), src.lever_arm.y(), src.lever_arm.z(),
              src.rotation_rpy.x(), src.rotation_rpy.y(), src.rotation_rpy.z());
            calib_status = buf;
            logger->info("[Calibrate] {}", calib_status);
          }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel calibration")) {
          calib_active = false;
          // Clean up 3D markers + revert camera color to white
          auto vw = guik::LightViewer::instance();
          for (size_t i = 0; i < calib_pairs.size(); i++) vw->remove_drawable("calib_pt_" + std::to_string(i));
          if (calib_cam_src >= 0 && calib_cam_idx >= 0 && calib_cam_src < static_cast<int>(image_sources.size()) &&
              calib_cam_idx < static_cast<int>(image_sources[calib_cam_src].frames.size())) {
            const auto& cf = image_sources[calib_cam_src].frames[calib_cam_idx];
            if (cf.located) {
              const Eigen::Vector3f p = cf.T_world_cam.translation().cast<float>();
              Eigen::Affine3f btf = Eigen::Affine3f::Identity(); btf.translate(p); btf.linear() = cf.T_world_cam.rotation().cast<float>();
              btf = btf * Eigen::Scaling(Eigen::Vector3f(0.12f, 0.18f, 0.12f));
              vw->update_drawable("cam_" + std::to_string(calib_cam_src) + "_" + std::to_string(calib_cam_idx),
                glk::Primitives::cube(),
                guik::FlatColor(1.0f, 1.0f, 1.0f, 0.9f, btf).add("info_values",
                  Eigen::Vector4i(static_cast<int>(PickType::CAMERA), calib_cam_src, 0, calib_cam_idx)));
            }
          }
          calib_pairs.clear();
          calib_status.clear();
        }
        // Save/Load calibration pairs
        ImGui::Separator();
        if (ImGui::Button("Save pairs")) {
          const std::string save_path = loaded_map_path + "/calib_pairs_cam" + std::to_string(calib_cam_idx) + ".json";
          nlohmann::json j;
          j["cam_src"] = calib_cam_src;
          j["cam_idx"] = calib_cam_idx;
          j["time_shift"] = image_sources[calib_cam_src].time_shift;
          j["pairs"] = nlohmann::json::array();
          for (const auto& p : calib_pairs) {
            j["pairs"].push_back({
              {"pt_3d", {p.pt_3d.x(), p.pt_3d.y(), p.pt_3d.z()}},
              {"pt_2d", {p.pt_2d.x(), p.pt_2d.y()}}
            });
          }
          std::ofstream ofs(save_path);
          ofs << std::setprecision(10) << j.dump(2);
          calib_status = "Saved " + std::to_string(calib_pairs.size()) + " pairs to " + save_path;
          logger->info("[Calibrate] {}", calib_status);
        }
        ImGui::SameLine();
        if (ImGui::Button("Load pairs")) {
          const std::string load_path = loaded_map_path + "/calib_pairs_cam" + std::to_string(calib_cam_idx) + ".json";
          std::ifstream ifs(load_path);
          if (ifs) {
            auto j = nlohmann::json::parse(ifs, nullptr, false);
            if (!j.is_discarded() && j.contains("pairs")) {
              // Clean up old markers
              auto vw = guik::LightViewer::instance();
              for (size_t i = 0; i < calib_pairs.size(); i++) vw->remove_drawable("calib_pt_" + std::to_string(i));
              calib_pairs.clear();
              for (const auto& jp : j["pairs"]) {
                CalibPair p;
                p.pt_3d = Eigen::Vector3d(jp["pt_3d"][0], jp["pt_3d"][1], jp["pt_3d"][2]);
                p.pt_2d = Eigen::Vector2d(jp["pt_2d"][0], jp["pt_2d"][1]);
                calib_pairs.push_back(p);
                // Render 3D marker
                Eigen::Affine3f mtf = Eigen::Affine3f::Identity();
                mtf.translate(p.pt_3d.cast<float>());
                mtf.scale(calib_sphere_size);
                vw->update_drawable("calib_pt_" + std::to_string(calib_pairs.size() - 1),
                  glk::Primitives::sphere(),
                  guik::FlatColor(0.0f, 1.0f, 0.0f, 0.5f, mtf).make_transparent());
              }
              // Switch to the saved camera if available
              if (j.contains("cam_src") && j.contains("cam_idx")) {
                const int saved_src = j["cam_src"];
                const int saved_idx = j["cam_idx"];
                if (saved_src == calib_cam_src && saved_idx < static_cast<int>(image_sources[calib_cam_src].frames.size())) {
                  switch_calib_cam(saved_idx);
                }
              }
              calib_waiting_3d = true;
              char buf[128]; std::snprintf(buf, sizeof(buf), "Loaded %zu pairs from file", calib_pairs.size());
              calib_status = buf;
              logger->info("[Calibrate] {}", calib_status);
            }
          } else {
            calib_status = "No saved pairs found at " + load_path;
          }
        }

        // Sphere size slider — update all existing markers when changed
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Sphere size", &calib_sphere_size, 0.01f, 0.5f, "%.2f")) {
          auto vw = guik::LightViewer::instance();
          for (size_t i = 0; i < calib_pairs.size(); i++) {
            Eigen::Affine3f mtf = Eigen::Affine3f::Identity();
            mtf.translate(calib_pairs[i].pt_3d.cast<float>());
            mtf.scale(calib_sphere_size);
            vw->update_drawable("calib_pt_" + std::to_string(i),
              glk::Primitives::sphere(),
              guik::FlatColor(0.0f, 1.0f, 0.0f, 0.5f, mtf).make_transparent());
          }
        }
        ImGui::TextDisabled("Short left-click = pick point. Drag = navigate.");
        ImGui::TextDisabled("Hold Ctrl for super zoom in image.");
      }
    }
    ImGui::End();
    if (!show_image_viewer) {
      if (image_viewer_texture) { glDeleteTextures(1, &image_viewer_texture); image_viewer_texture = 0; }
      if (calib_active) {
        calib_active = false;
        auto vw = guik::LightViewer::instance();
        for (size_t i = 0; i < calib_pairs.size(); i++) vw->remove_drawable("calib_pt_" + std::to_string(i));
        calib_pairs.clear();
      }
    }
  });

  // Camera gizmo visibility manager (runs independently of Locate Cameras window)
  viewer->register_ui_callback("camera_visibility", [this] {
    static bool prev_cameras = false;
    if (image_sources.empty()) { prev_cameras = draw_cameras; return; }
    if (!draw_cameras && prev_cameras) {
      auto vw = guik::LightViewer::instance();
      for (size_t si = 0; si < image_sources.size(); si++) {
        for (size_t fi = 0; fi < image_sources[si].frames.size(); fi++) {
          vw->remove_drawable("cam_" + std::to_string(si) + "_" + std::to_string(fi));
          vw->remove_drawable("cam_fov_" + std::to_string(si) + "_" + std::to_string(fi));
        }
      }
    } else if (draw_cameras && !prev_cameras) {
      auto vw = guik::LightViewer::instance();
      for (size_t si = 0; si < image_sources.size(); si++) {
        for (size_t fi = 0; fi < image_sources[si].frames.size(); fi++) {
          if (!image_sources[si].frames[fi].located) continue;
          const auto& T = image_sources[si].frames[fi].T_world_cam;
          const Eigen::Vector3f pos = T.translation().cast<float>();
          const Eigen::Matrix3f R = T.rotation().cast<float>();
          const Eigen::Vector3f fwd = R.col(0).normalized(), right = R.col(1).normalized(), up = R.col(2).normalized();
          const float fl = 0.6f, fw = 0.3f, fh = 0.2f;
          const Eigen::Vector3f bc = pos + fwd * fl;
          std::vector<Eigen::Vector3f> verts = {
            pos, bc+right*fw+up*fh, pos, bc-right*fw+up*fh, pos, bc-right*fw-up*fh, pos, bc+right*fw-up*fh,
            bc+right*fw+up*fh, bc-right*fw+up*fh, bc-right*fw+up*fh, bc-right*fw-up*fh,
            bc-right*fw-up*fh, bc+right*fw-up*fh, bc+right*fw-up*fh, bc+right*fw+up*fh
          };
          vw->update_drawable("cam_fov_" + std::to_string(si) + "_" + std::to_string(fi),
            std::make_shared<glk::ThinLines>(verts.data(), static_cast<int>(verts.size()), false),
            guik::FlatColor(1.0f, 1.0f, 1.0f, 0.7f));
          Eigen::Affine3f btf = Eigen::Affine3f::Identity(); btf.translate(pos); btf.linear() = R;
          btf = btf * Eigen::Scaling(Eigen::Vector3f(0.12f, 0.18f, 0.12f));
          vw->update_drawable("cam_" + std::to_string(si) + "_" + std::to_string(fi),
            glk::Primitives::cube(),
            guik::FlatColor(1.0f, 1.0f, 1.0f, 0.9f, btf).add("info_values",
              Eigen::Vector4i(static_cast<int>(PickType::CAMERA), static_cast<int>(si), 0, static_cast<int>(fi))));
        }
      }
    }
    prev_cameras = draw_cameras;
  });

  // 3D point picking for calibration (intercepts short left-click when calib is active)
  viewer->register_ui_callback("calib_3d_pick", [this] {
    if (!calib_active || !calib_waiting_3d) return;
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    // Track mouse down time to distinguish click from drag
    static bool mouse_was_down = false;
    static double mouse_down_time = 0.0;
    if (ImGui::IsMouseDown(0) && !mouse_was_down) {
      mouse_was_down = true;
      mouse_down_time = ImGui::GetTime();
    }
    if (!ImGui::IsMouseReleased(0)) {
      if (!ImGui::IsMouseDown(0)) mouse_was_down = false;
      return;
    }
    mouse_was_down = false;
    const double click_duration = ImGui::GetTime() - mouse_down_time;
    if (click_duration > 0.25) return;  // was a drag, not a click

    auto vw = guik::LightViewer::instance();
    const auto mouse = ImGui::GetMousePos();
    const Eigen::Vector2i mpos(static_cast<int>(mouse.x), static_cast<int>(mouse.y));
    const float depth = vw->pick_depth(mpos);
    if (depth >= 1.0f) return;  // clicked background

    const Eigen::Vector3f point = vw->unproject(mpos, depth);
    // Add a new pair with 3D point, waiting for 2D
    CalibPair pair;
    pair.pt_3d = point.cast<double>();
    pair.pt_2d = Eigen::Vector2d::Zero();
    calib_pairs.push_back(pair);
    calib_waiting_3d = false;
    calib_status = "Now click the same point in the image";

    // Visual marker in 3D
    Eigen::Affine3f marker_tf = Eigen::Affine3f::Identity();
    marker_tf.translate(point);
    marker_tf.scale(calib_sphere_size);
    vw->update_drawable("calib_pt_" + std::to_string(calib_pairs.size() - 1),
      glk::Primitives::sphere(),
      guik::FlatColor(0.0f, 1.0f, 0.0f, 0.5f, marker_tf).make_transparent());

    logger->info("[Calibrate] 3D point {}: ({:.3f}, {:.3f}, {:.3f})", calib_pairs.size(), point.x(), point.y(), point.z());
  });

  // Locate Cameras floating window
  viewer->register_ui_callback("colorize_window", [this] {
    if (!show_colorize_window) return;
    ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Locate Cameras", &show_colorize_window)) {
      if (image_sources.empty()) {
        ImGui::Text("No image sources loaded.\nUse Colorize > Image folder > Add folder...");
      } else {
        // Source selector
        std::vector<const char*> src_names;
        for (const auto& s : image_sources) src_names.push_back(s.name.c_str());
        ImGui::Combo("Source", &colorize_source_idx, src_names.data(), src_names.size());

        auto& src = image_sources[colorize_source_idx];
        ImGui::Text("%zu images (%zu located)", src.frames.size(),
          std::count_if(src.frames.begin(), src.frames.end(), [](const CameraFrame& f) { return f.located; }));

        ImGui::Separator();

        // Location criteria
        ImGui::Combo("Criteria", &colorize_locate_mode, "Time\0Coordinates\0");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Time: match image timestamp to trajectory.\nCoordinates: match GPS position to trajectory.");

        // Time shift with +/- step buttons
        bool time_changed = false;
        float ts_f = static_cast<float>(src.time_shift);
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputFloat("##ts", &ts_f, 0.0f, 0.0f, "%.3f")) { src.time_shift = ts_f; time_changed = true; }
        ImGui::SameLine();
        if (ImGui::Button("<")) { src.time_shift -= colorize_time_step; time_changed = true; }
        ImGui::SameLine();
        if (ImGui::Button(">")) { src.time_shift += colorize_time_step; time_changed = true; }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputFloat("step##ts_step", &colorize_time_step, 0.0f, 0.0f, "%.3f");
        ImGui::SameLine();
        ImGui::Text("Time shift");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Offset in seconds. Use < > to step.\nAdjust to align camera timing with LiDAR.");

        // Lever arm with per-axis step buttons
        bool extrinsic_changed = false;
        {
          float la[3] = {static_cast<float>(src.lever_arm.x()), static_cast<float>(src.lever_arm.y()), static_cast<float>(src.lever_arm.z())};
          const char* labels[] = {"X##la", "Y##la", "Z##la"};
          for (int ax = 0; ax < 3; ax++) {
            ImGui::SetNextItemWidth(70);
            if (ImGui::InputFloat(labels[ax], &la[ax], 0, 0, "%.4f")) extrinsic_changed = true;
            ImGui::SameLine();
            char mb[16], pb[16]; std::snprintf(mb, 16, "-##la%d", ax); std::snprintf(pb, 16, "+##la%d", ax);
            if (ImGui::SmallButton(mb)) { la[ax] -= colorize_lever_step; extrinsic_changed = true; }
            ImGui::SameLine();
            if (ImGui::SmallButton(pb)) { la[ax] += colorize_lever_step; extrinsic_changed = true; }
            if (ax < 2) ImGui::SameLine();
          }
          src.lever_arm = Eigen::Vector3d(la[0], la[1], la[2]);
          ImGui::SetNextItemWidth(50);
          ImGui::InputFloat("step##la_step", &colorize_lever_step, 0, 0, "%.3f");
          ImGui::SameLine(); ImGui::TextDisabled("Lever arm (m)");
        }
        // Rotation with per-axis step buttons
        {
          float rpy[3] = {static_cast<float>(src.rotation_rpy.x()), static_cast<float>(src.rotation_rpy.y()), static_cast<float>(src.rotation_rpy.z())};
          const char* labels[] = {"R##rp", "P##rp", "Y##rp"};
          for (int ax = 0; ax < 3; ax++) {
            ImGui::SetNextItemWidth(70);
            if (ImGui::InputFloat(labels[ax], &rpy[ax], 0, 0, "%.3f")) extrinsic_changed = true;
            ImGui::SameLine();
            char mb[16], pb[16]; std::snprintf(mb, 16, "-##rp%d", ax); std::snprintf(pb, 16, "+##rp%d", ax);
            if (ImGui::SmallButton(mb)) { rpy[ax] -= colorize_rot_step; extrinsic_changed = true; }
            ImGui::SameLine();
            if (ImGui::SmallButton(pb)) { rpy[ax] += colorize_rot_step; extrinsic_changed = true; }
            if (ax < 2) ImGui::SameLine();
          }
          src.rotation_rpy = Eigen::Vector3d(rpy[0], rpy[1], rpy[2]);
          ImGui::SetNextItemWidth(50);
          ImGui::InputFloat("step##rp_step", &colorize_rot_step, 0, 0, "%.2f");
          ImGui::SameLine(); ImGui::TextDisabled("Rotation (deg)");
        }
        // Treat extrinsic changes like time changes for live preview
        if (extrinsic_changed) time_changed = true;

        // Live preview
        ImGui::Checkbox("Live preview", &colorize_live_preview);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Auto-update colorize preview when time shift changes.\nUses last colorized camera or submap.");

        ImGui::Separator();

        // Auto re-colorize on time shift change with live preview
        if (time_changed && colorize_live_preview) {
          if (!trajectory_built) build_trajectory();
          std::vector<TimedPose> timed_traj(trajectory_data.size());
          for (size_t i = 0; i < trajectory_data.size(); i++)
            timed_traj[i] = {trajectory_data[i].stamp, trajectory_data[i].pose};

          // Re-locate only the preview cameras (not all)
          if (colorize_last_cam_src >= 0 && colorize_last_cam_idx >= 0 &&
              colorize_last_cam_src < static_cast<int>(image_sources.size())) {
            // Single camera preview — re-locate just this camera (apply full extrinsic: lever + RPY)
            auto& cam = image_sources[colorize_last_cam_src].frames[colorize_last_cam_idx];
            if (cam.timestamp > 0.0) {
              const double ts = cam.timestamp + src.time_shift;
              const Eigen::Isometry3d T_world_lidar = Colorizer::interpolate_pose(timed_traj, ts);
              const Eigen::Isometry3d T_lidar_cam = Colorizer::build_extrinsic(src.lever_arm, src.rotation_rpy);
              cam.T_world_cam = T_world_lidar * T_lidar_cam;
              cam.located = true;
            }
            // Re-project — find submap by timestamp
            const double lp_cam_time = cam.timestamp + src.time_shift;
            int best_sm = -1; double best_dt = 1e9;
            for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
              if (!submaps[si] || submaps[si]->frames.empty()) continue;
              const double t0 = submaps[si]->frames.front()->stamp;
              const double t1 = submaps[si]->frames.back()->stamp;
              if (lp_cam_time >= t0 && lp_cam_time <= t1) { best_sm = si; break; }
              const double dt = std::min(std::abs(lp_cam_time - t0), std::abs(lp_cam_time - t1));
              if (dt < best_dt) { best_dt = dt; best_sm = si; }
            }
            if (best_sm >= 0) {
              auto hd = load_hd_for_submap(best_sm, false);
              if (hd && hd->size() > 0) {
                const Eigen::Isometry3d T_wo = submaps[best_sm]->T_world_origin;
                std::vector<Eigen::Vector3f> wpts(hd->size()); std::vector<float> ints(hd->size(), 0.0f);
                for (size_t i = 0; i < hd->size(); i++) {
                  wpts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                  if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
                }
                std::vector<CameraFrame> cams = {cam};
                auto cr = Colorizer::project_colors(cams, src.intrinsics, wpts, ints, colorize_max_range, colorize_blend, colorize_min_range, colorize_mask);
                colorize_last_result = cr;
                { auto vw = guik::LightViewer::instance();
                  const size_t n = cr.points.size();
                  float imin = std::numeric_limits<float>::max(), imax = std::numeric_limits<float>::lowest();
                  for (size_t i = 0; i < n && i < cr.intensities.size(); i++) { imin = std::min(imin, cr.intensities[i]); imax = std::max(imax, cr.intensities[i]); }
                  if (imin >= imax) { imin = 0; imax = 255; }
                  std::vector<Eigen::Vector4d> p4(n); std::vector<Eigen::Vector4f> c4(n);
                  for (size_t i = 0; i < n; i++) {
                    p4[i] = Eigen::Vector4d(cr.points[i].x(), cr.points[i].y(), cr.points[i].z(), 1.0);
                    Eigen::Vector3f rgb = cr.colors[i];
                    if (colorize_intensity_blend && i < cr.intensities.size()) {
                      float inv = (cr.intensities[i] - imin) / (imax - imin);
                      if (colorize_nonlinear_int) inv = std::sqrt(inv);
                      rgb = rgb * (1.0f - colorize_intensity_mix) + intensity_to_color(inv) * colorize_intensity_mix;
                    }
                    c4[i] = Eigen::Vector4f(rgb.x(), rgb.y(), rgb.z(), 1.0f);
                  }
                  auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), p4.size()); cb->add_color(c4);
                  vw->update_drawable("colorize_preview", cb, guik::Rainbow().set_color_mode(guik::ColorMode::VERTEX_COLOR)); }
              }
            }
          } else if (colorize_last_submap >= 0) {
            // Submap preview — re-locate all sources, then filter by submap's time range (matches right-click colorize)
            for (auto& s : image_sources) Colorizer::locate_by_time(s, timed_traj);
            const auto& sm = submaps[colorize_last_submap];
            const double t_first = sm->frames.front()->stamp;
            const double t_last = sm->frames.back()->stamp;
            const double t_margin = 1.0;
            std::vector<CameraFrame> nearby;
            for (const auto& s : image_sources) {
              for (const auto& c : s.frames) {
                if (!c.located || c.timestamp <= 0.0) continue;
                const double ct = c.timestamp + s.time_shift;
                if (ct >= t_first - t_margin && ct <= t_last + t_margin) nearby.push_back(c);
              }
            }
            if (!nearby.empty()) {
              auto hd = load_hd_for_submap(colorize_last_submap, false);
              if (hd && hd->size() > 0) {
                const Eigen::Isometry3d T_wo = sm->T_world_origin;
                std::vector<Eigen::Vector3f> wpts(hd->size()); std::vector<float> ints(hd->size(), 0.0f);
                for (size_t i = 0; i < hd->size(); i++) {
                  wpts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                  if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
                }
                auto cr = Colorizer::project_colors(nearby, src.intrinsics, wpts, ints, colorize_max_range, colorize_blend, colorize_min_range, colorize_mask);
                colorize_last_result = cr;
                { auto vw = guik::LightViewer::instance();
                  const size_t n = cr.points.size();
                  float imin = std::numeric_limits<float>::max(), imax = std::numeric_limits<float>::lowest();
                  for (size_t i = 0; i < n && i < cr.intensities.size(); i++) { imin = std::min(imin, cr.intensities[i]); imax = std::max(imax, cr.intensities[i]); }
                  if (imin >= imax) { imin = 0; imax = 255; }
                  std::vector<Eigen::Vector4d> p4(n); std::vector<Eigen::Vector4f> c4(n);
                  for (size_t i = 0; i < n; i++) {
                    p4[i] = Eigen::Vector4d(cr.points[i].x(), cr.points[i].y(), cr.points[i].z(), 1.0);
                    Eigen::Vector3f rgb = cr.colors[i];
                    if (colorize_intensity_blend && i < cr.intensities.size()) {
                      float inv = (cr.intensities[i] - imin) / (imax - imin);
                      if (colorize_nonlinear_int) inv = std::sqrt(inv);
                      rgb = rgb * (1.0f - colorize_intensity_mix) + intensity_to_color(inv) * colorize_intensity_mix;
                    }
                    c4[i] = Eigen::Vector4f(rgb.x(), rgb.y(), rgb.z(), 1.0f);
                  }
                  auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), p4.size()); cb->add_color(c4);
                  vw->update_drawable("colorize_preview", cb, guik::Rainbow().set_color_mode(guik::ColorMode::VERTEX_COLOR)); }
              }
            }
          }
          // Update camera gizmo position for the preview camera
          if (draw_cameras && colorize_last_cam_src >= 0 && colorize_last_cam_idx >= 0) {
            const auto& cam = image_sources[colorize_last_cam_src].frames[colorize_last_cam_idx];
            if (cam.located) {
              auto vw = guik::LightViewer::instance();
              const Eigen::Vector3f pos = cam.T_world_cam.translation().cast<float>();
              const Eigen::Matrix3f R = cam.T_world_cam.rotation().cast<float>();
              const Eigen::Vector3f fwd = R.col(0).normalized(), right = R.col(1).normalized(), up = R.col(2).normalized();
              const float fl = 0.6f, fw = 0.3f, fh = 0.2f;
              const Eigen::Vector3f bc = pos + fwd * fl;
              std::vector<Eigen::Vector3f> verts = {
                pos, bc+right*fw+up*fh, pos, bc-right*fw+up*fh, pos, bc-right*fw-up*fh, pos, bc+right*fw-up*fh,
                bc+right*fw+up*fh, bc-right*fw+up*fh, bc-right*fw+up*fh, bc-right*fw-up*fh,
                bc-right*fw-up*fh, bc+right*fw-up*fh, bc+right*fw-up*fh, bc+right*fw+up*fh
              };
              const int si = colorize_last_cam_src, fi = colorize_last_cam_idx;
              vw->update_drawable("cam_fov_" + std::to_string(si) + "_" + std::to_string(fi),
                std::make_shared<glk::ThinLines>(verts.data(), static_cast<int>(verts.size()), false),
                guik::FlatColor(1.0f, 1.0f, 1.0f, 0.7f));
              Eigen::Affine3f btf = Eigen::Affine3f::Identity(); btf.translate(pos); btf.linear() = R;
              btf = btf * Eigen::Scaling(Eigen::Vector3f(0.12f, 0.18f, 0.12f));
              vw->update_drawable("cam_" + std::to_string(si) + "_" + std::to_string(fi),
                glk::Primitives::cube(),
                guik::FlatColor(1.0f, 1.0f, 1.0f, 0.9f, btf).add("info_values",
                  Eigen::Vector4i(static_cast<int>(PickType::CAMERA), si, 0, fi)));
            }
          }
        }

        // Locate button
        if (ImGui::Button("Locate along path")) {
          // Save colorize config on each locate
          if (!loaded_map_path.empty()) {
            nlohmann::json cfg; cfg["sources"] = nlohmann::json::array();
            for (const auto& s : image_sources) {
              nlohmann::json sj;
              sj["path"] = s.path; sj["mask_path"] = s.mask_path; sj["time_shift"] = s.time_shift;
              sj["lever_arm"] = {s.lever_arm.x(), s.lever_arm.y(), s.lever_arm.z()};
              sj["rotation_rpy"] = {s.rotation_rpy.x(), s.rotation_rpy.y(), s.rotation_rpy.z()};
              sj["fx"] = s.intrinsics.fx; sj["fy"] = s.intrinsics.fy;
              sj["cx"] = s.intrinsics.cx; sj["cy"] = s.intrinsics.cy;
              sj["width"] = s.intrinsics.width; sj["height"] = s.intrinsics.height;
              sj["k1"] = s.intrinsics.k1; sj["k2"] = s.intrinsics.k2;
              sj["p1"] = s.intrinsics.p1; sj["p2"] = s.intrinsics.p2; sj["k3"] = s.intrinsics.k3;
              cfg["sources"].push_back(sj);
            }
            std::ofstream ofs(loaded_map_path + "/colorize_config.json");
            ofs << std::setprecision(10) << cfg.dump(2);
          }
          if (!trajectory_built) build_trajectory();
          // Build timed pose vector from trajectory
          std::vector<TimedPose> timed_traj(trajectory_data.size());
          for (size_t i = 0; i < trajectory_data.size(); i++) {
            timed_traj[i] = {trajectory_data[i].stamp, trajectory_data[i].pose};
          }

          int count = 0;
          if (colorize_locate_mode == 0) {
            count = Colorizer::locate_by_time(src, timed_traj);
          } else {
            count = Colorizer::locate_by_coordinates(src, timed_traj,
              gnss_utm_zone, gnss_utm_easting_origin, gnss_utm_northing_origin, gnss_datum_alt);
          }
          logger->info("[Colorize] Located {} / {} cameras", count, src.frames.size());
          draw_cameras = true;

          // Render camera gizmos
          if (draw_cameras) {
            auto vw = guik::LightViewer::instance();
            int cam_count = 0;
            for (size_t fi = 0; fi < src.frames.size(); fi++) {
              if (!src.frames[fi].located) continue;
              const auto& T = src.frames[fi].T_world_cam;
              const Eigen::Vector3f pos = T.translation().cast<float>();
              const Eigen::Matrix3f R = T.rotation().cast<float>();
              const Eigen::Vector3f fwd = R.col(0).normalized();
              const Eigen::Vector3f right = R.col(1).normalized();
              const Eigen::Vector3f up = R.col(2).normalized();

              // FOV pyramid: tip at camera pos, 4 lines to rectangle corners ahead
              const float fov_len = 0.6f, fov_w = 0.3f, fov_h = 0.2f;
              const Eigen::Vector3f base_center = pos + fwd * fov_len;
              const Eigen::Vector3f c0 = base_center + right * fov_w + up * fov_h;
              const Eigen::Vector3f c1 = base_center - right * fov_w + up * fov_h;
              const Eigen::Vector3f c2 = base_center - right * fov_w - up * fov_h;
              const Eigen::Vector3f c3 = base_center + right * fov_w - up * fov_h;

              // 8 lines: 4 from tip to corners + 4 base edges (16 vertices, pairs)
              std::vector<Eigen::Vector3f> verts = {
                pos, c0, pos, c1, pos, c2, pos, c3,  // tip to corners
                c0, c1, c1, c2, c2, c3, c3, c0       // base rectangle
              };
              // FOV lines
              auto line_buf = std::make_shared<glk::ThinLines>(verts.data(), static_cast<int>(verts.size()), false);
              vw->update_drawable("cam_fov_" + std::to_string(colorize_source_idx) + "_" + std::to_string(fi),
                line_buf, guik::FlatColor(1.0f, 1.0f, 1.0f, 0.7f));

              // Camera body — solid pickable cube
              Eigen::Affine3f box_tf = Eigen::Affine3f::Identity();
              box_tf.translate(pos);
              box_tf.linear() = R;
              box_tf = box_tf * Eigen::Scaling(Eigen::Vector3f(0.12f, 0.18f, 0.12f));
              const Eigen::Vector4i cam_info(static_cast<int>(PickType::CAMERA), colorize_source_idx, 0, static_cast<int>(fi));
              vw->update_drawable("cam_" + std::to_string(colorize_source_idx) + "_" + std::to_string(fi),
                glk::Primitives::cube(),
                guik::FlatColor(1.0f, 1.0f, 1.0f, 0.9f, box_tf).add("info_values", cam_info));
              cam_count++;
            }
            logger->info("[Colorize] Rendered {} camera gizmos", cam_count);
          }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Place cameras along the SLAM trajectory\nusing the selected criteria.");

        ImGui::SameLine();
        static bool prev_draw_cameras = false;
        ImGui::Checkbox("Show", &draw_cameras);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show/hide camera gizmos in the 3D view.");

        // Handle show/hide transitions
        if (!draw_cameras && prev_draw_cameras) {
          // Just turned off — remove all gizmos
          auto vw = guik::LightViewer::instance();
          for (size_t si = 0; si < image_sources.size(); si++) {
            for (size_t fi = 0; fi < image_sources[si].frames.size(); fi++) {
              vw->remove_drawable("cam_" + std::to_string(si) + "_" + std::to_string(fi));
              vw->remove_drawable("cam_fov_" + std::to_string(si) + "_" + std::to_string(fi));
            }
          }
        } else if (draw_cameras && !prev_draw_cameras) {
          // Just turned on — re-render all located cameras
          auto vw = guik::LightViewer::instance();
          for (size_t si = 0; si < image_sources.size(); si++) {
            for (size_t fi = 0; fi < image_sources[si].frames.size(); fi++) {
              if (!image_sources[si].frames[fi].located) continue;
              const auto& T = image_sources[si].frames[fi].T_world_cam;
              const Eigen::Vector3f pos = T.translation().cast<float>();
              const Eigen::Matrix3f R = T.rotation().cast<float>();
              const Eigen::Vector3f fwd = R.col(0).normalized();
              const Eigen::Vector3f right = R.col(1).normalized();
              const Eigen::Vector3f up = R.col(2).normalized();
              const float fov_len = 0.6f, fov_w = 0.3f, fov_h = 0.2f;
              const Eigen::Vector3f bc = pos + fwd * fov_len;
              std::vector<Eigen::Vector3f> verts = {
                pos, bc + right*fov_w + up*fov_h, pos, bc - right*fov_w + up*fov_h,
                pos, bc - right*fov_w - up*fov_h, pos, bc + right*fov_w - up*fov_h,
                bc + right*fov_w + up*fov_h, bc - right*fov_w + up*fov_h,
                bc - right*fov_w + up*fov_h, bc - right*fov_w - up*fov_h,
                bc - right*fov_w - up*fov_h, bc + right*fov_w - up*fov_h,
                bc + right*fov_w - up*fov_h, bc + right*fov_w + up*fov_h
              };
              vw->update_drawable("cam_fov_" + std::to_string(si) + "_" + std::to_string(fi),
                std::make_shared<glk::ThinLines>(verts.data(), static_cast<int>(verts.size()), false),
                guik::FlatColor(1.0f, 1.0f, 1.0f, 0.7f));
              Eigen::Affine3f box_tf = Eigen::Affine3f::Identity();
              box_tf.translate(pos); box_tf.linear() = R;
              box_tf = box_tf * Eigen::Scaling(Eigen::Vector3f(0.12f, 0.18f, 0.12f));
              const Eigen::Vector4i cam_info(static_cast<int>(PickType::CAMERA), static_cast<int>(si), 0, static_cast<int>(fi));
              vw->update_drawable("cam_" + std::to_string(si) + "_" + std::to_string(fi),
                glk::Primitives::cube(),
                guik::FlatColor(1.0f, 1.0f, 1.0f, 0.9f, box_tf).add("info_values", cam_info));
            }
          }
        }
        prev_draw_cameras = draw_cameras;

        // Intrinsics (compact input fields)
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Camera Intrinsics")) {
          float fx = static_cast<float>(src.intrinsics.fx), fy = static_cast<float>(src.intrinsics.fy);
          float cxv = static_cast<float>(src.intrinsics.cx), cyv = static_cast<float>(src.intrinsics.cy);
          ImGui::SetNextItemWidth(80); if (ImGui::InputFloat("fx##i", &fx, 0, 0, "%.0f")) src.intrinsics.fx = fx;
          ImGui::SameLine(); ImGui::SetNextItemWidth(80); if (ImGui::InputFloat("fy##i", &fy, 0, 0, "%.0f")) src.intrinsics.fy = fy;
          ImGui::SetNextItemWidth(80); if (ImGui::InputFloat("cx##i", &cxv, 0, 0, "%.0f")) src.intrinsics.cx = cxv;
          ImGui::SameLine(); ImGui::SetNextItemWidth(80); if (ImGui::InputFloat("cy##i", &cyv, 0, 0, "%.0f")) src.intrinsics.cy = cyv;
          int iw = src.intrinsics.width, ih = src.intrinsics.height;
          ImGui::SetNextItemWidth(80); if (ImGui::InputInt("W##i", &iw)) src.intrinsics.width = iw;
          ImGui::SameLine(); ImGui::SetNextItemWidth(80); if (ImGui::InputInt("H##i", &ih)) src.intrinsics.height = ih;
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Pinhole intrinsics. Default: Elgato Facecam Pro 90° FOV.");
          ImGui::Separator();
          ImGui::Text("Distortion (Brown-Conrady)");
          float dk1 = static_cast<float>(src.intrinsics.k1), dk2 = static_cast<float>(src.intrinsics.k2);
          float dp1 = static_cast<float>(src.intrinsics.p1), dp2 = static_cast<float>(src.intrinsics.p2);
          float dk3 = static_cast<float>(src.intrinsics.k3);
          ImGui::SetNextItemWidth(80); if (ImGui::InputFloat("k1##d", &dk1, 0, 0, "%.6f")) src.intrinsics.k1 = dk1;
          ImGui::SameLine(); ImGui::SetNextItemWidth(80); if (ImGui::InputFloat("k2##d", &dk2, 0, 0, "%.6f")) src.intrinsics.k2 = dk2;
          ImGui::SetNextItemWidth(80); if (ImGui::InputFloat("p1##d", &dp1, 0, 0, "%.6f")) src.intrinsics.p1 = dp1;
          ImGui::SameLine(); ImGui::SetNextItemWidth(80); if (ImGui::InputFloat("p2##d", &dp2, 0, 0, "%.6f")) src.intrinsics.p2 = dp2;
          ImGui::SetNextItemWidth(80); if (ImGui::InputFloat("k3##d", &dk3, 0, 0, "%.6f")) src.intrinsics.k3 = dk3;
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Radial (k1,k2,k3) and tangential (p1,p2) distortion.\nImport from Metashape or calibration tool.\nLeave zeros for no distortion.");
        }

        ImGui::DragFloat("Min range (m)", &colorize_min_range, 0.5f, 0.0f, 50.0f, "%.1f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Skip points closer than this to the camera.\nReduces close-up distortion artifacts.");
        ImGui::DragFloat("Max range (m)", &colorize_max_range, 1.0f, 5.0f, 200.0f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max distance from camera to project points.\nCloser = faster, further = more coverage.");
        ImGui::Checkbox("Blend cameras", &colorize_blend);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("On: average colors from all cameras seeing a point.\nOff: use only the closest camera's color.");

        // Static mask
        if (ImGui::Button("Load mask")) {
          auto files = pfd::open_file("Select mask image", src.path, {"Image files", "*.png *.jpg *.bmp"}).result();
          if (!files.empty()) {
            colorize_mask = cv::imread(files[0], cv::IMREAD_UNCHANGED);
            if (!colorize_mask.empty()) { src.mask_path = files[0]; logger->info("[Colorize] Loaded mask from {}", files[0]); }
          }
        }
        // Simple: look for mask.png in the image source folder
        if (colorize_mask.empty() && !src.path.empty()) {
          const std::string auto_mask = src.path + "/mask.png";
          if (boost::filesystem::exists(auto_mask)) {
            colorize_mask = cv::imread(auto_mask, cv::IMREAD_UNCHANGED);
            if (!colorize_mask.empty()) logger->info("[Colorize] Auto-loaded mask from {}", auto_mask);
          }
        }
        if (!colorize_mask.empty()) {
          ImGui::SameLine();
          const bool mask_size_ok = (colorize_mask.cols == src.intrinsics.width && colorize_mask.rows == src.intrinsics.height);
          if (mask_size_ok) ImGui::TextDisabled("Mask: %dx%d ch=%d", colorize_mask.cols, colorize_mask.rows, colorize_mask.channels());
          else ImGui::TextColored(ImVec4(1,0.4f,0,1), "Mask: %dx%d (expected %dx%d!)", colorize_mask.cols, colorize_mask.rows, src.intrinsics.width, src.intrinsics.height);
          ImGui::SameLine();
          if (ImGui::SmallButton("Preview##mask")) {
            // Overlay mask on first image: checkerboard where masked, image where not
            if (!src.frames.empty()) {
              auto img = cv::imread(src.frames[0].filepath);
              if (!img.empty()) {
                cv::Mat mask_resized;
                if (colorize_mask.cols != img.cols || colorize_mask.rows != img.rows)
                  cv::resize(colorize_mask, mask_resized, cv::Size(img.cols, img.rows));
                else mask_resized = colorize_mask;
                // Build display: image where mask is white, checkerboard where mask is black
                for (int y = 0; y < img.rows; y++) {
                  for (int x = 0; x < img.cols; x++) {
                    bool is_masked = false;
                    if (mask_resized.channels() == 1) is_masked = mask_resized.at<uint8_t>(y, x) == 0;
                    else if (mask_resized.channels() == 3) { auto& p = mask_resized.at<cv::Vec3b>(y, x); is_masked = (p[0]==0 && p[1]==0 && p[2]==0); }
                    else if (mask_resized.channels() == 4) { auto& p = mask_resized.at<cv::Vec4b>(y, x); is_masked = (p[3]==0) || (p[0]==0 && p[1]==0 && p[2]==0); }
                    if (is_masked) {
                      uint8_t c = ((x/16 + y/16) % 2 == 0) ? 180 : 80;
                      img.at<cv::Vec3b>(y, x) = cv::Vec3b(c, c, c);
                    }
                  }
                }
                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                image_original_w = img.cols; image_original_h = img.rows;
                if (img.cols > 1920) { double s = 1920.0 / img.cols; cv::resize(img, img, cv::Size(), s, s); }
                if (image_viewer_texture) glDeleteTextures(1, &image_viewer_texture);
                glGenTextures(1, &image_viewer_texture);
                glBindTexture(GL_TEXTURE_2D, image_viewer_texture);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);
                glBindTexture(GL_TEXTURE_2D, 0);
                image_viewer_w = img.cols; image_viewer_h = img.rows;
                image_viewer_title = "Mask Preview";
                show_image_viewer = true;
              }
            }
          }
          ImGui::SameLine();
          if (ImGui::SmallButton("Clear mask")) colorize_mask = cv::Mat();
        } else {
          ImGui::SameLine();
          ImGui::TextDisabled("No mask (place mask.png in image folder)");
        }

        ImGui::Separator();
        bool blend_changed = false;
        if (ImGui::Checkbox("Intensity blend", &colorize_intensity_blend)) blend_changed = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Blend projected RGB with LiDAR intensity.\nUseful for alignment verification using road markings.");
        if (colorize_intensity_blend) {
          if (ImGui::SliderFloat("Mix##intblend", &colorize_intensity_mix, 0.0f, 1.0f, "%.2f")) blend_changed = true;
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("0 = pure RGB, 1 = pure intensity.");
          if (ImGui::Checkbox("Non-linear intensity", &colorize_nonlinear_int)) blend_changed = true;
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Compress intensity range (sqrt) to boost\nroad marking contrast. Useful for Livox scanners.");
        }
        // Re-render preview with new blend without re-projecting
        if (blend_changed && !colorize_last_result.points.empty()) {
          auto vw = guik::LightViewer::instance();
          const size_t n = colorize_last_result.points.size();
          float imin = std::numeric_limits<float>::max(), imax = std::numeric_limits<float>::lowest();
          for (size_t i = 0; i < n && i < colorize_last_result.intensities.size(); i++) {
            imin = std::min(imin, colorize_last_result.intensities[i]);
            imax = std::max(imax, colorize_last_result.intensities[i]);
          }
          if (imin >= imax) { imin = 0; imax = 255; }
          std::vector<Eigen::Vector4d> p4(n); std::vector<Eigen::Vector4f> c4(n);
          for (size_t i = 0; i < n; i++) {
            p4[i] = Eigen::Vector4d(colorize_last_result.points[i].x(), colorize_last_result.points[i].y(), colorize_last_result.points[i].z(), 1.0);
            Eigen::Vector3f rgb = colorize_last_result.colors[i];
            if (colorize_intensity_blend && i < colorize_last_result.intensities.size()) {
              float inv = (colorize_last_result.intensities[i] - imin) / (imax - imin);
              if (colorize_nonlinear_int) inv = std::sqrt(inv);
              rgb = rgb * (1.0f - colorize_intensity_mix) + intensity_to_color(inv) * colorize_intensity_mix;
            }
            c4[i] = Eigen::Vector4f(rgb.x(), rgb.y(), rgb.z(), 1.0f);
          }
          auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), p4.size());
          cb->add_color(c4);
          vw->update_drawable("colorize_preview", cb, guik::Rainbow().set_color_mode(guik::ColorMode::VERTEX_COLOR));
        }

        // Colorize all submaps + clear
        static bool colorize_all_running = false;
        static std::string colorize_all_status;
        if (colorize_all_running) {
          ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", colorize_all_status.c_str());
        } else {
          if (ImGui::Button("Colorize all submaps (preview)")) {
            colorize_all_running = true;
            colorize_all_status = "Starting...";
            const auto& isrc = image_sources[colorize_source_idx];
            std::thread([this, &isrc] {
              if (!trajectory_built) build_trajectory();
              std::vector<TimedPose> timed_traj(trajectory_data.size());
              for (size_t i = 0; i < trajectory_data.size(); i++) timed_traj[i] = {trajectory_data[i].stamp, trajectory_data[i].pose};

              std::vector<Eigen::Vector4d> all_pts;
              std::vector<Eigen::Vector4f> all_cols;
              ColorizeResult agg;  // store full-map result for blend re-render
              size_t total_colored = 0;

              for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
                const auto& sm = submaps[si];
                if (!sm || sm->frames.empty()) continue;
                if (hidden_sessions.count(sm->session_id)) continue;

                char buf[64]; std::snprintf(buf, sizeof(buf), "Submap %d/%zu...", si + 1, submaps.size());
                colorize_all_status = buf;

                // Find cameras for this submap by timestamp
                const double t0 = sm->frames.front()->stamp, t1 = sm->frames.back()->stamp;
                std::vector<CameraFrame> cams;
                for (const auto& src2 : image_sources) {
                  for (const auto& cam : src2.frames) {
                    if (!cam.located || cam.timestamp <= 0) continue;
                    const double ct = cam.timestamp + src2.time_shift;
                    if (ct >= t0 - 1.0 && ct <= t1 + 1.0) cams.push_back(cam);
                  }
                }
                if (cams.empty()) continue;

                // Load HD points
                auto hd = load_hd_for_submap(si, false);
                if (!hd || hd->size() == 0) continue;
                const Eigen::Isometry3d T_wo = sm->T_world_origin;
                std::vector<Eigen::Vector3f> wpts(hd->size());
                std::vector<float> ints(hd->size(), 0.0f);
                for (size_t i = 0; i < hd->size(); i++) {
                  wpts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                  if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
                }

                auto cr = Colorizer::project_colors(cams, isrc.intrinsics, wpts, ints, colorize_max_range, colorize_blend, colorize_min_range, colorize_mask);
                for (size_t i = 0; i < cr.points.size(); i++) {
                  if (cr.colors[i].x() == 0.5f && cr.colors[i].y() == 0.5f && cr.colors[i].z() == 0.5f) continue; // skip gray (uncolored)
                  all_pts.push_back(Eigen::Vector4d(cr.points[i].x(), cr.points[i].y(), cr.points[i].z(), 1.0));
                  all_cols.push_back(Eigen::Vector4f(cr.colors[i].x(), cr.colors[i].y(), cr.colors[i].z(), 1.0f));
                  agg.points.push_back(cr.points[i]);
                  agg.colors.push_back(cr.colors[i]);
                  if (i < cr.intensities.size()) agg.intensities.push_back(cr.intensities[i]);
                }
                total_colored += cr.colored;
              }

              // Render — store in colorize_last_result so blend slider re-renders the full map
              guik::LightViewer::instance()->invoke([this, all_pts, all_cols, agg, total_colored] {
                auto vw = guik::LightViewer::instance();
                lod_hide_all_submaps = true;
                colorize_last_result = agg;
                // Clear focused context so live preview won't swap back to a single submap/cam
                colorize_last_submap = -1;
                colorize_last_cam_src = -1;
                colorize_last_cam_idx = -1;
                if (!all_pts.empty()) {
                  auto cb = std::make_shared<glk::PointCloudBuffer>(all_pts.data(), all_pts.size());
                  cb->add_color(all_cols);
                  vw->update_drawable("colorize_preview", cb, guik::Rainbow().set_color_mode(guik::ColorMode::VERTEX_COLOR));
                }
                char buf[128]; std::snprintf(buf, sizeof(buf), "Done: %zu colored points from %zu total", total_colored, all_pts.size());
                colorize_all_status = buf;
                colorize_all_running = false;
                logger->info("[Colorize] {}", colorize_all_status);
              });
            }).detach();
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Colorize ALL submaps using nearby cameras.\nRenders as preview overlay.");
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear preview")) {
          auto vw = guik::LightViewer::instance();
          vw->remove_drawable("colorize_preview");
          lod_hide_all_submaps = false;
        }

        // Apply colorize to HD (write aux_rgb.bin per frame)
        static bool apply_rgb_running = false;
        static std::string apply_rgb_status;
        if (apply_rgb_running) {
          ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", apply_rgb_status.c_str());
        } else {
          if (ImGui::Button("Apply colorize to HD")) {
            apply_rgb_running = true;
            apply_rgb_status = "Starting...";
            const auto mask_copy = colorize_mask.clone();
            std::thread([this, mask_copy] {
              if (!trajectory_built) build_trajectory();
              const auto start_time = std::chrono::steady_clock::now();
              const auto& isrc = image_sources[colorize_source_idx];
              int frames_written = 0;

              for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
                const auto& sm = submaps[si];
                if (!sm || sm->frames.empty() || hidden_sessions.count(sm->session_id)) continue;

                char buf[64]; std::snprintf(buf, sizeof(buf), "Submap %d/%zu...", si + 1, submaps.size());
                apply_rgb_status = buf;

                // Find cameras by timestamp
                const double t0 = sm->frames.front()->stamp, t1 = sm->frames.back()->stamp;
                std::vector<CameraFrame> cams;
                for (const auto& src2 : image_sources) {
                  for (const auto& cam : src2.frames) {
                    if (!cam.located || cam.timestamp <= 0) continue;
                    const double ct = cam.timestamp + src2.time_shift;
                    if (ct >= t0 - 1.0 && ct <= t1 + 1.0) cams.push_back(cam);
                  }
                }
                if (cams.empty()) continue;

                // Load full submap (same as preview) — project once, split by frame
                auto hd = load_hd_for_submap(si, false);
                if (!hd || hd->size() == 0) continue;
                const Eigen::Isometry3d T_wo = sm->T_world_origin;
                std::vector<Eigen::Vector3f> wpts(hd->size());
                std::vector<float> ints(hd->size(), 0.0f);
                for (size_t i = 0; i < hd->size(); i++) {
                  wpts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                  if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
                }

                auto cr = Colorizer::project_colors(cams, isrc.intrinsics, wpts, ints,
                  colorize_max_range, colorize_blend, colorize_min_range, mask_copy);
                logger->info("[Apply] Submap {}: {} colored / {} total from {} cameras",
                  si, cr.colored, cr.total, cams.size());

                // Split colors back to per-frame aux_rgb.bin
                const auto hd_it = session_hd_paths.find(sm->session_id);
                if (hd_it == session_hd_paths.end()) continue;
                const Eigen::Isometry3d T_odom0 = sm->frames.front()->T_world_imu;
                size_t pt_offset = 0;  // tracks position in the merged submap

                for (const auto& fr : sm->frames) {
                  char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                  const std::string fd = hd_it->second + "/" + dn;
                  auto fi = glim::frame_info_from_meta(fd,
                    glim::compute_frame_world_pose(sm->T_world_origin, sm->T_origin_endpoint_L, T_odom0, fr->T_world_imu, fr->T_lidar_imu));
                  if (fi.num_points == 0) continue;

                  // Count how many points from this frame passed the range filter in load_hd_for_submap
                  std::vector<float> rng;
                  glim::load_bin(fd + "/range.bin", rng, fi.num_points);
                  int frame_hd_pts = 0;
                  for (int pi = 0; pi < fi.num_points; pi++) {
                    const float r = (!rng.empty()) ? rng[pi] : 0.0f;
                    if (r >= 1.5f || rng.empty()) frame_hd_pts++;
                  }

                  // Write aux_rgb.bin — map merged submap indices back to frame indices
                  std::vector<float> rgb_data(fi.num_points * 3);
                  int hd_idx = 0;
                  for (int pi = 0; pi < fi.num_points; pi++) {
                    const float r = (!rng.empty()) ? rng[pi] : 0.0f;
                    if (r >= 1.5f || rng.empty()) {
                      if (pt_offset + hd_idx < cr.colors.size()) {
                        const auto& c = cr.colors[pt_offset + hd_idx];
                        rgb_data[pi * 3 + 0] = c.x();
                        rgb_data[pi * 3 + 1] = c.y();
                        rgb_data[pi * 3 + 2] = c.z();
                      }
                      hd_idx++;
                    } else {
                      // Point filtered by range — use intensity fallback
                      float gray = 0.5f;
                      rgb_data[pi * 3 + 0] = gray;
                      rgb_data[pi * 3 + 1] = gray;
                      rgb_data[pi * 3 + 2] = gray;
                    }
                  }
                  pt_offset += frame_hd_pts;

                  std::ofstream f(fd + "/aux_rgb.bin", std::ios::binary);
                  f.write(reinterpret_cast<const char*>(rgb_data.data()), sizeof(float) * rgb_data.size());
                  frames_written++;
                }
              }

              const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
              char buf[128]; std::snprintf(buf, sizeof(buf), "Done: %d frames colored (%.1f sec)", frames_written, elapsed);
              apply_rgb_status = buf;
              apply_rgb_running = false;
              logger->info("[Colorize] {}", apply_rgb_status);
            }).detach();
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("WRITE aux_rgb.bin to every HD frame.\nPersistent — appears in color dropdown after HD reload.");
        }

        // Info about first/last timestamps
        if (!src.frames.empty()) {
          double min_ts = std::numeric_limits<double>::max(), max_ts = 0.0;
          for (const auto& f : src.frames) {
            if (f.timestamp > 0.0) { min_ts = std::min(min_ts, f.timestamp); max_ts = std::max(max_ts, f.timestamp); }
          }
          if (max_ts > 0.0) {
            ImGui::Separator();
            ImGui::Text("Time range: %.1f sec", max_ts - min_ts);
            if (trajectory_built && !trajectory_data.empty()) {
              ImGui::Text("Traj range: %.1f - %.1f", trajectory_data.front().stamp, trajectory_data.back().stamp);
              ImGui::Text("Img range:  %.1f - %.1f", min_ts, max_ts);
            }
          }
        }
      }
    }
    ImGui::End();
  });

  // Alignment check window — image + projected LiDAR overlay, scale-aware
  viewer->register_ui_callback("align_view", [this] {
    if (!align_show) return;
    ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Alignment check", &align_show)) {
      if (image_sources.empty()) { ImGui::TextDisabled("No image sources loaded."); ImGui::End(); return; }
      align_cam_src = std::clamp(align_cam_src, 0, static_cast<int>(image_sources.size()) - 1);
      auto& src = image_sources[align_cam_src];
      if (src.frames.empty()) { ImGui::TextDisabled("Selected source has no frames."); ImGui::End(); return; }
      align_cam_idx = std::clamp(align_cam_idx, 0, static_cast<int>(src.frames.size()) - 1);

      // --- Top controls ---
      if (image_sources.size() > 1) {
        std::vector<std::string> labels;
        for (size_t i = 0; i < image_sources.size(); i++) labels.push_back("src " + std::to_string(i));
        std::vector<const char*> lptrs; for (auto& s : labels) lptrs.push_back(s.c_str());
        ImGui::SetNextItemWidth(100);
        ImGui::Combo("Source", &align_cam_src, lptrs.data(), lptrs.size());
        ImGui::SameLine();
      }
      ImGui::SetNextItemWidth(200);
      if (ImGui::SliderInt("Image", &align_cam_idx, 0, static_cast<int>(src.frames.size()) - 1)) {}
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Camera image index (not LiDAR frame).");
      ImGui::SameLine();
      if (ImGui::ArrowButton("##align_prev", ImGuiDir_Left)) align_cam_idx = std::max(0, align_cam_idx - 1);
      ImGui::SameLine();
      if (ImGui::ArrowButton("##align_next", ImGuiDir_Right)) align_cam_idx = std::min(static_cast<int>(src.frames.size()) - 1, align_cam_idx + 1);
      ImGui::SameLine();
      ImGui::Text("%s", boost::filesystem::path(src.frames[align_cam_idx].filepath).filename().string().c_str());

      ImGui::SetNextItemWidth(120); ImGui::SliderFloat("View scale", &align_display_scale, 0.05f, 4.0f, "%.2fx");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Display scale vs native pixels.\nMath always runs at native resolution.");
      ImGui::SameLine(); if (ImGui::Button("Fit")) {
        // Will be computed below once we know window size
        align_display_scale = -1.0f;  // sentinel
      }
      ImGui::SameLine(); ImGui::SetNextItemWidth(100);
      ImGui::SliderFloat("Pt size", &align_point_size, 0.5f, 6.0f, "%.1f");
      ImGui::SameLine(); ImGui::SetNextItemWidth(100);
      ImGui::Combo("Color", &align_point_color_mode, "Intensity\0Range\0Depth\0");
      ImGui::SameLine(); ImGui::SetNextItemWidth(100);
      ImGui::SliderFloat("Min bright", &align_bright_threshold, 0.0f, 1.0f, "%.2f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Only show points with intensity above this threshold.\nUseful for comparing road markings.");

      ImGui::SetNextItemWidth(120); ImGui::SliderFloat("Max range", &align_max_range, 2.0f, 100.0f, "%.1fm");
      ImGui::SameLine(); ImGui::SetNextItemWidth(120);
      ImGui::SliderFloat("Min range", &align_min_range, 0.1f, 10.0f, "%.1fm");
      ImGui::SameLine(); ImGui::SetNextItemWidth(120);
      ImGui::SliderFloat("Alpha", &align_point_alpha, 0.05f, 1.0f, "%.2f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overlay point transparency — blend dots with image.");
      ImGui::SameLine();
      if (ImGui::Checkbox("Rectified", &align_rectified)) align_loaded_path.clear();  // force reload
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("OFF: raw image + distorted projection (matches colorize phase).\nON: undistorted image + pinhole projection (isolates extrinsic error).");
      if (align_rectified) {
        ImGui::TextDisabled("Mode: rectified image vs pinhole projection — residual = extrinsic only.");
      } else {
        ImGui::TextDisabled("Mode: raw image vs distorted projection — residual = extrinsic + distortion model.");
      }

      // --- Load image if needed ---
      const auto& cam = src.frames[align_cam_idx];
      if (cam.filepath != align_loaded_path || align_rect_applied != align_rectified) {
        cv::Mat img = cv::imread(cam.filepath);
        if (!img.empty()) {
          align_img_w = img.cols; align_img_h = img.rows;
          // Optional rectification: undistort at native resolution using source intrinsics
          if (align_rectified) {
            cv::Mat K = (cv::Mat_<double>(3, 3) <<
              src.intrinsics.fx, 0, src.intrinsics.cx,
              0, src.intrinsics.fy, src.intrinsics.cy,
              0, 0, 1);
            cv::Mat D = (cv::Mat_<double>(1, 5) <<
              src.intrinsics.k1, src.intrinsics.k2,
              src.intrinsics.p1, src.intrinsics.p2, src.intrinsics.k3);
            cv::Mat rect;
            cv::undistort(img, rect, K, D);
            img = rect;
          }
          cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
          // Downscale texture if huge; display math uses native size
          const int max_tex = 2048;
          cv::Mat tex_img = img;
          if (img.cols > max_tex) {
            const double s = static_cast<double>(max_tex) / img.cols;
            cv::resize(img, tex_img, cv::Size(), s, s);
          }
          if (align_texture) { glDeleteTextures(1, &align_texture); align_texture = 0; }
          glGenTextures(1, &align_texture);
          glBindTexture(GL_TEXTURE_2D, align_texture);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
          glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_img.cols, tex_img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_img.data);
          glBindTexture(GL_TEXTURE_2D, 0);
          align_tex_w = tex_img.cols; align_tex_h = tex_img.rows;
          align_loaded_path = cam.filepath;
          align_rect_applied = align_rectified;
        }
      }

      // --- Locate camera using current colorize extrinsic (live-linked) ---
      if (!trajectory_built) build_trajectory();
      std::vector<TimedPose> timed_traj(trajectory_data.size());
      for (size_t i = 0; i < trajectory_data.size(); i++) timed_traj[i] = {trajectory_data[i].stamp, trajectory_data[i].pose};
      Eigen::Isometry3d T_world_cam = Eigen::Isometry3d::Identity();
      bool cam_ok = false;
      if (cam.timestamp > 0.0 && !timed_traj.empty()) {
        const double ts = cam.timestamp + src.time_shift;
        if (ts >= timed_traj.front().stamp - 2.0 && ts <= timed_traj.back().stamp + 2.0) {
          const Eigen::Isometry3d T_world_lidar = Colorizer::interpolate_pose(timed_traj, ts);
          const Eigen::Isometry3d T_lidar_cam = Colorizer::build_extrinsic(src.lever_arm, src.rotation_rpy);
          T_world_cam = T_world_lidar * T_lidar_cam;
          cam_ok = true;
        }
      }

      // --- Find submap at camera timestamp; cache its world points ---
      int best_sm = -1;
      if (cam_ok) {
        double best_dt = 1e9;
        const double ct = cam.timestamp + src.time_shift;
        for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
          if (!submaps[si] || submaps[si]->frames.empty()) continue;
          const double t0 = submaps[si]->frames.front()->stamp, t1 = submaps[si]->frames.back()->stamp;
          if (ct >= t0 && ct <= t1) { best_sm = si; break; }
          const double dt = std::min(std::abs(ct - t0), std::abs(ct - t1));
          if (dt < best_dt) { best_dt = dt; best_sm = si; }
        }
      }
      if (best_sm >= 0 && best_sm != align_last_submap_id) {
        align_submap_world_pts.clear(); align_submap_ints.clear();
        auto hd = load_hd_for_submap(best_sm, false);
        if (hd && hd->size() > 0) {
          const Eigen::Isometry3d T_wo = submaps[best_sm]->T_world_origin;
          align_submap_world_pts.resize(hd->size());
          align_submap_ints.assign(hd->size(), 0.0f);
          for (size_t i = 0; i < hd->size(); i++) {
            align_submap_world_pts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
            if (hd->intensities) align_submap_ints[i] = static_cast<float>(hd->intensities[i]);
          }
        }
        align_last_submap_id = best_sm;
      }

      // --- Identify nearest LiDAR frame to this camera (for frame-assignment check) ---
      long nearest_frame_id = -1;
      double nearest_frame_stamp = 0.0;
      double nearest_dt = 0.0;
      if (cam_ok && best_sm >= 0 && !submaps[best_sm]->frames.empty()) {
        const double ct = cam.timestamp + src.time_shift;
        double best_dt = 1e9;
        for (const auto& f : submaps[best_sm]->frames) {
          const double dt = std::abs(f->stamp - ct);
          if (dt < best_dt) { best_dt = dt; nearest_frame_id = f->id; nearest_frame_stamp = f->stamp; nearest_dt = f->stamp - ct; }
        }
      }
      // Info line
      if (cam_ok) {
        ImGui::TextDisabled("cam_t=%.3f  submap=%d  nearest_lidar_frame=%ld  lidar_t=%.3f  dt=%+.3fs",
          cam.timestamp + src.time_shift, best_sm, nearest_frame_id, nearest_frame_stamp, nearest_dt);
      } else {
        ImGui::TextDisabled("Camera not locatable (timestamp out of trajectory range).");
      }

      // --- Compute canvas area ---
      const ImVec2 avail = ImGui::GetContentRegionAvail();
      if (align_display_scale < 0.0f && align_img_w > 0 && align_img_h > 0) {
        const float fit_w = avail.x / static_cast<float>(align_img_w);
        const float fit_h = (avail.y - 10.0f) / static_cast<float>(align_img_h);
        align_display_scale = std::max(0.05f, std::min(fit_w, fit_h));
      }
      const float disp_w = align_img_w * align_display_scale;
      const float disp_h = align_img_h * align_display_scale;

      ImGui::BeginChild("align_canvas", avail, false, ImGuiWindowFlags_HorizontalScrollbar);
      const ImVec2 cur = ImGui::GetCursorScreenPos();
      if (align_texture && align_img_w > 0) {
        ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(align_texture)), ImVec2(disp_w, disp_h));
      } else {
        ImGui::Dummy(ImVec2(disp_w > 0 ? disp_w : 400, disp_h > 0 ? disp_h : 300));
      }

      // --- Project and draw points ---
      if (cam_ok && !align_submap_world_pts.empty()) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const Eigen::Isometry3d T_cw = T_world_cam.inverse();
        const Eigen::Matrix3d R_cam = T_cw.rotation();
        const Eigen::Vector3d t_cam = T_cw.translation();
        const double fx = src.intrinsics.fx, fy = src.intrinsics.fy;
        const double cx_d = src.intrinsics.cx, cy_d = src.intrinsics.cy;
        const bool has_dist = !align_rectified && (src.intrinsics.k1 != 0 || src.intrinsics.k2 != 0 || src.intrinsics.p1 != 0 || src.intrinsics.p2 != 0);
        const Eigen::Vector3f cam_pos = T_world_cam.translation().cast<float>();
        const float max_r_sq = align_max_range * align_max_range;
        const float min_r_sq = align_min_range * align_min_range;
        // Intensity range for color mapping
        float imin = std::numeric_limits<float>::max(), imax = std::numeric_limits<float>::lowest();
        for (float v : align_submap_ints) { imin = std::min(imin, v); imax = std::max(imax, v); }
        if (imin >= imax) { imin = 0; imax = 255; }
        int drawn = 0;
        for (size_t pi = 0; pi < align_submap_world_pts.size(); pi++) {
          const float dsq = (align_submap_world_pts[pi] - cam_pos).squaredNorm();
          if (dsq > max_r_sq || dsq < min_r_sq) continue;
          const Eigen::Vector3d p_cam = R_cam * align_submap_world_pts[pi].cast<double>() + t_cam;
          const double depth = p_cam.x();
          if (depth <= 0.1) continue;
          double xn = -p_cam.y() / depth;
          double yn = -p_cam.z() / depth;
          if (has_dist) {
            const double r2 = xn * xn + yn * yn, r4 = r2 * r2, r6 = r4 * r2;
            const double radial = 1.0 + src.intrinsics.k1 * r2 + src.intrinsics.k2 * r4 + src.intrinsics.k3 * r6;
            const double xd = xn * radial + 2.0 * src.intrinsics.p1 * xn * yn + src.intrinsics.p2 * (r2 + 2.0 * xn * xn);
            const double yd = yn * radial + src.intrinsics.p1 * (r2 + 2.0 * yn * yn) + 2.0 * src.intrinsics.p2 * xn * yn;
            xn = xd; yn = yd;
          }
          const double u = fx * xn + cx_d;
          const double v = fy * yn + cy_d;
          if (u < 0 || u >= align_img_w || v < 0 || v >= align_img_h) continue;
          // Intensity gate
          const float in_norm = align_submap_ints.empty() ? 0.0f : std::clamp((align_submap_ints[pi] - imin) / (imax - imin), 0.0f, 1.0f);
          if (align_bright_threshold > 0.0f && in_norm < align_bright_threshold) continue;
          // Color
          const int a8 = std::clamp(static_cast<int>(align_point_alpha * 255.0f), 0, 255);
          ImU32 col;
          if (align_point_color_mode == 0) {
            const int g = static_cast<int>(in_norm * 255.0f);
            col = IM_COL32(g, g, 0, a8);
          } else if (align_point_color_mode == 1) {
            const float rn = std::sqrt(dsq) / align_max_range;
            const int r = static_cast<int>((1.0f - rn) * 255.0f);
            const int b = static_cast<int>(rn * 255.0f);
            col = IM_COL32(r, 80, b, a8);
          } else {
            const float dn = std::clamp(static_cast<float>(depth) / align_max_range, 0.0f, 1.0f);
            const int g = static_cast<int>((1.0f - dn) * 255.0f);
            col = IM_COL32(0, g, 255 - g, a8);
          }
          const ImVec2 sp(cur.x + static_cast<float>(u) * align_display_scale,
                          cur.y + static_cast<float>(v) * align_display_scale);
          dl->AddCircleFilled(sp, align_point_size, col);
          drawn++;
        }
        // Status line on top of image
        char buf[160]; std::snprintf(buf, sizeof(buf),
          "sm=%d  pts_drawn=%d  native=%dx%d  scale=%.2fx  shift=%.2fs  RPY=(%.2f,%.2f,%.2f)",
          best_sm, drawn, align_img_w, align_img_h, align_display_scale,
          src.time_shift, src.rotation_rpy.x(), src.rotation_rpy.y(), src.rotation_rpy.z());
        dl->AddText(ImVec2(cur.x + 6, cur.y + 6), IM_COL32(255, 255, 0, 255), buf);
      } else if (!cam_ok) {
        ImGui::GetWindowDrawList()->AddText(ImVec2(cur.x + 6, cur.y + 6),
          IM_COL32(255, 80, 80, 255), "Camera not locatable (timestamp out of trajectory range).");
      }
      ImGui::EndChild();
    }
    ImGui::End();
  });

  // Data Filter tool window
  viewer->register_ui_callback("data_filter_window", [this] {
    if (!show_data_filter) return;
    ImGui::SetNextWindowSize(ImVec2(350, 320), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Data Filter", &show_data_filter)) {
      ImGui::Combo("Mode", &df_mode, "SOR\0Dynamic\0Range\0Scalar\0");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Range: remove distant noise when closer points exist.\nDynamic: remove objects that moved between passes.");

      // Ground-only checkbox (Range mode only, right after dropdown)
      if (df_mode == 2) {
        bool has_ground = false;
        if (!hd_frames_path.empty()) {
          for (const auto& submap : submaps) {
            if (!submap || submap->frames.empty()) continue;
            std::string shd = hd_frames_path;
            for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
            char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", submap->frames.front()->id);
            if (boost::filesystem::exists(shd + "/" + dn + "/aux_ground.bin")) { has_ground = true; break; }
          }
        }
        if (!has_ground) { ImGui::BeginDisabled(); rf_ground_only = false; }
        ImGui::Checkbox("Affect only ground", &rf_ground_only);
        if (!has_ground) {
          ImGui::EndDisabled();
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Requires aux_ground.bin per frame.\nGenerate with Data Filter > Dynamic > Classify ground to scalar.");
        } else {
          if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Only filter ground-classified points.\nNon-ground points pass through untouched.\nUseful for tight road cleanup without affecting walls/vegetation.");
        }
      }

      ImGui::Separator();

      // Shared parameter
      if (df_mode == 2) {
        // Range filter parameters
        ImGui::Combo("Criteria", &rf_criteria, "Range\0GPS Time\0");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Range: remove distant points when closer exist.\nGPS Time: remove overlapping pass points when earlier/later exist.");
        ImGui::SliderFloat("Voxel size (m)", &rf_voxel_size, 0.05f, 5.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Spatial grid cell size for grouping points.");

        if (rf_criteria == 0) {
          ImGui::SliderFloat("Safe range (m)", &rf_safe_range, 5.0f, 50.0f, "%.0f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Points within this range are ALWAYS kept.");
          ImGui::SliderFloat("Range delta (m)", &rf_range_delta, 1.0f, 50.0f, "%.0f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove points >delta further than closest safe-range\npoint in the voxel.");
          ImGui::SliderFloat("Far delta (m)", &rf_far_delta, 5.0f, 100.0f, "%.0f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Secondary delta for voxels with NO safe-range points.\nRemoves points > (min_range + far_delta) in the voxel.");
          ImGui::SliderInt("Min close points", &rf_min_close_pts, 1, 20);
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum close-range points in a voxel before\nthe primary delta applies.");
        } else {
          ImGui::Combo("Keep", &rf_gps_keep, "Dominant\0Newest\0Oldest\0");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Dominant: keep cluster with most points.\nNewest: keep latest temporal cluster.\nOldest: keep earliest temporal cluster.");
        }
      } else if (df_mode == 1) {
        // Dynamic filter parameters
        ImGui::DragFloat("Voxel size (m)", &df_voxel_size, 0.01f, 0.1f, 5.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Spatial grid cell size for point grouping.");
        ImGui::DragFloat("Range threshold (m)", &df_range_threshold, 0.1f, 0.1f, 50.0f, "%.1f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How close measured vs expected range must be\nto count as STATIC. Smaller = catches more dynamics.");
        ImGui::DragFloat("Observation range (m)", &df_observation_range, 1.0f, 5.0f, 200.0f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max sensor-to-point distance for frame comparison.");
        ImGui::SliderInt("Max compare frames", &df_min_observations, 5, 50);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum frames to vote against per point.");

        // Ground exclusion
        ImGui::Separator();
        ImGui::Checkbox("Exclude ground (PatchWork++)", &df_exclude_ground_pw);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Skip ground using PatchWork++ plane fitting.");
        if (df_exclude_ground_pw) {
          ImGui::SameLine();
          if (ImGui::Button("Config##pw")) { show_pw_config = !show_pw_config; }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open PatchWork++ configuration window.");
          ImGui::SameLine();
          if (ImGui::Button("Preview chunk##pw")) {
            rf_processing = true;
            rf_status = "Classifying ground (chunk)...";
            const bool accumulate = pw_accumulate;
            const int acc_count = pw_accumulate_count;
            const bool refine_z = df_refine_ground;
            std::thread([this, accumulate, acc_count, refine_z] {
              if (!trajectory_built) build_trajectory();
              auto vw = guik::LightViewer::instance();
              const Eigen::Matrix4f vm = vw->view_matrix();
              const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));

              // Find nearest trajectory point
              double min_d = 1e9; size_t best_idx = 0;
              for (size_t k = 0; k < trajectory_data.size(); k++) {
                const double d = (trajectory_data[k].pose.translation().cast<float>() - cam_pos).cast<double>().norm();
                if (d < min_d) { min_d = d; best_idx = k; }
              }
              // Build one chunk
              const double chunk_hs = df_chunk_size * 0.5;
              const Eigen::Vector3d c = trajectory_data[best_idx].pose.translation();
              const size_t next = std::min(best_idx + 1, trajectory_data.size() - 1);
              Eigen::Vector3d fwd = trajectory_data[next].pose.translation() - c;
              fwd.z() = 0; if (fwd.norm() < 0.01) fwd = Eigen::Vector3d::UnitX(); else fwd.normalize();
              const Eigen::Vector3d up = Eigen::Vector3d::UnitZ(), right = fwd.cross(up).normalized();
              Eigen::Matrix3d R; R.col(0) = fwd; R.col(1) = right; R.col(2) = up;
              glim::Chunk chunk{c, R, R.transpose(), chunk_hs, 50.0};

              // Build flat frame list within chunk
              struct FrameEntry { std::string dir; Eigen::Isometry3d T_world_lidar; int num_points; };
              std::vector<FrameEntry> chunk_frames;
              const auto chunk_aabb = chunk.world_aabb();
              for (const auto& submap : submaps) {
                if (!submap) continue;
                if (hidden_sessions.count(submap->session_id)) continue;
                std::string shd = hd_frames_path;
                for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
                const Eigen::Isometry3d T0 = submap->frames.front()->T_world_imu;
                for (const auto& fr : submap->frames) {
                  char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                  auto fi = glim::frame_info_from_meta(shd + "/" + dn,
                    glim::compute_frame_world_pose(submap->T_world_origin, submap->T_origin_endpoint_L, T0, fr->T_world_imu, fr->T_lidar_imu));
                  if (fi.num_points > 0 && chunk_aabb.intersects(fi.world_bbox)
                      && (fi.T_world_lidar.translation().cast<float>() - chunk.center.cast<float>()).norm() < chunk_hs + 50.0)
                    chunk_frames.push_back({fi.dir, fi.T_world_lidar, fi.num_points});
                }
              }

              rf_status = "Classifying " + std::to_string(chunk_frames.size()) + " frames...";
              std::vector<Eigen::Vector3f> ground_pts, nonground_pts;

              for (size_t fi = 0; fi < chunk_frames.size(); fi++) {
                const auto& entry = chunk_frames[fi];
                std::vector<Eigen::Vector3f> pts; std::vector<float> rng, ints;
                if (!glim::load_bin(entry.dir + "/points.bin", pts, entry.num_points)) continue;
                glim::load_bin(entry.dir + "/range.bin", rng, entry.num_points);
                glim::load_bin(entry.dir + "/intensities.bin", ints, entry.num_points);
                const int n = entry.num_points;

                // Classify ground (with frame accumulation if enabled)
                std::vector<bool> pw_gnd;
                if (accumulate) {
                  std::vector<Eigen::Vector3f> merged_pts(pts);
                  std::vector<float> merged_ints(ints);
                  const Eigen::Isometry3f T_cur_inv = entry.T_world_lidar.inverse().cast<float>();
                  int start = static_cast<int>(fi) - acc_count;
                  int end = static_cast<int>(fi) + acc_count;
                  if (start < 0) { end = std::min(end - start, static_cast<int>(chunk_frames.size()) - 1); start = 0; }
                  if (end >= static_cast<int>(chunk_frames.size())) { start = std::max(start - (end - static_cast<int>(chunk_frames.size()) + 1), 0); end = static_cast<int>(chunk_frames.size()) - 1; }
                  for (int ni = start; ni <= end; ni++) {
                    if (ni == static_cast<int>(fi)) continue;
                    const auto& nb = chunk_frames[ni];
                    std::vector<Eigen::Vector3f> nb_pts; std::vector<float> nb_ints;
                    if (!glim::load_bin(nb.dir + "/points.bin", nb_pts, nb.num_points)) continue;
                    glim::load_bin(nb.dir + "/intensities.bin", nb_ints, nb.num_points);
                    const Eigen::Matrix3f R_to = (T_cur_inv * nb.T_world_lidar.cast<float>()).rotation();
                    const Eigen::Vector3f t_to = (T_cur_inv * nb.T_world_lidar.cast<float>()).translation();
                    for (int pi = 0; pi < nb.num_points; pi++) {
                      merged_pts.push_back(R_to * nb_pts[pi] + t_to);
                      merged_ints.push_back(pi < static_cast<int>(nb_ints.size()) ? nb_ints[pi] : 0.0f);
                    }
                  }
                  auto merged_gnd = glim::MapCleanerFilter::classify_ground_patchwork(
                    merged_pts, static_cast<int>(merged_pts.size()), 1.7f, merged_ints);
                  pw_gnd.resize(n);
                  for (int i = 0; i < n; i++) pw_gnd[i] = !merged_gnd.empty() && i < static_cast<int>(merged_gnd.size()) && merged_gnd[i];
                } else {
                  pw_gnd = glim::MapCleanerFilter::classify_ground_patchwork(pts, n, 1.7f, ints);
                }

                const Eigen::Matrix3f Rf = entry.T_world_lidar.rotation().cast<float>();
                const Eigen::Vector3f t = entry.T_world_lidar.translation().cast<float>();
                for (int i = 0; i < n; i++) {
                  if (!rng.empty() && rng[i] < 1.5f) continue;
                  const Eigen::Vector3f wp = Rf * pts[i] + t;
                  if (!chunk.contains(wp)) continue;
                  if (!pw_gnd.empty() && pw_gnd[i]) ground_pts.push_back(wp); else nonground_pts.push_back(wp);
                }
              }

              // Z-column refinement: revoke false ground above column minimum
              if (refine_z && !ground_pts.empty()) {
                const float col_res = 1.0f, col_inv = 1.0f / col_res, ground_z_tol = 0.5f;
                std::unordered_map<uint64_t, float> col_min_z;
                for (const auto& p : ground_pts) {
                  const uint64_t ck = (static_cast<uint64_t>(static_cast<int>(std::floor(p.x() * col_inv)) + 1048576) << 21)
                                    | static_cast<uint64_t>(static_cast<int>(std::floor(p.y() * col_inv)) + 1048576);
                  auto it = col_min_z.find(ck);
                  if (it == col_min_z.end() || p.z() < it->second) col_min_z[ck] = p.z();
                }
                std::vector<Eigen::Vector3f> refined_ground;
                int revoked = 0;
                for (const auto& p : ground_pts) {
                  const uint64_t ck = (static_cast<uint64_t>(static_cast<int>(std::floor(p.x() * col_inv)) + 1048576) << 21)
                                    | static_cast<uint64_t>(static_cast<int>(std::floor(p.y() * col_inv)) + 1048576);
                  if (p.z() > col_min_z[ck] + ground_z_tol) { nonground_pts.push_back(p); revoked++; }
                  else refined_ground.push_back(p);
                }
                ground_pts = std::move(refined_ground);
                if (revoked > 0) logger->info("[Ground preview] Z-column revoked {} points", revoked);
              }

              vw->invoke([this, ground_pts, nonground_pts] {
                auto v = guik::LightViewer::instance(); lod_hide_all_submaps = true; rf_preview_active = true;
                v->remove_drawable("rf_preview_kept"); v->remove_drawable("rf_preview_removed");
                if (!ground_pts.empty()) v->update_drawable("rf_preview_kept", std::make_shared<glk::PointCloudBuffer>(ground_pts[0].data(), sizeof(Eigen::Vector3f), ground_pts.size()), guik::FlatColor(1.0f, 1.0f, 0.0f, 1.0f));
                if (!nonground_pts.empty()) v->update_drawable("rf_preview_removed", std::make_shared<glk::PointCloudBuffer>(nonground_pts[0].data(), sizeof(Eigen::Vector3f), nonground_pts.size()), guik::FlatColor(0.2f, 0.9f, 0.2f, 1.0f));
              });
              char buf[256]; std::snprintf(buf, sizeof(buf), "Ground: %zu ground (yellow), %zu non-ground (green)", ground_pts.size(), nonground_pts.size());
              rf_status = buf;
              rf_processing = false;
            }).detach();
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Preview ground classification on one chunk.\nUses frame accumulation and Z refinement if enabled.\nYellow = ground, Green = non-ground.");
        }
        ImGui::Checkbox("Refine ground (Z column)", &df_refine_ground);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Revoke false ground labels for points above\nthe lowest Z in each XY column.");
        if (ImGui::Button("Classify ground to scalar")) {
          rf_processing = true;
          rf_status = "Saving ground classification...";
          const bool accumulate = pw_accumulate && df_exclude_ground_pw;
          const int acc_count = pw_accumulate_count;
          const bool refine_z = df_refine_ground;
          const float chunk_size = df_chunk_size;
          const float chunk_spacing = df_chunk_spacing;
          std::thread([this, accumulate, acc_count, refine_z, chunk_size, chunk_spacing] {
            if (!trajectory_built) build_trajectory();
            const auto start_time = std::chrono::steady_clock::now();

            // Build flat list of all frame entries with poses
            struct FrameEntry {
              std::string dir;
              Eigen::Isometry3d T_world_lidar;
              int num_points;
            };
            std::vector<FrameEntry> all_frames;
            for (const auto& submap : submaps) {
              if (!submap) continue;
              if (hidden_sessions.count(submap->session_id)) continue;
              std::string shd = hd_frames_path;
              for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
              const Eigen::Isometry3d T0 = submap->frames.front()->T_world_imu;
              for (const auto& fr : submap->frames) {
                char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                const std::string fd = shd + "/" + dn;
                auto fi = glim::frame_info_from_meta(fd,
                  glim::compute_frame_world_pose(submap->T_world_origin, submap->T_origin_endpoint_L, T0, fr->T_world_imu, fr->T_lidar_imu));
                if (fi.num_points > 0) all_frames.push_back({fi.dir, fi.T_world_lidar, fi.num_points});
              }
            }

            // Build chunks along trajectory
            auto chunks = glim::build_chunks(trajectory_data, trajectory_total_dist, chunk_spacing, chunk_size * 0.5);
            logger->info("[Ground] {} frames, {} chunks (size={:.0f}m, spacing={:.0f}m, accumulate={}, refine_z={})",
              all_frames.size(), chunks.size(), chunk_size, chunk_spacing, accumulate, refine_z);

            // Track which frames have been written (to avoid double-write at chunk overlaps)
            std::unordered_set<std::string> written_frames;
            int frames_written = 0;

            // Sliding window frame cache (avoids redundant disk I/O for accumulation neighbors)
            struct CachedFrameData {
              std::vector<Eigen::Vector3f> pts;
              std::vector<float> ints;
              int num_points;
            };
            std::unordered_map<std::string, std::shared_ptr<CachedFrameData>> frame_cache;

            for (size_t ci = 0; ci < chunks.size(); ci++) {
              const auto& chunk = chunks[ci];
              const auto chunk_aabb = chunk.world_aabb();
              glim::Chunk core_chunk = chunk;
              core_chunk.half_size = chunk_size * 0.5;

              if (ci % 5 == 0) {
                char buf[128]; std::snprintf(buf, sizeof(buf), "Ground: chunk %zu/%zu (cache: %zu)...", ci + 1, chunks.size(), frame_cache.size());
                rf_status = buf;
              }

              // Find frames overlapping this chunk (sensor within chunk range)
              // Include extra margin for accumulation neighbors
              const float frame_search_radius = chunk_size + 50.0f + (accumulate ? acc_count * 5.0f : 0.0f);
              struct ChunkFrame { size_t all_idx; };
              std::vector<ChunkFrame> chunk_frame_indices;
              std::unordered_set<std::string> needed_dirs;
              for (size_t fi = 0; fi < all_frames.size(); fi++) {
                const auto& entry = all_frames[fi];
                if ((entry.T_world_lidar.translation().cast<float>() - chunk.center.cast<float>()).norm() > frame_search_radius) continue;
                chunk_frame_indices.push_back({fi});
                needed_dirs.insert(entry.dir);
                // Also mark accumulation neighbors as needed
                if (accumulate) {
                  int start = std::max(0, static_cast<int>(fi) - acc_count);
                  int end = std::min(static_cast<int>(all_frames.size()) - 1, static_cast<int>(fi) + acc_count);
                  for (int ni = start; ni <= end; ni++) needed_dirs.insert(all_frames[ni].dir);
                }
              }
              if (chunk_frame_indices.empty()) continue;

              // Evict frames no longer needed
              std::vector<std::string> evict_keys;
              for (const auto& [dir, _] : frame_cache) {
                if (!needed_dirs.count(dir)) evict_keys.push_back(dir);
              }
              for (const auto& k : evict_keys) frame_cache.erase(k);

              // Load missing frames into cache
              for (const auto& dir : needed_dirs) {
                if (frame_cache.count(dir)) continue;
                // Find the frame entry to get num_points
                int np = 0;
                for (const auto& cf : chunk_frame_indices) {
                  if (all_frames[cf.all_idx].dir == dir) { np = all_frames[cf.all_idx].num_points; break; }
                }
                if (np == 0) {
                  // Might be an accumulation neighbor not in chunk_frame_indices — find from all_frames
                  for (const auto& f : all_frames) { if (f.dir == dir) { np = f.num_points; break; } }
                }
                if (np == 0) continue;
                auto cf = std::make_shared<CachedFrameData>();
                cf->num_points = np;
                if (!glim::load_bin(dir + "/points.bin", cf->pts, np)) continue;
                glim::load_bin(dir + "/intensities.bin", cf->ints, np);
                frame_cache[dir] = cf;
              }

              // Phase 1: Run PatchWork++ per frame, store per-frame ground labels + world-space positions
              struct FrameResult {
                std::string dir;
                std::vector<float> ground_values;
                std::vector<Eigen::Vector3f> world_pts;
                Eigen::Isometry3d T_world_lidar;
                int num_points;
                bool in_core;
              };
              std::vector<FrameResult> frame_results;

              for (const auto& cf : chunk_frame_indices) {
                const auto& entry = all_frames[cf.all_idx];
                if (written_frames.count(entry.dir)) continue;

                auto cache_it = frame_cache.find(entry.dir);
                if (cache_it == frame_cache.end()) continue;
                const auto& cached = cache_it->second;
                const int n = cached->num_points;

                std::vector<float> ground_values(n, 0.0f);
                if (df_exclude_ground_pw) {
                  if (accumulate) {
                    std::vector<Eigen::Vector3f> merged_pts(cached->pts);
                    std::vector<float> merged_ints(cached->ints);
                    const Eigen::Isometry3f T_cur_inv = entry.T_world_lidar.inverse().cast<float>();
                    int start = static_cast<int>(cf.all_idx) - acc_count;
                    int end = static_cast<int>(cf.all_idx) + acc_count;
                    if (start < 0) { end = std::min(end - start, static_cast<int>(all_frames.size()) - 1); start = 0; }
                    if (end >= static_cast<int>(all_frames.size())) { start = std::max(start - (end - static_cast<int>(all_frames.size()) + 1), 0); end = static_cast<int>(all_frames.size()) - 1; }
                    for (int ni = start; ni <= end; ni++) {
                      if (ni == static_cast<int>(cf.all_idx)) continue;
                      const auto& nb = all_frames[ni];
                      auto nb_cache = frame_cache.find(nb.dir);
                      if (nb_cache == frame_cache.end()) continue;
                      const auto& nb_data = nb_cache->second;
                      const Eigen::Matrix3f R_to_cur = (T_cur_inv * nb.T_world_lidar.cast<float>()).rotation();
                      const Eigen::Vector3f t_to_cur = (T_cur_inv * nb.T_world_lidar.cast<float>()).translation();
                      for (int pi = 0; pi < nb_data->num_points; pi++) {
                        merged_pts.push_back(R_to_cur * nb_data->pts[pi] + t_to_cur);
                        merged_ints.push_back(pi < static_cast<int>(nb_data->ints.size()) ? nb_data->ints[pi] : 0.0f);
                      }
                    }
                    auto pw_gnd = glim::MapCleanerFilter::classify_ground_patchwork(
                      merged_pts, static_cast<int>(merged_pts.size()), 1.7f, merged_ints);
                    for (int i = 0; i < n; i++) {
                      if (!pw_gnd.empty() && i < static_cast<int>(pw_gnd.size()) && pw_gnd[i]) ground_values[i] = 1.0f;
                    }
                  } else {
                    auto pw_gnd = glim::MapCleanerFilter::classify_ground_patchwork(cached->pts, n, 1.7f, cached->ints);
                    for (int i = 0; i < n; i++) { if (!pw_gnd.empty() && pw_gnd[i]) ground_values[i] = 1.0f; }
                  }
                }

                // Transform to world space for Z refinement
                const Eigen::Matrix3f Rf = entry.T_world_lidar.rotation().cast<float>();
                const Eigen::Vector3f tf = entry.T_world_lidar.translation().cast<float>();
                std::vector<Eigen::Vector3f> world_pts(n);
                for (int i = 0; i < n; i++) world_pts[i] = Rf * cached->pts[i] + tf;

                const bool sensor_in_core = core_chunk.contains(tf);
                frame_results.push_back({entry.dir, std::move(ground_values), std::move(world_pts),
                  entry.T_world_lidar, n, sensor_in_core});
              }

              // Phase 2: Cross-frame Z-column refinement on merged chunk data
              if (refine_z) {
                const float col_res = 1.0f, col_inv = 1.0f / col_res, ground_z_tol = 0.5f;
                // Build global min Z per XY column from ALL ground points in chunk
                std::unordered_map<uint64_t, float> col_min_z;
                for (const auto& fr : frame_results) {
                  for (int i = 0; i < fr.num_points; i++) {
                    if (fr.ground_values[i] < 0.5f) continue;
                    const auto& wp = fr.world_pts[i];
                    const uint64_t ck = (static_cast<uint64_t>(static_cast<int>(std::floor(wp.x() * col_inv)) + 1048576) << 21)
                                      | static_cast<uint64_t>(static_cast<int>(std::floor(wp.y() * col_inv)) + 1048576);
                    auto it = col_min_z.find(ck);
                    if (it == col_min_z.end() || wp.z() < it->second) col_min_z[ck] = wp.z();
                  }
                }
                // Revoke ground for points above column min + tolerance
                int total_revoked = 0;
                for (auto& fr : frame_results) {
                  for (int i = 0; i < fr.num_points; i++) {
                    if (fr.ground_values[i] < 0.5f) continue;
                    const auto& wp = fr.world_pts[i];
                    const uint64_t ck = (static_cast<uint64_t>(static_cast<int>(std::floor(wp.x() * col_inv)) + 1048576) << 21)
                                      | static_cast<uint64_t>(static_cast<int>(std::floor(wp.y() * col_inv)) + 1048576);
                    if (wp.z() > col_min_z[ck] + ground_z_tol) { fr.ground_values[i] = 0.0f; total_revoked++; }
                  }
                }
                if (total_revoked > 0 && ci % 10 == 0)
                  logger->info("[Ground] Chunk {}/{}: Z-column revoked {} points", ci + 1, chunks.size(), total_revoked);
              }

              // Phase 3: Write results (only core-area frames, avoid double-writes)
              for (const auto& fr : frame_results) {
                if (!fr.in_core) continue;
                if (written_frames.count(fr.dir)) continue;
                std::ofstream f(fr.dir + "/aux_ground.bin", std::ios::binary);
                f.write(reinterpret_cast<const char*>(fr.ground_values.data()), sizeof(float) * fr.num_points);
                written_frames.insert(fr.dir);
                frames_written++;
              }
            }

            // Write any remaining frames not covered by chunks (shouldn't happen but safety)
            for (const auto& entry : all_frames) {
              if (written_frames.count(entry.dir)) continue;
              std::vector<Eigen::Vector3f> pts; std::vector<float> ints;
              if (!glim::load_bin(entry.dir + "/points.bin", pts, entry.num_points)) continue;
              glim::load_bin(entry.dir + "/intensities.bin", ints, entry.num_points);
              std::vector<float> ground_values(entry.num_points, 0.0f);
              if (df_exclude_ground_pw) {
                auto pw_gnd = glim::MapCleanerFilter::classify_ground_patchwork(pts, entry.num_points, 1.7f, ints);
                for (int i = 0; i < entry.num_points; i++) { if (!pw_gnd.empty() && pw_gnd[i]) ground_values[i] = 1.0f; }
              }
              std::ofstream f(entry.dir + "/aux_ground.bin", std::ios::binary);
              f.write(reinterpret_cast<const char*>(ground_values.data()), sizeof(float) * entry.num_points);
              written_frames.insert(entry.dir);
              frames_written++;
            }

            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            char buf[256]; std::snprintf(buf, sizeof(buf), "Ground saved: %d frames, %zu chunks (%.1f sec)", frames_written, chunks.size(), elapsed);
            rf_status = buf;
            logger->info("[Ground] {}", rf_status);
            rf_processing = false;
          }).detach();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Classify ground on all HD frames.\nSaves aux_ground.bin scalar field per frame.\nAppears in color mode dropdown on HD reload.");
        ImGui::Separator();

        // Trail refinement
        ImGui::Checkbox("Refine trails", &df_refine_trails);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cluster candidates into elongated trails.\nRejects isolated false positives, fills gaps.");
        if (df_refine_trails) {
          ImGui::SameLine();
          if (ImGui::Button("Config##trail")) { show_trail_config = !show_trail_config; }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open trail refinement configuration.");
        }
      } else if (df_mode == 0) {
        // SOR filter parameters
        ImGui::DragFloat("Search radius (m)", &sor_radius, 0.01f, 0.05f, 5.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Radius for neighbor search.\nPoints with fewer neighbors than threshold are removed.");
        ImGui::DragInt("Min neighbors", &sor_min_neighbors, 1, 1, 50);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum number of neighbors within radius.\nPoints below this are considered outliers.");
        ImGui::DragFloat("Chunk size (m)", &sor_chunk_size, 10.0f, 20.0f, 500.0f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Size of spatial processing cube.\nLarger = more context, more memory.");
      } else if (df_mode == 3) {
        // Scalar visibility — same pattern as range highlight but for any scalar field
        if (!aux_attribute_names.empty()) {
          std::vector<const char*> field_ptrs;
          for (const auto& n : aux_attribute_names) field_ptrs.push_back(n.c_str());
          if (sv_field_idx >= static_cast<int>(field_ptrs.size())) sv_field_idx = 0;

          bool field_changed = ImGui::Combo("Scalar field", &sv_field_idx, field_ptrs.data(), field_ptrs.size());

          // When field changes, switch the main viewer to that colormap
          if (field_changed) {
            color_mode = 3 + sv_field_idx;
            aux_attr_samples.clear();
            aux_cmap_range = Eigen::Vector2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
            update_viewer();
            sv_threshold = 0.0f;
          }

          // Use the viewer's computed percentile range for slider bounds
          const float field_min = aux_cmap_range.x();
          const float field_max = aux_cmap_range.y();

          if (ImGui::SliderFloat("Highlight", &sv_threshold, field_min, field_max, "%.3f")) {
            // Same as range highlight: switch to colormap view, tint above threshold yellow
            auto viewer = guik::LightViewer::instance();
            const auto& attr_name = aux_attribute_names[sv_field_idx];
            const double base = (attr_name == "gps_time") ? gps_time_base : 0.0;

            // Set cmap range to [field_min, threshold] so points above threshold saturate
            viewer->shader_setting().add<Eigen::Vector2f>("cmap_range", Eigen::Vector2f(field_min, sv_threshold));
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Points above this value are highlighted.\nSame as range highlight but for the selected scalar.");

          ImGui::Checkbox("Hide below", &sv_hide_below);
          ImGui::SameLine();
          ImGui::Checkbox("Hide above", &sv_hide_above);

          if (ImGui::Button("Update view")) {
            rf_processing = true;
            rf_preview_active = true;
            lod_hide_all_submaps = true;
            rf_status = "Loading scalar data...";
            std::thread([this] {
              auto vw = guik::LightViewer::instance();
              const Eigen::Matrix4f vm = vw->view_matrix();
              const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));
              const auto& attr_name = aux_attribute_names[sv_field_idx];
              const double base = (attr_name == "gps_time") ? gps_time_base : 0.0;

              std::vector<Eigen::Vector3f> below_pts, above_pts;
              std::vector<float> below_int, above_int;

              for (const auto& submap : submaps) {
                if (!submap) continue;
                if ((submap->T_world_origin.translation().cast<float>() - cam_pos).norm() > lod_hd_range) continue;
                std::string shd = hd_frames_path;
                for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
                const Eigen::Isometry3d T0 = submap->frames.front()->T_world_imu;
                for (const auto& fr : submap->frames) {
                  char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                  const std::string fd = shd + "/" + dn;
                  std::ifstream mf(fd + "/frame_meta.json");
                  auto meta = nlohmann::json::parse(mf, nullptr, false);
                  if (meta.is_discarded()) continue;
                  const int n = meta.value("num_points", 0);
                  if (n == 0) continue;
                  std::vector<Eigen::Vector3f> pts; std::vector<float> rng, ints;
                  if (!glim::load_bin(fd + "/points.bin", pts, n)) continue;
                  glim::load_bin(fd + "/range.bin", rng, n);
                  glim::load_bin(fd + "/intensities.bin", ints, n);
                  // Load the selected scalar field
                  std::vector<float> scalar(n, 0.0f);
                  if (attr_name == "intensity") { scalar = ints; }
                  else if (attr_name == "range") { scalar = rng; }
                  else {
                    // Try loading aux_<name>.bin
                    std::vector<float> aux_f;
                    if (glim::load_bin(fd + "/aux_" + attr_name + ".bin", aux_f, n)) {
                      scalar = aux_f;
                    } else {
                      // Try as double
                      std::vector<double> aux_d;
                      if (glim::load_bin(fd + "/aux_" + attr_name + ".bin", aux_d, n)) {
                        for (int i = 0; i < n; i++) scalar[i] = static_cast<float>(aux_d[i] - base);
                      }
                    }
                  }

                  const auto T = glim::compute_frame_world_pose(submap->T_world_origin, submap->T_origin_endpoint_L, T0, fr->T_world_imu, fr->T_lidar_imu);
                  const Eigen::Matrix3f R = T.rotation().cast<float>();
                  const Eigen::Vector3f t = T.translation().cast<float>();
                  for (int i = 0; i < n; i++) {
                    if (rng.size() > 0 && rng[i] < 1.5f) continue;
                    const Eigen::Vector3f wp = R * pts[i] + t;
                    const float sv = (attr_name == "intensity" || attr_name == "range") ? scalar[i] : scalar[i];
                    if (sv < sv_threshold) {
                      below_pts.push_back(wp); below_int.push_back(ints.empty() ? 0.0f : ints[i]);
                    } else {
                      above_pts.push_back(wp); above_int.push_back(ints.empty() ? 0.0f : ints[i]);
                    }
                  }
                }
              }

              // Render: show both sides, skip hidden
              vw->invoke([this, below_pts, above_pts, below_int, above_int] {
                auto v = guik::LightViewer::instance();
                v->remove_drawable("rf_preview_kept");
                v->remove_drawable("rf_preview_removed");
                // Above threshold = green (kept)
                if (!above_pts.empty() && !sv_hide_above) {
                  const int n = above_pts.size();
                  std::vector<Eigen::Vector4d> p4(n);
                  for (int i = 0; i < n; i++) p4[i] = Eigen::Vector4d(above_pts[i].x(), above_pts[i].y(), above_pts[i].z(), 1.0);
                  auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), n);
                  cb->add_buffer("intensity", above_int);
                  cb->set_colormap_buffer("intensity");
                  v->update_drawable("rf_preview_kept", cb, guik::FlatColor(0.0f, 0.8f, 0.2f, 1.0f));
                }
                // Below threshold = red (removed)
                if (!below_pts.empty() && !sv_hide_below) {
                  const int n = below_pts.size();
                  std::vector<Eigen::Vector4d> p4(n);
                  for (int i = 0; i < n; i++) p4[i] = Eigen::Vector4d(below_pts[i].x(), below_pts[i].y(), below_pts[i].z(), 1.0);
                  auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), n);
                  v->update_drawable("rf_preview_removed", cb, guik::FlatColor(1.0f, 0.0f, 0.0f, 0.5f).make_transparent());
                }
              });

              rf_status = "Scalar: " + std::to_string(above_pts.size()) + " above, " + std::to_string(below_pts.size()) + " below";
              rf_processing = false;
            }).detach();
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Load visible HD data and split by threshold.\nGreen = above, Red = below.\nHide toggles control which side is shown.");
        } else {
          ImGui::Text("No scalar fields available.\nLoad a map with aux attributes.");
        }
      }

      // Reset defaults button
      if (ImGui::Button("Reset defaults")) {
        if (df_mode == 2) {
          rf_voxel_size = 1.0f; rf_safe_range = 20.0f; rf_range_delta = 10.0f;
          rf_far_delta = 30.0f; rf_min_close_pts = 3;
        } else if (df_mode == 1) {
          df_voxel_size = 0.64f; df_range_threshold = 0.8f; df_observation_range = 30.0f;
          df_min_observations = 15; df_refine_ground = true; df_refine_trails = true;
          df_trail_min_length = 7.0f; df_trail_min_aspect = 5.0f; df_trail_min_density = 11.0f;
          df_refine_voxel = 0.23f; df_chunk_size = 120.0f; df_chunk_spacing = 60.0f;
        } else if (df_mode == 0) {
          sor_radius = 0.3f; sor_min_neighbors = 5; sor_chunk_size = 100.0f;
        } else {
          sv_threshold = 0.5f; sv_hide_below = false; sv_hide_above = false;
        }
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset current mode parameters to defaults.");

      // Range highlight (only for Range/Dynamic modes)
      if (df_mode == 2 || df_mode == 1) {
      ImGui::Separator();
      ImGui::Text("Range highlight");
      if (!rf_preview_data.empty()) {
        if (ImGui::SliderFloat("Highlight range (m)", &rf_range_highlight, 0.0f, 200.0f, "%.0f")) {
          if (rf_range_highlight > 0.0f) {
            lod_hide_all_submaps = true;
            // Re-render preview data: intensity-colored below threshold, red above
            std::vector<Eigen::Vector3f> ok_pts, far_pts;
            std::vector<float> ok_int;
            for (const auto& p : rf_preview_data) {
              if (p.range <= rf_range_highlight) {
                ok_pts.push_back(p.pos);
                ok_int.push_back(p.intensity);
              } else {
                far_pts.push_back(p.pos);
              }
            }
            auto vw = guik::LightViewer::instance();
            if (!ok_pts.empty()) {
              const int n = ok_pts.size();
              // Compute intensity range for colormap
              float int_min = std::numeric_limits<float>::max();
              float int_max = std::numeric_limits<float>::lowest();
              for (float v : ok_int) { int_min = std::min(int_min, v); int_max = std::max(int_max, v); }
              if (int_min >= int_max) { int_min = 0.0f; int_max = 255.0f; }
              vw->shader_setting().add<Eigen::Vector2f>("cmap_range", Eigen::Vector2f(int_min, int_max));

              std::vector<Eigen::Vector4d> p4(n);
              for (int i = 0; i < n; i++) p4[i] = Eigen::Vector4d(ok_pts[i].x(), ok_pts[i].y(), ok_pts[i].z(), 1.0);
              auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), n);
              cb->add_buffer("intensity", ok_int);
              cb->set_colormap_buffer("intensity");
              vw->update_drawable("rf_preview_kept", cb, guik::Rainbow().set_color_mode(guik::ColorMode::VERTEX_COLORMAP));
            }
            if (!far_pts.empty()) {
              const int n = far_pts.size();
              std::vector<Eigen::Vector4d> p4(n);
              for (int i = 0; i < n; i++) p4[i] = Eigen::Vector4d(far_pts[i].x(), far_pts[i].y(), far_pts[i].z(), 1.0);
              auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), n);
              vw->update_drawable("rf_preview_removed", cb, guik::FlatColor(1.0f, 0.9f, 0.0f, 0.6f).make_transparent());
            } else {
              vw->remove_drawable("rf_preview_removed");
            }
          }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Intensity-colored below threshold, red above.\nRequires preview data (run Preview first).");
      } else {
        ImGui::BeginDisabled();
        ImGui::SliderFloat("Highlight range (m)", &rf_range_highlight, 0.0f, 200.0f, "%.0f");
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Run Preview first to enable range highlighting.");
      }
      } // end Range highlight (df_mode 0 or 1)

      ImGui::Separator();

      if (!hd_available) ImGui::BeginDisabled();

      if (rf_processing) {
        ImGui::Text("%s", rf_status.c_str());
      } else {
        // Preview buttons
        if (df_mode == 0 || df_mode == 3) if (ImGui::Button("Preview (visible area)")) {
          rf_processing = true;
          rf_preview_active = true;
          lod_hide_all_submaps = true;
          rf_intensity_mode = false;
          rf_status = "Loading frames...";
          std::thread([this] {
            auto vw = guik::LightViewer::instance();
            const Eigen::Matrix4f vm = vw->view_matrix();
            const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));

            // Collect all points from nearby HD frames into a shared world-space voxel grid
            struct PointEntry {
              Eigen::Vector3f world_pos;
              float range;
              float intensity;
              float normal_z;
              float gps_time;
              bool ground_pw;
              int frame_idx;
              int point_idx;
            };

            const float active_voxel_size = (df_mode == 1) ? df_voxel_size : rf_voxel_size;
            const float inv_voxel = 1.0f / active_voxel_size;
            std::unordered_map<uint64_t, std::vector<PointEntry>> voxels;
            int frame_count = 0;

            for (const auto& submap : submaps) {
              if (!submap) continue;
              const float dist = (submap->T_world_origin.translation().cast<float>() - cam_pos).norm();
              if (dist > lod_hd_range) continue;

              std::string session_hd = hd_frames_path;
              for (const auto& sess : sessions) {
                if (sess.id == submap->session_id && !sess.hd_frames_path.empty()) {
                  session_hd = sess.hd_frames_path; break;
                }
              }

              const Eigen::Isometry3d T_ep = submap->T_world_origin * submap->T_origin_endpoint_L;
              const Eigen::Isometry3d T_odom0 = submap->frames.front()->T_world_imu;

              for (const auto& frame : submap->frames) {
                char dir_name[16];
                std::snprintf(dir_name, sizeof(dir_name), "%08ld", frame->id);
                const std::string frame_dir = session_hd + "/" + dir_name;
                const std::string meta_path = frame_dir + "/frame_meta.json";
                if (!boost::filesystem::exists(meta_path)) continue;

                std::ifstream meta_ifs(meta_path);
                const auto meta = nlohmann::json::parse(meta_ifs, nullptr, false);
                if (meta.is_discarded()) continue;
                const int num_pts = meta.value("num_points", 0);
                if (num_pts == 0) continue;

                std::vector<Eigen::Vector3f> pts(num_pts);
                std::vector<float> range(num_pts);
                { std::ifstream f(frame_dir + "/points.bin", std::ios::binary);
                  if (!f) continue;
                  f.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * num_pts); }
                { std::ifstream f(frame_dir + "/range.bin", std::ios::binary);
                  if (!f) continue;
                  f.read(reinterpret_cast<char*>(range.data()), sizeof(float) * num_pts); }
                std::vector<float> intensity(num_pts, 0.0f);
                { std::ifstream f(frame_dir + "/intensities.bin", std::ios::binary);
                  if (f) f.read(reinterpret_cast<char*>(intensity.data()), sizeof(float) * num_pts); }
                std::vector<Eigen::Vector3f> normals(num_pts, Eigen::Vector3f::Zero());
                { std::ifstream f(frame_dir + "/normals.bin", std::ios::binary);
                  if (f) f.read(reinterpret_cast<char*>(normals.data()), sizeof(Eigen::Vector3f) * num_pts); }
                std::vector<float> frame_times(num_pts, 0.0f);
                { std::ifstream f(frame_dir + "/times.bin", std::ios::binary);
                  if (f) f.read(reinterpret_cast<char*>(frame_times.data()), sizeof(float) * num_pts); }
                const double frame_stamp = frame->stamp;

                const Eigen::Isometry3d T_w_imu = T_ep * T_odom0.inverse() * frame->T_world_imu;
                const Eigen::Isometry3d T_w_lidar = T_w_imu * frame->T_lidar_imu.inverse();
                const Eigen::Matrix3f R = T_w_lidar.rotation().cast<float>();
                const Eigen::Vector3f t_vec = T_w_lidar.translation().cast<float>();

                // PatchWork++ ground classification for this frame (cached → scalar file → recompute)
                std::vector<bool> pw_ground;
                if (df_mode == 1 && df_exclude_ground_pw) {
                  auto cache_it = pw_ground_cache.find(frame_dir);
                  if (cache_it != pw_ground_cache.end() && static_cast<int>(cache_it->second.size()) == num_pts) {
                    pw_ground = cache_it->second;
                  } else if (pw_reuse_scalar) {
                    std::vector<float> gnd_scalar;
                    if (glim::load_bin(frame_dir + "/aux_ground.bin", gnd_scalar, num_pts) && static_cast<int>(gnd_scalar.size()) == num_pts) {
                      pw_ground.resize(num_pts);
                      for (int gi = 0; gi < num_pts; gi++) pw_ground[gi] = gnd_scalar[gi] >= 0.5f;
                      pw_ground_cache[frame_dir] = pw_ground;
                    }
                  }
                  if (pw_ground.empty()) {
                    pw_ground = glim::MapCleanerFilter::classify_ground_patchwork(pts, num_pts, 1.7f, intensity);
                    pw_ground_cache[frame_dir] = pw_ground;
                  }
                }

                for (int i = 0; i < num_pts; i++) {
                  if (range[i] < 1.5f) continue;
                  const Eigen::Vector3f wp = R * pts[i] + t_vec;
                  const Eigen::Vector3f wn = (R * normals[i]).normalized();
                  const bool gpw = !pw_ground.empty() && pw_ground[i];
                  const uint64_t key = glim::voxel_key(wp, inv_voxel);
                  const float gps_t = static_cast<float>(frame_stamp - gps_time_base) + frame_times[i];
                  voxels[key].push_back({wp, range[i], intensity[i], std::abs(wn.z()), gps_t, gpw, frame_count, i});
                }
                frame_count++;
              }
            }

            // Safety check: abort if too many voxels (OOM risk)
            if (voxels.size() > 2000000) {
              char buf[256];
              std::snprintf(buf, sizeof(buf),
                "ABORTED: %zu voxels (>2M). Increase voxel size or reduce SD range.",
                voxels.size());
              rf_status = buf;
              rf_processing = false;
              return;
            }
            rf_status = "Filtering " + std::to_string(voxels.size()) + " voxels...";

            // Filter: per-voxel criterion based on mode
            std::vector<Eigen::Vector3f> kept_points, removed_points;
            std::vector<float> kept_intensities, kept_ranges;
            std::vector<float> removed_ranges, removed_intensities;
            size_t preview_kept = 0, preview_removed = 0;

            if (df_mode == 2 && rf_criteria == 0) {
              // --- RANGE MODE (range criteria) ---
              for (const auto& [key, entries] : voxels) {
                float max_close_range = 0.0f;
                int close_count = 0;
                for (const auto& e : entries) {
                  if (e.range <= rf_safe_range) {
                    max_close_range = std::max(max_close_range, e.range);
                    close_count++;
                  }
                }

                if (close_count < rf_min_close_pts) {
                  float min_range = std::numeric_limits<float>::max();
                  for (const auto& e : entries) min_range = std::min(min_range, e.range);
                  const float far_threshold = min_range + rf_far_delta;
                  for (const auto& e : entries) {
                    if (e.range <= far_threshold) {
                      kept_points.push_back(e.world_pos); kept_intensities.push_back(e.intensity); kept_ranges.push_back(e.range); preview_kept++;
                    } else {
                      removed_points.push_back(e.world_pos); removed_intensities.push_back(e.intensity); removed_ranges.push_back(e.range); preview_removed++;
                    }
                  }
                  continue;
                }

                const float threshold = max_close_range + rf_range_delta;
                for (const auto& e : entries) {
                  if (e.range <= rf_safe_range || e.range <= threshold) {
                    kept_points.push_back(e.world_pos); kept_intensities.push_back(e.intensity); kept_ranges.push_back(e.range); preview_kept++;
                  } else {
                    removed_points.push_back(e.world_pos); removed_intensities.push_back(e.intensity); removed_ranges.push_back(e.range); preview_removed++;
                  }
                }
              }
            } else if (df_mode == 2 && rf_criteria == 1) {
              // --- RANGE MODE (GPS time criteria) ---
              // Per voxel: cluster points by GPS time, keep the dominant cluster
              const float time_gap = 5.0f;  // seconds — points within this gap are same cluster
              for (const auto& [key, entries] : voxels) {
                if (entries.size() <= 1) {
                  for (const auto& e : entries) { kept_points.push_back(e.world_pos); kept_intensities.push_back(e.intensity); kept_ranges.push_back(e.range); preview_kept++; }
                  continue;
                }
                // Sort by GPS time
                std::vector<int> sorted_idx(entries.size());
                std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
                std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) { return entries[a].gps_time < entries[b].gps_time; });

                // Cluster by time gap
                std::vector<std::vector<int>> clusters;
                clusters.push_back({sorted_idx[0]});
                for (size_t i = 1; i < sorted_idx.size(); i++) {
                  if (entries[sorted_idx[i]].gps_time - entries[sorted_idx[i - 1]].gps_time > time_gap) {
                    clusters.push_back({});
                  }
                  clusters.back().push_back(sorted_idx[i]);
                }

                if (clusters.size() <= 1) {
                  // Only one cluster — keep all
                  for (const auto& e : entries) { kept_points.push_back(e.world_pos); kept_intensities.push_back(e.intensity); kept_ranges.push_back(e.range); preview_kept++; }
                  continue;
                }

                // Select cluster to keep
                int best_cluster = 0;
                if (rf_gps_keep == 0) { for (int ci = 1; ci < static_cast<int>(clusters.size()); ci++) { if (clusters[ci].size() > clusters[best_cluster].size()) best_cluster = ci; } }
                else if (rf_gps_keep == 1) { best_cluster = static_cast<int>(clusters.size()) - 1; }
                // else rf_gps_keep == 2: best_cluster = 0 (oldest)

                // Keep dominant, remove others
                std::unordered_set<int> keep_set(clusters[best_cluster].begin(), clusters[best_cluster].end());
                for (int ei = 0; ei < static_cast<int>(entries.size()); ei++) {
                  if (keep_set.count(ei)) {
                    kept_points.push_back(entries[ei].world_pos); kept_intensities.push_back(entries[ei].intensity); kept_ranges.push_back(entries[ei].range); preview_kept++;
                  } else {
                    removed_points.push_back(entries[ei].world_pos); removed_intensities.push_back(entries[ei].intensity); removed_ranges.push_back(entries[ei].range); preview_removed++;
                  }
                }
              }
            } else if (df_mode == 1) {
              // --- DYNAMIC MODE (MapCleaner algorithm) ---
              rf_status = "Collecting frames...";

              // Collect frame metadata for nearby HD frames
              std::vector<glim::MapCleanerFilter::FrameData> mc_frames;
              for (const auto& submap : submaps) {
                if (!submap) continue;
                const float sdist = (submap->T_world_origin.translation().cast<float>() - cam_pos).norm();
                if (sdist > lod_hd_range + 20.0f) continue;
                std::string session_hd = hd_frames_path;
                for (const auto& sess : sessions) {
                  if (sess.id == submap->session_id && !sess.hd_frames_path.empty()) {
                    session_hd = sess.hd_frames_path; break;
                  }
                }
                const Eigen::Isometry3d T_odom0 = submap->frames.front()->T_world_imu;
                for (const auto& frame : submap->frames) {
                  char dir_name[16];
                  std::snprintf(dir_name, sizeof(dir_name), "%08ld", frame->id);
                  const std::string frame_dir = session_hd + "/" + dir_name;
                  if (!boost::filesystem::exists(frame_dir + "/frame_meta.json")) continue;
                  std::ifstream mf(frame_dir + "/frame_meta.json");
                  const auto meta = nlohmann::json::parse(mf, nullptr, false);
                  if (meta.is_discarded()) continue;
                  const int npts = meta.value("num_points", 0);
                  if (npts == 0) continue;
                  const Eigen::Isometry3d T_w_lidar = glim::compute_frame_world_pose(
                    submap->T_world_origin, submap->T_origin_endpoint_L, T_odom0, frame->T_world_imu, frame->T_lidar_imu);
                  mc_frames.push_back({frame_dir, T_w_lidar, npts});
                }
              }

              // Flatten voxel points for MapCleaner + compute ground flags
              std::vector<Eigen::Vector3f> mc_points;
              std::vector<float> mc_ranges;
              std::vector<bool> mc_ground;
              std::vector<std::pair<uint64_t, int>> mc_refs;
              for (const auto& [key, entries] : voxels) {
                for (int ei = 0; ei < static_cast<int>(entries.size()); ei++) {
                  mc_points.push_back(entries[ei].world_pos);
                  mc_ranges.push_back(entries[ei].range);
                  const bool is_gnd = df_exclude_ground_pw && entries[ei].ground_pw;
                  mc_ground.push_back(is_gnd);
                  mc_refs.push_back({key, ei});
                }
              }

              // Configure and run
              rf_status = "Running MapCleaner (" + std::to_string(mc_frames.size()) + " frames, " + std::to_string(mc_points.size()) + " points)...";
              glim::MapCleanerFilter::Params mc_params;
              mc_params.range_threshold = df_range_threshold;
              mc_params.lidar_range = df_observation_range;
              mc_params.voxel_size = df_voxel_size;
              mc_params.frame_skip = (mc_frames.size() > 150) ? static_cast<int>(mc_frames.size() / 150) : 0;
              mc_params.exclude_ground_pw = df_exclude_ground_pw;

              glim::MapCleanerFilter filter(mc_params);
              auto mc_result = filter.compute(mc_frames, mc_points, mc_ranges, mc_ground);
              logger->info("[Dynamic] MapCleaner: {} static, {} dynamic ({} frames)",
                mc_result.num_static, mc_result.num_dynamic, mc_frames.size());

              for (size_t i = 0; i < mc_points.size(); i++) {
                const auto& e = voxels.at(mc_refs[i].first)[mc_refs[i].second];
                if (mc_result.is_dynamic[i]) {
                  removed_points.push_back(e.world_pos); removed_intensities.push_back(e.intensity); removed_ranges.push_back(e.range); preview_removed++;
                } else {
                  kept_points.push_back(e.world_pos); kept_intensities.push_back(e.intensity); kept_ranges.push_back(e.range); preview_kept++;
                }
              }
            } else if (df_mode == 0) {
              // --- SOR MODE ---
              // Flatten all voxel points for KD-tree
              std::vector<Eigen::Vector3f> all_pts;
              std::vector<float> all_ints, all_rngs;
              for (const auto& [key, entries] : voxels) {
                for (const auto& e : entries) {
                  all_pts.push_back(e.world_pos);
                  all_ints.push_back(e.intensity);
                  all_rngs.push_back(e.range);
                }
              }
              // Build KD-tree for neighbor search using gtsam_points::KdTree (Vector4d)
              std::vector<Eigen::Vector4d> pts4(all_pts.size());
              for (size_t i = 0; i < all_pts.size(); i++) pts4[i] = Eigen::Vector4d(all_pts[i].x(), all_pts[i].y(), all_pts[i].z(), 1.0);
              gtsam_points::KdTree kdt(pts4.data(), pts4.size());
              const float r2 = sor_radius * sor_radius;

              rf_status = "SOR: checking " + std::to_string(all_pts.size()) + " points...";
              for (size_t i = 0; i < all_pts.size(); i++) {
                // Count neighbors within radius
                std::vector<size_t> k_indices(sor_min_neighbors + 1);
                std::vector<double> k_sq_dists(sor_min_neighbors + 1);
                const int found = kdt.knn_search(pts4[i].data(), sor_min_neighbors + 1, k_indices.data(), k_sq_dists.data());
                // Check if the Nth nearest neighbor (excluding self) is within radius
                int nn = 0;
                for (int j = 0; j < found; j++) {
                  if (k_indices[j] == i) continue;  // skip self
                  if (k_sq_dists[j] <= static_cast<double>(r2)) nn++;
                }
                if (nn >= sor_min_neighbors) {
                  kept_points.push_back(all_pts[i]); kept_intensities.push_back(all_ints[i]); kept_ranges.push_back(all_rngs[i]); preview_kept++;
                } else {
                  removed_points.push_back(all_pts[i]); removed_intensities.push_back(all_ints[i]); removed_ranges.push_back(all_rngs[i]); preview_removed++;
                }
              }
            } else if (df_mode == 3) {
              // --- SCALAR VISIBILITY MODE ---
              // Split all points by the selected scalar field threshold
              const std::string field_name = (sv_field_idx < static_cast<int>(aux_attribute_names.size()))
                ? aux_attribute_names[sv_field_idx] : "ground";

              for (const auto& [key, entries] : voxels) {
                for (const auto& e : entries) {
                  // Get scalar value — use aux attribute from the submap frame
                  // For preview, we approximate using intensity/range/normal_z/ground_pw
                  float scalar_val = 0.0f;
                  if (field_name == "intensity") scalar_val = e.intensity;
                  else if (field_name == "range") scalar_val = e.range;
                  else if (field_name == "ground") scalar_val = (df_exclude_ground_pw && e.ground_pw) ? 1.0f : 0.0f;
                  else scalar_val = e.range;  // fallback

                  const bool below = scalar_val < sv_threshold;
                  const bool hidden = (below && sv_hide_below) || (!below && sv_hide_above);
                  if (!hidden) {
                    kept_points.push_back(e.world_pos); kept_intensities.push_back(e.intensity); kept_ranges.push_back(e.range); preview_kept++;
                  } else {
                    removed_points.push_back(e.world_pos); removed_intensities.push_back(e.intensity); removed_ranges.push_back(e.range); preview_removed++;
                  }
                }
              }
            }

            // Cache preview data for range highlight re-coloring
            rf_preview_data.clear();
            rf_preview_data.reserve(preview_kept + preview_removed);
            for (size_t pi = 0; pi < kept_points.size(); pi++) {
              rf_preview_data.push_back({kept_points[pi], kept_ranges[pi], kept_intensities[pi], 0.0f, false, true});
            }
            for (size_t pi = 0; pi < removed_points.size(); pi++) {
              rf_preview_data.push_back({removed_points[pi], removed_ranges[pi], removed_intensities[pi], 0.0f, false, false});
            }

            auto kept_buf = std::make_shared<std::vector<Eigen::Vector3f>>(std::move(kept_points));
            auto kept_int = std::make_shared<std::vector<float>>(std::move(kept_intensities));
            auto removed_buf = std::make_shared<std::vector<Eigen::Vector3f>>(std::move(removed_points));

            const bool hide_b = sv_hide_below && (df_mode == 3);
            const bool hide_a = sv_hide_above && (df_mode == 3);
            vw->invoke([this, kept_buf, kept_int, removed_buf, preview_kept, preview_removed, hide_b, hide_a] {
              auto viewer = guik::LightViewer::instance();
              // In scalar mode: "kept" = above threshold, "removed" = below threshold
              // hide_above hides "kept", hide_below hides "removed"
              if (!kept_buf->empty() && !hide_a) {
                const int n = kept_buf->size();
                std::vector<Eigen::Vector4d> pts4(n);
                for (int i = 0; i < n; i++) pts4[i] = Eigen::Vector4d((*kept_buf)[i].x(), (*kept_buf)[i].y(), (*kept_buf)[i].z(), 1.0);
                auto cb = std::make_shared<glk::PointCloudBuffer>(pts4.data(), n);
                if (kept_int->size() == static_cast<size_t>(n)) {
                  cb->add_buffer("intensity", *kept_int);
                  cb->set_colormap_buffer("intensity");
                }
                viewer->update_drawable("rf_preview_kept", cb, guik::FlatColor(0.0f, 0.8f, 0.2f, 1.0f));
              } else {
                viewer->remove_drawable("rf_preview_kept");
              }
              if (!removed_buf->empty() && !hide_b) {
                const int n = removed_buf->size();
                std::vector<Eigen::Vector4d> pts4(n);
                for (int i = 0; i < n; i++) pts4[i] = Eigen::Vector4d((*removed_buf)[i].x(), (*removed_buf)[i].y(), (*removed_buf)[i].z(), 1.0);
                auto cb = std::make_shared<glk::PointCloudBuffer>(pts4.data(), n);
                viewer->update_drawable("rf_preview_removed", cb, guik::FlatColor(1.0f, 0.0f, 0.0f, 0.5f).make_transparent());
              } else {
                viewer->remove_drawable("rf_preview_removed");
              }
            });

            char buf[256];
            std::snprintf(buf, sizeof(buf), "Preview: %d frames, %zu voxels, %.1f M kept, %.1f M removed (%.1f%%)",
                          frame_count, voxels.size(),
                          static_cast<double>(preview_kept) / 1e6,
                          static_cast<double>(preview_removed) / 1e6,
                          (preview_kept + preview_removed > 0) ? 100.0 * preview_removed / (preview_kept + preview_removed) : 0.0);
            rf_status = buf;
            rf_processing = false;
          }).detach();
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("Cross-frame filter in view area (no disk writes).\nGreen = kept, Red = would be removed.\nVoxels span multiple frames to detect redundancy.");
        }

        if (df_mode == 1) {
          ImGui::Checkbox("Reuse ground scalar", &pw_reuse_scalar);
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Use existing aux_ground.bin instead of recomputing PatchWork++.\nEnable after running Classify ground to scalar with tuned params.");
        }

        if (df_mode == 1 || df_mode == 2) {
          if (df_mode == 0 || df_mode == 3) ImGui::SameLine();  // SOR/Scalar: inline with Preview
          if (ImGui::Button("Process chunk")) {
            rf_processing = true;
            rf_preview_active = true;
            lod_hide_all_submaps = true;
            rf_intensity_mode = false;
            rf_status = "Processing chunk with overlap...";
            std::thread([this] {
              if (!trajectory_built) build_trajectory();
              auto vw = guik::LightViewer::instance();
              const Eigen::Matrix4f vm = vw->view_matrix();
              const Eigen::Vector3f cam_pos = source_finder_active ? source_finder_pos
                : Eigen::Vector3f(-(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3)));

              // Find nearest trajectory point to camera
              double min_dist_traj = std::numeric_limits<double>::max();
              double chunk_dist = 0.0;
              for (const auto& tp : trajectory_data) {
                const double d = (tp.pose.translation().cast<float>() - cam_pos).cast<double>().norm();
                if (d < min_dist_traj) { min_dist_traj = d; chunk_dist = tp.cumulative_dist; }
              }

              // Build one chunk centered here with overlap
              const double core_size = (df_mode == 1) ? df_chunk_size : rf_chunk_size;
              const double overlap = core_size * 0.5;
              const double chunk_total = core_size + 2.0 * overlap;
              // Override: build a single chunk at the found position
              glim::Chunk chunk;
              {
                size_t idx = 0;
                for (size_t k = 1; k < trajectory_data.size(); k++) {
                  if (trajectory_data[k].cumulative_dist >= chunk_dist) { idx = k; break; }
                }
                const Eigen::Vector3d c = trajectory_data[idx].pose.translation();
                const size_t next = std::min(idx + 1, trajectory_data.size() - 1);
                Eigen::Vector3d fwd = trajectory_data[next].pose.translation() - trajectory_data[idx].pose.translation();
                fwd.z() = 0.0;
                if (fwd.norm() < 0.01) fwd = Eigen::Vector3d::UnitX(); else fwd.normalize();
                const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
                const Eigen::Vector3d right = fwd.cross(up).normalized();
                Eigen::Matrix3d R; R.col(0) = fwd; R.col(1) = right; R.col(2) = up;
                chunk = {c, R, R.transpose(), chunk_total * 0.5, 50.0};
              }
              glim::Chunk core_chunk = chunk;
              core_chunk.half_size = core_size * 0.5;

              // Index all frames
              rf_status = "Indexing frames...";
              std::vector<glim::MapCleanerFilter::FrameData> all_mc_frames;
              for (const auto& submap : submaps) {
                if (!submap) continue;
                if (hidden_sessions.count(submap->session_id)) continue;
                std::string session_hd = hd_frames_path;
                for (const auto& sess : sessions) {
                  if (sess.id == submap->session_id && !sess.hd_frames_path.empty()) {
                    session_hd = sess.hd_frames_path; break;
                  }
                }
                const Eigen::Isometry3d T_odom0 = submap->frames.front()->T_world_imu;
                for (const auto& frame : submap->frames) {
                  char dir_name[16];
                  std::snprintf(dir_name, sizeof(dir_name), "%08ld", frame->id);
                  const std::string frame_dir = session_hd + "/" + dir_name;
                  auto fi = glim::frame_info_from_meta(frame_dir,
                    glim::compute_frame_world_pose(submap->T_world_origin, submap->T_origin_endpoint_L, T_odom0, frame->T_world_imu, frame->T_lidar_imu));
                  if (fi.num_points > 0) {
                    // Quick distance check
                    if ((fi.T_world_lidar.translation().cast<float>() - chunk.center.cast<float>()).norm() < chunk_total + df_observation_range) {
                      all_mc_frames.push_back({fi.dir, fi.T_world_lidar, fi.num_points});
                    }
                  }
                }
              }

              // Load points within chunk
              rf_status = "Loading chunk points...";
              std::vector<Eigen::Vector3f> chunk_pts;
              std::vector<float> chunk_ranges, chunk_intensities;
              std::vector<bool> chunk_ground;
              std::vector<float> chunk_normal_z;
              std::vector<bool> chunk_ground_pw;
              std::vector<bool> chunk_is_ground_scalar;  // from aux_ground.bin (for range ground-only)
              std::vector<float> chunk_gps_times;         // for GPS time criteria in range mode
              const bool range_ground_only = (df_mode == 2) && rf_ground_only;
              const bool need_gps_time = (df_mode == 2) && (rf_criteria == 1);
              const auto chunk_aabb = chunk.world_aabb();

              for (const auto& fd : all_mc_frames) {
                std::vector<Eigen::Vector3f> pts;
                std::vector<float> rng, ints(fd.num_points, 0.0f);
                if (!glim::load_bin(fd.dir + "/points.bin", pts, fd.num_points)) continue;
                if (!glim::load_bin(fd.dir + "/range.bin", rng, fd.num_points)) continue;
                glim::load_bin(fd.dir + "/intensities.bin", ints, fd.num_points);
                // Load times for GPS time criteria
                std::vector<float> frame_times;
                double frame_stamp = 0.0;
                if (need_gps_time) {
                  glim::load_bin(fd.dir + "/times.bin", frame_times, fd.num_points);
                  std::ifstream mf(fd.dir + "/frame_meta.json");
                  auto meta = nlohmann::json::parse(mf, nullptr, false);
                  if (!meta.is_discarded()) frame_stamp = meta.value("stamp", 0.0);
                }
                // Load ground scalar for range ground-only mode
                std::vector<float> frame_ground_scalar;
                if (range_ground_only) glim::load_bin(fd.dir + "/aux_ground.bin", frame_ground_scalar, fd.num_points);
                std::vector<Eigen::Vector3f> nrm(fd.num_points, Eigen::Vector3f::Zero());
                if (df_exclude_ground_pw) glim::load_bin(fd.dir + "/normals.bin", nrm, fd.num_points);
                std::vector<bool> pw_gnd;
                if (df_exclude_ground_pw) {
                  auto cache_it = pw_ground_cache.find(fd.dir);
                  if (cache_it != pw_ground_cache.end() && static_cast<int>(cache_it->second.size()) == fd.num_points) {
                    pw_gnd = cache_it->second;
                  } else if (pw_reuse_scalar) {
                    std::vector<float> gnd_scalar;
                    if (glim::load_bin(fd.dir + "/aux_ground.bin", gnd_scalar, fd.num_points) && static_cast<int>(gnd_scalar.size()) == fd.num_points) {
                      pw_gnd.resize(fd.num_points);
                      for (int gi = 0; gi < fd.num_points; gi++) pw_gnd[gi] = gnd_scalar[gi] >= 0.5f;
                      pw_ground_cache[fd.dir] = pw_gnd;
                    }
                  }
                  if (pw_gnd.empty()) {
                    pw_gnd = glim::MapCleanerFilter::classify_ground_patchwork(pts, fd.num_points, 1.7f, ints);
                    pw_ground_cache[fd.dir] = pw_gnd;
                  }
                }

                const Eigen::Matrix3f R = fd.T_world_lidar.rotation().cast<float>();
                const Eigen::Vector3f t = fd.T_world_lidar.translation().cast<float>();
                for (int i = 0; i < fd.num_points; i++) {
                  if (rng[i] < 1.5f) continue;
                  const Eigen::Vector3f wp = R * pts[i] + t;
                  if (!chunk.contains(wp)) continue;
                  chunk_pts.push_back(wp);
                  chunk_ranges.push_back(rng[i]);
                  chunk_intensities.push_back(ints[i]);
                  const Eigen::Vector3f wn = (R * nrm[i]).normalized();
                  const float nz = std::abs(wn.z());
                  const bool gpw = !pw_gnd.empty() && pw_gnd[i];
                  if (range_ground_only) {
                    chunk_is_ground_scalar.push_back(
                      i < static_cast<int>(frame_ground_scalar.size()) && frame_ground_scalar[i] >= 0.5f);
                  }
                  if (need_gps_time) {
                    const float base = (gps_time_base > 0.0) ? static_cast<float>(frame_stamp - gps_time_base) : 0.0f;
                    chunk_gps_times.push_back(base + (i < static_cast<int>(frame_times.size()) ? frame_times[i] : 0.0f));
                  }
                  chunk_normal_z.push_back(nz);
                  chunk_ground_pw.push_back(gpw);
                  chunk_ground.push_back(df_exclude_ground_pw && gpw);
                }
              }

              // Mode-specific processing
              std::vector<Eigen::Vector3f> kept_points, removed_points;
              std::vector<float> kept_ints, removed_ints;
              rf_preview_data.clear();

              if (df_mode == 1) {
              // ========== DYNAMIC MODE ==========
              // Pre-MapCleaner ground refinement: revoke false ground labels
              if (df_refine_ground && df_exclude_ground_pw) {
                const float col_res = 1.0f, col_inv = 1.0f / col_res, ground_z_tol = 0.5f;
                // Find min Z per XY column
                std::unordered_map<uint64_t, float> col_min_z;
                for (size_t i = 0; i < chunk_pts.size(); i++) {
                  const uint64_t ck = (static_cast<uint64_t>(static_cast<int>(std::floor(chunk_pts[i].x() * col_inv)) + 1048576) << 21)
                                    | static_cast<uint64_t>(static_cast<int>(std::floor(chunk_pts[i].y() * col_inv)) + 1048576);
                  auto it = col_min_z.find(ck);
                  if (it == col_min_z.end() || chunk_pts[i].z() < it->second) col_min_z[ck] = chunk_pts[i].z();
                }
                // Revoke ground for points above column min + tolerance
                int revoked = 0;
                for (size_t i = 0; i < chunk_pts.size(); i++) {
                  if (!chunk_ground[i]) continue;
                  const uint64_t ck = (static_cast<uint64_t>(static_cast<int>(std::floor(chunk_pts[i].x() * col_inv)) + 1048576) << 21)
                                    | static_cast<uint64_t>(static_cast<int>(std::floor(chunk_pts[i].y() * col_inv)) + 1048576);
                  if (chunk_pts[i].z() > col_min_z[ck] + ground_z_tol) {
                    chunk_ground[i] = false;  // revoke for MapCleaner voting only
                    // DO NOT clear chunk_normal_z or chunk_ground_pw — gap-fill needs them for protection
                    revoked++;
                  }
                }
                // Also revoke ground for high-intensity points (reflective plates, signs)
                if (!chunk_intensities.empty()) {
                  // Compute intensity percentile for ground points
                  std::vector<float> gnd_ints;
                  for (size_t i = 0; i < chunk_pts.size(); i++) {
                    if (chunk_ground[i]) gnd_ints.push_back(chunk_intensities[i]);
                  }
                  if (!gnd_ints.empty()) {
                    std::sort(gnd_ints.begin(), gnd_ints.end());
                    const float int_p95 = gnd_ints[static_cast<size_t>(gnd_ints.size() * 0.95)];
                    const float int_threshold = int_p95 * 2.0f;  // points with 2x the 95th percentile ground intensity = not ground
                    int int_revoked = 0;
                    for (size_t i = 0; i < chunk_pts.size(); i++) {
                      if (!chunk_ground[i]) continue;
                      if (chunk_intensities[i] > int_threshold) {
                        chunk_ground[i] = false;  // revoke for MapCleaner only
                        int_revoked++;
                      }
                    }
                    if (int_revoked > 0) logger->info("[Refine] Revoked {} ground by intensity (threshold={:.0f})", int_revoked, int_threshold);
                  }
                }
                if (revoked > 0) logger->info("[Refine] Revoked {} ground by Z column", revoked);
              }

              rf_status = "Running MapCleaner (" + std::to_string(chunk_pts.size()) + " pts, " + std::to_string(all_mc_frames.size()) + " frames)...";
              glim::MapCleanerFilter::Params mc_params;
              mc_params.range_threshold = df_range_threshold;
              mc_params.lidar_range = df_observation_range;
              mc_params.voxel_size = df_voxel_size;
              mc_params.frame_skip = (all_mc_frames.size() > 200) ? static_cast<int>(all_mc_frames.size() / 200) : 0;
              mc_params.exclude_ground_pw = df_exclude_ground_pw;

              logger->info("[Dynamic chunk] {} frames (skip={}), {} pts, chunk_size={:.0f}m",
                all_mc_frames.size(), mc_params.frame_skip, chunk_pts.size(), core_size);
              glim::MapCleanerFilter filter(mc_params);
              auto result = filter.compute(all_mc_frames, chunk_pts, chunk_ranges, chunk_ground);
              logger->info("[Dynamic chunk] {} static, {} dynamic", result.num_static, result.num_dynamic);

              // Classify core-area points as kept/removed
              for (size_t i = 0; i < chunk_pts.size(); i++) {
                if (!core_chunk.contains(chunk_pts[i])) continue;  // only show core area
                const float nz = (i < static_cast<int>(chunk_normal_z.size())) ? chunk_normal_z[i] : 0.0f;
                const bool gpw = (i < static_cast<int>(chunk_ground_pw.size())) && chunk_ground_pw[i];
                if (result.is_dynamic[i]) {
                  removed_points.push_back(chunk_pts[i]);
                  removed_ints.push_back(chunk_intensities[i]);
                  rf_preview_data.push_back({chunk_pts[i], chunk_ranges[i], chunk_intensities[i], nz, gpw, false});
                } else {
                  kept_points.push_back(chunk_pts[i]);
                  kept_ints.push_back(chunk_intensities[i]);
                  rf_preview_data.push_back({chunk_pts[i], chunk_ranges[i], chunk_intensities[i], nz, gpw, true});
                }
              }

              // (Ground Z refinement already applied before MapCleaner)

              // Trail clustering refinement
              if (df_refine_trails) {
                const float rv = df_refine_voxel, inv_rv = 1.0f / rv, voxel_vol = rv * rv * rv;
                std::unordered_map<uint64_t, std::vector<int>> cand_vox, all_vox;
                for (int i = 0; i < static_cast<int>(rf_preview_data.size()); i++) {
                  const auto& pp = rf_preview_data[i];
                  const uint64_t k = glim::voxel_key(pp.pos, inv_rv);
                  all_vox[k].push_back(i);
                  // Only add as candidate if NOT ground (ground should never be a trail candidate)
                  if (!pp.kept) {
                    const bool is_gnd = df_exclude_ground_pw && pp.ground_pw;
                    if (!is_gnd) cand_vox[k].push_back(i);
                    else { rf_preview_data[i].kept = true; }  // force ground back to kept
                  }
                }
                // BFS clustering
                std::unordered_map<uint64_t, int> vox_cluster;
                std::vector<std::vector<uint64_t>> clusters;
                int nc = 0;
                for (const auto& [k, _] : cand_vox) {
                  if (vox_cluster.count(k)) continue;
                  std::vector<uint64_t> ck; std::queue<uint64_t> q;
                  q.push(k); vox_cluster[k] = nc;
                  while (!q.empty()) {
                    const uint64_t c = q.front(); q.pop(); ck.push_back(c);
                    const int cx = static_cast<int>((c >> 42) & 0x1FFFFF) - 1048576;
                    const int cy = static_cast<int>((c >> 21) & 0x1FFFFF) - 1048576;
                    const int cz = static_cast<int>(c & 0x1FFFFF) - 1048576;
                    for (int dz=-1;dz<=1;dz++) for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++) {
                      if (!dx && !dy && !dz) continue;
                      const uint64_t nk = glim::voxel_key(cx+dx, cy+dy, cz+dz);
                      if (cand_vox.count(nk) && !vox_cluster.count(nk)) { vox_cluster[nk] = nc; q.push(nk); }
                    }
                  }
                  clusters.push_back(std::move(ck)); nc++;
                }
                // Evaluate clusters
                std::unordered_set<uint64_t> trail_voxels;
                for (int ci = 0; ci < nc; ci++) {
                  Eigen::Vector3f bmin = Eigen::Vector3f::Constant(1e9f), bmax = Eigen::Vector3f::Constant(-1e9f);
                  int tp = 0;
                  for (const auto& vk : clusters[ci]) {
                    auto it = cand_vox.find(vk);
                    if (it == cand_vox.end()) continue;
                    for (int idx : it->second) { bmin = bmin.cwiseMin(rf_preview_data[idx].pos); bmax = bmax.cwiseMax(rf_preview_data[idx].pos); tp++; }
                  }
                  const Eigen::Vector3f ext = bmax - bmin;
                  const float longest = ext.maxCoeff(), shortest = std::max(0.01f, ext.minCoeff());
                  const float density = tp / std::max(0.001f, static_cast<float>(clusters[ci].size()) * voxel_vol);
                  if (longest >= df_trail_min_length && longest/shortest >= df_trail_min_aspect && density >= df_trail_min_density) {
                    for (const auto& vk : clusters[ci]) trail_voxels.insert(vk);
                    logger->info("[Refine] Trail: {:.1f}x{:.1f}x{:.1f}m, density={:.0f}, {} pts", ext.x(), ext.y(), ext.z(), density, tp);
                  }
                }
                // Reject non-trail candidates, fill gaps (excluding ground)
                int rejected = 0, filled = 0;
                for (const auto& [vk, indices] : cand_vox) {
                  if (!trail_voxels.count(vk)) { for (int idx : indices) { rf_preview_data[idx].kept = true; rejected++; } }
                }
                // Gap fill: only fill kept points that are ABOVE the trail's dynamic points in the same voxel
                // This prevents road surface below the trail from being swept up
                for (const auto& vk : trail_voxels) {
                  // Find the Z range of existing dynamic points in this voxel
                  auto cit = cand_vox.find(vk);
                  if (cit == cand_vox.end()) continue;
                  float trail_min_z = std::numeric_limits<float>::max();
                  float trail_max_z = std::numeric_limits<float>::lowest();
                  for (int idx : cit->second) {
                    trail_min_z = std::min(trail_min_z, rf_preview_data[idx].pos.z());
                    trail_max_z = std::max(trail_max_z, rf_preview_data[idx].pos.z());
                  }
                  // Only fill kept points within the trail's Z range (not below)
                  auto ait = all_vox.find(vk);
                  if (ait == all_vox.end()) continue;
                  for (int idx : ait->second) {
                    if (!rf_preview_data[idx].kept) continue;
                    const auto& pp = rf_preview_data[idx];
                    // Skip ground
                    if (df_exclude_ground_pw && pp.ground_pw) continue;
                    // Only fill if clearly above ground (at or above trail min Z)
                    if (pp.pos.z() < trail_min_z) continue;
                    rf_preview_data[idx].kept = false; filled++;
                  }
                }
                logger->info("[Refine] {} rejected, {} gaps filled, {} trail voxels", rejected, filled, trail_voxels.size());
              }

              // Rebuild kept/removed from refined preview data
              kept_points.clear(); removed_points.clear(); kept_ints.clear();
              for (const auto& p : rf_preview_data) {
                if (p.kept) { kept_points.push_back(p.pos); kept_ints.push_back(p.intensity); }
                else removed_points.push_back(p.pos);
              }

              } else if (df_mode == 2) {
              // ========== RANGE MODE ==========
              // Build voxel grid from chunk points (ground-only: only ground enters the grid)
              const float inv_voxel = 1.0f / rf_voxel_size;
              struct VoxEntry { size_t idx; float range; float gps_time; };
              std::unordered_map<uint64_t, std::vector<VoxEntry>> voxels;
              for (size_t i = 0; i < chunk_pts.size(); i++) {
                if (!core_chunk.contains(chunk_pts[i])) continue;
                if (range_ground_only && (i >= chunk_is_ground_scalar.size() || !chunk_is_ground_scalar[i])) continue;
                const uint64_t key = glim::voxel_key(chunk_pts[i], inv_voxel);
                const float gt = (i < chunk_gps_times.size()) ? chunk_gps_times[i] : 0.0f;
                voxels[key].push_back({i, chunk_ranges[i], gt});
              }

              // Per-voxel discrimination
              std::vector<bool> is_removed(chunk_pts.size(), false);
              if (rf_criteria == 0) {
                // Range criteria
                for (const auto& [key, entries] : voxels) {
                  float max_close_range = 0.0f;
                  int close_count = 0;
                  for (const auto& e : entries) {
                    if (e.range <= rf_safe_range) { max_close_range = std::max(max_close_range, e.range); close_count++; }
                  }
                  if (close_count < rf_min_close_pts) {
                    float min_range = std::numeric_limits<float>::max();
                    for (const auto& e : entries) min_range = std::min(min_range, e.range);
                    const float far_threshold = min_range + rf_far_delta;
                    for (const auto& e : entries) { if (e.range > far_threshold) is_removed[e.idx] = true; }
                    continue;
                  }
                  const float threshold = max_close_range + rf_range_delta;
                  for (const auto& e : entries) {
                    if (e.range <= rf_safe_range) continue;
                    if (e.range > threshold) is_removed[e.idx] = true;
                  }
                }
              } else {
                // GPS Time criteria — keep dominant temporal cluster per voxel
                const float time_gap = 5.0f;
                for (const auto& [key, entries] : voxels) {
                  if (entries.size() <= 1) continue;
                  std::vector<int> si(entries.size()); std::iota(si.begin(), si.end(), 0);
                  std::sort(si.begin(), si.end(), [&](int a, int b) { return entries[a].gps_time < entries[b].gps_time; });
                  std::vector<std::vector<int>> clusters;
                  clusters.push_back({si[0]});
                  for (size_t k = 1; k < si.size(); k++) {
                    if (entries[si[k]].gps_time - entries[si[k-1]].gps_time > time_gap) clusters.push_back({});
                    clusters.back().push_back(si[k]);
                  }
                  if (clusters.size() <= 1) continue;
                  int best = 0;
                  if (rf_gps_keep == 0) { for (int tci = 1; tci < static_cast<int>(clusters.size()); tci++) { if (clusters[tci].size() > clusters[best].size()) best = tci; } }
                  else if (rf_gps_keep == 1) { best = static_cast<int>(clusters.size()) - 1; }  // newest = last cluster (sorted by time)
                  // else rf_gps_keep == 2: best = 0 (oldest = first cluster, already default)
                  std::unordered_set<int> keep_set(clusters[best].begin(), clusters[best].end());
                  for (int ei = 0; ei < static_cast<int>(entries.size()); ei++) {
                    if (!keep_set.count(ei)) is_removed[entries[ei].idx] = true;
                  }
                }
              }

              // Build kept/removed lists (core area only)
              for (size_t i = 0; i < chunk_pts.size(); i++) {
                if (!core_chunk.contains(chunk_pts[i])) continue;
                if (is_removed[i]) {
                  removed_points.push_back(chunk_pts[i]);
                  rf_preview_data.push_back({chunk_pts[i], chunk_ranges[i], chunk_intensities[i], 0.0f, false, false});
                } else {
                  kept_points.push_back(chunk_pts[i]);
                  kept_ints.push_back(chunk_intensities[i]);
                  rf_preview_data.push_back({chunk_pts[i], chunk_ranges[i], chunk_intensities[i], 0.0f, false, true});
                }
              }
              size_t total_core = kept_points.size() + removed_points.size();
              logger->info("[Range chunk] {} kept, {} removed out of {} core points (ground_only={})",
                kept_points.size(), removed_points.size(), total_core, range_ground_only);
              } // end mode branch

              // Render (same pattern as regular preview — with intensity buffer)
              auto kept_buf2 = std::make_shared<std::vector<Eigen::Vector3f>>(std::move(kept_points));
              auto kept_int2 = std::make_shared<std::vector<float>>(std::move(kept_ints));
              auto removed_buf2 = std::make_shared<std::vector<Eigen::Vector3f>>(std::move(removed_points));
              vw->invoke([this, kept_buf2, kept_int2, removed_buf2] {
                auto vw = guik::LightViewer::instance();
                if (!kept_buf2->empty()) {
                  const int n = kept_buf2->size();
                  std::vector<Eigen::Vector4d> pts4(n);
                  for (int i = 0; i < n; i++) pts4[i] = Eigen::Vector4d((*kept_buf2)[i].x(), (*kept_buf2)[i].y(), (*kept_buf2)[i].z(), 1.0);
                  auto cb = std::make_shared<glk::PointCloudBuffer>(pts4.data(), n);
                  if (kept_int2->size() == static_cast<size_t>(n)) {
                    cb->add_buffer("intensity", *kept_int2);
                    cb->set_colormap_buffer("intensity");
                  }
                  vw->update_drawable("rf_preview_kept", cb, guik::FlatColor(0.0f, 0.8f, 0.2f, 1.0f));
                }
                if (!removed_buf2->empty()) {
                  const int n = removed_buf2->size();
                  std::vector<Eigen::Vector4d> pts4(n);
                  for (int i = 0; i < n; i++) pts4[i] = Eigen::Vector4d((*removed_buf2)[i].x(), (*removed_buf2)[i].y(), (*removed_buf2)[i].z(), 1.0);
                  auto cb = std::make_shared<glk::PointCloudBuffer>(pts4.data(), n);
                  vw->update_drawable("rf_preview_removed", cb, guik::FlatColor(1.0f, 0.0f, 0.0f, 0.5f).make_transparent());
                }
              });

              char buf[256];
              std::snprintf(buf, sizeof(buf), "Chunk: %zu kept, %zu dynamic, %zu total",
                kept_points.size(), removed_points.size(), chunk_pts.size());
              rf_status = buf;
              rf_processing = false;
            }).detach();
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Process one chunk with full overlap at current position.\nShows exact apply-quality result for this area.");
        }

        ImGui::SameLine();
        if (ImGui::Button("Clear preview")) {
          auto vw = guik::LightViewer::instance();
          vw->remove_drawable("rf_preview_kept");
          vw->remove_drawable("rf_preview_removed");
          rf_status.clear();
          rf_preview_active = false;
          rf_intensity_mode = false;
          rf_range_highlight = 0.0f;
          rf_preview_data.clear();
          lod_hide_all_submaps = false;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove preview overlay and restore normal view.");

        ImGui::SameLine();
        if (ImGui::Button("Filter preview")) {
          auto vw = guik::LightViewer::instance();
          vw->remove_drawable("rf_preview_removed");
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hide removed points (red) to see cleaned result.");

        ImGui::SameLine();
        if (ImGui::Button("Toggle intensity")) {
          rf_intensity_mode = !rf_intensity_mode;
          auto vw = guik::LightViewer::instance();
          auto drawable = vw->find_drawable("rf_preview_kept");
          if (drawable.first) {
            if (rf_intensity_mode) {
              // Set intensity colormap range from cached data
              float int_min = std::numeric_limits<float>::max();
              float int_max = std::numeric_limits<float>::lowest();
              for (const auto& p : rf_preview_data) {
                if (p.kept) { int_min = std::min(int_min, p.intensity); int_max = std::max(int_max, p.intensity); }
              }
              if (int_min >= int_max) { int_min = 0.0f; int_max = 255.0f; }
              vw->shader_setting().add<Eigen::Vector2f>("cmap_range", Eigen::Vector2f(int_min, int_max));
              drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLORMAP);
            } else {
              drawable.first->set_color_mode(guik::ColorMode::FLAT_COLOR);
            }
          }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle between flat green and intensity coloring\non the kept preview points.");

        ImGui::Separator();

        // Show chunks visualization
        ImGui::Checkbox("Display chunks", &rf_show_chunks);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show processing chunk boundaries as wireframes.");
        if (rf_show_chunks) {
          if (!trajectory_built) build_trajectory();
          auto vw = guik::LightViewer::instance();
          const double active_chunk_size = (df_mode == 1) ? df_chunk_size : rf_chunk_size;
          const double active_chunk_spacing = (df_mode == 1) ? df_chunk_spacing : rf_chunk_spacing;
          const double hs = active_chunk_size * 0.5;
          int chunk_count = 0;
          for (double d = 0.0; d < trajectory_total_dist; d += active_chunk_spacing) {
            size_t idx = 0;
            for (size_t k = 1; k < trajectory_data.size(); k++) {
              if (trajectory_data[k].cumulative_dist >= d) { idx = k; break; }
            }
            const Eigen::Vector3f center = trajectory_data[idx].pose.translation().cast<float>();
            // Get heading for rotation
            const size_t next = std::min(idx + 1, trajectory_data.size() - 1);
            Eigen::Vector3f fwd = (trajectory_data[next].pose.translation() - trajectory_data[idx].pose.translation()).cast<float>();
            fwd.z() = 0.0f;
            if (fwd.norm() < 0.01f) fwd = Eigen::Vector3f::UnitX();
            else fwd.normalize();
            const Eigen::Vector3f right = fwd.cross(Eigen::Vector3f::UnitZ()).normalized();
            // Build rotation matrix
            Eigen::Matrix3f rot;
            rot.col(0) = fwd;
            rot.col(1) = right;
            rot.col(2) = Eigen::Vector3f::UnitZ();
            Eigen::Affine3f box_tf = Eigen::Affine3f::Identity();
            box_tf.translate(center);
            box_tf.linear() = rot;
            box_tf = box_tf * Eigen::Scaling(Eigen::Vector3f(static_cast<float>(hs), static_cast<float>(hs), 50.0f));
            vw->update_drawable("rf_chunk_" + std::to_string(chunk_count), glk::Primitives::wire_cube(),
              guik::FlatColor(1.0f, 0.0f, 0.0f, 0.8f, box_tf));
            chunk_count++;
          }
          // Clean up old chunks beyond current count
          for (int ci = chunk_count; ci < chunk_count + 100; ci++) {
            vw->remove_drawable("rf_chunk_" + std::to_string(ci));
          }
        } else {
          // Remove chunk visualizations
          for (int ci = 0; ci < 10000; ci++) {
            auto vw = guik::LightViewer::instance();
            if (!vw->find_drawable("rf_chunk_" + std::to_string(ci)).first) break;
            vw->remove_drawable("rf_chunk_" + std::to_string(ci));
          }
        }

        if (df_mode == 2) {
          ImGui::SliderFloat("Chunk size (m)", &rf_chunk_size, 20.0f, 200.0f, "%.0f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Size of each processing chunk.");
          ImGui::SliderFloat("Chunk spacing (m)", &rf_chunk_spacing, 10.0f, 100.0f, "%.0f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Distance between chunk centers.");
        } else {
          ImGui::SliderFloat("Chunk size (m)", &df_chunk_size, 40.0f, 500.0f, "%.0f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Size of each processing chunk.\nLarger = more trail context.");
          ImGui::SliderFloat("Chunk spacing (m)", &df_chunk_spacing, 20.0f, 250.0f, "%.0f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Distance between chunk centers.");
        }

        if (df_mode == 1 && ImGui::Button("Apply dynamic filter to HD")) {
          ImGui::OpenPopup("DynApplyGroundReuse");
        }
        if (ImGui::BeginPopup("DynApplyGroundReuse")) {
          ImGui::Text("Reuse existing ground classification?");
          ImGui::Separator();
          bool launch = false;
          bool reuse_ground = false;
          if (ImGui::Button("Yes — reuse aux_ground.bin")) { reuse_ground = true; launch = true; ImGui::CloseCurrentPopup(); }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Uses ground from last Classify ground to scalar.\nFaster, recommended if ground is already tuned.");
          if (ImGui::Button("No — recompute PatchWork++")) { reuse_ground = false; launch = true; ImGui::CloseCurrentPopup(); }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Runs PatchWork++ fresh per frame.\nSlower, use if ground hasn't been classified yet.");
          ImGui::Separator();
          if (ImGui::Button("Cancel")) { ImGui::CloseCurrentPopup(); }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Abort without applying.");
          if (launch) {
          rf_processing = true;
          rf_status = "Starting dynamic filter...";
          std::thread([this, reuse_ground] {
            if (!trajectory_built) build_trajectory();
            const auto start_time = std::chrono::steady_clock::now();

            // Build chunks along trajectory
            const double overlap = df_chunk_size * 0.5;  // 50% overlap on each side
            const double chunk_total = df_chunk_size + 2.0 * overlap;
            auto chunks = glim::build_chunks(trajectory_data, trajectory_total_dist, df_chunk_spacing, chunk_total * 0.5);
            logger->info("[Dynamic apply] {} chunks (size={:.0f}m + {:.0f}m overlap each side)", chunks.size(), df_chunk_size, overlap);

            // Index ALL frames with metadata
            rf_status = "Indexing HD frames...";
            struct FrameEntry {
              glim::MapCleanerFilter::FrameData fd;
              Eigen::Vector3f sensor_pos;
            };
            std::vector<FrameEntry> all_frame_entries;
            for (const auto& submap : submaps) {
              if (!submap) continue;
              if (hidden_sessions.count(submap->session_id)) continue;
              std::string session_hd = hd_frames_path;
              for (const auto& sess : sessions) {
                if (sess.id == submap->session_id && !sess.hd_frames_path.empty()) {
                  session_hd = sess.hd_frames_path; break;
                }
              }
              const Eigen::Isometry3d T_odom0 = submap->frames.front()->T_world_imu;
              for (const auto& frame : submap->frames) {
                char dir_name[16];
                std::snprintf(dir_name, sizeof(dir_name), "%08ld", frame->id);
                const std::string frame_dir = session_hd + "/" + dir_name;
                auto fi = glim::frame_info_from_meta(frame_dir,
                  glim::compute_frame_world_pose(submap->T_world_origin, submap->T_origin_endpoint_L, T_odom0, frame->T_world_imu, frame->T_lidar_imu),
                  submap->id, submap->session_id);
                if (fi.num_points > 0) {
                  all_frame_entries.push_back({{fi.dir, fi.T_world_lidar, fi.num_points}, fi.T_world_lidar.translation().cast<float>()});
                }
              }
            }
            logger->info("[Dynamic apply] {} total frames indexed", all_frame_entries.size());

            // Accumulated removals across all chunks
            std::unordered_map<std::string, std::unordered_set<int>> frame_removals;

            // Process each chunk
            glim::MapCleanerFilter::Params mc_params;
            mc_params.range_threshold = df_range_threshold;
            mc_params.lidar_range = df_observation_range;
            mc_params.voxel_size = df_voxel_size;
            mc_params.exclude_ground_pw = df_exclude_ground_pw;

            for (size_t ci = 0; ci < chunks.size(); ci++) {
              const auto& chunk = chunks[ci];
              const auto chunk_aabb = chunk.world_aabb();

              // Core area (for writing removals — no overlap)
              glim::Chunk core_chunk = chunk;
              core_chunk.half_size = df_chunk_size * 0.5;

              char buf[256];
              std::snprintf(buf, sizeof(buf), "Chunk %zu/%zu: loading...", ci + 1, chunks.size());
              rf_status = buf;

              // Find frames overlapping this chunk (including overlap area)
              std::vector<glim::MapCleanerFilter::FrameData> chunk_mc_frames;
              for (const auto& fe : all_frame_entries) {
                // Quick distance check
                if ((fe.sensor_pos - chunk.center.cast<float>()).norm() > chunk_total + mc_params.lidar_range) continue;
                chunk_mc_frames.push_back(fe.fd);
              }
              if (chunk_mc_frames.empty()) continue;

              // Auto frame skip for this chunk
              mc_params.frame_skip = (chunk_mc_frames.size() > 200) ? static_cast<int>(chunk_mc_frames.size() / 200) : 0;

              // Load points from chunk frames into world space
              std::vector<Eigen::Vector3f> chunk_pts;
              std::vector<float> chunk_ranges;
              std::vector<bool> chunk_ground;
              struct ChunkPtSource { int frame_idx; int point_idx; bool in_core; };
              std::vector<ChunkPtSource> chunk_sources;

              for (int fi = 0; fi < static_cast<int>(chunk_mc_frames.size()); fi++) {
                const auto& fd = chunk_mc_frames[fi];
                std::vector<Eigen::Vector3f> pts;
                std::vector<float> rng, ints(fd.num_points, 0.0f);
                if (!glim::load_bin(fd.dir + "/points.bin", pts, fd.num_points)) continue;
                if (!glim::load_bin(fd.dir + "/range.bin", rng, fd.num_points)) continue;
                glim::load_bin(fd.dir + "/intensities.bin", ints, fd.num_points);
                std::vector<Eigen::Vector3f> nrm(fd.num_points, Eigen::Vector3f::Zero());
                if (df_exclude_ground_pw) glim::load_bin(fd.dir + "/normals.bin", nrm, fd.num_points);
                // PatchWork++ ground classification (cached or from popup choice)
                std::vector<bool> pw_gnd;
                if (df_exclude_ground_pw) {
                  auto cache_it = pw_ground_cache.find(fd.dir);
                  if (cache_it != pw_ground_cache.end() && static_cast<int>(cache_it->second.size()) == fd.num_points) {
                    pw_gnd = cache_it->second;
                  } else if (reuse_ground) {
                    std::vector<float> gnd_scalar;
                    if (glim::load_bin(fd.dir + "/aux_ground.bin", gnd_scalar, fd.num_points) && static_cast<int>(gnd_scalar.size()) == fd.num_points) {
                      pw_gnd.resize(fd.num_points);
                      for (int gi = 0; gi < fd.num_points; gi++) pw_gnd[gi] = gnd_scalar[gi] >= 0.5f;
                      pw_ground_cache[fd.dir] = pw_gnd;
                    }
                  }
                  if (pw_gnd.empty()) {
                    pw_gnd = glim::MapCleanerFilter::classify_ground_patchwork(pts, fd.num_points, 1.7f, ints);
                    pw_ground_cache[fd.dir] = pw_gnd;
                  }
                }

                const Eigen::Matrix3f R = fd.T_world_lidar.rotation().cast<float>();
                const Eigen::Vector3f t = fd.T_world_lidar.translation().cast<float>();
                for (int i = 0; i < fd.num_points; i++) {
                  if (rng[i] < 1.5f) continue;
                  const Eigen::Vector3f wp = R * pts[i] + t;
                  if (!chunk.contains(wp)) continue;
                  chunk_pts.push_back(wp);
                  chunk_ranges.push_back(rng[i]);
                  const bool gpw = !pw_gnd.empty() && pw_gnd[i];
                  chunk_ground.push_back(df_exclude_ground_pw && gpw);
                  chunk_sources.push_back({fi, i, core_chunk.contains(wp)});
                }
              }

              if (chunk_pts.empty()) continue;

              // Pre-MapCleaner ground refinement (same as process chunk)
              if (df_refine_ground && df_exclude_ground_pw) {
                const float col_res = 1.0f, col_inv = 1.0f / col_res, ground_z_tol = 0.5f;
                std::unordered_map<uint64_t, float> col_min_z;
                for (size_t i = 0; i < chunk_pts.size(); i++) {
                  const uint64_t ck = (static_cast<uint64_t>(static_cast<int>(std::floor(chunk_pts[i].x() * col_inv)) + 1048576) << 21)
                                    | static_cast<uint64_t>(static_cast<int>(std::floor(chunk_pts[i].y() * col_inv)) + 1048576);
                  auto it = col_min_z.find(ck);
                  if (it == col_min_z.end() || chunk_pts[i].z() < it->second) col_min_z[ck] = chunk_pts[i].z();
                }
                for (size_t i = 0; i < chunk_pts.size(); i++) {
                  if (!chunk_ground[i]) continue;
                  const uint64_t ck = (static_cast<uint64_t>(static_cast<int>(std::floor(chunk_pts[i].x() * col_inv)) + 1048576) << 21)
                                    | static_cast<uint64_t>(static_cast<int>(std::floor(chunk_pts[i].y() * col_inv)) + 1048576);
                  if (chunk_pts[i].z() > col_min_z[ck] + ground_z_tol) {
                    chunk_ground[i] = false;
                  }
                }
              }

              std::snprintf(buf, sizeof(buf), "Chunk %zu/%zu: MapCleaner (%zu pts, %zu frames)...",
                ci + 1, chunks.size(), chunk_pts.size(), chunk_mc_frames.size());
              rf_status = buf;

              // Run MapCleaner on this chunk
              glim::MapCleanerFilter filter(mc_params);
              auto result = filter.compute(chunk_mc_frames, chunk_pts, chunk_ranges, chunk_ground);

              // Trail refinement (same logic as Process chunk preview)
              if (df_refine_trails) {
                const float rv = df_refine_voxel, inv_rv = 1.0f / rv, voxel_vol = rv * rv * rv;
                std::unordered_map<uint64_t, std::vector<size_t>> cand_vox;
                for (size_t i = 0; i < chunk_pts.size(); i++) {
                  if (!result.is_dynamic[i]) continue;
                  if (chunk_ground[i]) { result.is_dynamic[i] = false; continue; }  // ground → force static
                  cand_vox[glim::voxel_key(chunk_pts[i], inv_rv)].push_back(i);
                }
                // BFS clustering
                std::unordered_map<uint64_t, int> vox_cluster;
                std::vector<std::vector<uint64_t>> clusters;
                int nc = 0;
                for (const auto& [k, _] : cand_vox) {
                  if (vox_cluster.count(k)) continue;
                  std::vector<uint64_t> ck; std::queue<uint64_t> q;
                  q.push(k); vox_cluster[k] = nc;
                  while (!q.empty()) {
                    const uint64_t c = q.front(); q.pop(); ck.push_back(c);
                    const int cx = static_cast<int>((c >> 42) & 0x1FFFFF) - 1048576;
                    const int cy = static_cast<int>((c >> 21) & 0x1FFFFF) - 1048576;
                    const int cz = static_cast<int>(c & 0x1FFFFF) - 1048576;
                    for (int dz=-1;dz<=1;dz++) for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++) {
                      if (!dx && !dy && !dz) continue;
                      const uint64_t nk = glim::voxel_key(cx+dx, cy+dy, cz+dz);
                      if (cand_vox.count(nk) && !vox_cluster.count(nk)) { vox_cluster[nk] = nc; q.push(nk); }
                    }
                  }
                  clusters.push_back(std::move(ck)); nc++;
                }
                // Evaluate clusters — keep only trail-shaped ones
                std::unordered_set<uint64_t> trail_voxels;
                for (int tci = 0; tci < nc; tci++) {
                  Eigen::Vector3f bmin = Eigen::Vector3f::Constant(1e9f), bmax = Eigen::Vector3f::Constant(-1e9f);
                  int tp = 0;
                  for (const auto& vk : clusters[tci]) {
                    auto it = cand_vox.find(vk);
                    if (it == cand_vox.end()) continue;
                    for (size_t idx : it->second) { bmin = bmin.cwiseMin(chunk_pts[idx]); bmax = bmax.cwiseMax(chunk_pts[idx]); tp++; }
                  }
                  const Eigen::Vector3f ext = bmax - bmin;
                  const float longest = ext.maxCoeff(), shortest = std::max(0.01f, ext.minCoeff());
                  const float density = tp / std::max(0.001f, static_cast<float>(clusters[tci].size()) * voxel_vol);
                  if (longest >= df_trail_min_length && longest/shortest >= df_trail_min_aspect && density >= df_trail_min_density) {
                    for (const auto& vk : clusters[tci]) trail_voxels.insert(vk);
                  }
                }
                // Reject non-trail candidates
                int rejected = 0;
                for (const auto& [vk, indices] : cand_vox) {
                  if (!trail_voxels.count(vk)) {
                    for (size_t idx : indices) { result.is_dynamic[idx] = false; rejected++; }
                  }
                }
                logger->info("[Dynamic apply] Chunk {}/{}: trail refine: {} rejected, {} trail voxels",
                  ci + 1, chunks.size(), rejected, trail_voxels.size());
              }

              // Only mark removals for points in the CORE area, NEVER ground
              for (size_t i = 0; i < chunk_pts.size(); i++) {
                if (result.is_dynamic[i] && chunk_sources[i].in_core && !chunk_ground[i]) {
                  const auto& fd = chunk_mc_frames[chunk_sources[i].frame_idx];
                  frame_removals[fd.dir].insert(chunk_sources[i].point_idx);
                }
              }

              logger->info("[Dynamic apply] Chunk {}/{}: {} dynamic in core area",
                ci + 1, chunks.size(), result.num_dynamic);
            }

            // Write filtered frames — with final ground safety check
            rf_status = "Writing filtered frames (ground safety check)...";
            size_t total_removed = 0, total_kept = 0, ground_saved = 0;
            int frames_modified = 0;
            for (auto& [frame_dir, remove_set] : frame_removals) {
              const std::string meta_path = frame_dir + "/frame_meta.json";
              std::ifstream meta_ifs(meta_path);
              const auto meta = nlohmann::json::parse(meta_ifs, nullptr, false);
              meta_ifs.close();
              if (meta.is_discarded()) continue;
              const int num_pts = meta.value("num_points", 0);
              if (num_pts == 0) continue;

              // FINAL SAFETY: run PatchWork++ to protect ground
              if (df_exclude_ground_pw) {
                std::vector<Eigen::Vector3f> pts;
                std::vector<float> ints;
                glim::load_bin(frame_dir + "/points.bin", pts, num_pts);
                glim::load_bin(frame_dir + "/intensities.bin", ints, num_pts);

                std::vector<bool> is_ground(num_pts, false);
                bool ground_loaded = false;
                if (!pts.empty()) {
                  auto cache_it = pw_ground_cache.find(frame_dir);
                  if (cache_it != pw_ground_cache.end() && static_cast<int>(cache_it->second.size()) == num_pts) {
                    is_ground = cache_it->second; ground_loaded = true;
                  } else if (reuse_ground) {
                    std::vector<float> gnd_scalar;
                    if (glim::load_bin(frame_dir + "/aux_ground.bin", gnd_scalar, num_pts) && static_cast<int>(gnd_scalar.size()) == num_pts) {
                      for (int gi = 0; gi < num_pts; gi++) is_ground[gi] = gnd_scalar[gi] >= 0.5f;
                      ground_loaded = true;
                    }
                  }
                  if (!ground_loaded) {
                    is_ground = glim::MapCleanerFilter::classify_ground_patchwork(pts, num_pts, 1.7f, ints);
                  }
                }

                // Remove ground points from removal set
                size_t before = remove_set.size();
                for (int i = 0; i < num_pts; i++) {
                  if (is_ground[i]) remove_set.erase(i);
                }
                ground_saved += before - remove_set.size();
              }

              if (remove_set.empty()) continue;

              std::vector<int> kept_indices;
              kept_indices.reserve(num_pts - remove_set.size());
              for (int i = 0; i < num_pts; i++) {
                if (!remove_set.count(i)) kept_indices.push_back(i);
              }
              const int new_count = static_cast<int>(kept_indices.size());
              total_removed += remove_set.size();
              total_kept += new_count;

              glim::filter_bin_file(frame_dir + "/points.bin", sizeof(Eigen::Vector3f), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/normals.bin", sizeof(Eigen::Vector3f), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/intensities.bin", sizeof(float), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/times.bin", sizeof(float), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/range.bin", sizeof(float), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/rings.bin", sizeof(uint16_t), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/aux_ground.bin", sizeof(float), num_pts, kept_indices, new_count);

              {
                std::ofstream ofs(meta_path);
                ofs << std::setprecision(15) << std::fixed;
                ofs << "{\n";
                ofs << "  \"frame_id\": " << meta.value("frame_id", 0) << ",\n";
                ofs << "  \"stamp\": " << meta.value("stamp", 0.0) << ",\n";
                ofs << "  \"scan_end_time\": " << meta.value("scan_end_time", 0.0) << ",\n";
                ofs << "  \"num_points\": " << new_count << ",\n";
                if (meta.contains("T_world_lidar")) ofs << "  \"T_world_lidar\": " << meta["T_world_lidar"].dump() << ",\n";
                if (meta.contains("bbox_world_min")) ofs << "  \"bbox_world_min\": " << meta["bbox_world_min"].dump() << ",\n";
                if (meta.contains("bbox_world_max")) ofs << "  \"bbox_world_max\": " << meta["bbox_world_max"].dump() << "\n";
                ofs << "}\n";
              }
              frames_modified++;
            }

            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            char final_buf[256];
            std::snprintf(final_buf, sizeof(final_buf), "Done: %zu removed, %zu kept, %zu ground saved, %d frames (%.1f sec)",
              total_removed, total_kept, ground_saved, frames_modified, elapsed);
            rf_status = final_buf;
            logger->info("[Dynamic apply] {}", rf_status);
            rf_processing = false;
          }).detach();
          } // end if (launch)
          ImGui::EndPopup();
        } // end BeginPopup
        if (df_mode == 1 && ImGui::IsItemHovered()) {
          ImGui::SetTooltip("DESTRUCTIVE: runs MapCleaner chunk-by-chunk along trajectory.\nBackup first with Tools > Utils > Backup HD frames.");
        }

        if (df_mode == 2 && ImGui::Button("Apply to HD frames (chunked)")) {
          rf_processing = true;
          rf_status = "Building trajectory...";
          const bool apply_ground_only = rf_ground_only;
          std::thread([this, apply_ground_only] {
            if (!trajectory_built) build_trajectory();
            const auto start_time = std::chrono::steady_clock::now();

            // Step 1: Place path-aligned chunk centers along trajectory
            auto chunks = glim::build_chunks(trajectory_data, trajectory_total_dist, rf_chunk_spacing, rf_chunk_size * 0.5);
            logger->info("[DataFilter] {} chunks along {:.0f} m trajectory", chunks.size(), trajectory_total_dist);

            // Step 2: Build frame index with world-space bounding boxes
            std::vector<glim::FrameInfo> all_frames;
            rf_status = "Indexing HD frames...";
            for (const auto& submap : submaps) {
              if (!submap) continue;
              if (hidden_sessions.count(submap->session_id)) continue;
              std::string session_hd = hd_frames_path;
              for (const auto& sess : sessions) {
                if (sess.id == submap->session_id && !sess.hd_frames_path.empty()) {
                  session_hd = sess.hd_frames_path; break;
                }
              }
              const Eigen::Isometry3d T_odom0 = submap->frames.front()->T_world_imu;
              for (const auto& frame : submap->frames) {
                char dir_name[16];
                std::snprintf(dir_name, sizeof(dir_name), "%08ld", frame->id);
                const std::string frame_dir = session_hd + "/" + dir_name;
                const Eigen::Isometry3d T_w_lidar = glim::compute_frame_world_pose(
                  submap->T_world_origin, submap->T_origin_endpoint_L, T_odom0, frame->T_world_imu, frame->T_lidar_imu);
                auto fi = glim::frame_info_from_meta(frame_dir, T_w_lidar, submap->id, submap->session_id);
                if (fi.num_points > 0) all_frames.push_back(std::move(fi));
              }
            }
            logger->info("[DataFilter] Indexed {} HD frames", all_frames.size());

            // Step 3: Per-frame removal indices (accumulated across chunks)
            std::unordered_map<std::string, std::unordered_set<int>> frame_removals;  // frame_dir → set of point indices to remove

            // Step 4: Process each chunk
            const float inv_voxel = 1.0f / rf_voxel_size;
            for (size_t ci = 0; ci < chunks.size(); ci++) {
              const auto& chunk = chunks[ci];

              if (ci % 10 == 0) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "Processing chunk %zu / %zu...", ci + 1, chunks.size());
                rf_status = buf;
              }

              // Find frames overlapping this chunk (sensor position within chunk + max_range)
              struct ChunkFrameData {
                std::string dir;
                std::vector<Eigen::Vector3f> world_points;
                std::vector<float> ranges;
                std::vector<float> gps_times;
                std::vector<int> original_indices;
                std::vector<bool> is_ground;  // from aux_ground.bin (for ground-only mode)
              };
              std::vector<ChunkFrameData> chunk_frames;

              const auto chunk_aabb = chunk.world_aabb();
              for (const auto& fi : all_frames) {
                if (fi.num_points == 0) continue;
                if (!chunk_aabb.intersects(fi.world_bbox)) continue;

                std::vector<Eigen::Vector3f> pts;
                std::vector<float> range;
                if (!glim::load_bin(fi.dir + "/points.bin", pts, fi.num_points)) continue;
                if (!glim::load_bin(fi.dir + "/range.bin", range, fi.num_points)) continue;
                std::vector<float> ftimes(fi.num_points, 0.0f);
                if (rf_criteria == 1) glim::load_bin(fi.dir + "/times.bin", ftimes, fi.num_points);
                std::vector<float> frame_ground;
                if (apply_ground_only) glim::load_bin(fi.dir + "/aux_ground.bin", frame_ground, fi.num_points);

                const Eigen::Matrix3f R = fi.T_world_lidar.rotation().cast<float>();
                const Eigen::Vector3f t = fi.T_world_lidar.translation().cast<float>();

                ChunkFrameData cfd;
                cfd.dir = fi.dir;
                for (int i = 0; i < fi.num_points; i++) {
                  if (range[i] < 1.5f) continue;
                  const Eigen::Vector3f wp = R * pts[i] + t;
                  if (chunk.contains(wp)) {
                    const bool is_gnd = i < static_cast<int>(frame_ground.size()) && frame_ground[i] >= 0.5f;
                    cfd.world_points.push_back(wp);
                    cfd.ranges.push_back(range[i]);
                    cfd.gps_times.push_back(static_cast<float>(fi.stamp - gps_time_base) + ftimes[i]);
                    cfd.original_indices.push_back(i);
                    cfd.is_ground.push_back(is_gnd);
                  }
                }
                if (!cfd.world_points.empty()) chunk_frames.push_back(std::move(cfd));
              }

              // Build cross-frame voxel grid for this chunk (ground-only: skip non-ground)
              struct VoxelEntry { int cf_idx; int pt_idx; float range; float gps_time; };
              std::unordered_map<uint64_t, std::vector<VoxelEntry>> voxels;
              for (int cfi = 0; cfi < static_cast<int>(chunk_frames.size()); cfi++) {
                const auto& cf = chunk_frames[cfi];
                for (int pi = 0; pi < static_cast<int>(cf.world_points.size()); pi++) {
                  if (apply_ground_only && (pi >= static_cast<int>(cf.is_ground.size()) || !cf.is_ground[pi])) continue;
                  const uint64_t key = glim::voxel_key(cf.world_points[pi], inv_voxel);
                  const float gt = (pi < static_cast<int>(cf.gps_times.size())) ? cf.gps_times[pi] : 0.0f;
                  voxels[key].push_back({cfi, pi, cf.ranges[pi], gt});
                }
              }

              // Filter: per-voxel discrimination
              if (rf_criteria == 0) {
                // Range criteria
                for (const auto& [key, entries] : voxels) {
                  float max_close_range = 0.0f;
                  int close_count = 0;
                  for (const auto& e : entries) {
                    if (e.range <= rf_safe_range) { max_close_range = std::max(max_close_range, e.range); close_count++; }
                  }
                  if (close_count < rf_min_close_pts) {
                    float min_range = std::numeric_limits<float>::max();
                    for (const auto& e : entries) min_range = std::min(min_range, e.range);
                    const float far_threshold = min_range + rf_far_delta;
                    for (const auto& e : entries) {
                      if (e.range > far_threshold) { frame_removals[chunk_frames[e.cf_idx].dir].insert(chunk_frames[e.cf_idx].original_indices[e.pt_idx]); }
                    }
                    continue;
                  }
                  const float threshold = max_close_range + rf_range_delta;
                  for (const auto& e : entries) {
                    if (e.range <= rf_safe_range) continue;
                    if (e.range > threshold) { frame_removals[chunk_frames[e.cf_idx].dir].insert(chunk_frames[e.cf_idx].original_indices[e.pt_idx]); }
                  }
                }
              } else {
                // GPS time criteria — keep dominant temporal cluster per voxel
                const float time_gap = 5.0f;
                for (const auto& [key, entries] : voxels) {
                  if (entries.size() <= 1) continue;
                  std::vector<int> si(entries.size()); std::iota(si.begin(), si.end(), 0);
                  std::sort(si.begin(), si.end(), [&](int a, int b) { return entries[a].gps_time < entries[b].gps_time; });
                  std::vector<std::vector<int>> clusters;
                  clusters.push_back({si[0]});
                  for (size_t i = 1; i < si.size(); i++) {
                    if (entries[si[i]].gps_time - entries[si[i-1]].gps_time > time_gap) clusters.push_back({});
                    clusters.back().push_back(si[i]);
                  }
                  if (clusters.size() <= 1) continue;
                  int best = 0;
                  if (rf_gps_keep == 0) { for (int ci = 1; ci < static_cast<int>(clusters.size()); ci++) { if (clusters[ci].size() > clusters[best].size()) best = ci; } }
                  else if (rf_gps_keep == 1) { best = static_cast<int>(clusters.size()) - 1; }
                  std::unordered_set<int> keep_set(clusters[best].begin(), clusters[best].end());
                  for (int ei = 0; ei < static_cast<int>(entries.size()); ei++) {
                    if (!keep_set.count(ei)) { frame_removals[chunk_frames[entries[ei].cf_idx].dir].insert(chunk_frames[entries[ei].cf_idx].original_indices[entries[ei].pt_idx]); }
                  }
                }
              }
            }

            // Step 5: Apply removals — rewrite each affected frame
            rf_status = "Writing filtered frames...";
            size_t total_removed = 0, total_kept = 0;
            int frames_modified = 0;

            for (auto& [frame_dir, remove_set] : frame_removals) {
              if (remove_set.empty()) continue;

              std::ifstream meta_ifs(frame_dir + "/frame_meta.json");
              const auto meta = nlohmann::json::parse(meta_ifs, nullptr, false);
              if (meta.is_discarded()) continue;
              const int num_pts = meta.value("num_points", 0);

              // Build kept indices
              std::vector<int> kept_indices;
              kept_indices.reserve(num_pts);
              for (int i = 0; i < num_pts; i++) {
                if (!remove_set.count(i)) kept_indices.push_back(i);
              }
              const int new_count = static_cast<int>(kept_indices.size());
              total_removed += remove_set.size();
              total_kept += new_count;

              // Rewrite binary files
              glim::filter_bin_file(frame_dir + "/points.bin", sizeof(Eigen::Vector3f), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/normals.bin", sizeof(Eigen::Vector3f), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/intensities.bin", sizeof(float), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/times.bin", sizeof(float), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/range.bin", sizeof(float), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/rings.bin", sizeof(uint16_t), num_pts, kept_indices, new_count);
              glim::filter_bin_file(frame_dir + "/aux_ground.bin", sizeof(float), num_pts, kept_indices, new_count);

              // Update frame_meta.json
              {
                std::ofstream ofs(frame_dir + "/frame_meta.json");
                ofs << std::setprecision(15) << std::fixed;
                ofs << "{\n";
                ofs << "  \"frame_id\": " << meta.value("frame_id", 0) << ",\n";
                ofs << "  \"stamp\": " << meta.value("stamp", 0.0) << ",\n";
                ofs << "  \"scan_end_time\": " << meta.value("scan_end_time", 0.0) << ",\n";
                ofs << "  \"num_points\": " << new_count << ",\n";
                if (meta.contains("T_world_lidar")) ofs << "  \"T_world_lidar\": " << meta["T_world_lidar"].dump() << ",\n";
                if (meta.contains("bbox_world_min")) ofs << "  \"bbox_world_min\": " << meta["bbox_world_min"].dump() << ",\n";
                if (meta.contains("bbox_world_max")) ofs << "  \"bbox_world_max\": " << meta["bbox_world_max"].dump() << "\n";
                ofs << "}\n";
              }
              frames_modified++;
            }

            // Count kept points from unmodified frames
            for (const auto& fi : all_frames) {
              if (frame_removals.count(fi.dir)) continue;
              std::ifstream meta_ifs(fi.dir + "/frame_meta.json");
              const auto meta = nlohmann::json::parse(meta_ifs, nullptr, false);
              if (!meta.is_discarded()) total_kept += meta.value("num_points", 0);
            }

            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now() - start_time).count();

            char buf[512];
            std::snprintf(buf, sizeof(buf),
              "Done: %zu chunks, %d frames modified, %.1f M kept, %.1f M removed (%.1f%%), %lds",
              chunks.size(), frames_modified,
              static_cast<double>(total_kept) / 1e6,
              static_cast<double>(total_removed) / 1e6,
              (total_kept + total_removed > 0) ? 100.0 * total_removed / (total_kept + total_removed) : 0.0,
              elapsed);
            rf_status = buf;
            rf_processing = false;
            total_hd_points = total_kept;
            logger->info("[DataFilter] {}", rf_status);
          }).detach();
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("DESTRUCTIVE: applies range filter along the\nfull trajectory. Backup first with Tools > Utils > Backup HD frames.");
        }

        if (df_mode == 0 && ImGui::Button("Apply SOR to HD frames")) {
          rf_processing = true;
          rf_status = "Starting SOR filter...";
          std::thread([this] {
            const auto start_time = std::chrono::steady_clock::now();

            // Index all frames with bboxes
            rf_status = "Indexing frames...";
            std::vector<glim::FrameInfo> all_frames;
            Eigen::AlignedBox3d global_bbox;
            for (const auto& submap : submaps) {
              if (!submap) continue;
              if (hidden_sessions.count(submap->session_id)) continue;
              std::string shd = hd_frames_path;
              for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
              const Eigen::Isometry3d T0 = submap->frames.front()->T_world_imu;
              for (const auto& fr : submap->frames) {
                char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                auto fi = glim::frame_info_from_meta(shd + "/" + dn,
                  glim::compute_frame_world_pose(submap->T_world_origin, submap->T_origin_endpoint_L, T0, fr->T_world_imu, fr->T_lidar_imu));
                if (fi.num_points > 0) {
                  global_bbox.extend(fi.world_bbox);
                  all_frames.push_back(std::move(fi));
                }
              }
            }
            logger->info("[SOR apply] {} frames, bbox [{:.0f},{:.0f},{:.0f}]-[{:.0f},{:.0f},{:.0f}]",
              all_frames.size(), global_bbox.min().x(), global_bbox.min().y(), global_bbox.min().z(),
              global_bbox.max().x(), global_bbox.max().y(), global_bbox.max().z());

            // Build spatial grid (axis-aligned cubes)
            const double cs = sor_chunk_size;
            const Eigen::Vector3d gmin = global_bbox.min(), gmax = global_bbox.max();
            const int nx = std::max(1, static_cast<int>(std::ceil((gmax.x() - gmin.x()) / cs)));
            const int ny = std::max(1, static_cast<int>(std::ceil((gmax.y() - gmin.y()) / cs)));
            const int total_chunks = nx * ny;
            logger->info("[SOR apply] {} x {} = {} spatial chunks ({}m)", nx, ny, total_chunks, cs);

            std::unordered_map<std::string, std::unordered_set<int>> frame_removals;
            const float r2 = sor_radius * sor_radius;
            int chunks_done = 0;

            for (int iy = 0; iy < ny; iy++) {
              for (int ix = 0; ix < nx; ix++) {
                chunks_done++;
                if (chunks_done % 5 == 0) {
                  char buf[256]; std::snprintf(buf, sizeof(buf), "SOR chunk %d/%d...", chunks_done, total_chunks);
                  rf_status = buf;
                }

                // Chunk AABB
                Eigen::AlignedBox3d chunk_aabb;
                chunk_aabb.min() = Eigen::Vector3d(gmin.x() + ix * cs, gmin.y() + iy * cs, gmin.z());
                chunk_aabb.max() = Eigen::Vector3d(gmin.x() + (ix + 1) * cs, gmin.y() + (iy + 1) * cs, gmax.z());

                // Load points
                struct SorPt { Eigen::Vector3f wp; std::string dir; int orig_idx; };
                std::vector<SorPt> pts;
                for (const auto& fi : all_frames) {
                  if (fi.num_points == 0 || !chunk_aabb.intersects(fi.world_bbox)) continue;
                  std::vector<Eigen::Vector3f> fpts; std::vector<float> frng;
                  if (!glim::load_bin(fi.dir + "/points.bin", fpts, fi.num_points)) continue;
                  if (!glim::load_bin(fi.dir + "/range.bin", frng, fi.num_points)) continue;
                  const Eigen::Matrix3f R = fi.T_world_lidar.rotation().cast<float>();
                  const Eigen::Vector3f t = fi.T_world_lidar.translation().cast<float>();
                  for (int i = 0; i < fi.num_points; i++) {
                    if (frng[i] < 1.5f) continue;
                    const Eigen::Vector3f wp = R * fpts[i] + t;
                    if (wp.x() >= chunk_aabb.min().x() && wp.x() < chunk_aabb.max().x() &&
                        wp.y() >= chunk_aabb.min().y() && wp.y() < chunk_aabb.max().y()) {
                      pts.push_back({wp, fi.dir, i});
                    }
                  }
                }
                if (pts.empty()) continue;

                // KD-tree + SOR
                std::vector<Eigen::Vector4d> pts4(pts.size());
                for (size_t i = 0; i < pts.size(); i++) pts4[i] = Eigen::Vector4d(pts[i].wp.x(), pts[i].wp.y(), pts[i].wp.z(), 1.0);
                gtsam_points::KdTree kdt(pts4.data(), pts4.size());
                for (size_t i = 0; i < pts.size(); i++) {
                  std::vector<size_t> ki(sor_min_neighbors + 1);
                  std::vector<double> kd(sor_min_neighbors + 1);
                  kdt.knn_search(pts4[i].data(), sor_min_neighbors + 1, ki.data(), kd.data());
                  int nn = 0;
                  for (int j = 0; j < sor_min_neighbors + 1; j++) {
                    if (ki[j] == i) continue;
                    if (kd[j] <= static_cast<double>(r2)) nn++;
                  }
                  if (nn < sor_min_neighbors) {
                    frame_removals[pts[i].dir].insert(pts[i].orig_idx);
                  }
                }
              }
            }

            // Write
            rf_status = "Writing filtered frames...";
            size_t total_removed = 0; int frames_modified = 0;
            for (const auto& [frame_dir, remove_set] : frame_removals) {
              const std::string mp = frame_dir + "/frame_meta.json";
              std::ifstream mf(mp); auto meta = nlohmann::json::parse(mf, nullptr, false); mf.close();
              if (meta.is_discarded()) continue;
              const int np = meta.value("num_points", 0);
              std::vector<int> kept; kept.reserve(np);
              for (int i = 0; i < np; i++) { if (!remove_set.count(i)) kept.push_back(i); }
              const int nc = static_cast<int>(kept.size());
              total_removed += remove_set.size();
              glim::filter_bin_file(frame_dir + "/points.bin", sizeof(Eigen::Vector3f), np, kept, nc);
              glim::filter_bin_file(frame_dir + "/normals.bin", sizeof(Eigen::Vector3f), np, kept, nc);
              glim::filter_bin_file(frame_dir + "/intensities.bin", sizeof(float), np, kept, nc);
              glim::filter_bin_file(frame_dir + "/times.bin", sizeof(float), np, kept, nc);
              glim::filter_bin_file(frame_dir + "/range.bin", sizeof(float), np, kept, nc);
              glim::filter_bin_file(frame_dir + "/rings.bin", sizeof(uint16_t), np, kept, nc);
              glim::filter_bin_file(frame_dir + "/aux_ground.bin", sizeof(float), np, kept, nc);
              { std::ofstream ofs(mp); ofs << std::setprecision(15) << std::fixed;
                ofs << "{\n  \"frame_id\": " << meta.value("frame_id", 0) << ",\n";
                ofs << "  \"stamp\": " << meta.value("stamp", 0.0) << ",\n";
                ofs << "  \"scan_end_time\": " << meta.value("scan_end_time", 0.0) << ",\n";
                ofs << "  \"num_points\": " << nc << ",\n";
                if (meta.contains("T_world_lidar")) ofs << "  \"T_world_lidar\": " << meta["T_world_lidar"].dump() << ",\n";
                if (meta.contains("bbox_world_min")) ofs << "  \"bbox_world_min\": " << meta["bbox_world_min"].dump() << ",\n";
                if (meta.contains("bbox_world_max")) ofs << "  \"bbox_world_max\": " << meta["bbox_world_max"].dump() << "\n";
                ofs << "}\n"; }
              frames_modified++;
            }
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            char fb[256]; std::snprintf(fb, sizeof(fb), "SOR done: %zu removed, %d frames (%.1f sec)", total_removed, frames_modified, elapsed);
            rf_status = fb; logger->info("[SOR apply] {}", rf_status);
            rf_processing = false;
          }).detach();
        }
        if (df_mode == 0 && ImGui::IsItemHovered()) {
          ImGui::SetTooltip("DESTRUCTIVE: removes outlier points from HD frames.\nUses spatial grid (no trajectory needed).\nBackup first.");
        }
      }

      if (!hd_available) {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          ImGui::SetTooltip("No HD frames available.");
        }
        ImGui::EndDisabled();
      }

      if (!rf_status.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", rf_status.c_str());
      }
    }
    ImGui::End();
  });

  progress_modal.reset(new guik::ProgressModal("offline_viewer_progress"));

#ifdef GTSAM_POINTS_USE_CUDA
  gtsam_points::LinearizationHook::register_hook([] { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif
}

void OfflineViewer::main_menu() {
  bool start_open_map = !init_map_path.empty();
  bool start_close_map = false;
  bool start_save_map = false;
  bool start_export_map = false;

  if (ImGui::BeginMainMenuBar()) {
    // =====================================================================
    // File menu
    // =====================================================================
    if (ImGui::BeginMenu("File")) {
      if (!async_global_mapping) {
        if (ImGui::MenuItem("Open New Map")) {
          start_open_map = true;
        }
      } else {
        if (ImGui::MenuItem("Open Additional Map")) {
          start_open_map = true;
        }
      }

      if (ImGui::MenuItem("Close Map")) {
        if (pfd::message("Warning", "Close the map?").result() == pfd::button::ok) {
          start_close_map = true;
        }
      }

      if (ImGui::BeginMenu("Save")) {
        if (ImGui::MenuItem("Save Map")) {
          start_save_map = true;
        }
        if (ImGui::MenuItem("Export Points")) {
          start_export_map = true;
        }

        // Export options
        ImGui::Separator();
        if (!hd_available) ImGui::BeginDisabled();
        ImGui::Checkbox("Export HD", &export_hd);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          if (hd_available) {
            ImGui::SetTooltip("Export full-resolution HD frames instead of SD submaps.");
          } else {
            ImGui::SetTooltip("No HD frames available.\nRun SLAM with hd_frame_saver module.");
          }
        }
        if (!hd_available) ImGui::EndDisabled();

        if (!gnss_datum_available) {
          ImGui::BeginDisabled();
        }
        ImGui::Checkbox("Trim by tile", &trim_by_tile);
        if (!gnss_datum_available) {
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("No GNSS datum available (gnss_datum.json not found)");
          }
          ImGui::EndDisabled();
        }

        // Geoid correction controls (active when GNSS datum is available)
        if (!gnss_datum_available) {
          ImGui::BeginDisabled();
        }
        ImGui::Separator();
        ImGui::Text("Geoid correction");
        ImGui::RadioButton("None",          &geoid_correction_mode, 0);
        ImGui::RadioButton("Manual offset", &geoid_correction_mode, 1);
        ImGui::RadioButton("Auto EGM2008",  &geoid_correction_mode, 2);
        if (geoid_correction_mode == 1) {
          ImGui::SetNextItemWidth(120.0f);
          ImGui::InputFloat("N (m)", &geoid_manual_offset, 0.1f, 1.0f, "%.3f");
          if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Geoid undulation in metres.\nH_ortho = h_ellipsoidal - N");
          }
        }
        if (!gnss_datum_available) {
          ImGui::EndDisabled();
        }
        ImGui::EndMenu();
      }

      if (ImGui::MenuItem("Quit")) {
        if (pfd::message("Warning", "Quit?").result() == pfd::button::ok) {
          request_to_terminate = true;
        }
      }

      ImGui::EndMenu();
    }

    // =====================================================================
    // Coordinates menu
    // =====================================================================
    if (ImGui::BeginMenu("Coordinates")) {
      if (!gnss_datum_available) {
        ImGui::BeginDisabled();
      }

      // --- Export Coordinate System (lateral submenu) ---
      if (ImGui::BeginMenu("Export Coordinate System")) {
        if (ImGui::MenuItem("UTM WGS84", nullptr, coord_system == 0)) {
          coord_system = 0;
        }
        if (ImGui::BeginMenu("JGD2011")) {
          // Auto-detect entry
          {
            char auto_label[64];
            if (detected_jgd_zone > 0) {
              std::snprintf(auto_label, sizeof(auto_label), "Auto-detect: %s (Zone %s)",
                            detected_pref_en.c_str(), jgd2011_zone_name(detected_jgd_zone));
            } else {
              std::snprintf(auto_label, sizeof(auto_label), "Auto-detect");
            }
            if (ImGui::MenuItem(auto_label, nullptr, coord_system == 1 && jgd2011_pref_idx < 0)) {
              coord_system = 1;
              jgd2011_pref_idx = -1;
            }
            if (jgd2011_pref_idx < 0 && detected_jgd_zone == 0 && ImGui::IsItemHovered()) {
              ImGui::SetTooltip("Will auto-detect prefecture on first JGD2011 export.");
            }
          }
          // Prefecture submenu for manual override
          if (ImGui::BeginMenu("Prefecture")) {
            for (int i = 0; i < kPrefZoneTableSize; i++) {
              char label[64];
              std::snprintf(label, sizeof(label), "%s (Zone %s)",
                            kPrefZoneTable[i].en, jgd2011_zone_name(kPrefZoneTable[i].zone));
              if (ImGui::MenuItem(label, nullptr, coord_system == 1 && jgd2011_pref_idx == i)) {
                coord_system = 1;
                jgd2011_pref_idx = i;
              }
            }
            ImGui::EndMenu();
          }
          ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Custom...", nullptr, coord_system == 2)) {
          coord_system = 0;
          pfd::message("Custom Coordinate System", "Coming soon.");
        }
        ImGui::EndMenu();
      }

      // --- Consider zones on export (UTM only) ---
      ImGui::Separator();
      {
        const bool zone_check_disabled = (coord_system != 0);
        if (zone_check_disabled) ImGui::BeginDisabled();
        ImGui::MenuItem("Consider zones on export", nullptr, &consider_zones_on_export);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          if (coord_system == 0) {
            ImGui::SetTooltip("Reproject points to their correct UTM zone\nif they cross a zone boundary.");
          } else {
            ImGui::SetTooltip("UTM WGS84 only for now.\nJGD2011 zone handling coming with prefecture boundaries.");
          }
        }
        if (zone_check_disabled) ImGui::EndDisabled();
      }
      ImGui::Separator();

      // --- Tiles (lateral submenu) ---
      if (ImGui::BeginMenu("Tiles")) {
        if (ImGui::MenuItem("PNOA Spain (1x1 km, 2022-2025)", nullptr, grid_preset == 1)) {
          grid_preset = (grid_preset == 1) ? 0 : 1;
          if (grid_preset == 1) { coord_system = 0; grid_tile_size_km = 1.0f; }
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("1x1 km tiles, UTM WGS84\nNaming: PNOA_MMS_EEE_NNNN.ply (SW corner in km)");
        }
        if (ImGui::MenuItem("ICGC Cat (1x1 km, 2021-2023)", nullptr, grid_preset == 2)) {
          grid_preset = (grid_preset == 2) ? 0 : 2;
          if (grid_preset == 2) { coord_system = 0; grid_tile_size_km = 1.0f; }
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("1x1 km tiles, UTM WGS84\nNaming: EEENNN.ply (easting km, northing-4000000 km)");
        }
        if (ImGui::MenuItem("Japan (JGD2011)", nullptr, grid_preset == 3)) {
          grid_preset = (grid_preset == 3) ? 0 : 3;
          if (grid_preset == 3) {
            coord_system = 1;
            jgd2011_pref_idx = -1;  // auto-detect prefecture from datum
            grid_tile_size_km = 0.5f;
          }
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip(
            "Kokudo kihonzu zukaku 500 (1:500 map sheet)\n"
            "300m N-S x 400m E-W tiles, JGD2011\n"
            "Prefecture auto-detected from datum\n"
            "Naming: ZZLLRRCC.ply (zone + block + subdivision)");
        }
        if (ImGui::MenuItem("Custom tile grid (SHP in target coords)...")) {
          pfd::message("Custom Tile Grid", "Coming soon.");
        }
        ImGui::EndMenu();
      }

      // --- Settings ---
      ImGui::Separator();
      if (ImGui::BeginMenu("Settings")) {
        ImGui::SetNextItemWidth(100.0f);
        ImGui::InputFloat("Default tile size (km)", &grid_tile_size_km, 0.1f, 1.0f, "%.1f");
        if (grid_tile_size_km < 0.01f) grid_tile_size_km = 0.01f;
        if (ImGui::Button("Reset to defaults")) {
          coord_system = 0;
          jgd2011_pref_idx = -1;
          consider_zones_on_export = true;
          grid_preset = 0;
          grid_tile_size_km = 2.0f;
          trim_by_tile = false;
        }
        ImGui::EndMenu();
      }

      if (!gnss_datum_available) {
        ImGui::EndDisabled();
      }
      ImGui::EndMenu();
    }

    // =====================================================================
    // Sessions menu
    // =====================================================================
    // Remove fully unloaded sessions from the list (deferred to avoid iterator invalidation)
    sessions.erase(
      std::remove_if(sessions.begin(), sessions.end(), [](const SessionState& s) { return s.unloaded; }),
      sessions.end());

    if (sessions.size() > 1) {
      if (ImGui::BeginMenu("Sessions")) {
        for (auto& sess : sessions) {
          // Show parent of dump/ as the session name (e.g. "Bag_00_gps_full_Rig_01")
          const boost::filesystem::path dump_path(sess.path);
          const std::string display_name = dump_path.parent_path().filename().string();
          char label[256];
          std::snprintf(label, sizeof(label), "[%d] %s", sess.id, display_name.c_str());

          if (ImGui::BeginMenu(label)) {
            ImGui::Checkbox("Visible", &sess.visible);
            ImGui::Checkbox("Include in export", &sess.export_enabled);
            if (sess.id == 0) {
              ImGui::Text("(reference datum)");
            } else {
              if (ImGui::Button("Unload session")) {
                const int remove_id = sess.id;
                hidden_sessions.insert(remove_id);
                session_hd_paths.erase(remove_id);

                // Remove all drawables and null out submaps for this session
                auto vw = guik::LightViewer::instance();
                for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
                  if (submaps[si] && submaps[si]->session_id == remove_id) {
                    const int sid = submaps[si]->id;
                    vw->remove_drawable("submap_" + std::to_string(sid));
                    vw->remove_drawable("bbox_" + std::to_string(sid));
                    vw->remove_drawable("coord_" + std::to_string(sid));
                    vw->remove_drawable("sphere_" + std::to_string(sid));
                    if (si < static_cast<int>(render_states.size())) {
                      total_gpu_bytes -= render_states[si].gpu_bytes;
                      render_states[si].gpu_bytes = 0;
                      render_states[si].current_lod = SubmapLOD::UNLOADED;
                      render_states[si].bbox_computed = false;
                    }
                    submaps[si].reset();  // null out — all iterators check for null
                  }
                }

                // Remove factors referencing this session's submaps
                global_factors.erase(
                  std::remove_if(global_factors.begin(), global_factors.end(),
                    [this, remove_id](const auto& f) {
                      gtsam::Symbol s1(std::get<1>(f)), s2(std::get<2>(f));
                      auto check = [&](gtsam::Symbol s) {
                        return s.chr() == 'x' && s.index() < submaps.size() && !submaps[s.index()];
                      };
                      return check(s1) || check(s2);
                    }),
                  global_factors.end());

                // Mark session for removal from menu
                sess.unloaded = true;

                update_viewer();
              }
            }
            ImGui::EndMenu();
          }
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Show all")) {
          for (auto& s : sessions) { s.visible = true; s.export_enabled = true; }
        }
        ImGui::EndMenu();
      }
    }

    // =====================================================================
    // Tools menu
    // =====================================================================
    if (ImGui::BeginMenu("Tools")) {
      if (ImGui::BeginMenu("Camera")) {
        if (ImGui::MenuItem("Orbit", nullptr, camera_mode_sel == 0)) {
          camera_mode_sel = 0;
          guik::LightViewer::instance()->use_orbit_camera_control();
        }
        if (ImGui::MenuItem("Follow Trajectory", nullptr, camera_mode_sel == 2)) {
          camera_mode_sel = 2;
          if (!trajectory_built) build_trajectory();
          if (!trajectory_data.empty()) {
            const auto& start = trajectory_data[0].pose;
            auto fps = guik::LightViewer::instance()->use_fps_camera_control(60.0);
            const Eigen::Vector3f pos = start.translation().cast<float>();
            const Eigen::Vector3d fwd = start.rotation().col(0);  // X-forward in LiDAR frame
            const float yaw = std::atan2(fwd.y(), fwd.x()) * 180.0f / M_PI;
            const float pitch = std::asin(std::clamp(fwd.z(), -1.0, 1.0)) * 180.0f / M_PI;
            fps->set_pose(pos, yaw, pitch);
            fps->set_translation_speed(fpv_speed);
            fps->lock_fovy();
            follow_progress = 0.0f;
            follow_playing = true;  // start playing immediately
            follow_yaw_offset = 0.0f;
            follow_pitch_offset = 0.0f;
            follow_smooth_init = false;
            follow_last_time = ImGui::GetTime();
          }
        }
        if (ImGui::MenuItem("FPV", nullptr, camera_mode_sel == 1)) {
          // Get current camera position before switching
          auto vw = guik::LightViewer::instance();
          const Eigen::Matrix4f vm = vw->view_matrix();
          const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));
          const Eigen::Vector3f cam_fwd = -vm.block<1, 3>(2, 0).transpose();  // -Z row of view matrix
          const float yaw = std::atan2(cam_fwd.y(), cam_fwd.x()) * 180.0f / M_PI;
          const float pitch = std::asin(std::clamp(cam_fwd.z(), -1.0f, 1.0f)) * 180.0f / M_PI;

          camera_mode_sel = 1;
          auto fps = vw->use_fps_camera_control(60.0);
          fps->set_translation_speed(fpv_speed);
          fps->set_pose(cam_pos, yaw, pitch);
          fpv_smooth_init = false;
        }
        ImGui::Separator();
        // Projection
        if (ImGui::MenuItem("Perspective")) {
          auto vw = guik::LightViewer::instance();
          // Iridescence default is perspective — just reset orbit
          vw->use_orbit_camera_control();
          camera_mode_sel = 0;
        }
        if (ImGui::MenuItem("Orthographic")) {
          auto vw = guik::LightViewer::instance();
          vw->use_topdown_camera_control(200.0, 0.0);
          camera_mode_sel = 0;
        }
        ImGui::Separator();
        // Preset views
        {
          auto vw = guik::LightViewer::instance();
          const Eigen::Matrix4f vm = vw->view_matrix();
          const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));
          // Get a center point (current lookat target or map center)
          Eigen::Vector3f center = Eigen::Vector3f::Zero();
          if (!submaps.empty()) {
            for (const auto& sm : submaps) { if (sm) center += sm->T_world_origin.translation().cast<float>(); }
            center /= submaps.size();
          }
          const float dist = std::max(50.0f, (cam_pos - center).norm());

          if (ImGui::MenuItem("Top")) {
            auto fps = vw->use_fps_camera_control(60.0);
            fps->set_pose(Eigen::Vector3f(center.x(), center.y(), center.z() + dist), 0.0f, -89.0f);
            camera_mode_sel = 1;
          }
          if (ImGui::MenuItem("Front")) {
            auto fps = vw->use_fps_camera_control(60.0);
            fps->set_pose(Eigen::Vector3f(center.x() + dist, center.y(), center.z()), 180.0f, 0.0f);
            camera_mode_sel = 1;
          }
          if (ImGui::MenuItem("Left")) {
            auto fps = vw->use_fps_camera_control(60.0);
            fps->set_pose(Eigen::Vector3f(center.x(), center.y() + dist, center.z()), -90.0f, 0.0f);
            camera_mode_sel = 1;
          }
          if (ImGui::MenuItem("Right")) {
            auto fps = vw->use_fps_camera_control(60.0);
            fps->set_pose(Eigen::Vector3f(center.x(), center.y() - dist, center.z()), 90.0f, 0.0f);
            camera_mode_sel = 1;
          }
        }
        ImGui::Separator();
        if (ImGui::BeginMenu("Settings")) {
          if (ImGui::SliderFloat("FPV speed", &fpv_speed, 0.1f, 2.0f, "%.2f")) {
            if (camera_mode_sel == 1) {
              auto cam = std::dynamic_pointer_cast<guik::FPSCameraControl>(
                guik::LightViewer::instance()->get_camera_control());
              if (cam) cam->set_translation_speed(fpv_speed);
            }
          }
          ImGui::SliderFloat("Shift multiplier", &fpv_speed_mult, 2.0f, 20.0f, "%.1fx");
          ImGui::SliderFloat("FPV smoothness", &fpv_smoothness, 0.05f, 1.0f, "%.2f");
          if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Lower = smoother movement\n1.0 = no smoothing (raw)");
          }
          ImGui::Separator();
          ImGui::Text("Follow Trajectory");
          ImGui::SliderFloat("Speed (km/h)", &follow_speed_kmh, -500.0f, 500.0f, "%.0f");
          ImGui::SliderFloat("Smoothness", &follow_smoothness, 0.01f, 0.5f, "%.2f");
          if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Lower = smoother drone-like feel\nHigher = tighter track following");
          }
          ImGui::EndMenu();
        }
        ImGui::EndMenu();
      }
      if (ImGui::MenuItem("Display Settings", nullptr, show_display_settings)) {
        show_display_settings = !show_display_settings;
      }
      if (ImGui::MenuItem("Memory Manager", nullptr, show_memory_manager)) {
        show_memory_manager = !show_memory_manager;
      }
      if (ImGui::MenuItem("Data Filter", nullptr, show_data_filter)) {
        show_data_filter = !show_data_filter;
      }
      if (ImGui::BeginMenu("Utils")) {
        const bool has_hd = hd_available && !hd_frames_path.empty();

        // Backup HD frames
        if (!has_hd) ImGui::BeginDisabled();
        if (ImGui::MenuItem("Backup HD frames")) {
          const std::string backup_path = hd_frames_path + "_backup.tar.gz";
          const std::string src_dir = hd_frames_path;
          logger->info("[Utils] Backing up HD frames to {}", backup_path);
          std::thread([this, src_dir, backup_path] {
            const std::string parent = boost::filesystem::path(src_dir).parent_path().string();
            const std::string dirname = boost::filesystem::path(src_dir).filename().string();
            const std::string cmd = "tar -czf \"" + backup_path + "\" -C \"" + parent + "\" \"" + dirname + "\"";
            const int ret = std::system(cmd.c_str());
            if (ret == 0) {
              logger->info("[Utils] Backup complete: {}", backup_path);
            } else {
              logger->error("[Utils] Backup failed (exit code {})", ret);
            }
          }).detach();
          pfd::message("Backup Started", "Compressing HD frames in background.\nThis may take several minutes for large datasets.\nCheck the log for completion.");
        }
        if (!has_hd) {
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("No HD frames available to backup.");
          }
          ImGui::EndDisabled();
        }

        // Restore HD frames
        {
          std::string backup_file;
          if (has_hd) {
            backup_file = hd_frames_path + "_backup.tar.gz";
          }
          const bool has_backup = !backup_file.empty() && boost::filesystem::exists(backup_file);

          if (!has_backup) ImGui::BeginDisabled();
          if (ImGui::MenuItem("Restore HD frames from backup")) {
            if (pfd::message("Confirm Restore",
                "This will DELETE the current hd_frames/ directory\n"
                "and restore from the backup archive.\n\nProceed?",
                pfd::choice::ok_cancel, pfd::icon::warning).result() == pfd::button::ok) {
              const std::string src_dir = hd_frames_path;
              const std::string bf = backup_file;
              logger->info("[Utils] Restoring HD frames from {}", bf);
              std::thread([this, src_dir, bf] {
                // Remove existing
                boost::filesystem::remove_all(src_dir);
                // Extract backup
                const std::string parent = boost::filesystem::path(src_dir).parent_path().string();
                const std::string cmd = "tar -xzf \"" + bf + "\" -C \"" + parent + "\"";
                const int ret = std::system(cmd.c_str());
                if (ret == 0) {
                  logger->info("[Utils] Restore complete");
                  // Re-scan HD frames from the restored path
                  hd_available = false;
                  total_hd_points = 0;
                  detect_hd_frames(boost::filesystem::path(src_dir).parent_path().string());
                  // Also update session HD paths
                  for (auto& sess : sessions) {
                    if (sess.hd_frames_path == src_dir) {
                      sess.hd_frames_path = hd_frames_path;
                    }
                  }
                } else {
                  logger->error("[Utils] Restore failed (exit code {})", ret);
                }
              }).detach();
              pfd::message("Restore Started", "Extracting backup in background.\nCheck the log for completion.");
            }
          }
          if (!has_backup) {
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
              ImGui::SetTooltip("No backup found.\nBackup first with 'Backup HD frames'.");
            }
            ImGui::EndDisabled();
          }
        }

        ImGui::Separator();

        // Voxelize HD data
        if (!has_hd) ImGui::BeginDisabled();
        if (ImGui::MenuItem("Voxelize HD data", nullptr, show_voxelize_tool)) {
          show_voxelize_tool = !show_voxelize_tool;
        }
        if (!has_hd) {
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("No HD frames available.");
          ImGui::EndDisabled();
        }

        // Regenerate SD from HD
        if (!has_hd) ImGui::BeginDisabled();
        if (ImGui::BeginMenu("Regenerate SD from HD")) {
          static float regen_voxel_size = 0.20f;
          ImGui::DragFloat("Voxel size (m)", &regen_voxel_size, 0.01f, 0.05f, 1.0f, "%.2f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Voxel grid resolution for downsampling.\nSmaller = denser SD, more memory.\n0.20m is a good default.");
          if (ImGui::Button("Regenerate")) {
            ImGui::CloseCurrentPopup();
            if (pfd::message("Confirm SD Regeneration",
                "This will overwrite all submap SD point data\n"
                "by downsampling the current HD frames and\n"
                "recomputing covariances.\n\n"
                "Backup your map first!\n\nProceed?",
                pfd::choice::ok_cancel, pfd::icon::warning).result() == pfd::button::ok) {
              const double voxel_res = regen_voxel_size;
              progress_modal->open<bool>("regen_sd", [this, voxel_res](guik::ProgressInterface& progress) -> bool {
                progress.set_title("Regenerating SD from HD");
                progress.set_maximum(submaps.size());
                int regenerated = 0;
                for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
                  progress.set_text("Submap " + std::to_string(si) + "/" + std::to_string(submaps.size()));
                  progress.increment();
                  if (!submaps[si]) continue;
                  auto hd_cloud = load_hd_for_submap(si, false);  // points+intensity only, skip covs
                  if (!hd_cloud || hd_cloud->size() == 0) continue;
                  auto sd_cloud = gtsam_points::voxelgrid_sampling(hd_cloud, voxel_res, num_threads);
                  if (!sd_cloud || sd_cloud->size() == 0) continue;
                  // Compute normals + covariances on the downsampled cloud
                  {
                    const int k = 10;
                    gtsam_points::KdTree tree(sd_cloud->points, sd_cloud->num_points);
                    std::vector<int> neighbors(sd_cloud->num_points * k);
                    for (size_t j = 0; j < sd_cloud->num_points; j++) {
                      std::vector<size_t> k_indices(k, j);
                      std::vector<double> k_sq_dists(k);
                      tree.knn_search(sd_cloud->points[j].data(), k, k_indices.data(), k_sq_dists.data());
                      std::copy(k_indices.begin(), k_indices.begin() + k, neighbors.begin() + j * k);
                    }
                    glim::CloudCovarianceEstimation cov_est(num_threads);
                    std::vector<Eigen::Vector4d> normals;
                    std::vector<Eigen::Matrix4d> covs;
                    cov_est.estimate(sd_cloud->points_storage, neighbors, k, normals, covs);
                    sd_cloud->add_normals(normals);
                    sd_cloud->add_covs(covs);
                  }
                  // Add aux_attributes that the viewer needs for scalar field rendering
                  {
                    // intensity: copy from standard member to aux
                    if (sd_cloud->intensities) {
                      std::vector<float> aux_intensity(sd_cloud->num_points);
                      for (size_t j = 0; j < sd_cloud->num_points; j++) {
                        aux_intensity[j] = static_cast<float>(sd_cloud->intensities[j]);
                      }
                      sd_cloud->add_aux_attribute("intensity", aux_intensity);
                    }
                    // range: compute from point distance to origin (submap-local frame)
                    std::vector<float> aux_range(sd_cloud->num_points);
                    for (size_t j = 0; j < sd_cloud->num_points; j++) {
                      aux_range[j] = static_cast<float>(sd_cloud->points[j].head<3>().norm());
                    }
                    sd_cloud->add_aux_attribute("range", aux_range);
                    // gps_time: copy from standard times member to aux (double)
                    if (sd_cloud->times) {
                      std::vector<double> aux_gps_time(sd_cloud->num_points);
                      for (size_t j = 0; j < sd_cloud->num_points; j++) {
                        aux_gps_time[j] = sd_cloud->times[j];
                      }
                      sd_cloud->add_aux_attribute("gps_time", aux_gps_time);
                    }
                  }
                  const std::string submap_path = (boost::format("%s/%06d") % loaded_map_path % si).str();
                  sd_cloud->save_compact(submap_path);
                  std::const_pointer_cast<SubMap>(submaps[si])->frame = sd_cloud;
                  regenerated++;
                  logger->info("[Regen SD] Submap {}: {} HD pts -> {} SD pts", si, hd_cloud->size(), sd_cloud->size());
                }
                logger->info("[Regen SD] Done: regenerated {}/{} submaps (voxel={:.2f}m)", regenerated, submaps.size(), voxel_res);
                return true;
              });
            }
          }
          ImGui::EndMenu();
        }
        if (!has_hd) {
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("No HD frames available.\nHD frames are needed to regenerate SD data.");
          }
          ImGui::EndDisabled();
        }

        ImGui::EndMenu();
      }
      ImGui::EndMenu();
    }

    // Colorize menu
    if (ImGui::BeginMenu("Colorize")) {
      if (ImGui::BeginMenu("Image folder")) {
        if (ImGui::MenuItem("Add folder...")) {
          const std::string folder = pfd::select_folder("Select image folder").result();
          if (!folder.empty() && boost::filesystem::exists(folder)) {
            logger->info("[Colorize] Loading images from {}", folder);
            auto source = Colorizer::load_image_folder(folder);
            int with_gps = 0, with_time = 0;
            for (const auto& f : source.frames) {
              if (f.lat != 0.0 || f.lon != 0.0) with_gps++;
              if (f.timestamp > 0.0) with_time++;
            }
            logger->info("[Colorize] Loaded {} images ({} with GPS, {} with timestamp)", source.frames.size(), with_gps, with_time);
            image_sources.push_back(std::move(source));
            colorize_source_idx = static_cast<int>(image_sources.size()) - 1;
            // Save colorize config to dump
            if (!loaded_map_path.empty()) {
              nlohmann::json cfg;
              cfg["sources"] = nlohmann::json::array();
              for (const auto& s : image_sources) {
                nlohmann::json sj;
                sj["path"] = s.path;
                sj["time_shift"] = s.time_shift;
                sj["lever_arm"] = {s.lever_arm.x(), s.lever_arm.y(), s.lever_arm.z()};
                sj["rotation_rpy"] = {s.rotation_rpy.x(), s.rotation_rpy.y(), s.rotation_rpy.z()};
                sj["fx"] = s.intrinsics.fx; sj["fy"] = s.intrinsics.fy;
                sj["cx"] = s.intrinsics.cx; sj["cy"] = s.intrinsics.cy;
                sj["width"] = s.intrinsics.width; sj["height"] = s.intrinsics.height;
                sj["k1"] = s.intrinsics.k1; sj["k2"] = s.intrinsics.k2;
                sj["p1"] = s.intrinsics.p1; sj["p2"] = s.intrinsics.p2;
                sj["k3"] = s.intrinsics.k3;
                cfg["sources"].push_back(sj);
              }
              std::ofstream ofs(loaded_map_path + "/colorize_config.json");
              ofs << std::setprecision(10) << cfg.dump(2);
              logger->info("[Colorize] Saved config to {}/colorize_config.json", loaded_map_path);
            }
          }
        }
        if (!image_sources.empty()) {
          ImGui::Separator();
          int remove_idx = -1;
          for (size_t si = 0; si < image_sources.size(); si++) {
            if (ImGui::BeginMenu(image_sources[si].name.c_str())) {
              char info[128];
              std::snprintf(info, sizeof(info), "%zu images", image_sources[si].frames.size());
              ImGui::TextDisabled("%s", info);
              ImGui::TextDisabled("%s", image_sources[si].path.c_str());
              ImGui::Separator();
              if (ImGui::MenuItem("Remove")) {
                remove_idx = static_cast<int>(si);
                // Clean up gizmos
                auto vw = guik::LightViewer::instance();
                for (size_t fi = 0; fi < image_sources[si].frames.size(); fi++) {
                  vw->remove_drawable("cam_" + std::to_string(si) + "_" + std::to_string(fi));
                  vw->remove_drawable("cam_fov_" + std::to_string(si) + "_" + std::to_string(fi));
                }
              }
              ImGui::EndMenu();
            }
          }
          if (remove_idx >= 0) {
            image_sources.erase(image_sources.begin() + remove_idx);
            if (colorize_source_idx >= static_cast<int>(image_sources.size())) colorize_source_idx = std::max(0, static_cast<int>(image_sources.size()) - 1);
          }
        }
        ImGui::EndMenu();
      }
      if (ImGui::MenuItem("Locate Cameras", nullptr, show_colorize_window)) {
        show_colorize_window = !show_colorize_window;
      }
      if (ImGui::MenuItem("Alignment check", nullptr, align_show)) {
        align_show = !align_show;
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overlay image + projected LiDAR points to assess calibration.");
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }

  // --- Open map ---
  if (start_open_map) {
    logger->debug("open map");
    std::string map_path;

    guik::RecentFiles recent_files("offline_viewer_open");
    if (init_map_path.empty()) {
      map_path = pfd::select_folder("Select a dump directory", recent_files.most_recent()).result();
    } else {
      map_path = init_map_path;
      init_map_path.clear();
    }

    if (!map_path.empty()) {
      logger->debug("open map from {}", map_path);
      recent_files.push(map_path);
      loaded_map_path = map_path;

      if (boost::filesystem::exists(map_path + "/config")) {
        logger->info("Use config from {}", map_path + "/config");
        GlobalConfig::instance(map_path + "/config", true);
      } else {
        logger->warn("No config found in {}", map_path);
      }

      const Config config_ros(GlobalConfig::get_config_path("config_ros"));
      const std::vector<std::string> ext_module_names = config_ros.param<std::vector<std::string>>("glim_ros", "extension_modules", {});
      for (const auto& name : ext_module_names) {
        if (name.find("viewer") != std::string::npos || name.find("monitor") != std::string::npos) {
          continue;
        }
        if (imported_shared_libs.count(name)) {
          logger->debug("Extension module {} already loaded", name);
          continue;
        }
        logger->info("Export classes from {}", name);
        ExtensionModule::export_classes(name);
        imported_shared_libs.insert(name);
      }

      // Compute datum offset for multi-session alignment.
      // Read the new map's gnss_datum.json BEFORE loading, compare with reference datum.
      Eigen::Vector3d datum_offset = Eigen::Vector3d::Zero();
      {
        // Read new map's datum from the freshly-set GlobalConfig path
        const std::string new_datum_path = GlobalConfig::get_config_path("gnss_datum");
        int new_zone = 0;
        double new_E = 0, new_N = 0, new_alt = 0;
        bool new_has_datum = false;

        if (boost::filesystem::exists(new_datum_path)) {
          std::ifstream ifs(new_datum_path);
          const auto j = nlohmann::json::parse(ifs, nullptr, false);
          if (!j.is_discarded()) {
            new_zone = j.value("utm_zone", 0);
            new_E    = j.value("utm_easting_origin", 0.0);
            new_N    = j.value("utm_northing_origin", 0.0);
            new_alt  = j.value("altitude", 0.0);
            new_has_datum = true;
          }
        }

        if (ref_datum_set && new_has_datum) {
          // Check zone compatibility
          if (new_zone != ref_utm_zone) {
            logger->warn("[multi-map] UTM zone mismatch: reference zone={}, new map zone={}. "
                         "Cross-zone alignment is not supported — coordinates may be incorrect.",
                         ref_utm_zone, new_zone);
            pfd::message("UTM Zone Mismatch",
              "The new map uses UTM zone " + std::to_string(new_zone) +
              " but the reference map uses zone " + std::to_string(ref_utm_zone) +
              ".\n\nCross-zone alignment is not supported. Coordinates may be incorrect.");
          }
          datum_offset = Eigen::Vector3d(new_E - ref_utm_easting, new_N - ref_utm_northing, new_alt - ref_datum_alt);
          logger->info("[multi-map] Datum offset: dE={:.3f} m, dN={:.3f} m, dZ={:.3f} m",
                       datum_offset.x(), datum_offset.y(), datum_offset.z());
        } else if (!ref_datum_set && new_has_datum) {
          // First map with datum — store as reference
          ref_datum_set = true;
          ref_utm_zone = new_zone;
          ref_utm_easting = new_E;
          ref_utm_northing = new_N;
          ref_datum_alt = new_alt;
          logger->info("[multi-map] Reference datum set: zone={} E={:.3f} N={:.3f} alt={:.3f}",
                       ref_utm_zone, ref_utm_easting, ref_utm_northing, ref_datum_alt);
        }
      }

      std::shared_ptr<GlobalMapping> existing_mapping;
      if (async_global_mapping) {
        logger->info("global map already exists, loading new map into existing global map");
        existing_mapping = std::dynamic_pointer_cast<GlobalMapping>(async_global_mapping->get_global_mapping());
      }

      progress_modal->open<std::shared_ptr<GlobalMapping>>(
        "open",
        [this, map_path, existing_mapping, datum_offset](guik::ProgressInterface& progress) {
          return load_map(progress, map_path, existing_mapping, datum_offset);
        });
    }
  }

  auto open_result = progress_modal->run<std::shared_ptr<GlobalMapping>>("open");
  if (open_result) {
    if (!(*open_result)) {
      pfd::message("Error", "Failed to load map").result();
    } else {
      // Extract factor visualization BEFORE wrapping in AsyncGlobalMapping
      // (which may consume pending_factors on its background thread)
      const auto loaded_gm = *open_result;
      const bool is_additional_map = !sessions.empty();
      if (is_additional_map) {
        size_t extracted = 0;
        size_t total_pending = loaded_gm->pending_factors().size();
        size_t count_between = 0, count_matching = 0, count_imu = 0, count_1key = 0, count_null = 0, count_other = 0;
        for (const auto& factor : loaded_gm->pending_factors()) {
          if (!factor) { count_null++; continue; }
          if (factor->keys().size() < 2) { count_1key++; continue; }
          if (dynamic_cast<gtsam::BetweenFactor<gtsam::Pose3>*>(factor.get())) {
            global_factors.push_back(std::make_tuple(FactorType::BETWEEN, factor->keys()[0], factor->keys()[1]));
            count_between++;
            extracted++;
          } else if (dynamic_cast<gtsam_points::IntegratedMatchingCostFactor*>(factor.get())) {
            global_factors.push_back(std::make_tuple(FactorType::MATCHING_COST, factor->keys()[0], factor->keys()[1]));
            count_matching++;
            extracted++;
#ifdef GTSAM_POINTS_USE_CUDA
          } else if (dynamic_cast<gtsam_points::IntegratedVGICPFactorGPU*>(factor.get())) {
            global_factors.push_back(std::make_tuple(FactorType::MATCHING_COST, factor->keys()[0], factor->keys()[1]));
            count_matching++;
            extracted++;
#endif
          } else if (dynamic_cast<gtsam::ImuFactor*>(factor.get())) {
            global_factors.push_back(std::make_tuple(FactorType::IMU, factor->keys()[0], factor->keys()[2]));
            count_imu++;
            extracted++;
          } else {
            count_other++;
          }
        }
        logger->info("[multi-map] pending_factors total={}, extracted={} (between={}, matching={}, imu={}, 1-key={}, null={}, other={})",
                     total_pending, extracted, count_between, count_matching, count_imu, count_1key, count_null, count_other);

        // Re-render factor lines to include the newly extracted factors
        if (extracted > 0) {
          update_viewer();
        }
      }

      async_global_mapping.reset(new glim::AsyncGlobalMapping(*open_result, 1e6));
      load_gnss_datum();

      // Use the reference datum for all coordinate exports (not the latest map's datum)
      if (ref_datum_set) {
        gnss_utm_zone = ref_utm_zone;
        gnss_utm_easting_origin = ref_utm_easting;
        gnss_utm_northing_origin = ref_utm_northing;
        gnss_datum_alt = ref_datum_alt;
      }

      // Invalidate trajectory so it rebuilds with the new session
      trajectory_built = false;
      trajectory_data.clear();

      // Register session in the session list
      if (loaded_gm && !loaded_gm->session_infos.empty()) {
        const auto& latest = loaded_gm->session_infos.back();
        sessions.push_back({latest.id, latest.source_path, "", true, true});

        // Detect HD frames for this session
        detect_hd_frames(latest.source_path);
        if (hd_available) {
          sessions.back().hd_frames_path = hd_frames_path;
          session_hd_paths[latest.id] = hd_frames_path;
        }
      }
    }
  }

  // --- Auto-load colorize config after map load ---
  if (!loaded_map_path.empty() && image_sources.empty()) {
    const std::string cfg_path = loaded_map_path + "/colorize_config.json";
    if (boost::filesystem::exists(cfg_path)) {
      std::ifstream ifs(cfg_path);
      auto cfg = nlohmann::json::parse(ifs, nullptr, false);
      if (!cfg.is_discarded() && cfg.contains("sources")) {
        for (const auto& sj : cfg["sources"]) {
          const std::string path = sj.value("path", "");
          if (path.empty() || !boost::filesystem::exists(path)) continue;
          logger->info("[Colorize] Auto-loading images from {}", path);
          auto source = Colorizer::load_image_folder(path);
          source.time_shift = sj.value("time_shift", 0.0);
          source.mask_path = sj.value("mask_path", "");
          if (!source.mask_path.empty() && boost::filesystem::exists(source.mask_path)) {
            colorize_mask = cv::imread(source.mask_path, cv::IMREAD_UNCHANGED);
            if (!colorize_mask.empty()) logger->info("[Colorize] Auto-loaded mask from {}", source.mask_path);
          }
          if (sj.contains("lever_arm")) source.lever_arm = Eigen::Vector3d(sj["lever_arm"][0], sj["lever_arm"][1], sj["lever_arm"][2]);
          if (sj.contains("rotation_rpy")) source.rotation_rpy = Eigen::Vector3d(sj["rotation_rpy"][0], sj["rotation_rpy"][1], sj["rotation_rpy"][2]);
          source.intrinsics.fx = sj.value("fx", 1920.0); source.intrinsics.fy = sj.value("fy", 1920.0);
          source.intrinsics.cx = sj.value("cx", 1920.0); source.intrinsics.cy = sj.value("cy", 1080.0);
          source.intrinsics.width = sj.value("width", 3840); source.intrinsics.height = sj.value("height", 2160);
          source.intrinsics.k1 = sj.value("k1", 0.0); source.intrinsics.k2 = sj.value("k2", 0.0);
          source.intrinsics.p1 = sj.value("p1", 0.0); source.intrinsics.p2 = sj.value("p2", 0.0);
          source.intrinsics.k3 = sj.value("k3", 0.0);
          logger->info("[Colorize] Restored: {} images, time_shift={:.3f}, lever=[{:.3f},{:.3f},{:.3f}]",
            source.frames.size(), source.time_shift, source.lever_arm.x(), source.lever_arm.y(), source.lever_arm.z());
          image_sources.push_back(std::move(source));
        }
        if (!image_sources.empty()) colorize_source_idx = 0;
      }
    }
  }

  // --- Save map ---
  if (start_save_map) {
    if (!async_global_mapping) {
      logger->warn("No map data to save");
    } else {
      guik::RecentFiles recent_files("offline_viewer_save");
      const std::string path = pfd::select_folder("Select a directory to save the map", recent_files.most_recent()).result();
      if (!path.empty()) {
        recent_files.push(path);
        progress_modal->open<bool>("save", [this, path](guik::ProgressInterface& progress) { return save_map(progress, path); });
      }
    }
  }
  auto save_result = progress_modal->run<bool>("save");

  // --- Export points ---
  if (start_export_map) {
    guik::RecentFiles recent_files("offline_viewer_export");
    std::string path;
    if (trim_by_tile && gnss_datum_available) {
      path = pfd::select_folder("Select output directory for tiles", recent_files.most_recent()).result();
    } else {
      path = pfd::save_file("Select the file destination", recent_files.most_recent(), {"PLY", "*.ply"}).result();
    }
    if (!path.empty()) {
      recent_files.push(path);
      progress_modal->open<bool>("export", [this, path](guik::ProgressInterface& progress) { return export_map(progress, path); });
    }
  }
  auto export_result = progress_modal->run<bool>("export");

  // --- Regenerate SD ---
  auto regen_result = progress_modal->run<bool>("regen_sd");
  if (regen_result) {
    logger->info("[Regen SD] Regeneration complete, updating viewer");
    update_viewer();
  }

  // --- Close map ---
  if (start_close_map) {
    if (async_global_mapping) {
      logger->info("Closing map");
      async_global_mapping->join();
      async_global_mapping.reset();
      clear();
    } else {
      logger->warn("No map to close");
    }
  }
}

std::shared_ptr<glim::GlobalMapping> OfflineViewer::load_map(
  guik::ProgressInterface& progress,
  const std::string& path,
  std::shared_ptr<GlobalMapping> global_mapping,
  const Eigen::Vector3d& datum_offset) {
  progress.set_title("Load map");
  progress.set_text("Now loading");
  progress.set_maximum(1);

  if (global_mapping == nullptr) {
    glim::GlobalMappingParams params;
    params.isam2_relinearize_skip = 1;
    params.isam2_relinearize_thresh = 0.0;

    const auto result = pfd::message("Confirm", "Do optimization?", pfd::choice::yes_no).result();
    params.enable_optimization = (result == pfd::button::ok) || (result == pfd::button::yes);

    logger->info("enable_optimization={}", params.enable_optimization);
    global_mapping.reset(new glim::GlobalMapping(params));
  }

  if (!global_mapping->load(path, datum_offset)) {
    logger->error("failed to load {}", path);
    return nullptr;
  }

  return global_mapping;
}

bool OfflineViewer::save_map(guik::ProgressInterface& progress, const std::string& path) {
  progress.set_title("Save map");
  progress.set_text("Now saving");
  async_global_mapping->save(path);
  return true;
}

bool OfflineViewer::export_map(guik::ProgressInterface& progress, const std::string& path) {
  progress.set_title("Export points");
  progress.set_text("Concatenating submaps");
  progress.set_maximum(3);
  progress.increment();

  if (submaps.empty()) {
    logger->warn("No submaps available for export");
    return false;
  }

  // Build set of export-enabled sessions for filtering
  std::unordered_set<int> export_sessions;
  if (sessions.empty()) {
    // No session tracking (single map) — export all
    for (const auto& submap : submaps) {
      if (submap) export_sessions.insert(submap->session_id);
    }
  } else {
    for (const auto& sess : sessions) {
      if (sess.export_enabled) export_sessions.insert(sess.id);
    }
    logger->info("PLY export: exporting sessions {}", [&]() {
      std::string s;
      for (int id : export_sessions) { if (!s.empty()) s += ", "; s += std::to_string(id); }
      return s;
    }());
  }

  // =====================================================================
  // HD export path — load frames from disk, transform, write
  // =====================================================================
  if (export_hd && hd_available) {
    progress.set_text("Exporting HD frames");
    progress.increment();

    // Collect all HD points with coordinate transforms
    std::vector<double> out_x, out_y, out_z;
    std::vector<float> out_intensities, out_range;
    std::vector<double> out_gps_time;
    std::vector<Eigen::Vector3f> out_normals;
    std::vector<int> out_session_ids;
    size_t total_hd_exported = 0;

    const bool south = gnss_datum_available ? (gnss_datum_lat < 0.0) : false;

    for (const auto& submap : submaps) {
      if (!submap || !export_sessions.count(submap->session_id)) continue;

      // Resolve per-session HD path
      std::string session_hd_dir = hd_frames_path;  // default fallback
      for (const auto& sess : sessions) {
        if (sess.id == submap->session_id && !sess.hd_frames_path.empty()) {
          session_hd_dir = sess.hd_frames_path;
          break;
        }
      }

      for (const auto& frame : submap->frames) {
        char dir_name[16];
        std::snprintf(dir_name, sizeof(dir_name), "%08ld", frame->id);
        const std::string frame_dir = session_hd_dir + "/" + dir_name;
        const std::string meta_path = frame_dir + "/frame_meta.json";
        if (!boost::filesystem::exists(meta_path)) continue;

        std::ifstream meta_ifs(meta_path);
        const auto meta = nlohmann::json::parse(meta_ifs, nullptr, false);
        if (meta.is_discarded()) continue;
        const int num_pts = meta.value("num_points", 0);
        const double frame_stamp = meta.value("stamp", 0.0);
        if (num_pts == 0) continue;

        // Read binary files
        std::vector<Eigen::Vector3f> pts(num_pts);
        { std::ifstream f(frame_dir + "/points.bin", std::ios::binary);
          if (!f) continue;
          f.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * num_pts); }

        std::vector<Eigen::Vector3f> nrms(num_pts);
        bool has_nrms = false;
        { std::ifstream f(frame_dir + "/normals.bin", std::ios::binary);
          if (f) { f.read(reinterpret_cast<char*>(nrms.data()), sizeof(Eigen::Vector3f) * num_pts); has_nrms = true; } }

        std::vector<float> ints(num_pts);
        bool has_ints = false;
        { std::ifstream f(frame_dir + "/intensities.bin", std::ios::binary);
          if (f) { f.read(reinterpret_cast<char*>(ints.data()), sizeof(float) * num_pts); has_ints = true; } }

        std::vector<float> rng(num_pts);
        bool has_rng = false;
        { std::ifstream f(frame_dir + "/range.bin", std::ios::binary);
          if (f) { f.read(reinterpret_cast<char*>(rng.data()), sizeof(float) * num_pts); has_rng = true; } }

        std::vector<float> times(num_pts);
        bool has_times = false;
        { std::ifstream f(frame_dir + "/times.bin", std::ios::binary);
          if (f) { f.read(reinterpret_cast<char*>(times.data()), sizeof(float) * num_pts); has_times = true; } }

        // Compute optimized world pose
        const Eigen::Isometry3d T_world_endpoint_L = submap->T_world_origin * submap->T_origin_endpoint_L;
        const Eigen::Isometry3d T_odom_imu0 = submap->frames.front()->T_world_imu;
        const Eigen::Isometry3d T_world_imu = T_world_endpoint_L * T_odom_imu0.inverse() * frame->T_world_imu;
        const Eigen::Isometry3d T_world_lidar = T_world_imu * frame->T_lidar_imu.inverse();
        const Eigen::Matrix3d R = T_world_lidar.rotation();
        const Eigen::Vector3d t_vec = T_world_lidar.translation();

        constexpr float HD_MIN_RANGE = 1.5f;
        const Eigen::Matrix3f Rf = R.cast<float>();
        for (int pi = 0; pi < num_pts; pi++) {
          const float r = has_rng ? rng[pi] : pts[pi].norm();
          if (r < HD_MIN_RANGE) continue;

          const Eigen::Vector3d wp = R * pts[pi].cast<double>() + t_vec;
          out_x.push_back(wp.x());
          out_y.push_back(wp.y());
          out_z.push_back(wp.z());
          if (has_nrms) out_normals.push_back((Rf * nrms[pi]).normalized());
          if (has_ints) out_intensities.push_back(ints[pi]);
          if (has_rng) out_range.push_back(rng[pi]);
          if (has_times) out_gps_time.push_back(frame_stamp + static_cast<double>(times[pi]));
          out_session_ids.push_back(submap->session_id);
          total_hd_exported++;
        }

      }
    }

    if (total_hd_exported == 0) {
      logger->warn("No HD points to export");
      return false;
    }

    progress.set_text("Writing HD export");
    progress.increment();

    const size_t n = out_x.size();
    logger->info("HD export: {} points from HD frames", n);

    // Apply coordinate system transform (same logic as SD path)
    if (gnss_datum_available) {
      double geoid_N = 0.0;
      if (geoid_correction_mode == 1) {
        geoid_N = static_cast<double>(geoid_manual_offset);
      } else if (geoid_correction_mode == 2) {
        geoid_N = lookup_geoid_undulation(gnss_datum_lat, gnss_datum_lon);
      }

      if (coord_system == 0) {
        // UTM: add datum origin
        for (size_t i = 0; i < n; i++) {
          out_x[i] += gnss_utm_easting_origin;
          out_y[i] += gnss_utm_northing_origin;
          out_z[i] = gnss_datum_alt + out_z[i] - geoid_N;
        }
      } else if (coord_system == 1) {
        // JGD2011
        ensure_prefectures_loaded();
        int jgd_zone = 0;
        if (jgd2011_pref_idx >= 0 && jgd2011_pref_idx < kPrefZoneTableSize) {
          jgd_zone = kPrefZoneTable[jgd2011_pref_idx].zone;
        } else if (detected_jgd_zone > 0) {
          jgd_zone = detected_jgd_zone;
        } else {
          jgd_zone = jgd2011_auto_zone(gnss_datum_lat, gnss_datum_lon);
        }
        const TMProjectionParams params = jgd2011_zone_params(jgd_zone);
        for (size_t i = 0; i < n; i++) {
          const double abs_e = gnss_utm_easting_origin + out_x[i];
          const double abs_n = gnss_utm_northing_origin + out_y[i];
          const Eigen::Vector2d latlon = utm_inverse(abs_e, abs_n, gnss_utm_zone, south);
          const Eigen::Vector2d jgd = tm_forward(latlon.x(), latlon.y(), params);
          out_x[i] = jgd.x();
          out_y[i] = jgd.y();
          out_z[i] = gnss_datum_alt + out_z[i] - geoid_N;
        }
      }
    }

    // Write output (single file or tiles)
    if (trim_by_tile && gnss_datum_available) {
      const double tile_size_m = grid_tile_size_km * 1000.0;
      std::unordered_map<std::string, std::vector<size_t>> tile_indices;
      for (size_t i = 0; i < n; i++) {
        const std::string tname = tile_name_for_point(out_x[i], out_y[i], grid_preset, tile_size_m,
          (coord_system == 1) ? detected_jgd_zone : 0);
        tile_indices[tname].push_back(i);
      }

      boost::filesystem::create_directories(path);
      size_t total_tiles = 0;
      for (const auto& kv : tile_indices) {
        const auto& indices = kv.second;
        const size_t tn = indices.size();
        glk::PLYData tile_ply;

        std::vector<double> tx(tn), ty(tn), tz(tn);
        for (size_t j = 0; j < tn; j++) { tx[j] = out_x[indices[j]]; ty[j] = out_y[indices[j]]; tz[j] = out_z[indices[j]]; }
        tile_ply.add_prop<double>("x", tx.data(), tn);
        tile_ply.add_prop<double>("y", ty.data(), tn);
        tile_ply.add_prop<double>("z", tz.data(), tn);

        if (!out_normals.empty()) {
          tile_ply.normals.reserve(tn);
          for (size_t j = 0; j < tn; j++) tile_ply.normals.push_back(out_normals[indices[j]]);
        }
        if (!out_intensities.empty()) {
          std::vector<float> ti(tn);
          for (size_t j = 0; j < tn; j++) ti[j] = out_intensities[indices[j]];
          tile_ply.add_prop<float>("intensity", ti.data(), tn);
        }
        if (!out_range.empty()) {
          std::vector<float> tr(tn);
          for (size_t j = 0; j < tn; j++) tr[j] = out_range[indices[j]];
          tile_ply.add_prop<float>("range", tr.data(), tn);
        }
        if (!out_gps_time.empty()) {
          std::vector<double> tg(tn);
          for (size_t j = 0; j < tn; j++) tg[j] = out_gps_time[indices[j]];
          tile_ply.add_prop<double>("gps_time", tg.data(), tn);
        }
        if (sessions.size() > 1) {
          std::vector<int> ts(tn);
          for (size_t j = 0; j < tn; j++) ts[j] = out_session_ids[indices[j]];
          tile_ply.add_prop<int>("session_id", ts.data(), tn);
        }

        glk::save_ply_binary(path + "/" + kv.first + ".ply", tile_ply);
        total_tiles++;
      }
      logger->info("HD export: {} tiles, {} total points", total_tiles, n);
    } else {
      // Single file
      glk::PLYData ply;
      ply.add_prop<double>("x", out_x.data(), n);
      ply.add_prop<double>("y", out_y.data(), n);
      ply.add_prop<double>("z", out_z.data(), n);
      if (!out_normals.empty()) {
        ply.normals.reserve(n);
        for (size_t i = 0; i < n; i++) ply.normals.push_back(out_normals[i]);
      }
      if (!out_intensities.empty()) ply.add_prop<float>("intensity", out_intensities.data(), n);
      if (!out_range.empty()) ply.add_prop<float>("range", out_range.data(), n);
      if (!out_gps_time.empty()) ply.add_prop<double>("gps_time", out_gps_time.data(), n);
      if (sessions.size() > 1) ply.add_prop<int>("session_id", out_session_ids.data(), n);
      glk::save_ply_binary(path, ply);
      logger->info("HD export: {} points to {}", n, path);
    }
    return true;
  }

  // =====================================================================
  // SD export path (original)
  // =====================================================================

  // Determine which fields are present across all submaps
  bool has_normals = true;
  bool has_intensities = true;
  size_t total_points = 0;
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }
    if (!export_sessions.count(submap->session_id)) continue;
    total_points += submap->frame->size();
    if (!submap->frame->normals) {
      has_normals = false;
    }
    if (!submap->frame->has_intensities()) {
      has_intensities = false;
    }
  }

  if (total_points == 0) {
    logger->warn("No points available for export");
    return false;
  }

  // Collect float/double aux_attribute names present in all submaps, excluding primary PLY properties.
  // "intensity" is excluded because it collides with ply.intensities (primary double field);
  // it will be exported separately as "intensity_aux" for pipeline comparison.
  static const std::unordered_set<std::string> primary_ply_props = {"x", "y", "z", "nx", "ny", "nz", "intensity", "r", "g", "b", "a"};
  std::vector<std::string> aux_names;
  std::unordered_map<std::string, size_t> aux_elem_sizes;
  // Find first export-enabled submap as reference for aux attributes
  const SubMap* ref_submap = nullptr;
  for (const auto& sm : submaps) {
    if (sm && sm->frame && export_sessions.count(sm->session_id)) { ref_submap = sm.get(); break; }
  }
  if (ref_submap) {
    for (const auto& attrib : ref_submap->frame->aux_attributes) {
      const size_t elem_size = attrib.second.first;
      if (elem_size != sizeof(float) && elem_size != sizeof(double)) {
        continue;
      }
      if (primary_ply_props.count(attrib.first)) {
        continue;
      }
      bool all_have = true;
      for (const auto& sm : submaps) {
        if (!sm || !sm->frame || !export_sessions.count(sm->session_id)) {
          continue;
        }
        const auto it = sm->frame->aux_attributes.find(attrib.first);
        if (it == sm->frame->aux_attributes.end() || it->second.first != elem_size) {
          all_have = false;
          break;
        }
      }
      if (all_have) {
        aux_names.push_back(attrib.first);
        aux_elem_sizes[attrib.first] = elem_size;
      }
    }
  }

  progress.set_text("Writing to file");
  progress.increment();

  // Check whether aux_attributes["intensity"] (float) is present in all submaps.
  // This is distinct from frame->intensities (primary double field exported via ply.intensities).
  bool has_aux_intensity = true;
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame || !export_sessions.count(submap->session_id)) {
      continue;
    }
    const auto it = submap->frame->aux_attributes.find("intensity");
    if (it == submap->frame->aux_attributes.end() || it->second.first != sizeof(float)) {
      has_aux_intensity = false;
      break;
    }
  }

  // Split aux names by element size so double attrs (gps_time) are written as
  // "property double" in the PLY header — float32 loses ~128 s precision on GPS epoch values.
  std::vector<std::string> aux_names_float, aux_names_double;
  for (const auto& name : aux_names) {
    if (aux_elem_sizes.at(name) == sizeof(double)) {
      aux_names_double.push_back(name);
    } else {
      aux_names_float.push_back(name);
    }
  }

  glk::PLYData ply;
  ply.vertices.reserve(total_points);
  if (has_normals) {
    ply.normals.reserve(total_points);
  }
  if (has_intensities) {
    ply.intensities.reserve(total_points);
  }

  std::unordered_map<std::string, std::vector<float>> aux_data_float;
  std::unordered_map<std::string, std::vector<double>> aux_data_double;
  for (const auto& name : aux_names_float) {
    aux_data_float[name].reserve(total_points);
  }
  for (const auto& name : aux_names_double) {
    aux_data_double[name].reserve(total_points);
  }
  std::vector<float> aux_intensity_data;
  if (has_aux_intensity) {
    aux_intensity_data.reserve(total_points);
  }
  std::vector<int> session_id_data;
  session_id_data.reserve(total_points);

  size_t total_nan_filtered = 0;

  for (const auto& submap : submaps) {
    if (!submap || !submap->frame || !export_sessions.count(submap->session_id)) {
      continue;
    }
    const auto& frame = submap->frame;
    const int n = frame->size();
    const Eigen::Matrix3d R = submap->T_world_origin.rotation();

    // Build per-point valid mask: exclude points where any aux attribute is non-finite or
    // where gps_time == 0.0 exactly (sentinel left by voxels merged before gps_time was
    // populated — these would corrupt MIN-blend colorisation by pulling the range to zero).
    std::vector<bool> valid(n, true);
    for (const auto& name : aux_names_float) {
      const float* src = static_cast<const float*>(frame->aux_attributes.at(name).second);
      for (int i = 0; i < n; i++) {
        if (valid[i] && !std::isfinite(src[i])) valid[i] = false;
      }
    }
    for (const auto& name : aux_names_double) {
      const double* src = static_cast<const double*>(frame->aux_attributes.at(name).second);
      for (int i = 0; i < n; i++) {
        if (valid[i] && !std::isfinite(src[i])) valid[i] = false;
      }
    }
    // Filter gps_time == 0.0: these are voxels that inherited a zero stamp from a keyframe
    // that was processed before the per-point GPS timestamps were available.
    {
      const auto gps_it = frame->aux_attributes.find("gps_time");
      if (gps_it != frame->aux_attributes.end() && gps_it->second.first == sizeof(double)) {
        const double* gps_src = static_cast<const double*>(gps_it->second.second);
        for (int i = 0; i < n; i++) {
          if (valid[i] && gps_src[i] == 0.0) valid[i] = false;
        }
      }
    }
    if (has_aux_intensity) {
      const float* src = static_cast<const float*>(frame->aux_attributes.at("intensity").second);
      for (int i = 0; i < n; i++) {
        if (valid[i] && !std::isfinite(src[i])) valid[i] = false;
      }
    }
    const size_t sm_nan = static_cast<size_t>(std::count(valid.begin(), valid.end(), false));
    total_nan_filtered += sm_nan;

    // Write geometry arrays, skipping NaN points
    for (int i = 0; i < n; i++) {
      if (!valid[i]) continue;
      ply.vertices.push_back((submap->T_world_origin * frame->points[i]).head<3>().cast<float>());
      if (has_normals) {
        ply.normals.push_back((R * frame->normals[i].head<3>()).cast<float>().normalized());
      }
      if (has_intensities) {
        ply.intensities.push_back(static_cast<float>(frame->intensities[i]));
      }
      session_id_data.push_back(submap->session_id);
    }

    // Write float aux attributes, skipping NaN points
    for (const auto& name : aux_names_float) {
      const float* src = static_cast<const float*>(frame->aux_attributes.at(name).second);
      for (int i = 0; i < n; i++) {
        if (valid[i]) aux_data_float[name].push_back(src[i]);
      }
    }

    // Write double aux attributes as double (preserves full GPS time precision), skipping NaN points
    for (const auto& name : aux_names_double) {
      const double* src = static_cast<const double*>(frame->aux_attributes.at(name).second);
      for (int i = 0; i < n; i++) {
        if (valid[i]) aux_data_double[name].push_back(src[i]);
      }
    }

    if (has_aux_intensity) {
      const float* src = static_cast<const float*>(frame->aux_attributes.at("intensity").second);
      for (int i = 0; i < n; i++) {
        if (valid[i]) aux_intensity_data.push_back(src[i]);
      }
    }
  }

  if (total_nan_filtered > 0) {
    logger->info("PLY export: filtered {} / {} points with NaN aux attributes", total_nan_filtered, total_points);
  } else {
    logger->info("PLY export: no NaN points filtered ({} points total)", total_points);
  }

  for (const auto& name : aux_names_float) {
    ply.add_prop<float>(name, aux_data_float[name].data(), aux_data_float[name].size());
  }
  for (const auto& name : aux_names_double) {
    // Written as "property double <name>" — full 64-bit precision in the PLY file.
    ply.add_prop<double>(name, aux_data_double[name].data(), aux_data_double[name].size());
  }
  if (has_aux_intensity) {
    ply.add_prop<float>("intensity_aux", aux_intensity_data.data(), aux_intensity_data.size());
  }
  if (sessions.size() > 1) {
    ply.add_prop<int>("session_id", session_id_data.data(), session_id_data.size());
  }

  // Print gps_time range summary so we can verify precision in the exported file.
  const auto gps_it = aux_data_double.find("gps_time");
  if (gps_it != aux_data_double.end() && !gps_it->second.empty()) {
    const auto& gps_vec = gps_it->second;
    const double gps_min = *std::min_element(gps_vec.begin(), gps_vec.end());
    const double gps_max = *std::max_element(gps_vec.begin(), gps_vec.end());
    logger->info("PLY export: gps_time range [{:.9f}, {:.9f}] ({} points)", gps_min, gps_max, gps_vec.size());
  }

  // Apply coordinate system transform when GNSS datum is available.
  // The world frame is UTM-origin aligned (East=+X, North=+Y, Up=+Z).
  // Absolute coordinates need double precision to avoid banding.
  const size_t n = ply.vertices.size();
  std::vector<double> out_x, out_y, out_z;
  std::vector<int> out_utm_zones;
  std::vector<Eigen::Vector2d> cached_latlon;  // (lat, lon) per point, for JGD2011 per-tile re-projection
  int jgd_zone_used = 0;

  if (gnss_datum_available) {
    // Geoid correction: convert ellipsoidal height to orthometric.
    double geoid_N = 0.0;
    if (geoid_correction_mode == 1) {
      geoid_N = static_cast<double>(geoid_manual_offset);
      logger->info("PLY export: applying manual geoid offset N = {:.3f} m", geoid_N);
    } else if (geoid_correction_mode == 2) {
      geoid_N = lookup_geoid_undulation(gnss_datum_lat, gnss_datum_lon);
      logger->info("PLY export: applying EGM2008 geoid undulation N = {:.3f} m", geoid_N);
    }

    out_x.resize(n);
    out_y.resize(n);
    out_z.resize(n);
    const bool south = gnss_datum_lat < 0.0;

    if (coord_system == 0) {
      // UTM WGS84
      if (consider_zones_on_export) {
        out_utm_zones.resize(n);
        bool has_zone_crossings = false;

        for (size_t i = 0; i < n; i++) {
          const Eigen::Vector3d pt = ply.vertices[i].cast<double>();
          const double abs_e = gnss_utm_easting_origin  + pt.x();
          const double abs_n = gnss_utm_northing_origin + pt.y();

          const Eigen::Vector2d latlon = utm_inverse(abs_e, abs_n, gnss_utm_zone, south);
          const int correct_zone = ecef_to_utm_zone(latlon.x(), latlon.y());
          out_utm_zones[i] = correct_zone;

          if (correct_zone != gnss_utm_zone) {
            has_zone_crossings = true;
            const Eigen::Vector2d new_utm = wgs84_to_utm_xy(latlon.x(), latlon.y(), correct_zone);
            out_x[i] = new_utm.x();
            out_y[i] = new_utm.y();
          } else {
            out_x[i] = abs_e;
            out_y[i] = abs_n;
          }
          out_z[i] = gnss_datum_alt + pt.z() - geoid_N;
        }

        if (has_zone_crossings) {
          logger->info("PLY export: zone crossings detected");
        }
        logger->info("PLY export: {} vertices to UTM (zone correction ON, datum zone {})", n, gnss_utm_zone);
      } else {
        for (size_t i = 0; i < n; i++) {
          const Eigen::Vector3d pt = ply.vertices[i].cast<double>();
          out_x[i] = gnss_utm_easting_origin  + pt.x();
          out_y[i] = gnss_utm_northing_origin + pt.y();
          out_z[i] = gnss_datum_alt           + pt.z() - geoid_N;
        }
        logger->info(
          "PLY export: {} vertices to UTM zone {} (zone correction OFF, E_origin={:.3f} N_origin={:.3f})",
          n, gnss_utm_zone, gnss_utm_easting_origin, gnss_utm_northing_origin);
      }
    } else if (coord_system == 1) {
      // Lazy-load prefecture boundaries and auto-detect zone from datum
      ensure_prefectures_loaded();

      // Resolve effective zone: manual prefecture > auto-detect > stub fallback
      if (jgd2011_pref_idx >= 0 && jgd2011_pref_idx < kPrefZoneTableSize) {
        jgd_zone_used = kPrefZoneTable[jgd2011_pref_idx].zone;
      } else if (detected_jgd_zone > 0) {
        jgd_zone_used = detected_jgd_zone;
      } else {
        jgd_zone_used = jgd2011_auto_zone(gnss_datum_lat, gnss_datum_lon);
        logger->warn("[JGD2011] No prefecture detected, falling back to zone {} ({})",
                     jgd_zone_used, jgd2011_zone_name(jgd_zone_used));
      }

      // Phase 1: inverse UTM → lat/lon for all points (zone-independent, cached)
      cached_latlon.resize(n);
      for (size_t i = 0; i < n; i++) {
        const Eigen::Vector3d pt = ply.vertices[i].cast<double>();
        const double abs_e = gnss_utm_easting_origin  + pt.x();
        const double abs_n = gnss_utm_northing_origin + pt.y();
        cached_latlon[i] = utm_inverse(abs_e, abs_n, gnss_utm_zone, south);
      }

      // Phase 2: project all points with the initial zone
      const TMProjectionParams params = jgd2011_zone_params(jgd_zone_used);
      for (size_t i = 0; i < n; i++) {
        const Eigen::Vector2d jgd = tm_forward(cached_latlon[i].x(), cached_latlon[i].y(), params);
        out_x[i] = jgd.x();
        out_y[i] = jgd.y();
        out_z[i] = gnss_datum_alt + ply.vertices[i].cast<double>().z() - geoid_N;
      }
      logger->info(
        "PLY export: {} vertices to JGD2011 zone {} ({}) via UTM inverse + TM forward",
        n, jgd_zone_used, jgd2011_zone_name(jgd_zone_used));
    }
  }

  // -----------------------------------------------------------------------
  // Save — either single file or per-tile split
  // -----------------------------------------------------------------------
  if (trim_by_tile && gnss_datum_available && !out_x.empty()) {
    // Group points by tile
    const double tile_size_m = grid_tile_size_km * 1000.0;
    std::unordered_map<std::string, std::vector<size_t>> tile_indices;

    for (size_t i = 0; i < n; i++) {
      const std::string tname = tile_name_for_point(
        out_x[i], out_y[i], grid_preset, tile_size_m, jgd_zone_used);
      tile_indices[tname].push_back(i);
    }

    // Debug: coordinate range and tile count
    {
      double x_min = out_x[0], x_max = out_x[0];
      double y_min = out_y[0], y_max = out_y[0];
      for (size_t i = 1; i < n; i++) {
        x_min = std::min(x_min, out_x[i]); x_max = std::max(x_max, out_x[i]);
        y_min = std::min(y_min, out_y[i]); y_max = std::max(y_max, out_y[i]);
      }
      logger->info(
        "PLY tile export: {} points, {} unique tiles, preset={}, tile_size={:.0f} m",
        n, tile_indices.size(), grid_preset, tile_size_m);
      logger->info(
        "PLY tile export: easting range [{:.3f}, {:.3f}], northing range [{:.3f}, {:.3f}]",
        x_min, x_max, y_min, y_max);
    }

    // Create output directory
    boost::filesystem::create_directories(path);

    size_t total_tiles = 0;
    for (const auto& kv : tile_indices) {
      const std::string& tname = kv.first;
      const std::vector<size_t>& indices = kv.second;
      const size_t tn = indices.size();

      glk::PLYData tile_ply;

      // Double-precision x, y, z — may be re-projected for JGD2011 per-tile zones
      std::vector<double> tx(tn), ty(tn), tz(tn);

      // JGD2011 per-tile zone detection: check if this tile's centroid
      // falls in a different prefecture/zone than the initial projection zone.
      int tile_zone = jgd_zone_used;
      if (coord_system == 1 && !cached_latlon.empty() && !prefectures.empty()) {
        // Compute centroid lat/lon from cached per-point values
        double clat = 0.0, clon = 0.0;
        for (size_t j = 0; j < tn; j++) {
          clat += cached_latlon[indices[j]].x();
          clon += cached_latlon[indices[j]].y();
        }
        clat /= static_cast<double>(tn);
        clon /= static_cast<double>(tn);

        // PIP test centroid against all prefectures
        for (const auto& pref : prefectures) {
          bool found = false;
          for (const auto& ring : pref.rings) {
            if (point_in_ring(clon, clat, ring)) {
              tile_zone = pref.jgd_zone;
              found = true;
              break;
            }
          }
          if (found) break;
        }
      }

      if (coord_system == 1 && tile_zone != jgd_zone_used && !cached_latlon.empty()) {
        // Re-project this tile's points with the correct zone
        const TMProjectionParams tile_params = jgd2011_zone_params(tile_zone);
        for (size_t j = 0; j < tn; j++) {
          const auto& ll = cached_latlon[indices[j]];
          const Eigen::Vector2d jgd = tm_forward(ll.x(), ll.y(), tile_params);
          tx[j] = jgd.x();
          ty[j] = jgd.y();
          tz[j] = out_z[indices[j]];
        }
        logger->info("PLY tile {}: re-projected to zone {} ({}) ({} points)",
                     tname, tile_zone, jgd2011_zone_name(tile_zone), tn);
      } else {
        for (size_t j = 0; j < tn; j++) {
          tx[j] = out_x[indices[j]];
          ty[j] = out_y[indices[j]];
          tz[j] = out_z[indices[j]];
        }
      }

      tile_ply.add_prop<double>("x", tx.data(), tn);
      tile_ply.add_prop<double>("y", ty.data(), tn);
      tile_ply.add_prop<double>("z", tz.data(), tn);

      // Normals
      if (has_normals) {
        tile_ply.normals.reserve(tn);
        for (size_t j = 0; j < tn; j++) tile_ply.normals.push_back(ply.normals[indices[j]]);
      }
      // Intensities
      if (has_intensities) {
        tile_ply.intensities.reserve(tn);
        for (size_t j = 0; j < tn; j++) tile_ply.intensities.push_back(ply.intensities[indices[j]]);
      }
      // Float aux attributes
      for (const auto& aname : aux_names_float) {
        const auto& src = aux_data_float.at(aname);
        std::vector<float> tv(tn);
        for (size_t j = 0; j < tn; j++) tv[j] = src[indices[j]];
        tile_ply.add_prop<float>(aname, tv.data(), tn);
      }
      // Double aux attributes
      for (const auto& aname : aux_names_double) {
        const auto& src = aux_data_double.at(aname);
        std::vector<double> tv(tn);
        for (size_t j = 0; j < tn; j++) tv[j] = src[indices[j]];
        tile_ply.add_prop<double>(aname, tv.data(), tn);
      }
      // Intensity aux
      if (has_aux_intensity) {
        std::vector<float> tv(tn);
        for (size_t j = 0; j < tn; j++) tv[j] = aux_intensity_data[indices[j]];
        tile_ply.add_prop<float>("intensity_aux", tv.data(), tn);
      }
      // UTM zone property
      if (!out_utm_zones.empty()) {
        std::vector<int> tz_zones(tn);
        for (size_t j = 0; j < tn; j++) tz_zones[j] = out_utm_zones[indices[j]];
        tile_ply.add_prop<int>("utm_zone", tz_zones.data(), tn);
      }
      // Session ID property (multi-map)
      if (sessions.size() > 1) {
        std::vector<int> ts_ids(tn);
        for (size_t j = 0; j < tn; j++) ts_ids[j] = session_id_data[indices[j]];
        tile_ply.add_prop<int>("session_id", ts_ids.data(), tn);
      }

      const std::string tile_path = path + "/" + tname + ".ply";
      glk::save_ply_binary(tile_path, tile_ply);
      total_tiles++;
    }
    logger->info("PLY export: {} tiles exported, {} total points", total_tiles, n);
  } else if (!out_x.empty()) {
    // Single file with transformed coordinates
    ply.vertices.clear();
    ply.add_prop<double>("x", out_x.data(), n);
    ply.add_prop<double>("y", out_y.data(), n);
    ply.add_prop<double>("z", out_z.data(), n);
    if (!out_utm_zones.empty()) {
      ply.add_prop<int>("utm_zone", out_utm_zones.data(), n);
    }
    glk::save_ply_binary(path, ply);
  } else {
    // No datum — local coordinates
    glk::save_ply_binary(path, ply);
  }
  return true;
}

std::pair<size_t, size_t> OfflineViewer::apply_range_filter_to_frame(const std::string& frame_dir) {
  // Read frame_meta.json
  const std::string meta_path = frame_dir + "/frame_meta.json";
  if (!boost::filesystem::exists(meta_path)) return {0, 0};
  std::ifstream meta_ifs(meta_path);
  const auto meta = nlohmann::json::parse(meta_ifs, nullptr, false);
  if (meta.is_discarded()) return {0, 0};
  const int num_pts = meta.value("num_points", 0);
  if (num_pts == 0) return {0, 0};

  // Read range.bin (required for filtering)
  std::vector<float> range(num_pts);
  {
    std::ifstream f(frame_dir + "/range.bin", std::ios::binary);
    if (!f) return {static_cast<size_t>(num_pts), 0};  // no range data, keep all
    f.read(reinterpret_cast<char*>(range.data()), sizeof(float) * num_pts);
  }

  // Read points.bin (needed for voxelization)
  std::vector<Eigen::Vector3f> points(num_pts);
  {
    std::ifstream f(frame_dir + "/points.bin", std::ios::binary);
    if (!f) return {static_cast<size_t>(num_pts), 0};
    f.read(reinterpret_cast<char*>(points.data()), sizeof(Eigen::Vector3f) * num_pts);
  }

  // Build voxel grid: map voxel key → list of point indices
  const float inv_voxel = 1.0f / rf_voxel_size;
  std::unordered_map<uint64_t, std::vector<int>> voxels;
  for (int i = 0; i < num_pts; i++) {
    voxels[glim::voxel_key(points[i], inv_voxel)].push_back(i);
  }

  // Determine which points to keep
  std::vector<bool> keep(num_pts, true);
  size_t removed = 0;

  for (const auto& [key, indices] : voxels) {
    // Find min range and count close points in this voxel
    float min_range = std::numeric_limits<float>::max();
    int close_count = 0;
    for (int idx : indices) {
      min_range = std::min(min_range, range[idx]);
      if (range[idx] <= rf_safe_range) close_count++;
    }

    if (close_count < rf_min_close_pts) {
      // No safe-range anchor — apply secondary (far) delta from min range
      const float far_threshold = min_range + rf_far_delta;
      for (int idx : indices) {
        if (range[idx] > far_threshold) {
          keep[idx] = false;
          removed++;
        }
      }
      continue;
    }

    // Remove distant points beyond safe anchor + delta
    for (int idx : indices) {
      if (range[idx] <= rf_safe_range) continue;  // always keep safe-range points
      if (range[idx] - min_range > rf_range_delta) {
        keep[idx] = false;
        removed++;
      }
    }
  }

  if (removed == 0) return {static_cast<size_t>(num_pts), 0};

  // Build filtered index list
  std::vector<int> kept_indices;
  kept_indices.reserve(num_pts - removed);
  for (int i = 0; i < num_pts; i++) {
    if (keep[i]) kept_indices.push_back(i);
  }
  const int new_count = static_cast<int>(kept_indices.size());

  // Rewrite all per-point binary files
  glim::filter_bin_file(frame_dir + "/points.bin", sizeof(Eigen::Vector3f), num_pts, kept_indices, new_count);
  glim::filter_bin_file(frame_dir + "/normals.bin", sizeof(Eigen::Vector3f), num_pts, kept_indices, new_count);
  glim::filter_bin_file(frame_dir + "/intensities.bin", sizeof(float), num_pts, kept_indices, new_count);
  glim::filter_bin_file(frame_dir + "/times.bin", sizeof(float), num_pts, kept_indices, new_count);
  glim::filter_bin_file(frame_dir + "/range.bin", sizeof(float), num_pts, kept_indices, new_count);
  glim::filter_bin_file(frame_dir + "/rings.bin", sizeof(uint16_t), num_pts, kept_indices, new_count);
  glim::filter_bin_file(frame_dir + "/aux_ground.bin", sizeof(float), num_pts, kept_indices, new_count);

  // Update frame_meta.json with new point count
  {
    std::ofstream ofs(meta_path);
    ofs << std::setprecision(15) << std::fixed;
    ofs << "{\n";
    ofs << "  \"frame_id\": " << meta.value("frame_id", 0) << ",\n";
    ofs << "  \"stamp\": " << meta.value("stamp", 0.0) << ",\n";
    ofs << "  \"scan_end_time\": " << meta.value("scan_end_time", 0.0) << ",\n";
    ofs << "  \"num_points\": " << new_count << ",\n";
    // Preserve T_world_lidar and bbox from original
    if (meta.contains("T_world_lidar")) {
      ofs << "  \"T_world_lidar\": " << meta["T_world_lidar"].dump() << ",\n";
    }
    if (meta.contains("bbox_world_min")) {
      ofs << "  \"bbox_world_min\": " << meta["bbox_world_min"].dump() << ",\n";
    }
    if (meta.contains("bbox_world_max")) {
      ofs << "  \"bbox_world_max\": " << meta["bbox_world_max"].dump() << "\n";
    }
    ofs << "}\n";
  }

  return {static_cast<size_t>(new_count), removed};
}

}  // namespace glim
