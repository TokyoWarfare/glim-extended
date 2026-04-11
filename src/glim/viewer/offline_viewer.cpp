#include <glim/viewer/offline_viewer.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
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
      trajectory_data.push_back({T_world_lidar, cumul, submap->session_id, frame->id});
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

  // Range Filter tool window
  viewer->register_ui_callback("range_filter_window", [this] {
    if (!show_range_filter) return;
    ImGui::SetNextWindowSize(ImVec2(350, 280), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Range Filter", &show_range_filter)) {
      ImGui::Text("Remove distant noise when closer points exist");
      ImGui::Separator();

      ImGui::SliderFloat("Voxel size (m)", &rf_voxel_size, 0.05f, 5.0f, "%.2f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Spatial grid cell size for grouping points.");

      ImGui::SliderFloat("Safe range (m)", &rf_safe_range, 5.0f, 50.0f, "%.0f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Points within this range are ALWAYS kept.");

      ImGui::SliderFloat("Range delta (m)", &rf_range_delta, 1.0f, 50.0f, "%.0f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove points >delta further than closest safe-range\npoint in the voxel.");

      ImGui::SliderFloat("Far delta (m)", &rf_far_delta, 5.0f, 100.0f, "%.0f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Secondary delta for voxels with NO safe-range points.\nRemoves points > (min_range + far_delta) in the voxel.\nCleans up distant noise where no close data exists.");

      ImGui::SliderInt("Min close points", &rf_min_close_pts, 1, 20);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum close-range points in a voxel before\nthe primary delta applies. Below this threshold,\nthe far delta is used instead.");

      // Range highlight: re-color cached preview data by range threshold
      // Uses intensity as base color, red tint beyond threshold
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

      ImGui::Separator();

      if (!hd_available) ImGui::BeginDisabled();

      if (rf_processing) {
        ImGui::Text("%s", rf_status.c_str());
      } else {
        // Preview: cross-frame voxel filtering in visible area
        if (ImGui::Button("Preview (visible area)")) {
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
              int frame_idx;
              int point_idx;
            };

            const float inv_voxel = 1.0f / rf_voxel_size;
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

                const Eigen::Isometry3d T_w_imu = T_ep * T_odom0.inverse() * frame->T_world_imu;
                const Eigen::Isometry3d T_w_lidar = T_w_imu * frame->T_lidar_imu.inverse();
                const Eigen::Matrix3f R = T_w_lidar.rotation().cast<float>();
                const Eigen::Vector3f t_vec = T_w_lidar.translation().cast<float>();

                for (int i = 0; i < num_pts; i++) {
                  if (range[i] < 1.5f) continue;  // skip near-sensor noise
                  const Eigen::Vector3f wp = R * pts[i] + t_vec;
                  const int vx = static_cast<int>(std::floor(wp.x() * inv_voxel));
                  const int vy = static_cast<int>(std::floor(wp.y() * inv_voxel));
                  const int vz = static_cast<int>(std::floor(wp.z() * inv_voxel));
                  const uint64_t key = (static_cast<uint64_t>(vx + 1048576) << 42) |
                                       (static_cast<uint64_t>(vy + 1048576) << 21) |
                                       static_cast<uint64_t>(vz + 1048576);
                  voxels[key].push_back({wp, range[i], intensity[i], frame_count, i});
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

            // Cross-frame filter: within each world-space voxel, find close points
            // and remove distant ones from other frames
            std::vector<Eigen::Vector3f> kept_points, removed_points;
            std::vector<float> kept_intensities, kept_ranges;
            std::vector<Eigen::Vector3f> removed_positions;
            std::vector<float> removed_ranges, removed_intensities;
            size_t preview_kept = 0, preview_removed = 0;

            for (const auto& [key, entries] : voxels) {
              // Find max range among close points (within safe_range)
              float max_close_range = 0.0f;
              int close_count = 0;
              for (const auto& e : entries) {
                if (e.range <= rf_safe_range) {
                  max_close_range = std::max(max_close_range, e.range);
                  close_count++;
                }
              }

              if (close_count < rf_min_close_pts) {
                // No safe-range anchor — apply secondary (far) delta from min range in voxel
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

              // Remove points beyond max_close_range + delta
              const float threshold = max_close_range + rf_range_delta;
              for (const auto& e : entries) {
                if (e.range <= rf_safe_range || e.range <= threshold) {
                  kept_points.push_back(e.world_pos);
                  kept_intensities.push_back(e.intensity);
                  kept_ranges.push_back(e.range);
                  preview_kept++;
                } else {
                  removed_points.push_back(e.world_pos);
                  removed_intensities.push_back(e.intensity);
                  removed_ranges.push_back(e.range);
                  preview_removed++;
                }
              }
            }

            // Cache preview data for range highlight re-coloring
            rf_preview_data.clear();
            rf_preview_data.reserve(preview_kept + preview_removed);
            for (size_t pi = 0; pi < kept_points.size(); pi++) {
              rf_preview_data.push_back({kept_points[pi], kept_ranges[pi], kept_intensities[pi], true});
            }
            for (size_t pi = 0; pi < removed_points.size(); pi++) {
              rf_preview_data.push_back({removed_points[pi], removed_ranges[pi], removed_intensities[pi], false});
            }

            auto kept_buf = std::make_shared<std::vector<Eigen::Vector3f>>(std::move(kept_points));
            auto kept_int = std::make_shared<std::vector<float>>(std::move(kept_intensities));
            auto removed_buf = std::make_shared<std::vector<Eigen::Vector3f>>(std::move(removed_points));

            vw->invoke([this, kept_buf, kept_int, removed_buf, preview_kept, preview_removed] {
              auto viewer = guik::LightViewer::instance();
              if (!kept_buf->empty()) {
                const int n = kept_buf->size();
                std::vector<Eigen::Vector4d> pts4(n);
                for (int i = 0; i < n; i++) pts4[i] = Eigen::Vector4d((*kept_buf)[i].x(), (*kept_buf)[i].y(), (*kept_buf)[i].z(), 1.0);
                auto cb = std::make_shared<glk::PointCloudBuffer>(pts4.data(), n);
                if (kept_int->size() == static_cast<size_t>(n)) {
                  cb->add_buffer("intensity", *kept_int);
                  cb->set_colormap_buffer("intensity");
                }
                viewer->update_drawable("rf_preview_kept", cb, guik::FlatColor(0.0f, 0.8f, 0.2f, 1.0f));
              }
              if (!removed_buf->empty()) {
                const int n = removed_buf->size();
                std::vector<Eigen::Vector4d> pts4(n);
                for (int i = 0; i < n; i++) pts4[i] = Eigen::Vector4d((*removed_buf)[i].x(), (*removed_buf)[i].y(), (*removed_buf)[i].z(), 1.0);
                auto cb = std::make_shared<glk::PointCloudBuffer>(pts4.data(), n);
                viewer->update_drawable("rf_preview_removed", cb, guik::FlatColor(1.0f, 0.0f, 0.0f, 0.5f).make_transparent());
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
        if (rf_show_chunks) {
          if (!trajectory_built) build_trajectory();
          auto vw = guik::LightViewer::instance();
          const double hs = rf_chunk_size * 0.5;
          int chunk_count = 0;
          for (double d = 0.0; d < trajectory_total_dist; d += rf_chunk_spacing) {
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

        ImGui::SliderFloat("Chunk size (m)", &rf_chunk_size, 20.0f, 200.0f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Size of each processing chunk (width x height).\nLarger = more cross-frame context but more memory.");
        ImGui::SliderFloat("Chunk spacing (m)", &rf_chunk_spacing, 10.0f, 100.0f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Distance between chunk centers along trajectory.\nSmaller = more overlap, better coverage.");

        if (ImGui::Button("Apply to HD frames (chunked)")) {
          rf_processing = true;
          rf_status = "Building trajectory...";
          std::thread([this] {
            if (!trajectory_built) build_trajectory();
            const auto start_time = std::chrono::steady_clock::now();

            // Step 1: Place path-aligned chunk centers along trajectory
            struct Chunk {
              Eigen::Vector3d center;
              Eigen::Matrix3d R_world_chunk;  // chunk-local to world rotation
              Eigen::Matrix3d R_chunk_world;  // world to chunk-local rotation
              double half_size;
              double half_height;
            };
            std::vector<Chunk> chunks;
            const double hs = rf_chunk_size * 0.5;
            const double hh = 50.0;  // ±50m height (only occupies voxels where points exist)
            for (double d = 0.0; d < trajectory_total_dist; d += rf_chunk_spacing) {
              size_t idx = 0;
              for (size_t k = 1; k < trajectory_data.size(); k++) {
                if (trajectory_data[k].cumulative_dist >= d) { idx = k; break; }
              }
              const Eigen::Vector3d c = trajectory_data[idx].pose.translation();
              // Get heading from trajectory tangent
              const size_t next = std::min(idx + 1, trajectory_data.size() - 1);
              Eigen::Vector3d fwd = trajectory_data[next].pose.translation() - trajectory_data[idx].pose.translation();
              fwd.z() = 0.0;  // project to XY plane
              if (fwd.norm() < 0.01) fwd = Eigen::Vector3d::UnitX();
              else fwd.normalize();
              // Build local frame: forward, right, up
              const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
              const Eigen::Vector3d right = fwd.cross(up).normalized();
              Eigen::Matrix3d R;
              R.col(0) = fwd;    // local X = forward along path
              R.col(1) = right;  // local Y = right of path
              R.col(2) = up;     // local Z = up
              chunks.push_back({c, R, R.transpose(), hs, hh});
            }
            logger->info("[RangeFilter] {} chunks along {:.0f} m trajectory", chunks.size(), trajectory_total_dist);

            // Step 2: Build frame index with world-space bounding boxes from frame_meta.json
            struct FrameInfo {
              std::string dir;
              Eigen::Isometry3d T_world_lidar;
              Eigen::AlignedBox3d world_bbox;  // from frame_meta.json, transformed by optimized pose
              int num_points;
            };
            std::vector<FrameInfo> all_frames;
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

                const Eigen::Isometry3d T_w_imu = T_ep * T_odom0.inverse() * frame->T_world_imu;
                const Eigen::Isometry3d T_w_lidar = T_w_imu * frame->T_lidar_imu.inverse();

                // Compute world bbox from frame_meta's local bbox + optimized pose
                Eigen::AlignedBox3d wbox;
                if (meta.contains("bbox_world_min") && meta.contains("bbox_world_max")) {
                  // bbox in frame_meta is from odometry pose — recompute with optimized pose
                  const auto& bmin_j = meta["bbox_world_min"];
                  const auto& bmax_j = meta["bbox_world_max"];
                  const Eigen::Vector3d local_min(bmin_j[0].get<double>() - meta["T_world_lidar"][3].get<double>(),
                                                   bmin_j[1].get<double>() - meta["T_world_lidar"][7].get<double>(),
                                                   bmin_j[2].get<double>() - meta["T_world_lidar"][11].get<double>());
                  const Eigen::Vector3d local_max(bmax_j[0].get<double>() - meta["T_world_lidar"][3].get<double>(),
                                                   bmax_j[1].get<double>() - meta["T_world_lidar"][7].get<double>(),
                                                   bmax_j[2].get<double>() - meta["T_world_lidar"][11].get<double>());
                  // Transform 8 corners by optimized pose
                  for (int ci = 0; ci < 8; ci++) {
                    Eigen::Vector3d corner(
                      (ci & 1) ? local_max.x() : local_min.x(),
                      (ci & 2) ? local_max.y() : local_min.y(),
                      (ci & 4) ? local_max.z() : local_min.z());
                    wbox.extend(T_w_lidar * corner);
                  }
                } else {
                  // Fallback: sensor position ± 200m
                  const Eigen::Vector3d pos = T_w_lidar.translation();
                  wbox.extend(pos - Eigen::Vector3d::Constant(200.0));
                  wbox.extend(pos + Eigen::Vector3d::Constant(200.0));
                }

                all_frames.push_back({frame_dir, T_w_lidar, wbox, meta.value("num_points", 0)});
              }
            }
            logger->info("[RangeFilter] Indexed {} HD frames", all_frames.size());

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
                std::vector<int> original_indices;  // index in the frame's points.bin
              };
              std::vector<ChunkFrameData> chunk_frames;

              for (const auto& fi : all_frames) {
                if (fi.num_points == 0) continue;
                // Bbox ∩ chunk test: check if frame's world bbox overlaps the chunk
                // Transform chunk bounds to world-aligned AABB for fast box-box test
                const Eigen::Vector3d chunk_world_half(chunk.half_size, chunk.half_size, chunk.half_height);
                Eigen::AlignedBox3d chunk_aabb;
                for (int ci = 0; ci < 8; ci++) {
                  Eigen::Vector3d local(
                    (ci & 1) ? chunk_world_half.x() : -chunk_world_half.x(),
                    (ci & 2) ? chunk_world_half.y() : -chunk_world_half.y(),
                    (ci & 4) ? chunk_world_half.z() : -chunk_world_half.z());
                  chunk_aabb.extend(chunk.center + chunk.R_world_chunk * local);
                }
                if (!chunk_aabb.intersects(fi.world_bbox)) continue;

                // Load frame point data
                const int num_pts = fi.num_points;

                std::vector<Eigen::Vector3f> pts(num_pts);
                std::vector<float> range(num_pts);
                { std::ifstream f(fi.dir + "/points.bin", std::ios::binary);
                  if (!f) continue;
                  f.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * num_pts); }
                { std::ifstream f(fi.dir + "/range.bin", std::ios::binary);
                  if (!f) continue;
                  f.read(reinterpret_cast<char*>(range.data()), sizeof(float) * num_pts); }

                const Eigen::Matrix3f R = fi.T_world_lidar.rotation().cast<float>();
                const Eigen::Vector3f t = fi.T_world_lidar.translation().cast<float>();

                ChunkFrameData cfd;
                cfd.dir = fi.dir;
                for (int i = 0; i < num_pts; i++) {
                  if (range[i] < 1.5f) continue;
                  const Eigen::Vector3f wp = R * pts[i] + t;
                  // Point-in-chunk test in chunk-local coordinates
                  const Eigen::Vector3d local_pt = chunk.R_chunk_world * (wp.cast<double>() - chunk.center);
                  if (std::abs(local_pt.x()) <= chunk.half_size &&
                      std::abs(local_pt.y()) <= chunk.half_size &&
                      std::abs(local_pt.z()) <= chunk.half_height) {
                    cfd.world_points.push_back(wp);
                    cfd.ranges.push_back(range[i]);
                    cfd.original_indices.push_back(i);
                  }
                }
                if (!cfd.world_points.empty()) chunk_frames.push_back(std::move(cfd));
              }

              // Build cross-frame voxel grid for this chunk
              struct VoxelEntry { int cf_idx; int pt_idx; float range; };
              std::unordered_map<uint64_t, std::vector<VoxelEntry>> voxels;
              for (int cfi = 0; cfi < static_cast<int>(chunk_frames.size()); cfi++) {
                const auto& cf = chunk_frames[cfi];
                for (int pi = 0; pi < static_cast<int>(cf.world_points.size()); pi++) {
                  const auto& wp = cf.world_points[pi];
                  const int vx = static_cast<int>(std::floor(wp.x() * inv_voxel));
                  const int vy = static_cast<int>(std::floor(wp.y() * inv_voxel));
                  const int vz = static_cast<int>(std::floor(wp.z() * inv_voxel));
                  const uint64_t key = (static_cast<uint64_t>(vx + 1048576) << 42) |
                                       (static_cast<uint64_t>(vy + 1048576) << 21) |
                                       static_cast<uint64_t>(vz + 1048576);
                  voxels[key].push_back({cfi, pi, cf.ranges[pi]});
                }
              }

              // Filter: cross-frame range discrimination
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
                  // No safe-range anchor — apply secondary (far) delta
                  float min_range = std::numeric_limits<float>::max();
                  for (const auto& e : entries) min_range = std::min(min_range, e.range);
                  const float far_threshold = min_range + rf_far_delta;
                  for (const auto& e : entries) {
                    if (e.range > far_threshold) {
                      const auto& cf = chunk_frames[e.cf_idx];
                      frame_removals[cf.dir].insert(cf.original_indices[e.pt_idx]);
                    }
                  }
                  continue;
                }
                const float threshold = max_close_range + rf_range_delta;
                for (const auto& e : entries) {
                  if (e.range <= rf_safe_range) continue;
                  if (e.range > threshold) {
                    // Mark for removal
                    const auto& cf = chunk_frames[e.cf_idx];
                    frame_removals[cf.dir].insert(cf.original_indices[e.pt_idx]);
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
              auto filter_file = [&](const std::string& filename, size_t elem_size) {
                const std::string path = frame_dir + "/" + filename;
                if (!boost::filesystem::exists(path)) return;
                std::vector<char> src(num_pts * elem_size);
                { std::ifstream f(path, std::ios::binary); f.read(src.data(), src.size()); }
                std::vector<char> dst(new_count * elem_size);
                for (int j = 0; j < new_count; j++) {
                  std::memcpy(dst.data() + j * elem_size, src.data() + kept_indices[j] * elem_size, elem_size);
                }
                { std::ofstream f(path, std::ios::binary); f.write(dst.data(), dst.size()); }
              };

              filter_file("points.bin", sizeof(Eigen::Vector3f));
              filter_file("normals.bin", sizeof(Eigen::Vector3f));
              filter_file("intensities.bin", sizeof(float));
              filter_file("times.bin", sizeof(float));
              filter_file("range.bin", sizeof(float));
              filter_file("rings.bin", sizeof(uint16_t));

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
            logger->info("[RangeFilter] {}", rf_status);
          }).detach();
        }
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("DESTRUCTIVE: cross-frame filtering along trajectory.\nBackup first with Tools > Utils > Backup HD frames.");
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
      if (ImGui::MenuItem("Range Filter", nullptr, show_range_filter)) {
        show_range_filter = !show_range_filter;
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
    const auto& p = points[i];
    const int vx = static_cast<int>(std::floor(p.x() * inv_voxel));
    const int vy = static_cast<int>(std::floor(p.y() * inv_voxel));
    const int vz = static_cast<int>(std::floor(p.z() * inv_voxel));
    // Pack 3 ints into uint64 (21 bits each, handles ±1M voxels)
    const uint64_t key = (static_cast<uint64_t>(vx + 1048576) << 42) |
                         (static_cast<uint64_t>(vy + 1048576) << 21) |
                         static_cast<uint64_t>(vz + 1048576);
    voxels[key].push_back(i);
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

  // Helper: filter and rewrite a binary file
  auto filter_file = [&](const std::string& filename, size_t elem_size) {
    const std::string path = frame_dir + "/" + filename;
    if (!boost::filesystem::exists(path)) return;
    std::vector<char> src(num_pts * elem_size);
    { std::ifstream f(path, std::ios::binary); f.read(src.data(), src.size()); }
    std::vector<char> dst(new_count * elem_size);
    for (int j = 0; j < new_count; j++) {
      std::memcpy(dst.data() + j * elem_size, src.data() + kept_indices[j] * elem_size, elem_size);
    }
    { std::ofstream f(path, std::ios::binary); f.write(dst.data(), dst.size()); }
  };

  // Rewrite all per-point binary files
  filter_file("points.bin", sizeof(Eigen::Vector3f));    // 12 bytes
  filter_file("normals.bin", sizeof(Eigen::Vector3f));   // 12 bytes
  filter_file("intensities.bin", sizeof(float));          // 4 bytes
  filter_file("times.bin", sizeof(float));                // 4 bytes
  filter_file("range.bin", sizeof(float));                // 4 bytes
  filter_file("rings.bin", sizeof(uint16_t));             // 2 bytes

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
