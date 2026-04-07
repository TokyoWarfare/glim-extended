#include <glim/viewer/offline_viewer.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
#include <gtsam_points/config.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>
#include <glim/util/config.hpp>
#include <glim/util/geodetic.hpp>

#include <spdlog/spdlog.h>
#include <portable-file-dialogs.h>
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

        // Trim by tile checkbox
        ImGui::Separator();
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

      // Log previous datum before overwriting config (multi-map diagnostics)
      if (gnss_datum_available) {
        logger->info("[multi-map] Previous datum: zone={} E={:.3f} N={:.3f} alt={:.3f} (will be overwritten by new map's config)",
                     gnss_utm_zone, gnss_utm_easting_origin, gnss_utm_northing_origin, gnss_datum_alt);
      }

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

      std::shared_ptr<GlobalMapping> existing_mapping;
      if (async_global_mapping) {
        logger->info("global map already exists, loading new map into existing global map");
        existing_mapping = std::dynamic_pointer_cast<GlobalMapping>(async_global_mapping->get_global_mapping());
      }

      progress_modal->open<std::shared_ptr<GlobalMapping>>(
        "open",
        [this, map_path, existing_mapping](guik::ProgressInterface& progress) { return load_map(progress, map_path, existing_mapping); });
    }
  }

  auto open_result = progress_modal->run<std::shared_ptr<GlobalMapping>>("open");
  if (open_result) {
    if (!(*open_result)) {
      pfd::message("Error", "Failed to load map").result();
    } else {
      // Save previous datum for multi-map delta logging
      const bool had_prev_datum = gnss_datum_available;
      const double prev_E = gnss_utm_easting_origin;
      const double prev_N = gnss_utm_northing_origin;
      const double prev_alt = gnss_datum_alt;

      async_global_mapping.reset(new glim::AsyncGlobalMapping(*open_result, 1e6));
      load_gnss_datum();

      if (had_prev_datum && gnss_datum_available) {
        const double dE = gnss_utm_easting_origin - prev_E;
        const double dN = gnss_utm_northing_origin - prev_N;
        const double dZ = gnss_datum_alt - prev_alt;
        logger->warn(
          "[multi-map] Datum changed: dE={:.3f} m, dN={:.3f} m, dZ={:.3f} m. "
          "Submaps from previous map(s) retain their original poses — "
          "this offset is NOT applied to them.",
          dE, dN, dZ);
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
  std::shared_ptr<GlobalMapping> global_mapping) {
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

  if (!global_mapping->load(path)) {
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

  // Determine which fields are present across all submaps
  bool has_normals = true;
  bool has_intensities = true;
  size_t total_points = 0;
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }
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
  if (submaps[0] && submaps[0]->frame) {
    for (const auto& attrib : submaps[0]->frame->aux_attributes) {
      const size_t elem_size = attrib.second.first;
      if (elem_size != sizeof(float) && elem_size != sizeof(double)) {
        continue;
      }
      if (primary_ply_props.count(attrib.first)) {
        continue;
      }
      bool all_have = true;
      for (const auto& sm : submaps) {
        if (!sm || !sm->frame) {
          all_have = false;
          break;
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
    if (!submap || !submap->frame) {
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

  size_t total_nan_filtered = 0;

  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
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

}  // namespace glim
