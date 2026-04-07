#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <Eigen/Geometry>
#include <glim/mapping/global_mapping.hpp>
#include <glim/mapping/async_global_mapping.hpp>
#include <glim/viewer/interactive_viewer.hpp>

namespace guik {
class ProgressModal;
class ProgressInterface;
}  // namespace guik

namespace glim {

class OfflineViewer : public InteractiveViewer {
public:
  OfflineViewer(const std::string& init_map_path = "");
  virtual ~OfflineViewer() override;

private:
  virtual void setup_ui() override;

  void main_menu();

  std::shared_ptr<GlobalMapping> load_map(guik::ProgressInterface& progress, const std::string& path, std::shared_ptr<GlobalMapping> global_mapping);
  bool save_map(guik::ProgressInterface& progress, const std::string& path);
  bool export_map(guik::ProgressInterface& progress, const std::string& path);

  /// Tries to load gnss_datum.json from the current GlobalConfig config path.
  /// Sets gnss_datum_available and populates UTM origin on success.
  void load_gnss_datum();

  // Geoid undulation lookup (Issue 1 — ellipsoidal→orthometric correction).
  // Searches AMENT_PREFIX_PATH for share/glim_ext/EGM_tables/*.geoid files,
  // then tries <config_dir>/EGM_tables/ as a local override.
  // Returns 0.0 and logs a warning if no table covers (lat, lon).
  double lookup_geoid_undulation(double lat, double lon) const;

private:
  std::string init_map_path;
  std::unique_ptr<guik::ProgressModal> progress_modal;

  std::unordered_set<std::string> imported_shared_libs;
  std::unique_ptr<AsyncGlobalMapping> async_global_mapping;

  // GNSS datum loaded from gnss_datum.json (written by GNSSGlobal).
  // World frame = UTM frame (origin subtracted). No rotation needed.
  bool             gnss_datum_available     = false;
  int              gnss_utm_zone            = 0;
  double           gnss_utm_easting_origin  = 0.0;
  double           gnss_utm_northing_origin = 0.0;
  double           gnss_datum_alt           = 0.0;
  double           gnss_datum_lat           = 0.0;
  double           gnss_datum_lon           = 0.0;

  // Coordinate system selection (Coordinates menu).
  int  coord_system              = 0;     // 0=UTM WGS84, 1=JGD2011, 2=Custom (reserved)
  int  jgd2011_pref_idx          = -1;    // index into kPrefZoneTable; -1=auto-detect
  bool consider_zones_on_export  = true;  // per-point UTM zone correction (UTM only)

  // JGD2011 prefecture detection (lazy-loaded on first JGD2011 export).
  struct PrefectureEntry {
    std::string name_jp;   // e.g. "静岡県"
    std::string name_en;   // e.g. "Shizuoka"
    int jgd_zone;          // 1-19
    std::vector<std::vector<Eigen::Vector2d>> rings;  // exterior rings: (lon, lat) pairs
  };
  std::vector<PrefectureEntry> prefectures;
  bool prefectures_loaded = false;
  std::string detected_pref_jp;   // auto-detected from datum
  std::string detected_pref_en;
  int detected_jgd_zone = 0;

  /// Load japan_prefectures.geojson and detect zone from datum. Called lazily.
  void ensure_prefectures_loaded();

  // Grid presets (Coordinates menu).
  int   grid_preset        = 0;     // 0=None, 1=PNOA Spain, 2=ICGC Cataluna, 3=Virtual Shizuoka, 4=Custom
  float grid_tile_size_km  = 2.0f;
  bool  trim_by_tile       = false; // split export into per-tile PLY files

  // PLY export options (persistent across export invocations).
  int   geoid_correction_mode  = 0;    // 0=None  1=Manual  2=Auto EGM2008
  float geoid_manual_offset    = 0.0f; // metres, used when mode==1
};

}  // namespace glim
