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

  std::shared_ptr<GlobalMapping> load_map(guik::ProgressInterface& progress, const std::string& path, std::shared_ptr<GlobalMapping> global_mapping, const Eigen::Vector3d& datum_offset);
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
  std::string loaded_map_path;  // path to the currently loaded dump directory
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

  // Reference datum: first loaded map's datum, used as coordinate origin for all sessions.
  bool             ref_datum_set            = false;
  int              ref_utm_zone             = 0;
  double           ref_utm_easting          = 0.0;
  double           ref_utm_northing         = 0.0;
  double           ref_datum_alt            = 0.0;

  // Session visibility/export control (indexed by session_id).
  struct SessionState {
    int id;
    std::string path;
    std::string hd_frames_path;  // per-session HD frames directory
    bool visible = true;
    bool export_enabled = true;
    bool unloaded = false;  // permanently removed from viewer (cannot undo)
  };
  std::vector<SessionState> sessions;

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

  // Camera mode
  int camera_mode_sel = 0;  // 0=Orbit, 1=FPV, 2=Follow Trajectory
  float fpv_speed = 1.0f;
  float fpv_speed_mult = 10.0f;  // shift multiplier
  float fpv_smoothness = 0.25f;  // position smoothing (lower = smoother)
  Eigen::Vector3f fpv_smooth_pos = Eigen::Vector3f::Zero();
  float fpv_smooth_yaw = 0.0f;
  float fpv_smooth_pitch = 0.0f;
  bool fpv_smooth_init = false;

  // Shared trajectory data — used by follow camera, trajectory rendering, chunk processing
  struct TrajectoryPoint {
    Eigen::Isometry3d pose;  // full 6-DOF pose (T_world_lidar)
    double cumulative_dist;  // metres from start
    int session_id;
    long frame_id;
  };
  std::vector<TrajectoryPoint> trajectory_data;
  double trajectory_total_dist = 0.0;
  bool trajectory_built = false;
  void build_trajectory();  // builds/rebuilds trajectory_data from submaps

  // Follow trajectory mode (uses trajectory_data)
  double follow_total_dist = 0.0;
  float follow_progress = 0.0f;    // 0.0-1.0 slider position
  float follow_speed_kmh = 30.0f;  // playback speed
  bool follow_playing = false;     // auto-advance
  double follow_last_time = 0.0;   // for delta time computation
  float follow_smoothness = 0.05f;  // position smoothing factor (lower = smoother)
  float follow_yaw_offset = 0.0f;   // user turret rotation (mouse drag)
  float follow_pitch_offset = 0.0f;
  Eigen::Vector3d follow_smooth_pos = Eigen::Vector3d::Zero();  // smoothed camera position
  double follow_actual_speed_ms = 0.0;  // measured camera speed (m/s)
  float follow_smooth_yaw = 0.0f;
  float follow_smooth_pitch = 0.0f;
  bool follow_smooth_init = false;

  // Range filter tool
  bool show_range_filter = false;
  float rf_voxel_size = 1.0f;        // metres
  float rf_safe_range = 30.0f;      // metres — points within this always kept
  float rf_range_delta = 10.0f;     // metres — remove if >delta further than closest in voxel
  float rf_far_delta = 30.0f;       // metres — in voxels with no safe-range points, remove if > min_range + far_delta
  int   rf_min_close_pts = 3;       // min close points to trigger removal of distant ones
  float rf_range_highlight = 0.0f; // range threshold for red tinting (0=off)
  bool rf_preview_active = false;  // preview overlay is showing — hide other LOD data
  bool rf_intensity_mode = false;  // toggle intensity display on preview

  // Cached preview data (kept in CPU memory for range highlight re-coloring)
  struct PreviewPoint {
    Eigen::Vector3f pos;
    float range;
    float intensity;
    bool kept;  // true = kept by filter, false = removed
  };
  std::vector<PreviewPoint> rf_preview_data;
  float rf_chunk_size = 60.0f;     // metres — chunk size along trajectory
  float rf_chunk_spacing = 30.0f;  // metres — spacing between chunk centers
  bool  rf_show_chunks = false;    // visualize chunk boxes
  bool  rf_processing = false;
  std::string rf_status;

  /// Apply range filter to a single HD frame directory. Returns (kept, removed).
  std::pair<size_t, size_t> apply_range_filter_to_frame(const std::string& frame_dir);

  // PLY export options (persistent across export invocations).
  bool  export_hd              = false; // export HD frames instead of SD submaps
  int   geoid_correction_mode  = 0;    // 0=None  1=Manual  2=Auto EGM2008
  float geoid_manual_offset    = 0.0f; // metres, used when mode==1
};

}  // namespace glim
