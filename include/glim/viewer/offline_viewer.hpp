#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <atomic>
#include <unordered_set>
#include <Eigen/Geometry>
#include <glim/mapping/global_mapping.hpp>
#include <glim/mapping/async_global_mapping.hpp>
#include <glim/viewer/interactive_viewer.hpp>
#include <opencv2/core.hpp>
#include <glim/util/lidar_colorizer.hpp>
#include <glim/util/auto_calibrate.hpp>
#include <glim/util/virtual_cameras.hpp>

namespace guik {
class ProgressModal;
class ProgressInterface;
class ModelControl;
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
  int camera_mode_sel = 1;  // 0=Orbit, 1=FPV, 2=Follow Trajectory
  bool show_axis_gizmo = true;  // bottom-left world-axis indicator
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
    double stamp;            // frame timestamp (UNIX epoch)
    int session_id;
    long frame_id;
  };
  std::vector<TrajectoryPoint> trajectory_data;
  double trajectory_total_dist = 0.0;
  bool trajectory_built = false;
  void build_trajectory();  // builds/rebuilds trajectory_data from submaps

  // Snapshot the shared `trajectory_data` as a `TimedPose` vector suitable for
  // Colorizer::interpolate_pose / locate_by_time. Previously inlined ~9x across
  // the Colorize menu, Apply paths, alignment checker and Time Matcher.
  std::vector<TimedPose> timed_traj_snapshot() const;

  // Per-submap aux_rgb.bin loader for the build_calibration_context()
  // `load_hd_aux_rgb_for_submap` callback. Mirrors load_hd_for_submap's
  // per-frame iteration + 1.5 m range filter so the returned RGB vector is
  // parallel to the cloud's point order. Used by the virtual-camera path
  // (both per-RGB-camera and along-trajectory) to enable NativeRGB
  // rendering from per-point aux_rgb.bin without camera projection.
  std::function<std::vector<Eigen::Vector3f>(int)> make_aux_rgb_loader();

  // Snapshot of all current Colorize-window state into a BlendParams struct.
  // Called from every colorize site to avoid the 20-line duplicated initializer
  // list. Any new vs_* / colorize_* field lands here once and propagates
  // automatically to preview / live-preview / apply / align-check paths.
  /// Build BlendParams from the given source's ColorizeParams. Use the overload
  /// without `src` when the context is the Colorize window's active source
  /// (image_sources[colorize_source_idx]). Call sites that project onto a
  /// specific source (right-click "Colorize from this camera", Apply workers,
  /// alignment checker) should pass that source explicitly.
  BlendParams current_blend_params(const ImageSource& src) const;
  BlendParams current_blend_params() const;

  // Render a single camera's 3D gizmo (cube + FOV frustum for Pinhole, blue
  // sphere + forward ray for Spherical). Branches on the source's camera_type.
  // Safe to call per-frame -- update_drawable is idempotent. One source of
  // truth for gizmo rendering; called from Locate, Show-toggle, and
  // live-preview tracking paths so they don't drift out of sync.
  void render_camera_gizmo(int src_idx, int frame_idx);

  /// Shared anchor-panel widget. Draws "Anchor here" + table + drift badge for
  /// the given source. Called from both the Colorize window (anchor-at-current-
  /// preview) and the Alignment-check window (anchor-at-scrubbed-frame) so the
  /// UI stays identical in both places.
  ///
  /// cam_time: source-local timestamp to commit on "Anchor here" click.
  /// have_time: false disables the button (no current frame resolved).
  /// id_suffix: disambiguates ImGui ids when the panel appears twice.
  void render_anchor_panel(int src_idx, double cam_time, bool have_time, const char* id_suffix);

  // Follow trajectory mode (uses trajectory_data)
  double follow_total_dist = 0.0;
  float follow_progress = 0.0f;    // 0.0-1.0 slider position
  float follow_speed_kmh = 30.0f;  // playback speed
  bool follow_playing = false;     // auto-advance
  double follow_last_time = 0.0;   // for delta time computation
  float follow_smoothness = 0.05f;  // position smoothing factor (lower = smoother)
  float follow_yaw_offset = 0.0f;   // user turret rotation (mouse drag)
  float follow_pitch_offset = 0.0f;
  float follow_height_offset_m = 0.0f;  // vertical shift above spline (drone-view)
  Eigen::Vector3d follow_smooth_pos = Eigen::Vector3d::Zero();  // smoothed camera position
  double follow_actual_speed_ms = 0.0;  // measured camera speed (m/s)
  float follow_smooth_yaw = 0.0f;
  float follow_smooth_pitch = 0.0f;
  bool follow_smooth_init = false;

  // Data Cleaner tool (SOR / Range / Dynamic / Dynamic-Erasor point-removal filters)
  bool show_data_filter = false;
  int  df_mode = 0;                  // 0=SOR, 1=Range, 2=Dynamic, 3=Dynamic-Erasor

  // Sensor preset (currently only Livox Horizon; Pandar 128 / Velodyne / Ouster
  // slots will be added once we have data to tune them). On selection, the
  // preset's algorithm params are copied into the per-mode UI fields below.
  // Struct definition lives in glim/util/sensor_preset.hpp so other tools can
  // share the same defaults without including the viewer header.
  int  df_sensor_preset_idx = 0;

  // Data Display Filter tool -- scalar-driven visibility (split off from
  // Data Cleaner: semantically different, acts only on what is rendered /
  // flagged for subsequent tools, never removes points).
  bool show_data_display_filter = false;

  // Scalar visibility tool
  int   sv_field_idx = 0;             // selected scalar field index
  float sv_threshold = 0.5f;          // split threshold (A) -- in range mode derived from center - radius
  float sv_threshold_b = 1.0f;        // upper bound (B)       -- in range mode derived from center + radius
  bool  sv_range_mode = false;        // false = single-threshold (A), true = range [A, B]
  // Range-mode center/radius state. A big scalar (e.g. gps_time over a 5 km
  // session = ~300 s) makes a two-handle slider impossible to land precisely,
  // so range mode defines the window as center +/- radius instead. Center
  // comes from a picked 3D point (scalar value at the nearest cache point)
  // OR direct text entry. Slider max is user-settable so a session that
  // runs 30 s can have fine resolution while a 600 s one scales up.
  bool  sv_picking = false;           // armed: next 3D click sets the center
  float sv_range_center = 0.0f;
  float sv_range_radius = 5.0f;
  float sv_range_radius_max = 30.0f;
  // World-space position of the last successful pick -- rendered as a big
  // sphere so the user can verify which side of the track they landed on.
  bool  sv_pick_has_pos = false;
  Eigen::Vector3f sv_pick_pos = Eigen::Vector3f::Zero();
  // First-time seed flag: default the field dropdown to gps_time (the
  // scalar users almost always start with for isolation work) on first
  // entry to Scalar mode with aux_attribute_names populated.
  bool  sv_field_initialized = false;
  // Visibility history log. One entry per "Apply to visibility" click,
  // appended in order so the user can read the stack top-down ("first gps
  // time 80-120, then intensity > 50, then ..."). Cleared on Clear
  // visibility. In-memory only -- the actual stacking lives in the HD
  // frames' aux_visibility.bin files (additive AND).
  struct SvHistoryEntry {
    std::string field;
    float a = 0.0f;
    float b = 0.0f;
    bool  range_mode = false;
    int   hide_mode = 0;       // 0 = hide below A / outside, 1 = hide above A / inside
    int   points_hidden = 0;   // added by THIS apply (not cumulative)
  };
  std::vector<SvHistoryEntry> sv_history;

  // ---- Data Isolation tool ----
  // Right-click a LiDAR point to place a cylinder gizmo around it; Apply
  // sets iso_subvisibility (inherited from InteractiveViewer) so only points
  // inside the cylinder render. Non-persistent, non-destructive. Clear
  // restores the full view (AND-combined with aux_visibility.bin).
  bool  iso_show_window = false;
  bool  iso_placed = false;            // gizmo placed (waiting for edit / apply)
  Eigen::Vector3f iso_center = Eigen::Vector3f::Zero();
  float iso_radius = 5.0f;
  float iso_height = 10.0f;
  std::string iso_status;              // last Apply / Clear outcome
  bool  iso_processing = false;

  // ---- Trimmer (Tools -> Trimmer) ----
  // Screen-space polygon select over HD points; Delete destructively rewrites
  // all per-point binary files in affected frames. Three modes:
  //   Lasso    — mouse-down starts polyline, release auto-closes (first<->last)
  //   Polygon  — left-click adds vertex, right-click closes
  //   Rectangle— mouse-down/drag/release produces a 4-corner box
  // Front-only: points behind the camera never select. Respects both
  // aux_visibility.bin AND iso_subvisibility (only already-visible points
  // are candidates). Operates on HD cache built lazily on close.
  bool  show_trimmer = false;
  int   trimmer_mode = 0;                    // 0=Lasso, 1=Polygon, 2=Rectangle
  bool  trimmer_armed = false;               // next viewport action builds selection
  bool  trimmer_drawing = false;             // mid-drag (lasso/rect)
  bool  trimmer_closed = false;              // polygon finalised, selection valid
  std::vector<Eigen::Vector2f> trimmer_vertices;   // screen-space pixel polygon
  Eigen::Vector2f trimmer_rect_start = Eigen::Vector2f::Zero();
  // HD cache of candidate points (already visible: vis & iso-mask passed).
  // Built on first close after arming. Index back to source via parallel
  // arrays so Delete can group removes by (submap, frame) and rewrite each
  // HD frame's binaries once.
  std::vector<Eigen::Vector3f> trim_cache_pts;
  std::vector<int>             trim_cache_submap_idx;
  std::vector<int>             trim_cache_frame_idx;
  std::vector<int>             trim_cache_point_idx;
  bool                         trim_cache_ready = false;
  std::vector<uint8_t>         trim_selected;      // 1 = inside polygon
  int                          trim_status_selected = 0;
  bool                         trim_processing = false;
  std::string                  trim_status;

  void trimmer_build_cache();
  void trimmer_compute_selection();
  void trimmer_delete_selected();
  void trimmer_reset();
  void trimmer_undo_last();

  // In-memory single-shot undo for the most recent trim. We snapshot ONLY
  // the deleted points' bytes (per per-point file), not the full frame
  // directories -- a few thousand deleted points cost a few hundred KB
  // instead of MBs of irrelevant sibling data. On undo we append the bytes
  // back to each per-point file (re-inserting at the end is fine because
  // no downstream tool depends on per-point ordering -- LOD, MapCleaner,
  // etc. all consume the points indiscriminately). frame_meta.json's
  // num_points is bumped by n_deleted on restore. Cleared at the start of
  // every new trim, so only the LAST trim is recoverable.
  struct TrimUndoFrame {
    std::string frame_dir;
    int n_deleted = 0;
    std::map<std::string, std::vector<uint8_t>> deleted_bytes;  // per-file payloads, sized n_deleted * elem_size
  };
  std::vector<TrimUndoFrame> trim_undo_buffer;

  // ---- Batch processor (Tools -> Batch) ----
  // Snapshot-at-add JSON queue of cleaning / export tools that runs serially.
  // Recipes save to disk so the same pipeline can replay across maps from a
  // same-day capture / same rig. Tokens in path-shaped fields ($MAP, $HD,
  // $TIMESTAMP) are resolved at run time so the file is portable.
  enum class BatchKind {
    BackupHD,
    DeleteCloseToSensor,
    SOR,
    Range,
    Dynamic,
    VoxelizeHD,
    Colorize,
    VirtualCameras,
    RegenerateSDFromHD,
    ClassifyGround,
    Erasor
  };
  struct BatchEntry {
    BatchKind        kind = BatchKind::BackupHD;
    // Serialized JSON snapshot of slider/checkbox state at add-time. Stored
    // as string in the public header so glim_ros (which inherits from this
    // class) doesn't need nlohmann/json on its include path -- internal .cpp
    // parses/dumps at the boundary.
    std::string      params_json;
    std::string      note;          // user-editable description
    std::string      status;        // "pending" / "running" / "done" / "failed: <reason>"
  };
  bool                       show_batch_window = false;
  std::vector<BatchEntry>    batch_queue;
  bool                       batch_running = false;
  std::atomic_bool           batch_cancel{false};
  int                        batch_current_index = -1;
  std::string                batch_status;     // running state / step counter
  // Multi-line report from the last Validate run -- mirrors what the log
  // shows but stays visible in the window so the user doesn't have to flip
  // to the log pane. Cleared on next Validate / queue mutation.
  std::string                batch_validation_report;
  // Edit state. When >= 0, the matching tool's "Add to batch" button label
  // morphs into "Update batch values" and writes back to batch_queue[idx]
  // instead of pushing a new entry. Reset on Update / Cancel / window close.
  int                        batch_edit_index = -1;
  // Edit-popup fallback for tools that have no dedicated window (currently
  // only Backup HD: tunable is just one path string, doesn't warrant a full
  // floating window). Set by the entry's Edit button when the kind doesn't
  // map to a persistent window.
  bool                       batch_edit_popup_open = false;

  // Delete close to sensor floating window (Tools -> Utils opens it; Edit
  // on a batch entry of this kind also opens it pre-filled).
  bool   show_delete_close_window = false;
  float  delete_close_threshold   = 2.0f;   // tunable in window

  // Per-tool snapshot helpers. Each reads the current UI globals into a
  // serialized JSON string ready to feed back into run_X later. Returned as
  // string to keep nlohmann/json out of this public header.
  std::string batch_snapshot_backup_hd();
  std::string batch_snapshot_delete_close();
  std::string batch_snapshot_sor();
  std::string batch_snapshot_range();
  std::string batch_snapshot_dynamic();
  std::string batch_snapshot_voxelize_hd();
  std::string batch_snapshot_colorize();
  std::string batch_snapshot_virtual_cameras();
  std::string batch_snapshot_regenerate_sd_from_hd();
  std::string batch_snapshot_classify_ground();
  std::string batch_snapshot_erasor();
  // Prefill the Data Cleaner UI globals from a JSON snapshot. Used by the
  // batch Edit handler to seed the window with the entry's stored values
  // before letting the user tweak.
  void batch_prefill_sor(const std::string& params_json);
  void batch_prefill_range(const std::string& params_json);
  void batch_prefill_dynamic(const std::string& params_json);
  void batch_prefill_voxelize_hd(const std::string& params_json);
  void batch_prefill_colorize(const std::string& params_json);
  void batch_prefill_virtual_cameras(const std::string& params_json);
  void batch_prefill_regenerate_sd_from_hd(const std::string& params_json);
  void batch_prefill_classify_ground(const std::string& params_json);
  void batch_prefill_erasor(const std::string& params_json);

  // Per-tool runners. Take serialized JSON params, return true on success.
  // On failure populate `error_out` with a human-readable reason. All runners
  // are synchronous (caller is the batch worker thread or a std::thread spawned
  // from the tool's Apply button). Touch rf_status for live reporting but not
  // rf_processing (UI flag, owned by the caller).
  bool run_backup_hd(const std::string& params_json, std::string& error_out);
  bool run_delete_close_to_sensor(const std::string& params_json, std::string& error_out);
  bool run_sor(const std::string& params_json, std::string& error_out);
  bool run_range(const std::string& params_json, std::string& error_out);
  bool run_dynamic(const std::string& params_json, std::string& error_out);
  bool run_voxelize_hd(const std::string& params_json, std::string& error_out);
  bool run_colorize(const std::string& params_json, std::string& error_out);
  bool run_virtual_cameras(const std::string& params_json, std::string& error_out);
  bool run_regenerate_sd_from_hd(const std::string& params_json, std::string& error_out);
  bool run_classify_ground(const std::string& params_json, std::string& error_out);
  bool run_erasor(const std::string& params_json, std::string& error_out);

  // Promoted from a static-in-lambda so batch snapshot can read it.
  float regen_voxel_size = 0.20f;

  // Dispatcher / queue management.
  bool run_batch_entry(BatchEntry& entry, std::string& error_out);
  void run_batch_worker();
  void batch_add(BatchKind kind, std::string params_json, const std::string& note = "");
  void batch_save(const std::string& path);
  bool batch_load(const std::string& path);
  void batch_autosave();              // writes <map_path>/batch_process.json
  // Per-entry pre-flight check. Returns true if entry is runnable; populates
  // `error_out` with the reason if not, and `paths_in` / `paths_out` with the
  // resolved input / output file paths the run would touch (after $MAP/$HD/
  // $TIMESTAMP token expansion). Does not touch disk -- inspects params +
  // map state only. Whole-queue batch_validate() iterates and stamps each
  // entry's status with "validated" / "invalid: <reason>" plus logs the full
  // IN/OUT manifest so the user can sanity-check overnight runs.
  bool batch_validate_entry(const BatchEntry& e,
                             std::string& error_out,
                             std::vector<std::string>& paths_in,
                             std::vector<std::string>& paths_out) const;
  void batch_validate();
  // Resolve $MAP / $HD / $TIMESTAMP tokens in a path-shaped string.
  std::string batch_resolve_path(const std::string& templ) const;
  static const char* batch_kind_label(BatchKind k);
  // Hide radio (always hides one side; no "none" because the whole point of
  // this tool is to isolate a subset):
  //   single-threshold mode: 0 = hide below A, 1 = hide above A.
  //   range mode:            0 = hide outside [A,B], 1 = hide inside [A,B].
  int   sv_hide_mode = 0;
  // Cached HD points from the last Filter preview so slider / radio changes
  // can re-render instantly without another HD load. Cleared on Clear preview
  // or when the field selector changes. `sv_cache_scalars[i]` is the scalar
  // value of point i; `sv_cache_ints[i]` is the intensity (for any future
  // intensity colormap on the kept side). `sv_cache_ready` gates re-renders
  // so the first slider drag after Preview doesn't wipe a half-built cache.
  std::vector<Eigen::Vector3f> sv_cache_pts;
  std::vector<float>           sv_cache_scalars;
  std::vector<float>           sv_cache_ints;
  bool                         sv_cache_ready = false;
  // Memoised counts + scalar extents so the UI display doesn't iterate the
  // whole cache every ImGui frame. Refreshed inside sv_render_from_cache().
  int   sv_last_kept = 0;
  int   sv_last_dropped = 0;
  float sv_last_scalar_min = 0.0f;
  float sv_last_scalar_max = 0.0f;
  // When false the "dropped" side is truly hidden (Apply mode). When true
  // both sides render. The style of the dropped overlay is chosen by
  // sv_dropped_style -- diagnostic red, faint gray context, or intensity gray.
  bool  sv_show_dropped = true;
  // 0 = Red translucent (diagnostic, default for Filter preview).
  // 1 = Gray flat        (faint context for "is this junk?" assessment).
  // 2 = Intensity gray   (gray modulated by intensity for more texture).
  int   sv_dropped_style = 0;
  // When false, sv_render_from_cache skips the kept overlay (LOD already
  // renders those points natively after Apply). Reset to true on Clear preview.
  // Lets the dropped ghost persist past Apply for context.
  bool  sv_render_kept = true;
  void sv_render_from_cache();
  // Trimmer/segmentation gate -- when on, trimmer_build_cache excludes points
  // that the active scalar preview would drop. Lets the user lasso the kept
  // subset without first committing Apply to visibility.
  bool  trim_respect_preview_split = false;
  int   rf_criteria = 0;              // 0=Range, 1=GPS Time
  int   rf_gps_keep = 0;              // 0=Dominant (most points), 1=Newest, 2=Oldest
  float rf_voxel_size = 1.0f;        // metres (range mode default)
  float rf_voxel_height_mult = 1.0f;  // z-extent multiplier on rf_voxel_size (e.g. 2.0 = voxel is 2x taller in Z)
  float rf_safe_range = 20.0f;      // metres — points within this always kept (range mode)
  float rf_range_delta = 10.0f;     // metres — remove if >delta further than closest in voxel
  float rf_far_delta = 30.0f;       // metres — in voxels with no safe-range points, remove if > min_range + far_delta
  int   rf_min_close_pts = 3;       // min close points to trigger removal of distant ones
  float rf_range_highlight = 0.0f; // range threshold for red tinting (0=off)
  bool  rf_ground_only = false;   // range filter affects only ground-classified points (requires aux_ground.bin)
  bool rf_preview_active = false;  // preview overlay is showing — hide other LOD data
  bool rf_intensity_mode = false;  // toggle intensity display on preview

  // Dynamic filter params
  float df_voxel_size = 0.28f;         // voxel size for dynamic mode (separate from rf_voxel_size)
  float df_range_threshold = 0.3f;
  float df_observation_range = 30.0f;
  int   df_min_observations = 15;
  bool  df_exclude_ground_pw = true;
  bool  show_pw_config = false;
  bool  show_trail_config = false;
  bool  pw_accumulate = false;         // accumulate neighboring frames for PatchWork++
  int   pw_accumulate_count = 10;      // number of prior frames to include (or next frames at start)
  bool  pw_reuse_scalar = false;        // reuse aux_ground.bin if it exists instead of recomputing
  std::unordered_map<std::string, std::vector<bool>> pw_ground_cache;  // frame_dir → cached ground flags
  float df_chunk_size = 120.0f;        // chunk size for dynamic filter (larger = more trail context)
  float df_chunk_spacing = 60.0f;      // chunk spacing for dynamic filter
  bool  df_refine_ground = true;       // refine ground labels using Z + intensity
  bool  df_refine_trails = true;       // cluster candidates into trails, reject noise
  float df_trail_min_length = 7.0f;
  float df_trail_min_aspect = 5.0f;
  float df_trail_min_density = 11.0f;
  float df_refine_voxel = 0.28f;
  // Gap-fill: when on, Process Chunk preview also flips KEPT points inside
  // trail voxels to removed (above trail min-Z). Default OFF to match Apply
  // path behaviour -- with it on, Preview removes more than Apply will.
  bool  df_trail_gap_fill = false;

  // Dynamic mode robustness extensions (passed through to MapCleanerFilter).
  // Defaults preserve original behaviour. See map_cleaner.hpp for semantics.
  int   df_vote_margin = 0;
  int   df_min_static_votes = 1;
  float df_min_baseline_m = 0.0f;
  int   df_min_dynamic_cluster_size = 0;
  float df_dynamic_cluster_voxel = 0.3f;
  // Final-barrier Z-column ground safety. When > 0 AND df_exclude_ground_pw is
  // on, the write-time safety check also scans each frame's local min-Z per
  // 1 m XY column and protects any point within this tolerance from deletion.
  // Independent of PatchWork -- catches ground that PatchWork's single-frame
  // pass missed. Default 0.3 m. Set to 0 to disable.
  float df_safety_z_tol_m = 0.3f;

  // Cleaner mode: 0 = Voting (default, single-pass), 1 = Remove-Revert
  // (Removert-style two-pass, AND-merge for higher precision).
  int   df_cleaner_mode = 0;
  float df_coarse_thresh_mult = 2.0f;  // RemoveRevert coarse range_threshold mult
  float df_coarse_res_mult    = 2.0f;  // RemoveRevert coarse res_h/res_v mult

  // Dynamic-Erasor (Data Cleaner mode 3) -- polar pseudo-occupancy dynamic
  // removal. Defaults below match the Livox Horizon sensor preset; selecting
  // a different preset overwrites these in-place.
  int   df_erasor_num_rings = 30;
  int   df_erasor_num_sectors = 108;
  float df_erasor_max_range = 30.0f;
  float df_erasor_min_range = 1.0f;
  float df_erasor_ratio_threshold = 0.01f;
  float df_erasor_v_fov_half_deg = 12.5f;  // Livox Horizon default

  // SOR filter params
  float sor_radius = 0.3f;             // search radius (metres)
  int   sor_min_neighbors = 5;         // minimum neighbors within radius to keep
  float sor_chunk_size = 100.0f;       // spatial chunk size (metres, axis-aligned cube)

  // Livox-specific intensity-0 filter
  bool  show_livox_tool = false;
  int   livox_mode = 0;                // 0=Delete, 1=Mark as 2nd return, 2=Interpolate
  float livox_interp_radius_m = 0.3f;  // kNN radius for Mode 2
  bool  livox_running = false;
  std::string livox_status;
  bool  livox_cancel_requested = false;
  bool  livox_intensity_mode = false;  // toggle intensity colormap on preview
  struct LivoxPreviewPoint {
    Eigen::Vector3f pos;
    float intensity;
    bool  was_zero;
  };
  std::vector<LivoxPreviewPoint> livox_preview_data;  // cached kept preview points for intensity toggle
  std::vector<std::string> livox_preview_frame_dirs;  // frames touched by current preview (used by "Apply filter")

  // (Batch processor state lives in the upper section -- see "Batch processor
  // (Tools -> Batch)" earlier in this header. The previous live-defaults
  // skeleton was replaced by the snapshot-at-add framework.)

  // Voxelize HD tool
  bool  show_voxelize_tool = false;
  float vox_size = 0.03f;              // voxel size in metres
  int   vox_mode = 2;                  // 0=center, 1=weighted, 2=XY center + Z weighted
  bool  vox_use_center = true;         // legacy, derived from vox_mode
  float vox_chunk_size = 60.0f;        // chunk size for processing
  float vox_chunk_spacing = 30.0f;     // chunk spacing
  bool  vox_processing = false;
  std::string vox_status;
  bool  vox_ground_only = false;       // ground-only mode: 1 point per XY cell (requires aux_ground.bin)
  bool  vox_include_intensity = true;  // load + write intensities.bin per voxel
  bool  vox_include_rgb = true;        // load + write aux_rgb.bin per voxel (when source aux_rgb.bin exists)
  // Per-voxel normals: average the unit normals of contributing points and
  // re-normalize. Same aggregation pattern as RGB/intensity. When disabled,
  // no normals.bin is written. When enabled, falls through to zero-vector
  // for voxels whose source frames lacked normals.bin.
  bool  vox_include_normals = true;
  bool  lod_use_voxelized = false;     // LOD checkbox: load from hd_frames_voxelized/

  // Cached preview data (kept in CPU memory for range highlight re-coloring)
  struct PreviewPoint {
    Eigen::Vector3f pos;
    float range;
    float intensity;
    float normal_z;
    bool ground_pw;
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

  // Colorize tool
  std::vector<ImageSource> image_sources;
  bool show_colorize_window = false;
  int  colorize_source_idx = 0;       // selected image source

  // Runtime-only colorize state (per-source tuning lives in ImageSource::params,
  // not here -- see ColorizeParams in lidar_colorizer.hpp).
  cv::Mat colorize_mask;                // runtime cache for active source's mask image
  int   colorize_cam_radius = 3;        // cameras before/after submap to include (TODO: move to params)
  ColorizeResult colorize_last_result;  // cached for intensity blend adjustment
  bool  colorize_live_preview = false;  // UI toggle, shared across sources
  // Cap for the spherical cube-face cache (GB of RAM). UI binds a float slider
  // here and pushes the byte value into the static g_cube_face_cache_cap_bytes
  // (in offline_viewer.cpp). 0 = uncapped. Default 8 GB.
  double preview_cache_cap_gb = 8.0;

  // -- Camera Time Matcher ----------------------------------------------------
  // Side-by-side visual matching between a time-stamped source (left) and a
  // dumb-frames source without reliable timestamps (right, e.g. Osmo 360 video
  // frames). User scrubs the right source until it matches the left, clicks
  // Anchor, then Apply back-fills timestamps for all right-side frames using a
  // user-entered FPS or a solved rate from two anchors.
  bool  show_time_matcher = false;
  int   tm_left_src = 0;           // left source index in image_sources
  int   tm_right_src = 0;          // right source index
  int   tm_left_idx = 0;           // current frame index left
  int   tm_right_idx = 0;          // current frame index right
  unsigned int tm_left_tex = 0;    // GL texture (left preview)
  unsigned int tm_right_tex = 0;   // GL texture (right preview)
  int   tm_left_tex_w = 0, tm_left_tex_h = 0;
  int   tm_right_tex_w = 0, tm_right_tex_h = 0;
  std::string tm_left_loaded_path;
  std::string tm_right_loaded_path;
  float tm_right_fps = 30.0f;       // user-entered FPS for dumb source
  int   tm_anchor1_right_idx = -1;  // right-frame index of anchor 1
  double tm_anchor1_left_time = 0.0;
  int   tm_anchor2_right_idx = -1;  // -1 until second anchor is set
  double tm_anchor2_left_time = 0.0;
  bool  tm_two_anchor_mode = false;  // if true, solve rate from the two anchors; else use tm_right_fps
  float tm_left_scale = -1.0f;       // <=0 sentinel = "fit to viewport on first frame"
  float tm_right_scale = -1.0f;
  bool  tm_left_auto_fit = true;     // while true, scale tracks viewport size (window resize refits)
  bool  tm_right_auto_fit = true;

  // -- Virtual LiDAR cameras tool -------------------------------------------
  // Renders locked-pose, zero-distortion cube-face images from the LiDAR data
  // along the trajectory. Imported into Metashape as locked anchors, real
  // cameras BA-refine against them. See project_virtual_camera_anchors memory.
  bool  show_virtual_cameras_window = false;
  std::string vc_output_dir;
  float vc_interval_m = 10.0f;
  bool  vc_face_enabled[6] = { true, true, true, true, false, true };  // skip +Z (sky) by default
  int   vc_face_size = 1920;
  // VC export filter: when on AND a COLMAP export volume has been placed,
  // skip frames whose camera position falls outside the volume's XY footprint.
  // Default ON so a placed volume automatically scopes the test export; the
  // user can disable to export the full source even with a volume placed.
  bool  vc_pcam_restrict_to_colmap_volume = true;
  float vc_context_radius_m = 60.0f;
  bool  vc_ground_only = false;
  bool  vc_render_rgb = false;
  bool  vc_embed_exif_gps = true;
  // Trajectory-mode camera type: 0 = Pinhole (single image per anchor at
  // vc_traj_pinhole_w x vc_traj_pinhole_h), 1 = 360° Cubemap (6-face split,
  // uses vc_face_enabled + vc_face_size). Independent of any source's
  // camera_type since trajectory mode has no source.
  int   vc_traj_camera_type = 1;          // default cubemap
  int   vc_traj_pinhole_w   = 1920;
  int   vc_traj_pinhole_h   = 1080;
  double vc_traj_pinhole_hfov_deg = 90.0;  // horizontal FoV for pinhole render
  // Trajectory-mode preview navigation: current anchor index + dirty flag.
  // Slider/arrows walk through anchors; the thumbnail re-renders when dirty.
  int  vc_traj_preview_idx   = 0;
  bool vc_traj_preview_dirty = true;
  cv::Mat vc_traj_preview_image;          // last rendered thumbnail
  unsigned int vc_traj_preview_tex = 0;   // GL texture id for thumbnail
  // State
  bool  vc_running = false;
  std::string vc_status;
  size_t vc_anchors_placed_last = 0;
  size_t vc_faces_rendered_last = 0;
  std::vector<Eigen::Vector3f> vc_preview_anchors;    // world positions of placed anchors, for 3D preview
  std::vector<Eigen::Matrix3f> vc_preview_orient;     // world-frame rotation per anchor
  bool  colorize_intrinsics_dirty = false;  // set by intrinsic input fields; consumed top-of-frame
  float intrinsics_dist_step = 0.0005f;     // +/- button step for k1/k2/k3/p1/p2 (user-tunable)
  float colorize_time_step = 0.02f;  // step for +/- buttons (seconds)
  float colorize_lever_step = 0.01f; // step for lever arm +/- (metres)
  float colorize_rot_step = 0.1f;    // step for rotation +/- (degrees)
  int   colorize_last_submap = -1;   // last colorized submap ID (-1 = none)
  int   colorize_last_cam_src = -1;  // last colorized camera source
  int   colorize_last_cam_idx = -1;  // last colorized camera frame index
  std::vector<int> colorize_preview_sm_ids;  // per-submap drawable IDs created by Colorize-all preview (for cleanup)
  bool  colorize_all_cancel_requested = false;  // Stop button for full-map preview worker

  // Apply-to-HD method selector and chunk-based params
  int   apply_method = 0;                // 0 = Per-submap (legacy), 1 = Chunk-based
  float apply_chunk_size_m = 10.0f;      // core chunk size (m), moves along trajectory at this spacing
  float apply_chunk_margin_m = 10.0f;    // edge overlap -- frames/cameras within core + this still load
  bool  apply_cancel_requested = false;

  // In-app image viewer
  bool show_image_viewer = false;
  std::string image_viewer_title;
  unsigned int image_viewer_texture = 0;
  int image_viewer_w = 0, image_viewer_h = 0;  // displayed resolution
  int image_original_w = 0, image_original_h = 0;  // original resolution (for intrinsics mapping)

  // Calibration tool
  bool calib_active = false;           // calibration mode active
  int  calib_cam_src = -1;             // source index of calibration camera
  int  calib_cam_idx = -1;             // frame index of calibration camera
  bool calib_waiting_3d = true;        // true=waiting for 3D click, false=waiting for 2D click
  struct CalibPair {
    Eigen::Vector3d pt_3d;
    Eigen::Vector2d pt_2d;
  };
  std::vector<CalibPair> calib_pairs;
  float calib_sphere_size = 0.08f;     // green sphere radius for 3D reference points
  std::string calib_status;

  // Alignment check window: image + projected LiDAR points overlay
  bool align_show = false;
  int  align_cam_src = 0;
  int  align_cam_idx = 0;
  unsigned int align_texture = 0;
  int  align_tex_w = 0, align_tex_h = 0;   // texture size (may be downscaled)
  int  align_img_w = 0, align_img_h = 0;   // original image size (for pixel math)
  std::string align_loaded_path;           // currently loaded image path
  float align_display_scale = 0.4f;        // display / native pixel ratio
  Eigen::Vector2f align_pan = Eigen::Vector2f::Zero();
  float align_point_size = 2.0f;
  int   align_point_color_mode = 0;        // 0=intensity, 1=range, 2=depth
  float align_max_range = 50.0f;
  float align_min_range = 0.5f;
  float align_bright_threshold = 0.0f;     // 0=show all, >0=intensity cutoff (0-1)
  float align_point_alpha = 0.85f;         // overlay dot transparency
  bool  align_rectified = false;           // true = undistort image & skip distortion math (pure extrinsic check)
  bool  align_rect_applied = false;        // track which state the current texture was loaded in
  bool  align_live_preview = false;        // re-undistort image on intrinsic change; cheaper than colorize live preview
  PinholeIntrinsics align_last_intrinsics;  // snapshot of intrinsics used for the currently loaded rectified image
  int   align_colormap_sel = 0;            // 0 = Turbo (default); index into glk::colormap_names()
  Eigen::Vector2f align_intensity_range = Eigen::Vector2f(0.0f, 255.0f);  // 5%/95% percentile, cached per submap
  double align_frame_interval_s = 0.1;     // avg LiDAR frame interval in current submap (s); cached on submap load
  bool  align_image_grayscale = false;     // render background image as grayscale for contrast
  bool  align_image_hidden = false;        // fully hide background to see only the LiDAR overlay
  int   align_nearest_frames = 0;          // 0 = whole submap; >0 = +/- N LiDAR frames around cam time
  bool  align_grid_show = false;           // reference H/V line grid for eyeballing lens distortion
  int   align_grid_lines = 10;             // number of horizontal = vertical interior lines
  // User-placed reference lines (H or V in ideal pinhole coords). Each line is
  // stored as (type, ideal pixel coord). type: 0 = vertical (fixed u), 1 = horizontal (fixed v).
  std::vector<std::pair<int, double>> align_user_lines;
  int   align_add_line_mode = 0;           // 0 = none, 1 = arm vertical, 2 = arm horizontal
  bool  align_colorize_hide_uncolored = false;  // when colorize cache is active, skip sentinel points
  std::vector<Eigen::Vector3f> align_colorize_rgb;  // per-point RGB from "Colorize from this camera"
  bool  align_colorize_valid = false;      // true while the colorize cache is usable
  bool  align_colorize_dirty = false;      // request to (re)compute the cache this frame
  bool  align_colorize_auto = false;       // toggle: recompute RGB automatically when switching frames
  int   align_colorize_cam_src = -1;       // cache-for which source
  int   align_colorize_cam_idx = -1;       // cache-for which image index
  Eigen::Vector3d align_last_rpy = Eigen::Vector3d::Zero();    // extrinsic snapshot for live-refresh detection
  Eigen::Vector3d align_last_lever = Eigen::Vector3d::Zero();
  double align_last_time_shift = 0.0;
  int   align_last_submap_id = -1;
  std::vector<Eigen::Vector3f> align_submap_world_pts;
  std::vector<float>           align_submap_ints;
  std::vector<Eigen::Vector3f> align_submap_world_normals;  // parallel to align_submap_world_pts; empty if unavailable
  std::vector<double>          align_submap_world_times;    // parallel gps_time per point; empty if unavailable

  // COLMAP export (single-chunk 2D top-view trimming)
  bool  ce_show = false;                       // window visible
  bool  ce_placing = false;                    // next 3D click places the region
  Eigen::Vector3f ce_center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
  Eigen::Vector3f ce_size = Eigen::Vector3f(50.0f, 50.0f, 50.0f);
  bool  ce_region_placed = false;              // has the user placed a region yet?
  std::string ce_output_dir;                   // last chosen output dir
  bool  ce_copy_images = true;                 // false = symlink
  // Source selection for the points3D.ply cloud. Independent toggles: export
  // from voxelized HD, raw HD, or both concatenated. Both trimmed to the
  // region of interest so the output size stays bounded regardless of source.
  // Default: both ON (user picks whichever looks best downstream; dupes are
  // tolerated for the testing phase).
  bool  ce_export_voxelized = false;  // default OFF: HD meshes better; flip on for 3DGS init
  bool  ce_export_hd        = true;
  // Per-cloud field toggles. Default ON for everything; user can drop columns
  // to keep PLY size down or because a downstream tool doesn't read them.
  // Voxelized normals usually missing on disk (Voxelize panel doesn't yet
  // emit normals.bin) -- if ON the writer ships zero-vectors, harmless.
  bool  ce_export_hd_color      = true;
  bool  ce_export_hd_intensity  = true;
  bool  ce_export_hd_normal     = true;
  bool  ce_export_vox_color     = true;
  bool  ce_export_vox_intensity = true;
  bool  ce_export_vox_normal    = true;
  float ce_overlap_margin_m = 3.0f;
  // Default OFF since Reality Scan (the default target) handles Z-up world
  // natively. target_changed flips it on for COLMAP / BlocksExchange.
  bool  ce_rotate_to_y_up = false;             // export with 3DGS-style Y-up world
  float ce_yaw_deg = 0.0f;                      // world-XY yaw of the export region (deg)
  // 3D gizmo for the export region. TX | TY | RZ -- drag handles in the
  // viewport for translating the cube in XY and rotating its yaw, both
  // wired bidirectionally with the DragFloat widgets in the SFM panel.
  // Lazily constructed on first use so we don't pull in ImGuizmo at
  // ctor time (matches source_finder_gizmo's pattern in interactive_viewer).
  std::unique_ptr<guik::ModelControl> ce_region_gizmo;
  bool  ce_undistort_images = true;             // undistort images (PINHOLE) vs raw (OPENCV)
  // SFM target = single 4-way radio in the SFM Export panel. Each value
  // is a complete export specification (folder structure + mask naming +
  // bundle files). emit_colmap and export_blocks_exchange are NOT
  // separate UI toggles -- they're derived from this enum at worker-
  // launch time. To get a different format, re-run the export.
  //   0 = COLMAP / 3DGS         (cameras.txt + images.txt + sparse/0/...)
  //   1 = Reality Scan          (.geometry/ + .mask/ + point_cloud/)
  //   2 = Metashape             (images/ + masks/<stem>_mask.png + point_cloud/)
  //   3 = BlocksExchange        (blocks_exchange.xml + images/ + masks/ + point_cloud/)
  // Default: Reality Scan -- the most-promising path for our LiDAR-anchored
  // SFM workflow per current testing.
  int   ce_target_layout         = 1;
  // Per-image metadata channels (orthogonal to target_layout). UI Metadata
  // section toggles these directly; target_changed sets sensible defaults
  // (XMP on for RS, EXIF on for Metashape; both off for COLMAP / BE) but
  // user can override for testing (e.g. RS layout + EXIF on to see what RS
  // does when both XMP and EXIF are present).
  bool  ce_emit_exif_gps         = false;
  // Default ON (RS is the default target -- XMP is its primary pose-prior
  // channel). target_changed handler keeps this in sync when the user
  // switches targets.
  bool  ce_emit_xmp_sidecar      = true;
  // Full UTM mode -- when on (and session has a GNSS datum loaded),
  // every exported world coord (PLY points, camera positions in
  // images.txt / BE / XMP) gets the datum's UTM origin added in.
  // Result: real-world UTM eastings/northings, matching EXIF GPS
  // lat/lon. Required for Metashape / RealityScan georef workflows
  // where camera GPS must match the LiDAR cloud's frame. Default:
  // ON for everything except COLMAP/3DGS (which prefers small
  // numbers near origin for float precision).
  // Default ON since RS / Metashape / BE want real-world coords; flipped
  // off by target_changed for COLMAP/3DGS.
  bool  ce_full_utm              = true;
  // Single shared mask per source instead of per-image. Sensible default
  // for Metashape / BE workflows where masks are static (rig hardware,
  // no per-frame YOLO segmentation). Off otherwise -- per-image masks are
  // the right semantic for RS XMP and any future learned-mask flows.
  bool  ce_single_mask           = false;
  // Flight log CSV alongside the export. Reality Scan imports this via
  // Workflow > Trajectory and uses it as a pose-prior source -- a fallback
  // when XMP sidecars are not auto-detected. Format: `# name lon lat alt`
  // header + one row per kept camera. Skipped silently when no GNSS datum.
  // Default ON for RS target (set by target_changed); off otherwise.
  bool  ce_emit_flight_log       = true;
  // Header-only points3D.txt for the RS-via-COLMAP SLAM workflow. RS's
  // tutorial says to wipe the points from points3D.txt before opening the
  // COLMAP scene; an unwiped file surfaces as "file not found" in RS. We
  // write it header-only when this is on. Default OFF (normal COLMAP /
  // 3DGS exports want the points for sparse init).
  bool  ce_points3d_header_only  = false;
  bool  ce_use_pose_priors = true;              // master toggle for per-photo accuracy hints
  // 1-sigma pose uncertainties exported to RS (flight log accuracy columns) and
  // Metashape BE (<Accuracy> tags). Loose by default (1m / 5deg) -- handheld
  // SLAM has cm-to-dm drift over a session, but BA needs room to refine
  // against feature matches; tight priors freeze cameras and prevent re-
  // alignment when the LiDAR-prior and image-feature solutions disagree.
  float ce_pose_pos_sigma_m = 1.0f;             // position sigma (m) for BA prior
  float ce_pose_rot_sigma_deg = 5.0f;           // rotation sigma (deg) for BA prior
  bool  ce_running = false;
  std::string ce_status;
  // Last export summary
  size_t ce_last_points = 0;
  size_t ce_last_cameras = 0;
  size_t ce_last_images = 0;
  size_t ce_last_masks = 0;

  // Auto-calibration (LightGlue-assisted)
  bool  ac_show = false;                 // show auto-calibrate window
  int   ac_cam_src = 0;
  int   ac_cam_idx = 0;                  // anchor camera
  int   ac_n_frames_before = 15;
  int   ac_n_frames_after = 15;
  bool  ac_use_time_window = false;
  float ac_time_before_s = 3.0f;
  float ac_time_after_s = 3.0f;
  bool  ac_directional_filter = true;
  float ac_directional_threshold_deg = 60.0f;
  float ac_min_range = 0.5f;
  float ac_max_range = 80.0f;
  int   ac_render_width = 0;             // 0 = auto-populate from source intrinsics (native)
  int   ac_render_height = 0;
  bool  ac_optimize_intrinsics = false;
  bool  ac_lock_extrinsic_for_intr = false;
  std::string ac_python_script_path;     // resolved to scripts/lightglue_match.py
  std::string ac_python_interpreter = "python3";  // override with e.g. /path/to/venv/bin/python
  std::string ac_work_dir;               // tempdir for PNG/JSON exchange
  std::string ac_status;                 // latest status message
  bool  ac_running = false;
  // Latest run stats
  int   ac_last_matches = 0;
  int   ac_last_inliers = 0;
  double ac_residual_before = 0.0;
  double ac_residual_after = 0.0;
  // Save a copy of the pre-run extrinsic + intrinsics to support "Revert"
  bool  ac_have_backup = false;
  Eigen::Vector3d ac_backup_lever;
  Eigen::Vector3d ac_backup_rpy;
  PinholeIntrinsics ac_backup_intrinsics;
  // Proposed values from latest run — NOT written to src until user clicks Apply
  bool  ac_has_proposed = false;
  Eigen::Vector3d ac_proposed_lever = Eigen::Vector3d::Zero();
  Eigen::Vector3d ac_proposed_rpy = Eigen::Vector3d::Zero();
  PinholeIntrinsics ac_proposed_intrinsics;
  bool  ac_proposed_has_intrinsics = false;  // true if run also refined intrinsics

  // Time-shift sweep mode
  bool  ac_sweep_on = false;
  float ac_sweep_neg_range_s = 0.05f;  // sweep from (current_time_shift - neg_range)
  float ac_sweep_pos_range_s = 0.05f;  // ...to (current_time_shift + pos_range)
  float ac_sweep_step_s = 0.01f;
  int   ac_sweep_progress = 0;
  int   ac_sweep_total = 0;
  bool  ac_cancel_requested = false;  // set by UI Stop button, consumed by worker
  struct AcSweepResult {
    float time_shift;
    int matches;
    int inliers;
    float residual;
    Eigen::Vector3d lever;
    Eigen::Vector3d rpy;
    PinholeIntrinsics intrinsics;
    bool  has_intrinsics;
    bool  success;
    std::string reject_reason;
  };
  std::vector<AcSweepResult> ac_sweep_results;
  // LightGlue tuning (exposed to UI)
  int   ac_max_kp = 2048;
  float ac_min_score = 0.2f;
  // Match visualization
  unsigned int ac_real_tex = 0;
  unsigned int ac_rend_tex = 0;
  int   ac_real_tex_w = 0, ac_real_tex_h = 0;
  int   ac_rend_tex_w = 0, ac_rend_tex_h = 0;
  // Each match: (real_uv in render-space, rendered_uv in render-space, score)
  std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> ac_match_pairs;
  std::vector<float> ac_match_scores;
  int   ac_match_render_w = 0;  // the render resolution used for these matches
  int   ac_match_render_h = 0;
  bool  ac_show_match_viz = true;
  bool  ac_match_viz_needs_reload = false;  // set by worker thread; UI thread consumes + deletes/recreates GL textures
  // Sanity-check thresholds — reject PnP result if any exceeded (result implausible)
  float ac_max_residual_px = 20.0f;
  float ac_max_lever_shift_m = 1.0f;        // max delta from pre-run lever
  float ac_max_rotation_shift_deg = 15.0f;  // max delta from pre-run RPY

  // Range / dataset-wide auto-calibration. Walks a frame range, runs the
  // single-frame pipeline per frame, drops frames whose top-quality match
  // count falls below ac_range_min_high_matches, and computes a weighted
  // average of the surviving extrinsic proposals (Markley quaternion mean
  // for rotation; weighted vector mean for lever arm). The result is
  // written to ac_proposed_lever / ac_proposed_rpy + ac_has_proposed=true
  // so the existing Apply button commits it -- no auto-apply.
  bool  ac_range_use_all = true;             // ignore start/end, walk every located frame
  int   ac_range_start = 0;                  // located-frame index (NOT folder index)
  int   ac_range_end = -1;
  int   ac_range_min_high_matches = 20;      // drop frames with fewer top-quality matches
  bool  ac_range_running = false;            // worker active
  bool  ac_range_cancel = false;             // user clicked Stop
  int   ac_range_progress = 0;
  int   ac_range_total = 0;
  int   ac_range_accepted = 0;               // contributed to average
  int   ac_range_skipped = 0;                // dropped (too few high matches, sanity gate, render fail, etc)
  std::string ac_range_status;
  // Winner-mask / Weight viz cache (filled by "Compute assignment" button)
  // Anchor selection (UI-driven). When a row in the anchor table is clicked,
  // align_anchor_selected stores that index and align_anchor_selected_src the
  // source it belongs to. The 3D gizmo renderer scales the selected anchor's
  // cone x10 on Z so it's easy to locate in the viewer.
  int   align_anchor_selected = -1;
  int   align_anchor_selected_src = -1;
  int   align_winner_sm = -1;                    // submap id the cache was built for (-1 = stale)
  int   align_winner_src = -1;                   // image source the cache was built for
  std::vector<int>   align_winner_frame_idx;     // per point: winning frame index into src.frames; -1 if none
  std::vector<float> align_winner_weight_vec;    // per point: winning weight value
  float align_weight_max_cached = 0.0f;          // max weight across cached points (for colormap normalization)

  // ----------------- Virtual Camera export (per-RGB-camera mode) -----------------
  // Second placement mode for the existing Virtual Cameras window. Instead of
  // walking the trajectory and dropping waypoints, this mode renders a virtual
  // LiDAR-intensity photo at the estimated world pose of every real RGB camera
  // frame. The result is 1:1 co-located with the real images (within a few cm,
  // from the Colorize extrinsic), giving Metashape/RealityCapture locked
  // anchors to register real cameras into the LiDAR frame during SFM/BA.
  // Shares context builder + intensity rasterizer with Auto-calibrate.
  int  vc_placement_mode = 1;                     // 0 = Waypoints, 1 = Per RGB camera
  std::vector<bool> vc_pcam_source_enabled;       // parallel to image_sources[]
  int  vc_pcam_active_src = 0;                    // source index previewed
  int  vc_pcam_preview_frame = 0;                 // frame index inside active source
  CalibContextOptions    vc_pcam_ctx_opts;        // context window tuning
  IntensityRenderOptions vc_pcam_render_opts;     // splat + intensity + colormap knobs
  int  vc_pcam_render_w = 0;                      // 0 = use source intrinsics W
  int  vc_pcam_render_h = 0;                      // 0 = use source intrinsics H
  int  vc_pcam_format = 1;                        // 0=PNG, 1=JPG
  int  vc_pcam_jpg_quality = 90;
  struct VcamPreviewTex { unsigned int tex = 0; int w = 0; int h = 0; std::string label; };
  std::vector<VcamPreviewTex> vc_pcam_preview_textures;
  bool vc_pcam_preview_dirty = false;             // set by Preview, consumed by UI
  // Click-to-enlarge popup state: a thumbnail click opens a floating window
  // showing the image at up to ~2048 px. -1 = closed.
  int  vc_pcam_focused_tex = -1;
  // Percentiles of the last preview context's intensities. Feed the "Lock
  // intensity range" button so a single click captures the synthetic-exposure
  // baseline from whatever's currently on screen.
  bool  vc_pcam_have_last_percentiles = false;
  float vc_pcam_last_imin = 0.0f;
  float vc_pcam_last_ibulk = 230.0f;
  float vc_pcam_last_imax = 250.0f;

  // Per-scanner presets for the Virtual Camera / Per-RGB-camera mode. Bundles
  // the context + render + faces + output settings that are known to work well
  // on a particular LiDAR. Factory entries are hardcoded; save/overwrite/rename
  // is a follow-up pass.
  struct VcamPreset {
    std::string name;
    CalibContextOptions    ctx_opts;
    IntensityRenderOptions render_opts;
    int render_w = 0;
    int render_h = 0;
    bool face_enabled[6] = { true, true, true, true, false, false };
    int face_size = 1024;
    int format = 1;
    int jpg_quality = 90;
  };
  std::vector<VcamPreset> vc_pcam_presets;
  int vc_pcam_preset_idx = 0;
  bool vc_pcam_presets_initialised = false;  // lazy factory-seed guard
  void vc_pcam_seed_factory_presets();
  void vc_pcam_apply_preset(const VcamPreset& p);

  // Match-tester state. Reuses the auto-calibrate LightGlue pipeline to score
  // how well the current rasterization settings produce matchable images --
  // run against the active source+frame's real image and the just-rendered
  // virtual preview. One entry per face for Spherical sources, single entry
  // for Pinhole. `vc_pcam_match_log` is a human-readable message for the UI.
  float vc_pcam_lg_min_score = 0.3f;
  int   vc_pcam_lg_max_kp    = 2048;
  struct VcamMatchResult {
    std::string label;
    MatchQualityStats stats;
    // Side-by-side visualisation data (populated by the match runner; consumed
    // by the match-viz window). Match UVs are in render-space (both images get
    // resized to the render's W/H before being sent to LightGlue).
    std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> match_pairs;
    std::vector<float> match_scores;
    unsigned int real_tex = 0;
    unsigned int rend_tex = 0;
    int real_w = 0, real_h = 0;
    int rend_w = 0, rend_h = 0;
  };
  std::vector<VcamMatchResult> vc_pcam_match_results;
  std::string vc_pcam_match_log;
  bool vc_pcam_match_viz_show = false;      // side-by-side window open
  int  vc_pcam_match_viz_idx  = 0;          // which face's matches are being viewed
  // Batch-export progress (thread-safe cross-thread state).
  std::atomic<bool> vc_pcam_cancel{false};
  std::atomic<int>  vc_pcam_progress_cur{0};
  std::atomic<int>  vc_pcam_progress_total{0};

  // Helper: cube-face label (word form) for filenames / UI labels.
  static const char* vc_face_label(int face_idx);
};

}  // namespace glim
