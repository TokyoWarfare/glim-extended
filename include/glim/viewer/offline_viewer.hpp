#pragma once

#include <string>
#include <vector>
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

  // Data filter tool (range + dynamic modes)
  bool show_data_filter = false;
  int  df_mode = 0;                  // 0=SOR, 1=Dynamic, 2=Range, 3=Scalar

  // Scalar visibility tool
  int   sv_field_idx = 0;             // selected scalar field index
  float sv_threshold = 0.5f;          // split threshold
  bool  sv_hide_below = false;        // hide points below threshold
  bool  sv_hide_above = false;        // hide points above threshold
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

  // Batch processor — queue of tool-apply tasks executed sequentially with current UI defaults
  bool show_batch_window = false;
  int  batch_selected_tool = 0;
  bool batch_running = false;
  bool batch_cancel_requested = false;
  int  batch_current_task = 0;
  std::string batch_status;
  enum class BatchTool {
    SOR,
    Range,
    Dynamic,
    Scalar,
    Voxelize,
    LivoxDelete0,
    LivoxMark2ndReturn,
    LivoxInterpolate,
  };
  std::vector<BatchTool> batch_queue;

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
  float vc_context_radius_m = 60.0f;
  bool  vc_ground_only = false;
  bool  vc_render_rgb = false;
  bool  vc_embed_exif_gps = true;
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
  bool  ce_voxelized_only = true;              // HARD requirement: voxelized data only
  float ce_overlap_margin_m = 3.0f;
  bool  ce_rotate_to_y_up = true;              // export with 3DGS-style Y-up world
  float ce_yaw_deg = 0.0f;                      // world-XY yaw of the export region (deg)
  bool  ce_undistort_images = true;             // undistort images (PINHOLE) vs raw (OPENCV)
  bool  ce_write_bundler = false;               // also emit bundle.out (Metashape-importable)
  bool  ce_write_blocks_exchange = false;       // also emit blocks_exchange.xml (CC/Metashape/RC)
  bool  ce_use_pose_priors = true;              // master toggle for per-photo accuracy hints
  float ce_pose_pos_sigma_m = 0.05f;            // position sigma (m) for BA prior
  float ce_pose_rot_sigma_deg = 2.0f;           // rotation sigma (deg) for BA prior
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
