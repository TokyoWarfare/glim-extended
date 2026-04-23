#include <glim/viewer/offline_viewer.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <queue>
#include <regex>
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
#include <glim/util/auto_calibrate.hpp>
#include <glim/util/colmap_export.hpp>
#include <glim/util/image_viewport.hpp>
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
#include <glk/colormap.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <GL/gl3w.h>
#include <glk/thin_lines.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <guik/camera/fps_camera_control.hpp>
#include <guik/camera/basic_projection_control.hpp>
#include <glk/io/ply_io.hpp>
#include <guik/recent_files.hpp>
#include <guik/progress_modal.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace glim {

// ---------------------------------------------------------------------------
// Geoid undulation lookup -- EGM2008 table files
// ---------------------------------------------------------------------------

namespace {

// Locate the submap index whose first..last frame stamp range contains `cam_time`,
// falling back to the closest boundary by |dt|. Returns -1 when no submap has
// any frames. Previously duplicated across single-cam right-click, live-preview
// single-cam and the Alignment-check submap loader.
int find_submap_for_timestamp(const std::vector<glim::SubMap::ConstPtr>& submaps, double cam_time) {
  int best = -1;
  double best_dt = std::numeric_limits<double>::max();
  for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
    if (!submaps[si] || submaps[si]->frames.empty()) continue;
    const double t0 = submaps[si]->frames.front()->stamp;
    const double t1 = submaps[si]->frames.back()->stamp;
    if (cam_time >= t0 && cam_time <= t1) return si;  // inside range -- exact match
    const double dt = std::min(std::abs(cam_time - t0), std::abs(cam_time - t1));
    if (dt < best_dt) { best_dt = dt; best = si; }
  }
  return best;
}

// Lookup-table-free bridge from Iridescence colormaps to ImGui ImU32.
// Takes a [0, 1] scalar and the numeric index of a glk::COLORMAP (as
// surfaced to the UI via glk::colormap_names()). Useful anywhere we render
// overlay points via ImGui draw list instead of going through a drawable.
// Keep local for now; lift to a shared util the day we repeat this pattern
// in a second non-drawable overlay.
// Cube-face expansion cache for spherical sources. Keyed by equirect file path.
// Slicing an 7680x3840 equirect into 6 faces via cv::remap is ~40-80 ms; caching
// means we pay it once per frame, not per colorize invocation. Memory: each face
// is ~1920x1920 RGB = ~11 MB, so 6 faces ~= 66 MB per equirect. Long sessions
// with hundreds of spherical frames easily pass 10+ GB -- the cap below evicts
// FIFO when total bytes exceed it. Bump via the UI knob (viewer::preview_cache_cap_gb).
static std::unordered_map<std::string, std::array<std::shared_ptr<cv::Mat>, 6>> g_cube_face_cache;
static std::deque<std::string> g_cube_face_cache_order;   // insertion order -- FIFO eviction
static size_t g_cube_face_cache_bytes = 0;                // sum of 6 Mats' bytes per entry
static std::mutex g_cube_face_cache_mtx;

// Runtime-settable cap (bytes). UI writes here; get_or_build_cube_faces reads.
// 0 means no cap (previous behavior -- grow unbounded).
static std::atomic<size_t> g_cube_face_cache_cap_bytes{size_t{8} * 1024 * 1024 * 1024};  // 8 GB default

// Sum bytes across the 6 face Mats (nullptr entries count as 0).
static size_t face_set_bytes(const std::array<std::shared_ptr<cv::Mat>, 6>& faces) {
  size_t b = 0;
  for (int f = 0; f < 6; f++) if (faces[f]) b += faces[f]->total() * faces[f]->elemSize();
  return b;
}

// Load equirect file into 6 cube faces (with cache). Returns a shared_ptr array
// -- empty slots (nullptr) on imread failure. FIFO-evicts when cache exceeds cap.
static std::array<std::shared_ptr<cv::Mat>, 6> get_or_build_cube_faces(
    const std::string& equirect_path, int face_size) {
  std::lock_guard<std::mutex> lock(g_cube_face_cache_mtx);
  auto it = g_cube_face_cache.find(equirect_path);
  if (it != g_cube_face_cache.end()) return it->second;
  cv::Mat img = cv::imread(equirect_path);
  if (img.empty()) return {};
  auto faces = slice_equirect_cubemap(img, face_size);
  const size_t new_bytes = face_set_bytes(faces);
  g_cube_face_cache.emplace(equirect_path, faces);
  g_cube_face_cache_order.push_back(equirect_path);
  g_cube_face_cache_bytes += new_bytes;
  // Evict oldest entries until under cap. Keep AT LEAST the just-inserted one
  // even if it alone exceeds the cap (single huge frame still works, just
  // uncached; cache effectively disabled for that pathological case).
  const size_t cap = g_cube_face_cache_cap_bytes.load(std::memory_order_relaxed);
  while (cap > 0 && g_cube_face_cache_bytes > cap && g_cube_face_cache_order.size() > 1) {
    const std::string& old_key = g_cube_face_cache_order.front();
    auto oit = g_cube_face_cache.find(old_key);
    if (oit != g_cube_face_cache.end()) {
      g_cube_face_cache_bytes -= face_set_bytes(oit->second);
      g_cube_face_cache.erase(oit);
    }
    g_cube_face_cache_order.pop_front();
  }
  return faces;
}

// Stats helper for the UI ("X / Y GB, N frames cached").
static void get_cube_face_cache_stats(size_t& bytes, size_t& frames) {
  std::lock_guard<std::mutex> lock(g_cube_face_cache_mtx);
  bytes  = g_cube_face_cache_bytes;
  frames = g_cube_face_cache.size();
}

// Expansion output: pinhole cams ready to pass to ILidarColorizer::project(),
// plus the (shared) intrinsics to feed alongside.
struct ExpandedCams {
  std::vector<CameraFrame> cams;
  PinholeIntrinsics intrinsics;
};

// Per-(mask_path, face_size) cache of the 6 cube-face slices of an equirect
// mask. Keyed by path so re-selecting the same mask across preview runs reuses
// the slice. Size cap is unlimited (masks are ~MB each, usually one per source).
static std::mutex g_mask_face_cache_mutex;
static std::unordered_map<std::string, std::array<std::shared_ptr<cv::Mat>, 6>> g_mask_face_cache;

// Build (or reuse cached) 6 cube-face slices of an equirect mask. Accepts the
// mask as a cv::Mat (runtime-loaded) and a stable key (usually the mask file
// path). Returns all-null shared_ptrs when mask is empty.
static std::array<std::shared_ptr<cv::Mat>, 6> get_or_build_mask_faces(
    const std::string& key, const cv::Mat& mask, int face_size) {
  std::array<std::shared_ptr<cv::Mat>, 6> out;
  if (mask.empty() || key.empty()) return out;
  const std::string full_key = key + "#" + std::to_string(face_size) + "x" + std::to_string(mask.cols) + "x" + std::to_string(mask.rows);
  {
    std::lock_guard<std::mutex> lk(g_mask_face_cache_mutex);
    auto it = g_mask_face_cache.find(full_key);
    if (it != g_mask_face_cache.end()) return it->second;
  }
  // Miss -- slice now. slice_equirect_cubemap reuses the face_remap_cache built
  // for images (same size pair), so this is just 6 cv::remap calls.
  auto sliced = slice_equirect_cubemap(mask, face_size);
  std::lock_guard<std::mutex> lk(g_mask_face_cache_mutex);
  g_mask_face_cache[full_key] = sliced;
  return sliced;
}

// Turn a list of cams belonging to a single source into "pinhole-ready" cams.
// For Pinhole sources: passthrough (cams + src.intrinsics).
// For Spherical sources: each equirect frame becomes 6 virtual pinhole frames
//   (one per cube face) with image_override pointing into the per-path cache;
//   intrinsics become cube_face_intrinsics(face_size).
// A face_size of 1920 gives a 90 deg FOV face with sub-pixel sampling consistent
// with a 7680-wide equirect; bump to 2560-3840 on higher-density LiDAR if needed.
//
// The src_mask param lets Spherical sources get PER-FACE masks: without it, the
// colorizer would try to linearly-rescale a 2:1 equirect mask to a square cube
// face, producing X-axis tiling (a 4x-wide mask resamples 4 times across a face).
// Pass cv::Mat() when you don't have a mask; pinhole sources ignore it.
static ExpandedCams expand_source_cams_for_projection(
    const ImageSource& src,
    const std::vector<CameraFrame>& cams_in,
    const cv::Mat& src_mask = cv::Mat(),
    int face_size = 1920) {
  if (src.camera_type != CameraType::Spherical) {
    return { cams_in, src.intrinsics };
  }
  ExpandedCams out;
  out.intrinsics = cube_face_intrinsics(face_size);
  out.cams.reserve(cams_in.size() * 6);
  // Slice the equirect mask to 6 face-sized masks up front (once per expand
  // call). Cached by mask_path, so repeat calls with the same mask are free.
  auto mask_faces = get_or_build_mask_faces(src.mask_path, src_mask, face_size);
  for (const auto& cam : cams_in) {
    auto faces = get_or_build_cube_faces(cam.filepath, face_size);
    for (int f = 0; f < 6; f++) {
      if (!faces[f]) continue;   // imread failed or face missing
      CameraFrame vc;
      vc.filepath = cam.filepath;  // kept for logging/debug; image_override is what's read
      vc.timestamp = cam.timestamp;
      vc.lat = cam.lat; vc.lon = cam.lon; vc.alt = cam.alt;
      Eigen::Isometry3d T_face = Eigen::Isometry3d::Identity();
      T_face.linear() = cube_face_rotation(f);
      vc.T_world_cam = cam.T_world_cam * T_face;  // world = cam * face-local
      vc.located = cam.located;
      vc.image_override = faces[f];
      // Per-face mask override so the colorizer samples the right region of
      // the equirect mask for this face (square mask, matches face image dims).
      // Falls back to cam's own override if the caller didn't pass a mask,
      // then to params.mask in the colorizer (linear-scale path, pinhole only).
      vc.mask_override = mask_faces[f] ? mask_faces[f] : cam.mask_override;
      out.cams.push_back(std::move(vc));
    }
  }
  return out;
}

// Round-trip helpers for colorize_config.json. One function writes, one reads,
// both cover every persisted field of ImageSource + nested ColorizeParams. New
// tunables added to ColorizeParams go here once; every save/load site stays in
// sync automatically. `.value(...)` calls fall back to struct defaults when a
// key is missing, so old configs keep loading cleanly.
static nlohmann::json image_source_to_json(const glim::ImageSource& s) {
  nlohmann::json sj;
  sj["path"]        = s.path;
  sj["mask_path"]   = s.mask_path;
  sj["time_shift"]  = s.time_shift;
  sj["lever_arm"]   = {s.lever_arm.x(), s.lever_arm.y(), s.lever_arm.z()};
  sj["rotation_rpy"]= {s.rotation_rpy.x(), s.rotation_rpy.y(), s.rotation_rpy.z()};
  sj["fx"] = s.intrinsics.fx; sj["fy"] = s.intrinsics.fy;
  sj["cx"] = s.intrinsics.cx; sj["cy"] = s.intrinsics.cy;
  sj["width"] = s.intrinsics.width; sj["height"] = s.intrinsics.height;
  sj["k1"] = s.intrinsics.k1; sj["k2"] = s.intrinsics.k2;
  sj["p1"] = s.intrinsics.p1; sj["p2"] = s.intrinsics.p2; sj["k3"] = s.intrinsics.k3;
  sj["camera_type"] = static_cast<int>(s.camera_type);
  if (s.tm_anchor1_idx >= 0) {
    sj["tm_anchor1_idx"]  = s.tm_anchor1_idx;
    sj["tm_anchor1_time"] = s.tm_anchor1_time;
    sj["tm_anchor2_idx"]  = s.tm_anchor2_idx;
    sj["tm_anchor2_time"] = s.tm_anchor2_time;
    sj["tm_fps"]          = s.tm_fps;
  }
  // Time-shift anchors (linear-interp checkpoints across the track).
  // Persisted only when non-empty so configs for single-calib sources stay clean.
  // Schema is open for extension: when extrinsic drift anchors land, append
  // lever_arm / rotation_rpy fields here and the loader's .value() calls still
  // work on old files.
  if (!s.anchors.empty()) {
    nlohmann::json aj = nlohmann::json::array();
    for (const auto& a : s.anchors) {
      aj.push_back({
        {"cam_time",   a.cam_time},
        {"time_shift", a.time_shift}
      });
    }
    sj["calib_anchors"] = aj;
  }
  // Per-source ColorizeParams as a sub-object so it's easy to spot / edit by hand.
  const auto& p = s.params;
  nlohmann::json pj;
  pj["locate_mode"]         = p.locate_mode;
  pj["min_range"]           = p.min_range;
  pj["max_range"]           = p.max_range;
  pj["blend"]               = p.blend;
  pj["intensity_blend"]     = p.intensity_blend;
  pj["intensity_mix"]       = p.intensity_mix;
  pj["nonlinear_int"]       = p.nonlinear_int;
  pj["view_selector_mode"]  = static_cast<int>(p.view_selector_mode);
  pj["range_tau"]           = p.range_tau;
  pj["center_exp"]          = p.center_exp;
  pj["incidence_exp"]       = p.incidence_exp;
  pj["topK"]                = p.topK;
  pj["use_incidence_hard"]  = p.use_incidence_hard;
  pj["incidence_hard_deg"]  = p.incidence_hard_deg;
  pj["use_ncc"]             = p.use_ncc;
  pj["ncc_threshold"]       = p.ncc_threshold;
  pj["ncc_half"]            = p.ncc_half;
  pj["use_occlusion"]       = p.use_occlusion;
  pj["occlusion_tolerance"] = p.occlusion_tolerance;
  pj["occlusion_downscale"] = p.occlusion_downscale;
  pj["time_slice_hard"]     = p.time_slice_hard;
  pj["time_slice_soft"]     = p.time_slice_soft;
  pj["time_slice_sigma"]    = p.time_slice_sigma;
  pj["use_exposure_norm"]   = p.use_exposure_norm;
  pj["exposure_target"]     = p.exposure_target;
  pj["exposure_simple"]     = p.exposure_simple;
  sj["params"] = pj;
  return sj;
}

// Read the non-image-loading bits (the caller still calls
// Colorizer::load_image_folder for `path`, then hands the source in here).
static void image_source_apply_json(glim::ImageSource& source, const nlohmann::json& sj) {
  source.time_shift = sj.value("time_shift", 0.0);
  source.mask_path  = sj.value("mask_path", "");
  if (sj.contains("lever_arm"))
    source.lever_arm = Eigen::Vector3d(sj["lever_arm"][0], sj["lever_arm"][1], sj["lever_arm"][2]);
  if (sj.contains("rotation_rpy"))
    source.rotation_rpy = Eigen::Vector3d(sj["rotation_rpy"][0], sj["rotation_rpy"][1], sj["rotation_rpy"][2]);
  source.intrinsics.fx = sj.value("fx", 1920.0); source.intrinsics.fy = sj.value("fy", 1920.0);
  source.intrinsics.cx = sj.value("cx", 1920.0); source.intrinsics.cy = sj.value("cy", 1080.0);
  source.intrinsics.width = sj.value("width", 3840); source.intrinsics.height = sj.value("height", 2160);
  source.intrinsics.k1 = sj.value("k1", 0.0); source.intrinsics.k2 = sj.value("k2", 0.0);
  source.intrinsics.p1 = sj.value("p1", 0.0); source.intrinsics.p2 = sj.value("p2", 0.0);
  source.intrinsics.k3 = sj.value("k3", 0.0);
  if (sj.contains("camera_type"))
    source.camera_type = static_cast<glim::CameraType>(sj["camera_type"].get<int>());
  if (sj.contains("tm_anchor1_idx") && sj["tm_anchor1_idx"].get<int>() >= 0) {
    source.tm_anchor1_idx  = sj.value("tm_anchor1_idx", -1);
    source.tm_anchor1_time = sj.value("tm_anchor1_time", 0.0);
    source.tm_anchor2_idx  = sj.value("tm_anchor2_idx", -1);
    source.tm_anchor2_time = sj.value("tm_anchor2_time", 0.0);
    source.tm_fps          = sj.value("tm_fps", 30.0f);
  }
  // Time-shift anchors. Missing key -> empty vector -> scalar baseline
  // behavior preserved (old configs load cleanly). Extra keys on each anchor
  // (lever_arm/rotation_rpy from a future schema) are tolerated and ignored.
  source.anchors.clear();
  if (sj.contains("calib_anchors") && sj["calib_anchors"].is_array()) {
    source.anchors.reserve(sj["calib_anchors"].size());
    for (const auto& aj : sj["calib_anchors"]) {
      glim::CalibAnchor a;
      a.cam_time   = aj.value("cam_time", 0.0);
      a.time_shift = aj.value("time_shift", 0.0);
      source.anchors.push_back(a);
    }
    // Enforce sort invariant on load (tolerant of hand-edited configs that
    // might list anchors in insertion order rather than time order).
    std::sort(source.anchors.begin(), source.anchors.end(),
      [](const glim::CalibAnchor& x, const glim::CalibAnchor& y) { return x.cam_time < y.cam_time; });
  }
  // ColorizeParams: if missing entirely, seed camera-type-aware defaults; otherwise
  // read each field with struct-default fallback (additive / removable safely).
  auto& p = source.params;
  p = glim::default_colorize_params_for(source.camera_type);  // sensible baseline
  if (sj.contains("params")) {
    const auto& pj = sj["params"];
    p.locate_mode         = pj.value("locate_mode",         p.locate_mode);
    p.min_range           = pj.value("min_range",           p.min_range);
    p.max_range           = pj.value("max_range",           p.max_range);
    p.blend               = pj.value("blend",               p.blend);
    p.intensity_blend     = pj.value("intensity_blend",     p.intensity_blend);
    p.intensity_mix       = pj.value("intensity_mix",       p.intensity_mix);
    p.nonlinear_int       = pj.value("nonlinear_int",       p.nonlinear_int);
    if (pj.contains("view_selector_mode"))
      p.view_selector_mode = static_cast<glim::ViewSelectorMode>(pj["view_selector_mode"].get<int>());
    p.range_tau           = pj.value("range_tau",           p.range_tau);
    p.center_exp          = pj.value("center_exp",          p.center_exp);
    p.incidence_exp       = pj.value("incidence_exp",       p.incidence_exp);
    p.topK                = pj.value("topK",                p.topK);
    p.use_incidence_hard  = pj.value("use_incidence_hard",  p.use_incidence_hard);
    p.incidence_hard_deg  = pj.value("incidence_hard_deg",  p.incidence_hard_deg);
    p.use_ncc             = pj.value("use_ncc",             p.use_ncc);
    p.ncc_threshold       = pj.value("ncc_threshold",       p.ncc_threshold);
    p.ncc_half            = pj.value("ncc_half",            p.ncc_half);
    p.use_occlusion       = pj.value("use_occlusion",       p.use_occlusion);
    p.occlusion_tolerance = pj.value("occlusion_tolerance", p.occlusion_tolerance);
    p.occlusion_downscale = pj.value("occlusion_downscale", p.occlusion_downscale);
    p.time_slice_hard     = pj.value("time_slice_hard",     p.time_slice_hard);
    p.time_slice_soft     = pj.value("time_slice_soft",     p.time_slice_soft);
    p.time_slice_sigma    = pj.value("time_slice_sigma",    p.time_slice_sigma);
    p.use_exposure_norm   = pj.value("use_exposure_norm",   p.use_exposure_norm);
    p.exposure_target     = pj.value("exposure_target",     p.exposure_target);
    p.exposure_simple     = pj.value("exposure_simple",     p.exposure_simple);
  }
}

inline ImU32 scalar_to_imu32(int colormap_idx, float t, int alpha) {
  // glk::colormapf takes t in [0, 1] and does `int x = t * 255` internally to
  // index the 256-entry table. Pass the normalized value directly -- any extra
  // *255 here double-scales and collapses every non-trivial value to the top
  // of the table, producing a two-tone blob.
  const int max_idx = static_cast<int>(glk::COLORMAP::NUM_COLORMAPS) - 1;
  const auto cm = static_cast<glk::COLORMAP>(std::clamp(colormap_idx, 0, max_idx));
  const Eigen::Vector4f c = glk::colormapf(cm, std::clamp(t, 0.0f, 1.0f));
  const int r = std::clamp(static_cast<int>(c.x() * 255.0f), 0, 255);
  const int g = std::clamp(static_cast<int>(c.y() * 255.0f), 0, 255);
  const int b = std::clamp(static_cast<int>(c.z() * 255.0f), 0, 255);
  return IM_COL32(r, g, b, std::clamp(alpha, 0, 255));
}

// Intensity to blue-cyan-white color ramp (for intensity blend visualization)
inline Eigen::Vector3f intensity_to_color(float t) {
  // t in [0,1]: 0=dark blue, 0.5=cyan, 1.0=white
  if (t < 0.5f) {
    const float s = t * 2.0f;
    return Eigen::Vector3f(0.05f, 0.1f + 0.7f * s, 0.3f + 0.7f * s);  // dark blue -> cyan
  } else {
    const float s = (t - 0.5f) * 2.0f;
    return Eigen::Vector3f(s, 0.8f + 0.2f * s, 1.0f);  // cyan -> white
  }
}

// Build the named "colorize_preview" PointCloudBuffer drawable from a
// ColorizeResult, folding in ColorizeParams' intensity-blend knobs.
// Previously duplicated across 5 Colorize entry points (right-click per-camera,
// right-click per-submap, live-preview single-cam, live-preview submap,
// intensity-blend refresh). The submap-preview copy used a grayscale ramp
// instead of intensity_to_color(); the `use_grayscale_intensity` flag
// preserves that variant so pulling this out doesn't silently change behavior.
void push_colorize_preview_drawable(
    guik::LightViewer* vw,
    const glim::ColorizeResult& cr,
    const glim::ColorizeParams& cp,
    bool use_grayscale_intensity = false) {
  const size_t n = cr.points.size();
  float imin = std::numeric_limits<float>::max();
  float imax = std::numeric_limits<float>::lowest();
  for (size_t i = 0; i < n && i < cr.intensities.size(); i++) {
    imin = std::min(imin, cr.intensities[i]);
    imax = std::max(imax, cr.intensities[i]);
  }
  if (imin >= imax) { imin = 0.0f; imax = 255.0f; }

  std::vector<Eigen::Vector4d> p4(n);
  std::vector<Eigen::Vector4f> c4(n);
  for (size_t i = 0; i < n; i++) {
    p4[i] = Eigen::Vector4d(cr.points[i].x(), cr.points[i].y(), cr.points[i].z(), 1.0);
    Eigen::Vector3f rgb = cr.colors[i];
    if (cp.intensity_blend && i < cr.intensities.size()) {
      float inv = (cr.intensities[i] - imin) / (imax - imin);
      if (!use_grayscale_intensity && cp.nonlinear_int) inv = std::sqrt(inv);
      const Eigen::Vector3f ramp = use_grayscale_intensity
                                      ? Eigen::Vector3f(inv, inv, inv)
                                      : intensity_to_color(inv);
      rgb = rgb * (1.0f - cp.intensity_mix) + ramp * cp.intensity_mix;
    }
    c4[i] = Eigen::Vector4f(rgb.x(), rgb.y(), rgb.z(), 1.0f);
  }
  auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), p4.size());
  cb->add_color(c4);
  vw->update_drawable("colorize_preview", cb,
                       guik::Rainbow().set_color_mode(guik::ColorMode::VERTEX_COLOR));
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
      return false;  // data before all headers -- malformed
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
// JGD2011 prefecture -> zone mapping
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

void OfflineViewer::render_camera_gizmo(int src_idx, int frame_idx) {
  if (src_idx < 0 || src_idx >= static_cast<int>(image_sources.size())) return;
  const auto& source = image_sources[src_idx];
  if (frame_idx < 0 || frame_idx >= static_cast<int>(source.frames.size())) return;
  const auto& frame = source.frames[frame_idx];
  if (!frame.located) return;

  auto vw = guik::LightViewer::instance();
  const Eigen::Vector3f pos = frame.T_world_cam.translation().cast<float>();
  const Eigen::Matrix3f R   = frame.T_world_cam.rotation().cast<float>();
  const Eigen::Vector3f fwd   = R.col(0).normalized();
  const Eigen::Vector3f right = R.col(1).normalized();
  const Eigen::Vector3f up    = R.col(2).normalized();
  const bool is_spherical = (source.camera_type == CameraType::Spherical);
  const std::string fov_name = "cam_fov_" + std::to_string(src_idx) + "_" + std::to_string(frame_idx);
  const std::string body_name = "cam_" + std::to_string(src_idx) + "_" + std::to_string(frame_idx);

  // FOV indicator: pinhole gets a 4-edge frustum; spherical gets a single
  // forward-ray line (no frustum, 360 has no FOV).
  if (!is_spherical) {
    const float fov_len = 0.6f, fov_w = 0.3f, fov_h = 0.2f;
    const Eigen::Vector3f bc = pos + fwd * fov_len;
    std::vector<Eigen::Vector3f> v = {
      pos, bc + right*fov_w + up*fov_h, pos, bc - right*fov_w + up*fov_h,
      pos, bc - right*fov_w - up*fov_h, pos, bc + right*fov_w - up*fov_h,
      bc + right*fov_w + up*fov_h, bc - right*fov_w + up*fov_h,
      bc - right*fov_w + up*fov_h, bc - right*fov_w - up*fov_h,
      bc - right*fov_w - up*fov_h, bc + right*fov_w - up*fov_h,
      bc + right*fov_w - up*fov_h, bc + right*fov_w + up*fov_h
    };
    vw->update_drawable(fov_name,
      std::make_shared<glk::ThinLines>(v.data(), static_cast<int>(v.size()), false),
      guik::FlatColor(1.0f, 1.0f, 1.0f, 0.7f));
  } else {
    std::vector<Eigen::Vector3f> v = { pos, pos + fwd * 0.8f };
    vw->update_drawable(fov_name,
      std::make_shared<glk::ThinLines>(v.data(), static_cast<int>(v.size()), false),
      guik::FlatColor(0.4f, 0.7f, 1.0f, 0.9f));
  }

  // Body: cube for pinhole (white), sphere for spherical (blue).
  Eigen::Affine3f btf = Eigen::Affine3f::Identity();
  btf.translate(pos); btf.linear() = R;
  btf = btf * Eigen::Scaling(is_spherical ? Eigen::Vector3f(0.14f, 0.14f, 0.14f)
                                          : Eigen::Vector3f(0.12f, 0.18f, 0.12f));
  const Eigen::Vector4i info(static_cast<int>(PickType::CAMERA), src_idx, 0, frame_idx);
  if (is_spherical) {
    vw->update_drawable(body_name, glk::Primitives::sphere(),
      guik::FlatColor(0.35f, 0.65f, 1.0f, 0.9f, btf).add("info_values", info));
  } else {
    vw->update_drawable(body_name, glk::Primitives::cube(),
      guik::FlatColor(1.0f, 1.0f, 1.0f, 0.9f, btf).add("info_values", info));
  }
}

void OfflineViewer::render_anchor_panel(int src_idx, double cam_time, bool have_time,
                                         const char* id_suffix) {
  if (src_idx < 0 || src_idx >= static_cast<int>(image_sources.size())) return;
  auto& src = image_sources[src_idx];
  const size_t na = src.anchors.size();
  // Dedup tolerance: if a new anchor is within this many seconds of an
  // existing one, the existing one is UPDATED instead of a near-duplicate
  // being inserted. 0.5s = typically 15 frames at 30fps -- plenty of room to
  // re-land on the same anchor to re-tune it.
  constexpr double kAnchorMergeTol = 0.5;

  ImGui::TextDisabled("Time anchors (%zu)", na);
  ImGui::SameLine();
  {
    char btn_label[48]; std::snprintf(btn_label, sizeof(btn_label), "Anchor here##ca_%s", id_suffix);
    ImGui::BeginDisabled(!have_time);
    if (ImGui::SmallButton(btn_label)) {
      CalibAnchor a;
      a.cam_time   = cam_time;        // source-local; stable under time_shift tweaks
      a.time_shift = src.time_shift;
      // Keep anchors sorted by cam_time so effective_time_shift's
      // lower_bound lookup works without re-sorting on every query.
      auto it = std::lower_bound(src.anchors.begin(), src.anchors.end(), a.cam_time,
        [](const CalibAnchor& x, double t) { return x.cam_time < t; });
      // Find the closest neighbor on either side within merge tolerance;
      // update it instead of creating a near-duplicate anchor.
      int merge_idx = -1;
      if (it != src.anchors.end() && std::abs(it->cam_time - a.cam_time) < kAnchorMergeTol)
        merge_idx = static_cast<int>(it - src.anchors.begin());
      if (it != src.anchors.begin()) {
        auto prev = it - 1;
        if (std::abs(prev->cam_time - a.cam_time) < kAnchorMergeTol) {
          if (merge_idx < 0 || std::abs(prev->cam_time - a.cam_time) < std::abs(it->cam_time - a.cam_time))
            merge_idx = static_cast<int>(prev - src.anchors.begin());
        }
      }
      if (merge_idx >= 0) {
        src.anchors[merge_idx] = a;
        align_anchor_selected = merge_idx;
        logger->info("[Anchor] Updated existing anchor #{} at cam_t={:.6f} shift={:+.6f}s (src {})",
          merge_idx, a.cam_time, a.time_shift, src_idx);
      } else {
        const int new_idx = static_cast<int>(it - src.anchors.begin());
        src.anchors.insert(it, a);
        align_anchor_selected = new_idx;
        logger->info("[Anchor] Added anchor #{} at cam_t={:.6f} shift={:+.6f}s (src {})",
          new_idx, a.cam_time, a.time_shift, src_idx);
      }
      align_anchor_selected_src = src_idx;
    }
    ImGui::EndDisabled();
  }
  if (ImGui::IsItemHovered()) {
    if (have_time) {
      ImGui::SetTooltip(
        "Snapshot current time_shift (%+.6f s) as an anchor at cam_t=%.3f.\n"
        "With 2+ anchors, effective_time_shift() linearly interpolates per frame.\n"
        "\n"
        "Click within 0.5 s of an existing anchor UPDATES it instead of placing\n"
        "a near-duplicate. Order-independent: anchors always sort by cam_time.",
        src.time_shift, cam_time);
    } else {
      ImGui::SetTooltip(
        "No current frame to anchor. Scrub a frame in Alignment check, or\n"
        "right-click a camera / submap in the 3D view to seed a preview first.");
    }
  }
  // "Apply to #N": overwrites the selected anchor's time_shift with whatever
  // is currently in the widget (src.time_shift). Different from "Anchor here":
  //   - "Anchor here" places at the CURRENT preview frame (uses merge-tol).
  //   - "Apply to #N" targets the SELECTED anchor directly, keeping its cam_time.
  // Use after: click a row (pulls anchor's current value into widget), drag
  // widget to tune, click Apply. Live preview updates during drag via scratch.
  const bool have_sel = (align_anchor_selected_src == src_idx &&
                         align_anchor_selected >= 0 &&
                         align_anchor_selected < static_cast<int>(src.anchors.size()));
  if (have_sel) {
    const auto& target_ro = src.anchors[align_anchor_selected];
    ImGui::SameLine();
    char load_label[64];
    std::snprintf(load_label, sizeof(load_label), "Load #%d##ca_load_%s",
                  align_anchor_selected, id_suffix);
    if (ImGui::SmallButton(load_label)) {
      src.time_shift = target_ro.time_shift;
      logger->info("[Anchor] Loaded #{} (cam_t={:.3f}, shift={:+.6f}) into time_shift widget",
        align_anchor_selected, target_ro.cam_time, target_ro.time_shift);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Pull selected anchor #%d's time_shift (%+.6f s) into the widget so you\n"
      "can edit it. Live preview is scratch-driven, so the next frame reflects\n"
      "the loaded value. Drag the widget to tune, then hit 'Apply to #%d' to\n"
      "write it back.",
      align_anchor_selected, target_ro.time_shift, align_anchor_selected);

    ImGui::SameLine();
    char apply_label[64];
    std::snprintf(apply_label, sizeof(apply_label), "Apply to #%d##ca_apply_%s",
                  align_anchor_selected, id_suffix);
    if (ImGui::SmallButton(apply_label)) {
      auto& target = src.anchors[align_anchor_selected];
      const double old_ts = target.time_shift;
      target.time_shift = src.time_shift;
      logger->info("[Anchor] Applied time_shift {:+.6f}s -> {:+.6f}s on anchor #{} at cam_t={:.3f} (src {})",
        old_ts, target.time_shift, align_anchor_selected, target.cam_time, src_idx);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Write current time_shift (%+.6f s) into selected anchor #%d (cam_t=%.3f).\n"
      "Anchor's cam_time stays put -- only its time_shift value updates.\n"
      "\n"
      "Refine-an-anchor workflow:\n"
      "  1. Click the anchor row (highlights the 3D pin; widget unchanged).\n"
      "  2. Click 'Load #%d' to pull its value into the widget.\n"
      "  3. Drag time_shift -- live preview updates via scratch.\n"
      "  4. Click this to commit.",
      src.time_shift, align_anchor_selected, src.anchors[align_anchor_selected].cam_time,
      align_anchor_selected);
  }
  if (na >= 1) {
    ImGui::SameLine();
    char clr_label[48]; std::snprintf(clr_label, sizeof(clr_label), "Clear all##ca_%s", id_suffix);
    if (ImGui::SmallButton(clr_label)) {
      src.anchors.clear();
      if (align_anchor_selected_src == src_idx) align_anchor_selected = -1;
      logger->info("[Anchor] Cleared all anchors (src {})", src_idx);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Remove all anchors. The source reverts to its scalar time_shift baseline.");
  }
  // State badge. Below 2 anchors, interpolation is INERT -- scalar baseline
  // rules everywhere. Makes it obvious that a single anchor is a bookmark, not
  // an active override. Historically we clamped 1 anchor to a constant, which
  // unexpectedly nuked previously-correct sections.
  if (na == 1) {
    ImGui::TextColored(ImVec4(0.90f, 0.80f, 0.35f, 1.0f),
      "Bookmark only -- place a 2nd anchor elsewhere to activate interpolation.");
  } else if (na >= 2) {
    double ts_lo = src.anchors.front().time_shift, ts_hi = ts_lo;
    double ct_lo = src.anchors.front().cam_time,   ct_hi = ct_lo;
    for (const auto& a : src.anchors) {
      ts_lo = std::min(ts_lo, a.time_shift); ts_hi = std::max(ts_hi, a.time_shift);
      ct_lo = std::min(ct_lo, a.cam_time);   ct_hi = std::max(ct_hi, a.cam_time);
    }
    ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.0f, 1.0f),
      "Drift: %+.6f .. %+.6f s across %.1f s of track",
      ts_lo, ts_hi, ct_hi - ct_lo);
  }
  // Compact anchor table. Clicking a row selects it -- the 3D gizmo for that
  // anchor stretches Z x10 so it's easy to locate in the viewer.
  if (na > 0) {
    char hdr_label[48]; std::snprintf(hdr_label, sizeof(hdr_label), "Anchor list##ca_%s", id_suffix);
    if (ImGui::CollapsingHeader(hdr_label, ImGuiTreeNodeFlags_DefaultOpen)) {
      int remove_idx = -1;
      char tbl_id[48]; std::snprintf(tbl_id, sizeof(tbl_id), "anchors_table_%s", id_suffix);
      if (ImGui::BeginTable(tbl_id, 4,
            ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("#");
        ImGui::TableSetupColumn("cam_t(s)");
        ImGui::TableSetupColumn("shift(s)");
        ImGui::TableSetupColumn("");
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < src.anchors.size(); i++) {
          const auto& a = src.anchors[i];
          ImGui::TableNextRow();
          const bool is_sel = (align_anchor_selected == static_cast<int>(i) &&
                                align_anchor_selected_src == src_idx);
          ImGui::TableSetColumnIndex(0);
          char sel_lbl[32]; std::snprintf(sel_lbl, sizeof(sel_lbl), "%zu##ca_sel_%s_%zu", i, id_suffix, i);
          if (ImGui::Selectable(sel_lbl, is_sel, ImGuiSelectableFlags_SpanAllColumns)) {
            align_anchor_selected = static_cast<int>(i);
            align_anchor_selected_src = src_idx;
            // Pure selection: highlights the 3D gizmo only. Does NOT alter the
            // time_shift widget -- the user may be mid-drag on an unrelated
            // tuning and shouldn't have their scratch clobbered. An explicit
            // "Load #N" button (next to Apply) pulls the anchor's value into
            // the widget when the user decides to edit it.
          }
          ImGui::TableSetColumnIndex(1); ImGui::Text("%.3f", a.cam_time);
          ImGui::TableSetColumnIndex(2); ImGui::Text("%+.6f", a.time_shift);
          ImGui::TableSetColumnIndex(3);
          char btn[32]; std::snprintf(btn, sizeof(btn), "x##ca_%s_%zu", id_suffix, i);
          if (ImGui::SmallButton(btn)) remove_idx = static_cast<int>(i);
        }
        ImGui::EndTable();
      }
      if (remove_idx >= 0 && remove_idx < static_cast<int>(src.anchors.size())) {
        src.anchors.erase(src.anchors.begin() + remove_idx);
        if (align_anchor_selected_src == src_idx) {
          if (align_anchor_selected == remove_idx) align_anchor_selected = -1;
          else if (align_anchor_selected > remove_idx) align_anchor_selected--;
        }
        logger->info("[Anchor] Removed anchor #{} (src {}). {} remain.", remove_idx, src_idx, src.anchors.size());
      }
    }
  }
}

BlendParams OfflineViewer::current_blend_params(const ImageSource& src) const {
  // Pulls every tunable from the passed source's ColorizeParams so that every
  // preview / apply / assignment call site lands on the same strategy. Single
  // construction site means adding/removing a BlendParams field touches one
  // place.
  const auto& p = src.params;
  return BlendParams{
    p.max_range, p.min_range, p.blend, colorize_mask,
    p.range_tau, p.center_exp, p.incidence_exp, p.topK,
    p.use_incidence_hard ? static_cast<float>(std::cos(p.incidence_hard_deg * M_PI / 180.0)) : -1.0f,
    p.use_ncc ? p.ncc_threshold : -2.0f, p.ncc_half,
    p.use_occlusion, p.occlusion_tolerance, p.occlusion_downscale,
    p.time_slice_hard, p.time_slice_soft, p.time_slice_sigma,
    p.use_exposure_norm, p.exposure_target, p.exposure_simple
  };
}

BlendParams OfflineViewer::current_blend_params() const {
  // Convenience overload: use the Colorize window's active source. Falls back
  // to a default ColorizeParams if no sources exist yet (UI can run early).
  if (image_sources.empty()) {
    static const ImageSource empty_src;
    return current_blend_params(empty_src);
  }
  const int idx = std::clamp(colorize_source_idx, 0, static_cast<int>(image_sources.size()) - 1);
  return current_blend_params(image_sources[idx]);
}

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

void OfflineViewer::vc_pcam_seed_factory_presets() {
  if (vc_pcam_presets_initialised) return;
  vc_pcam_presets_initialised = true;

  // --- Livox Horizon ---
  // Forward-facing, sparse outside the narrow swath. Wide time window so the
  // camera looking forward still has side/back context from nearby frames as
  // the platform drives past. Linear ramp of 2 px (close) to 1 px (15 m+).
  VcamPreset horizon;
  horizon.name = "Livox Horizon";
  horizon.ctx_opts.use_time_window = true;
  horizon.ctx_opts.time_before_s   = 80.0;
  horizon.ctx_opts.time_after_s    = 80.0;
  horizon.ctx_opts.directional_filter = false;
  horizon.ctx_opts.min_range = 6.0f;
  horizon.ctx_opts.max_range = 150.0f;
  horizon.render_opts.splat_mode = IntensityRenderOptions::SplatMode::LinearRamp;
  horizon.render_opts.splat_ranges = { {0.0, 2}, {15.0, 1} };
  horizon.render_opts.non_linear_intensity = false;  // user reports linear has better road-marking contrast
  horizon.face_enabled[0] = true;  horizon.face_enabled[1] = true;
  horizon.face_enabled[2] = true;  horizon.face_enabled[3] = true;
  horizon.face_enabled[4] = false; horizon.face_enabled[5] = false;
  horizon.face_size = 4096;
  horizon.format = 1;           // JPG
  horizon.jpg_quality = 95;
  vc_pcam_presets.push_back(horizon);

  // --- Hesai XT32 ---
  // 32-beam mechanical -- denser than Horizon; slightly tighter window,
  // can pull in closer points. Placeholder values, expect tuning.
  VcamPreset xt32;
  xt32.name = "Hesai XT32";
  xt32.ctx_opts.use_time_window = true;
  xt32.ctx_opts.time_before_s   = 15.0;
  xt32.ctx_opts.time_after_s    = 15.0;
  xt32.ctx_opts.directional_filter = false;
  xt32.ctx_opts.min_range = 4.0f;
  xt32.ctx_opts.max_range = 150.0f;
  xt32.render_opts.splat_mode = IntensityRenderOptions::SplatMode::LinearRamp;
  xt32.render_opts.splat_ranges = { {0.0, 2}, {15.0, 1} };
  xt32.render_opts.non_linear_intensity = false;
  xt32.face_enabled[0] = true;  xt32.face_enabled[1] = true;
  xt32.face_enabled[2] = true;  xt32.face_enabled[3] = true;
  xt32.face_enabled[4] = false; xt32.face_enabled[5] = false;
  xt32.face_size = 4096;
  xt32.format = 1;
  xt32.jpg_quality = 95;
  vc_pcam_presets.push_back(xt32);

  // --- Pandar 128 ---
  // 128-beam, densest of the three. Tighter window acceptable because each
  // frame already carries a lot of data. Placeholder values.
  VcamPreset pandar;
  pandar.name = "Pandar 128";
  pandar.ctx_opts.use_time_window = true;
  pandar.ctx_opts.time_before_s   = 10.0;
  pandar.ctx_opts.time_after_s    = 10.0;
  pandar.ctx_opts.directional_filter = false;
  pandar.ctx_opts.min_range = 3.0f;
  pandar.ctx_opts.max_range = 200.0f;
  pandar.render_opts.splat_mode = IntensityRenderOptions::SplatMode::LinearRamp;
  pandar.render_opts.splat_ranges = { {0.0, 2}, {15.0, 1} };
  pandar.render_opts.non_linear_intensity = false;
  pandar.face_enabled[0] = true;  pandar.face_enabled[1] = true;
  pandar.face_enabled[2] = true;  pandar.face_enabled[3] = true;
  pandar.face_enabled[4] = false; pandar.face_enabled[5] = false;
  pandar.face_size = 4096;
  pandar.format = 1;
  pandar.jpg_quality = 95;
  vc_pcam_presets.push_back(pandar);
}

void OfflineViewer::vc_pcam_apply_preset(const VcamPreset& p) {
  vc_pcam_ctx_opts = p.ctx_opts;
  vc_pcam_render_opts = p.render_opts;
  vc_pcam_render_w = p.render_w;
  vc_pcam_render_h = p.render_h;
  for (int i = 0; i < 6; i++) vc_face_enabled[i] = p.face_enabled[i];
  vc_face_size = p.face_size;
  vc_pcam_format = p.format;
  vc_pcam_jpg_quality = p.jpg_quality;
  // Force a re-render with the new values the next time the UI ticks.
  vc_pcam_preview_dirty = true;
}

const char* OfflineViewer::vc_face_label(int face_idx) {
  switch (face_idx) {
    case 0: return "Front";
    case 1: return "Back";
    case 2: return "Left";
    case 3: return "Right";
    case 4: return "Up";
    case 5: return "Down";
    default: return "Face";
  }
}

std::vector<TimedPose> OfflineViewer::timed_traj_snapshot() const {
  std::vector<TimedPose> snap(trajectory_data.size());
  for (size_t i = 0; i < trajectory_data.size(); i++) {
    snap[i] = {trajectory_data[i].stamp, trajectory_data[i].pose};
  }
  return snap;
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
          logger->info("[JGD2011] Datum in {} ({}) -- Zone {} ({})",
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

  // World-axis gizmo (bottom-right). Always-on indicator that updates as the
  // camera rotates so the user can confirm the world's orientation. X red,
  // Y green, Z blue. Useful for sanity-checking exports vs other tools that
  // may use a different up-axis convention (e.g. LichtFeld defaults to Y-up
  // while we are Z-up).
  viewer->register_ui_callback("axis_gizmo", [this] {
    if (!show_axis_gizmo) return;
    auto vw = guik::LightViewer::instance();
    const Eigen::Matrix4f vm = vw->view_matrix();
    const Eigen::Matrix3f R = vm.block<3, 3>(0, 0);  // world -> view rotation
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    const ImVec2 origin(disp.x - 70.0f, disp.y - 70.0f);
    const float L = 36.0f;
    auto* dl = ImGui::GetForegroundDrawList();
    dl->AddCircleFilled(origin, L + 14.0f, IM_COL32(15, 15, 15, 170));
    dl->AddCircle(origin, L + 14.0f, IM_COL32(80, 80, 80, 200), 0, 1.5f);

    struct Axis { Eigen::Vector3f v; ImU32 color; const char* label; };
    Axis axes[3] = {
      {{1.0f, 0.0f, 0.0f}, IM_COL32(235, 75, 75, 255), "X"},
      {{0.0f, 1.0f, 0.0f}, IM_COL32(80, 210, 80, 255), "Y"},
      {{0.0f, 0.0f, 1.0f}, IM_COL32(85, 140, 235, 255), "Z"},
    };
    // Draw back-facing axes first so front-facing labels overlap them on top.
    std::sort(axes, axes + 3, [&R](const Axis& a, const Axis& b) {
      return (R * a.v).z() > (R * b.v).z();   // larger z (deeper into -view) first
    });
    for (const auto& a : axes) {
      const Eigen::Vector3f v = R * a.v;
      const ImVec2 endp(origin.x + v.x() * L, origin.y - v.y() * L);
      dl->AddLine(origin, endp, a.color, 2.0f);
      dl->AddCircleFilled(endp, 4.0f, a.color);
      Eigen::Vector2f dir(v.x(), -v.y());
      const float dn = dir.norm();
      if (dn > 0.05f) dir /= dn;
      else dir = Eigen::Vector2f(0.0f, 0.0f);
      const ImVec2 lbl(endp.x + dir.x() * 9.0f - 4.0f, endp.y + dir.y() * 9.0f - 7.0f);
      dl->AddText(lbl, a.color, a.label);
    }
  });

  // Session visibility filter -- hides submaps and spheres for disabled sessions
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
    // Smooth position only -- rotation stays crisp (Iridescence handles it natively)
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

    // Space toggles play/pause -- pause sets speed to 0, unpause recovers last speed
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
      const double unclamped = current_dist + advance;
      const double new_dist = std::clamp(unclamped, 0.0, follow_total_dist);
      follow_progress = static_cast<float>(new_dist / follow_total_dist);
      // Only stop when we actually tried to move past a boundary -- checking
      // follow_progress against 0 / 1 directly would fire on the very first
      // frame after mode switch (progress starts at 0.0 before motion).
      if (unclamped > follow_total_dist || unclamped < 0.0) {
        follow_speed_kmh = 0.0f;
        follow_playing = false;
      }
    }

    // Mouse drag for turret rotation (right-click drag -- smooth, no UI conflict)
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
    Eigen::Vector3d pos = 0.5 * (
      (2.0 * p1) +
      (-p0 + p2) * t +
      (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
      (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
    // Vertical shift ("drone view"): lift the camera above the trajectory
    // without changing yaw/pitch so the view still tracks the heading.
    pos.z() += static_cast<double>(follow_height_offset_m);

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
          ImGui::SetTooltip("Keep only ground-classified points.\nOne point per XY column -- removes ground noise.\nRequires aux_ground.bin from Dynamic filter.");
      }
      if (vox_ground_only) {
        ImGui::SameLine();
        ImGui::TextDisabled("(forces XY+Z weighted)");
      }

      ImGui::DragFloat("Chunk size (m)", &vox_chunk_size, 5.0f, 20.0f, 200.0f, "%.0f");
      ImGui::DragFloat("Chunk spacing (m)", &vox_chunk_spacing, 5.0f, 10.0f, 100.0f, "%.0f");
      ImGui::Checkbox("Include intensity", &vox_include_intensity);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Carry per-point intensity into each voxel (average) and write\n"
        "intensities.bin in the output frames. Off = no intensity file is\n"
        "emitted; downstream tools that rely on intensity will fall back.");
      ImGui::SameLine();
      ImGui::Checkbox("Include RGB", &vox_include_rgb);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Carry per-point RGB (from aux_rgb.bin, produced by Colorize > Apply) into\n"
        "each voxel (average) and write aux_rgb.bin in the output frames. Off =\n"
        "no RGB file is emitted. If a source frame is missing aux_rgb.bin the\n"
        "voxel is skipped for that file -- intensity (if on) still lands.");
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
            std::unordered_map<uint64_t, std::vector<std::pair<Eigen::Vector3f, float>>> voxel_data;  // key -> (pos, intensity)
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

            logger->info("[Voxelize preview] {} input points -> {} voxels (size={:.3f}m, ground_only={})", total_input, voxel_data.size(), vox_size, ground_only);
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
              std::vector<Eigen::Vector3f> rgbs;      // only populated when rgb_present
              std::vector<uint8_t> rgb_present;        // 1 = voxel had any RGB source
            };
            std::unordered_map<std::string, FrameOutput> frame_outputs;  // frame_dir -> output
            // Initialize empty outputs for all frames
            for (const auto& fi : all_frames) frame_outputs[fi.dir] = {};

            const float inv_vox = 1.0f / vox_size;
            size_t total_voxels = 0;

            // Frame cache: loaded world-space points per frame (sliding window)
            struct CachedFrame {
              std::vector<Eigen::Vector3f> world_pts;
              std::vector<float> intensities;
              std::vector<float> ranges;
              std::vector<Eigen::Vector3f> rgbs;   // empty unless source aux_rgb.bin existed
              bool has_rgb = false;
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
                if (vox_include_intensity) {
                  glim::load_bin(fi->dir + "/intensities.bin", ints, fi->num_points);
                }
                // Load ground mask if ground-only mode
                std::vector<float> ground;
                if (ground_only) glim::load_bin(fi->dir + "/aux_ground.bin", ground, fi->num_points);
                // Load RGB if the feature is on and the file exists for this frame.
                std::vector<Eigen::Vector3f> rgbs;
                bool frame_has_rgb = false;
                if (vox_include_rgb && boost::filesystem::exists(fi->dir + "/aux_rgb.bin")) {
                  rgbs.resize(fi->num_points, Eigen::Vector3f::Zero());
                  std::ifstream rgbf(fi->dir + "/aux_rgb.bin", std::ios::binary);
                  if (rgbf) {
                    rgbf.read(reinterpret_cast<char*>(rgbs.data()),
                              sizeof(Eigen::Vector3f) * fi->num_points);
                    frame_has_rgb = static_cast<bool>(rgbf);
                  }
                  if (!frame_has_rgb) rgbs.clear();
                }
                const Eigen::Matrix3f R = fi->T_world_lidar.rotation().cast<float>();
                const Eigen::Vector3f t = fi->T_world_lidar.translation().cast<float>();
                auto cf = std::make_shared<CachedFrame>();
                cf->dir = fi->dir;
                cf->has_rgb = frame_has_rgb;
                for (int i = 0; i < fi->num_points; i++) {
                  if (!rng.empty() && rng[i] < 1.5f) continue;
                  if (ground_only && (ground.empty() || ground[i] < 0.5f)) continue;
                  cf->world_pts.push_back(R * pts[i] + t);
                  cf->intensities.push_back(ints[i]);
                  cf->ranges.push_back(rng.empty() ? 0.0f : rng[i]);
                  if (frame_has_rgb) cf->rgbs.push_back(rgbs[i]);
                }
                frame_cache[fi->dir] = cf;
              }

              // Build voxel grid from cached frames
              struct VoxPt {
                Eigen::Vector3f wp;
                float intensity;
                float range;
                Eigen::Vector3f rgb;
                bool has_rgb;
                std::string dir;
              };
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
                  VoxPt vp;
                  vp.wp = cf->world_pts[i];
                  vp.intensity = cf->intensities[i];
                  vp.range = cf->ranges[i];
                  vp.has_rgb = cf->has_rgb;
                  vp.rgb = cf->has_rgb ? cf->rgbs[i] : Eigen::Vector3f::Zero();
                  vp.dir = cf->dir;
                  voxels[key].push_back(std::move(vp));
                }
              }

              // Process only core area voxels -- assign to frames round-robin
              int voxel_idx = 0;
              // Reuse the already-computed chunk_frame_infos instead of re-scanning all_frames
              std::vector<std::string> contributing_dirs;
              contributing_dirs.reserve(chunk_frame_infos.size());
              for (const auto* fi : chunk_frame_infos) contributing_dirs.push_back(fi->dir);

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

                // Average attributes. RGB averaged only over points that had RGB
                // (source frames missing aux_rgb.bin don't contribute); the flag
                // gates whether the voxel carries RGB at all.
                float avg_int = 0.0f, avg_rng = 0.0f;
                Eigen::Vector3f avg_rgb = Eigen::Vector3f::Zero();
                int rgb_n = 0;
                for (const auto& p : pts_in_voxel) {
                  avg_int += p.intensity; avg_rng += p.range;
                  if (p.has_rgb) { avg_rgb += p.rgb; rgb_n++; }
                }
                avg_int /= pts_in_voxel.size(); avg_rng /= pts_in_voxel.size();
                if (rgb_n > 0) avg_rgb /= static_cast<float>(rgb_n);

                // Assign to a frame -- round-robin across contributing frames
                const std::string& target_dir = contributing_dirs[voxel_idx % contributing_dirs.size()];
                frame_outputs[target_dir].points.push_back(pos);
                frame_outputs[target_dir].intensities.push_back(avg_int);
                frame_outputs[target_dir].ranges.push_back(avg_rng);
                frame_outputs[target_dir].rgbs.push_back(avg_rgb);
                frame_outputs[target_dir].rgb_present.push_back(rgb_n > 0 ? 1 : 0);
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
              // Write points as sensor-local (identity transform -- points are already in world space)
              // For the frame structure, store world-space points directly
              { std::ofstream f(out_dir + "/points.bin", std::ios::binary);
                f.write(reinterpret_cast<const char*>(output.points.data()), sizeof(Eigen::Vector3f) * n); }
              { std::ofstream f(out_dir + "/range.bin", std::ios::binary);
                f.write(reinterpret_cast<const char*>(output.ranges.data()), sizeof(float) * n); }
              if (vox_include_intensity) {
                std::ofstream f(out_dir + "/intensities.bin", std::ios::binary);
                f.write(reinterpret_cast<const char*>(output.intensities.data()), sizeof(float) * n);
              }
              // Only emit aux_rgb.bin if the flag is on AND at least one voxel
              // in this frame carried real RGB -- prevents a file of all-zero
              // fake colours downstream tools would happily paint on.
              if (vox_include_rgb) {
                bool any_rgb = false;
                for (uint8_t v : output.rgb_present) if (v) { any_rgb = true; break; }
                if (any_rgb) {
                  std::ofstream f(out_dir + "/aux_rgb.bin", std::ios::binary);
                  f.write(reinterpret_cast<const char*>(output.rgbs.data()), sizeof(Eigen::Vector3f) * n);
                }
              }

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

  // Livox intensity-0 filter tool
  viewer->register_ui_callback("livox_tool", [this] {
    if (!show_livox_tool) return;
    ImGui::SetNextWindowSize(ImVec2(380, 320), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Livox Intensity-0 Filter", &show_livox_tool)) {
      ImGui::TextDisabled("Livox Horizon 2nd-return detection via intensity == 0.");
      ImGui::Separator();

      const char* modes[] = {
        "Delete intensity 0",
        "Intensity 0 as 2nd return (write aux_second_return.bin)",
        "Intensity 0 interpolate (kNN from non-zero neighbors)"
      };
      ImGui::SetNextItemWidth(-1);
      ImGui::Combo("##livox_mode", &livox_mode, modes, IM_ARRAYSIZE(modes));
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Delete: removes intensity-0 points entirely (points.bin + parallel files shrink).\n"
        "Mark 2nd return: writes aux_second_return.bin per frame (1/0 flags). Keeps points.\n"
        "  Becomes a color-mode attribute on HD reload.\n"
        "Interpolate: replaces intensity of 0-points with kNN mean of non-zero neighbors.\n"
        "  Fills the intensity image; points stay in place.");
      if (livox_mode == 2) {
        ImGui::SetNextItemWidth(120);
        ImGui::DragFloat("kNN radius (m)##livox", &livox_interp_radius_m, 0.05f, 0.05f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Radius within which non-zero neighbors are averaged.");
      }

      ImGui::Separator();
      if (livox_running) {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", livox_status.c_str());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.25f, 0.25f, 1.0f));
        if (ImGui::Button(livox_cancel_requested ? "Stopping...##lv" : "Stop##lv")) livox_cancel_requested = true;
        ImGui::PopStyleColor();
        ImGui::End();
        return;
      }

      // ---- Preview on visible chunk ----
      if (ImGui::Button("Preview")) {
        livox_running = true; livox_cancel_requested = false;
        livox_status = "Loading frames near view center...";
        const int mode = livox_mode; const float radius = livox_interp_radius_m;
        livox_preview_data.clear();
        livox_preview_frame_dirs.clear();
        std::thread([this, mode, radius] {
          auto vw = guik::LightViewer::instance();
          const Eigen::Matrix4f vm = vw->view_matrix();
          const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));
          if (!trajectory_built) build_trajectory();
          double min_d = 1e9; size_t best = 0;
          for (size_t k = 0; k < trajectory_data.size(); k++) {
            const double d = (trajectory_data[k].pose.translation().cast<float>() - cam_pos).cast<double>().norm();
            if (d < min_d) { min_d = d; best = k; }
          }
          const Eigen::Vector3d c = trajectory_data[best].pose.translation();
          const size_t next = std::min(best + 1, trajectory_data.size() - 1);
          Eigen::Vector3d fwd = trajectory_data[next].pose.translation() - c;
          fwd.z() = 0; if (fwd.norm() < 0.01) fwd = Eigen::Vector3d::UnitX(); else fwd.normalize();
          const Eigen::Vector3d up = Eigen::Vector3d::UnitZ(), right = fwd.cross(up).normalized();
          Eigen::Matrix3d R; R.col(0) = fwd; R.col(1) = right; R.col(2) = up;
          glim::Chunk chunk{c, R, R.transpose(), 60.0, 50.0};  // 60m half-size chunk
          const auto chunk_aabb = chunk.world_aabb();

          std::vector<Eigen::Vector3f> kept_pts, removed_pts;
          std::vector<float> kept_ints, interp_ints;
          std::vector<uint8_t> kept_was_zero;  // parallel to kept_pts; 1 = was interpolated from zero
          std::vector<std::string> touched_dirs;
          size_t total_input = 0, total_zero = 0;

          for (const auto& submap : submaps) {
            if (livox_cancel_requested) break;
            if (!submap) continue;
            if (hidden_sessions.count(submap->session_id)) continue;
            std::string shd = hd_frames_path;
            for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
            const Eigen::Isometry3d T0 = submap->frames.front()->T_world_imu;
            for (const auto& fr : submap->frames) {
              if (livox_cancel_requested) break;
              char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
              const std::string fd = shd + "/" + dn;
              auto fi = glim::frame_info_from_meta(fd,
                glim::compute_frame_world_pose(submap->T_world_origin, submap->T_origin_endpoint_L, T0, fr->T_world_imu, fr->T_lidar_imu));
              if (fi.num_points == 0 || !chunk_aabb.intersects(fi.world_bbox)) continue;
              std::vector<Eigen::Vector3f> pts; std::vector<float> ints(fi.num_points, 0.0f), rng;
              if (!glim::load_bin(fd + "/points.bin", pts, fi.num_points)) continue;
              glim::load_bin(fd + "/intensities.bin", ints, fi.num_points);
              glim::load_bin(fd + "/range.bin", rng, fi.num_points);
              const Eigen::Matrix3f Rf = fi.T_world_lidar.rotation().cast<float>();
              const Eigen::Vector3f t = fi.T_world_lidar.translation().cast<float>();
              touched_dirs.push_back(fd);

              // Per-frame processing depending on mode
              if (mode == 2) {
                // Build voxel hash of ONLY non-zero-intensity points -- interpolation
                // must never pick up another zero-intensity neighbor (or zeros would
                // propagate into the replacement value).
                const float inv = 1.0f / std::max(0.05f, radius);
                std::unordered_map<uint64_t, std::vector<int>> vmap;
                for (int i = 0; i < fi.num_points; i++) {
                  if (ints[i] == 0.0f) continue;  // EXCLUDE zeros from neighbor set
                  const int vx = static_cast<int>(std::floor(pts[i].x() * inv));
                  const int vy = static_cast<int>(std::floor(pts[i].y() * inv));
                  const int vz = static_cast<int>(std::floor(pts[i].z() * inv));
                  vmap[glim::voxel_key(vx, vy, vz)].push_back(i);
                }
                const float r2 = radius * radius;
                for (int i = 0; i < fi.num_points; i++) {
                  if (!rng.empty() && rng[i] < 1.5f) continue;
                  const Eigen::Vector3f wp = Rf * pts[i] + t;
                  if (!chunk.contains(wp)) continue;
                  total_input++;
                  if (ints[i] != 0.0f) { kept_pts.push_back(wp); kept_ints.push_back(ints[i]); kept_was_zero.push_back(0); continue; }
                  total_zero++;
                  // kNN interpolate in FRAME-local space
                  const int vx = static_cast<int>(std::floor(pts[i].x() * inv));
                  const int vy = static_cast<int>(std::floor(pts[i].y() * inv));
                  const int vz = static_cast<int>(std::floor(pts[i].z() * inv));
                  double sum = 0; int cnt = 0;
                  for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++) {
                    auto it2 = vmap.find(glim::voxel_key(vx + dx, vy + dy, vz + dz));
                    if (it2 == vmap.end()) continue;
                    for (int nj : it2->second) {
                      if (ints[nj] == 0.0f) continue;  // defensive -- vmap already filtered, but keep explicit
                      if ((pts[nj] - pts[i]).squaredNorm() > r2) continue;
                      sum += ints[nj]; cnt++;
                    }
                  }
                  const float new_int = (cnt > 0) ? static_cast<float>(sum / cnt) : 0.0f;
                  interp_ints.push_back(new_int);
                  kept_pts.push_back(wp);
                  kept_ints.push_back(new_int);
                  kept_was_zero.push_back(1);
                }
              } else {
                for (int i = 0; i < fi.num_points; i++) {
                  if (!rng.empty() && rng[i] < 1.5f) continue;
                  const Eigen::Vector3f wp = Rf * pts[i] + t;
                  if (!chunk.contains(wp)) continue;
                  total_input++;
                  const bool is_zero = (ints[i] == 0.0f);
                  if (is_zero) total_zero++;
                  if (mode == 0) {
                    if (is_zero) removed_pts.push_back(wp);
                    else { kept_pts.push_back(wp); kept_ints.push_back(ints[i]); kept_was_zero.push_back(0); }
                  } else {  // mode == 1: mark but keep all in preview
                    if (is_zero) removed_pts.push_back(wp);
                    else { kept_pts.push_back(wp); kept_ints.push_back(ints[i]); kept_was_zero.push_back(0); }
                  }
                }
              }
            }
          }

          // Cache preview data for intensity toggle + Apply filter
          {
            std::vector<LivoxPreviewPoint> preview_data;
            preview_data.reserve(kept_pts.size());
            for (size_t i = 0; i < kept_pts.size(); i++) {
              preview_data.push_back({kept_pts[i], kept_ints[i], kept_was_zero[i] != 0});
            }
            livox_preview_data = std::move(preview_data);
            livox_preview_frame_dirs = std::move(touched_dirs);
            livox_intensity_mode = false;  // reset toggle on new preview
          }

          // Render
          vw->invoke([this, mode, kept_pts, removed_pts, kept_ints, total_input, total_zero] {
            auto v = guik::LightViewer::instance();
            lod_hide_all_submaps = true; rf_preview_active = true;
            v->remove_drawable("rf_preview_kept");
            v->remove_drawable("rf_preview_removed");
            if (!kept_pts.empty()) {
              const int n = kept_pts.size();
              std::vector<Eigen::Vector4d> p4(n);
              for (int i = 0; i < n; i++) p4[i] = Eigen::Vector4d(kept_pts[i].x(), kept_pts[i].y(), kept_pts[i].z(), 1.0);
              auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), p4.size());
              cb->add_buffer("intensity", kept_ints);
              cb->set_colormap_buffer("intensity");
              v->update_drawable("rf_preview_kept", cb, guik::FlatColor(0.0f, 0.8f, 0.2f, 1.0f));
            }
            if (!removed_pts.empty() && mode != 2) {
              const int n = removed_pts.size();
              std::vector<Eigen::Vector4d> p4(n);
              for (int i = 0; i < n; i++) p4[i] = Eigen::Vector4d(removed_pts[i].x(), removed_pts[i].y(), removed_pts[i].z(), 1.0);
              auto cb = std::make_shared<glk::PointCloudBuffer>(p4.data(), p4.size());
              v->update_drawable("rf_preview_removed", cb,
                guik::FlatColor(mode == 0 ? 1.0f : 0.9f, mode == 0 ? 0.0f : 0.4f, mode == 0 ? 0.0f : 0.1f, 0.7f).make_transparent());
            }
          });
          char buf[192]; std::snprintf(buf, sizeof(buf), "Preview: %zu points in chunk, %zu had intensity 0 (%.1f%%)",
            total_input, total_zero, total_input > 0 ? 100.0 * total_zero / total_input : 0.0);
          livox_status = buf;
          logger->info("[Livox] {}", livox_status);
          livox_running = false;
        }).detach();
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Preview the filter on the chunk around the current camera center.\n"
        "Green = kept (after filter), red/orange = intensity-0 (removed / flagged).");

      ImGui::SameLine();
      if (ImGui::Button("Apply filter##lv")) {
        // In-memory only -- hide the removed drawable so the preview shows the post-filter result.
        // Zero disk writes. To commit, use "Apply to all HD" below.
        auto v = guik::LightViewer::instance();
        v->remove_drawable("rf_preview_removed");
        livox_status = "Preview now showing post-filter result (removed points hidden).";
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Hide the red/orange (removed/flagged) points from the preview so you see what the\n"
        "post-filter cloud would look like. IN-MEMORY ONLY -- no disk writes.\n"
        "Use 'Apply to all HD' below to commit the filter to disk.");

      ImGui::SameLine();
      if (ImGui::Button("Intensity##lv")) {
        livox_intensity_mode = !livox_intensity_mode;
        auto v = guik::LightViewer::instance();
        auto drawable = v->find_drawable("rf_preview_kept");
        if (drawable.first) {
          if (livox_intensity_mode) {
            float imn = std::numeric_limits<float>::max(), imx = std::numeric_limits<float>::lowest();
            for (const auto& p : livox_preview_data) { imn = std::min(imn, p.intensity); imx = std::max(imx, p.intensity); }
            if (imn >= imx) { imn = 0.0f; imx = 255.0f; }
            v->shader_setting().add<Eigen::Vector2f>("cmap_range", Eigen::Vector2f(imn, imx));
            drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLORMAP);
          } else {
            drawable.first->set_color_mode(guik::ColorMode::FLAT_COLOR);
          }
        }
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Toggle between flat green and intensity colormap on the kept preview points.\n"
        "Use this to see the effect of interpolation (Mode 2) or just inspect intensity distribution.");

      ImGui::SameLine();
      if (ImGui::Button("Clear preview##lv")) {
        auto v = guik::LightViewer::instance();
        v->remove_drawable("rf_preview_kept");
        v->remove_drawable("rf_preview_removed");
        rf_preview_active = false;
        livox_intensity_mode = false;
        livox_preview_data.clear();
        livox_preview_frame_dirs.clear();
        lod_hide_all_submaps = false;
        livox_status.clear();
      }

      ImGui::Separator();
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.3f, 0.2f, 1.0f));
      if (ImGui::Button("Apply to all HD (destructive)")) {
        if (pfd::message("Confirm Apply",
            "This will modify HD frames on disk:\n"
            "  Mode 0: delete intensity-0 points in points.bin + all parallel .bin files\n"
            "  Mode 1: write aux_second_return.bin per frame (no existing data altered)\n"
            "  Mode 2: overwrite intensities.bin with kNN-interpolated values\n\n"
            "Backup first! Proceed?",
            pfd::choice::ok_cancel, pfd::icon::warning).result() == pfd::button::ok) {
          livox_running = true; livox_cancel_requested = false;
          livox_status = "Applying to all HD frames...";
          const int mode = livox_mode; const float radius = livox_interp_radius_m;
          std::thread([this, mode, radius] {
            const auto start_time = std::chrono::steady_clock::now();
            int frames_touched = 0;
            size_t total_zero = 0, total_pts = 0;

            for (const auto& submap : submaps) {
              if (livox_cancel_requested) break;
              if (!submap) continue;
              if (hidden_sessions.count(submap->session_id)) continue;
              std::string shd = hd_frames_path;
              for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
              for (const auto& fr : submap->frames) {
                if (livox_cancel_requested) break;
                char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                const std::string fd = shd + "/" + dn;
                const std::string meta = fd + "/frame_meta.json";
                if (!boost::filesystem::exists(meta)) continue;
                std::ifstream ifs(meta);
                const auto j = nlohmann::json::parse(ifs, nullptr, false);
                if (j.is_discarded()) continue;
                const int np = j.value("num_points", 0);
                if (np == 0) continue;

                std::vector<float> ints(np, 0.0f);
                if (!glim::load_bin(fd + "/intensities.bin", ints, np)) continue;

                if (mode == 0) {
                  // Delete: compute kept indices, rewrite all .bin files
                  std::vector<int> kept;
                  kept.reserve(np);
                  for (int i = 0; i < np; i++) if (ints[i] != 0.0f) kept.push_back(i);
                  const int new_count = static_cast<int>(kept.size());
                  total_zero += (np - new_count);
                  total_pts += np;
                  if (new_count == np) continue;  // nothing to delete
                  glim::filter_bin_file(fd + "/points.bin",      sizeof(Eigen::Vector3f), np, kept, new_count);
                  glim::filter_bin_file(fd + "/intensities.bin", sizeof(float),           np, kept, new_count);
                  glim::filter_bin_file(fd + "/range.bin",       sizeof(float),           np, kept, new_count);
                  glim::filter_bin_file(fd + "/times.bin",       sizeof(float),           np, kept, new_count);
                  glim::filter_bin_file(fd + "/normals.bin",     sizeof(Eigen::Vector3f), np, kept, new_count);
                  glim::filter_bin_file(fd + "/aux_ground.bin",  sizeof(float),           np, kept, new_count);
                  glim::filter_bin_file(fd + "/aux_rgb.bin",     sizeof(Eigen::Vector3f), np, kept, new_count);
                  // Update frame_meta.json num_points
                  nlohmann::json jm = j; jm["num_points"] = new_count;
                  std::ofstream om(meta); om << jm.dump(2);
                  frames_touched++;
                } else if (mode == 1) {
                  // Mark 2nd rebound: write aux_second_return.bin (float 0/1 per point)
                  std::vector<float> sec(np, 0.0f);
                  size_t z = 0;
                  for (int i = 0; i < np; i++) { if (ints[i] == 0.0f) { sec[i] = 1.0f; z++; } }
                  std::ofstream f(fd + "/aux_second_return.bin", std::ios::binary);
                  if (f) { f.write(reinterpret_cast<const char*>(sec.data()), sizeof(float) * np); }
                  total_zero += z; total_pts += np;
                  frames_touched++;
                } else {
                  // Interpolate: in-frame kNN replace of intensity-0 values
                  std::vector<Eigen::Vector3f> pts;
                  if (!glim::load_bin(fd + "/points.bin", pts, np)) continue;
                  const float inv = 1.0f / std::max(0.05f, radius);
                  std::unordered_map<uint64_t, std::vector<int>> vmap;
                  for (int i = 0; i < np; i++) {
                    if (ints[i] == 0.0f) continue;
                    const int vx = static_cast<int>(std::floor(pts[i].x() * inv));
                    const int vy = static_cast<int>(std::floor(pts[i].y() * inv));
                    const int vz = static_cast<int>(std::floor(pts[i].z() * inv));
                    vmap[glim::voxel_key(vx, vy, vz)].push_back(i);
                  }
                  const float r2 = radius * radius;
                  size_t z = 0, filled = 0;
                  for (int i = 0; i < np; i++) {
                    if (ints[i] != 0.0f) continue;
                    z++;
                    const int vx = static_cast<int>(std::floor(pts[i].x() * inv));
                    const int vy = static_cast<int>(std::floor(pts[i].y() * inv));
                    const int vz = static_cast<int>(std::floor(pts[i].z() * inv));
                    double sum = 0; int cnt = 0;
                    for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++) {
                      auto it2 = vmap.find(glim::voxel_key(vx + dx, vy + dy, vz + dz));
                      if (it2 == vmap.end()) continue;
                      for (int nj : it2->second) {
                        if (ints[nj] == 0.0f) continue;  // defensive -- vmap already filtered, but keep explicit
                        if ((pts[nj] - pts[i]).squaredNorm() > r2) continue;
                        sum += ints[nj]; cnt++;
                      }
                    }
                    if (cnt > 0) { ints[i] = static_cast<float>(sum / cnt); filled++; }
                  }
                  std::ofstream f(fd + "/intensities.bin", std::ios::binary);
                  f.write(reinterpret_cast<const char*>(ints.data()), sizeof(float) * np);
                  total_zero += z; total_pts += np;
                  frames_touched++;
                }
                if (frames_touched % 25 == 0) {
                  char b[128]; std::snprintf(b, sizeof(b), "Processed %d frames (%zu/%zu pts zero)...", frames_touched, total_zero, total_pts);
                  livox_status = b;
                }
              }
            }
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            char buf[192]; std::snprintf(buf, sizeof(buf), "Done: %d frames, %zu/%zu pts zero (%.1f%%), %.1f sec",
              frames_touched, total_zero, total_pts, total_pts > 0 ? 100.0 * total_zero / total_pts : 0.0, elapsed);
            livox_status = buf;
            logger->info("[Livox] {}", livox_status);
            livox_running = false;
          }).detach();
        }
      }
      ImGui::PopStyleColor();
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Apply to all HD frames in place. DESTRUCTIVE for Mode 0 (shrinks .bin files)\n"
        "and Mode 2 (overwrites intensities.bin). Mode 1 is purely additive.\n"
        "Backup HD frames first (Tools -> Utils -> Backup HD frames).");

      if (!livox_status.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", livox_status.c_str());
      }
    }
    ImGui::End();
  });

  // Batch processor window -- queue + reorder + run
  viewer->register_ui_callback("batch_process_window", [this] {
    if (!show_batch_window) return;
    ImGui::SetNextWindowSize(ImVec2(480, 420), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Batch Process", &show_batch_window)) {
      ImGui::TextDisabled("Queue apply-to-HD tasks. Each runs with current UI defaults.");
      ImGui::Separator();

      // Tool dropdown + Add button
      static const char* tool_names[] = {
        "SOR",
        "Range filter",
        "Dynamic (MapCleaner)",
        "Scalar filter",
        "Voxelize HD",
        "Livox: Delete intensity 0",
        "Livox: Mark 2nd return",
        "Livox: Interpolate",
      };
      ImGui::SetNextItemWidth(260);
      ImGui::Combo("##batchtool", &batch_selected_tool, tool_names, IM_ARRAYSIZE(tool_names));
      ImGui::SameLine();
      if (ImGui::Button("Add task##batch")) {
        batch_queue.push_back(static_cast<BatchTool>(batch_selected_tool));
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Append the selected tool to the queue.");

      ImGui::Separator();

      // Queue table
      if (batch_queue.empty()) {
        ImGui::TextDisabled("Queue is empty. Select a tool and click 'Add task'.");
      } else if (ImGui::BeginTable("##batchqueue", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("#", 0, 30.0f);
        ImGui::TableSetupColumn("Tool", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Move", 0, 70.0f);
        ImGui::TableSetupColumn("", 0, 60.0f);
        ImGui::TableHeadersRow();
        int to_remove = -1;
        int move_from = -1, move_to = -1;
        for (int i = 0; i < static_cast<int>(batch_queue.size()); i++) {
          ImGui::TableNextRow();
          if (batch_running && i == batch_current_task) {
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(70, 90, 40, 180));
          }
          ImGui::TableSetColumnIndex(0); ImGui::Text("%d", i + 1);
          ImGui::TableSetColumnIndex(1);
          ImGui::TextUnformatted(tool_names[static_cast<int>(batch_queue[i])]);
          ImGui::TableSetColumnIndex(2);
          ImGui::PushID(i);
          if (i > 0) {
            if (ImGui::SmallButton("^")) { move_from = i; move_to = i - 1; }
          } else {
            ImGui::TextDisabled(" ");
          }
          ImGui::SameLine();
          if (i < static_cast<int>(batch_queue.size()) - 1) {
            if (ImGui::SmallButton("v")) { move_from = i; move_to = i + 1; }
          } else {
            ImGui::TextDisabled(" ");
          }
          ImGui::TableSetColumnIndex(3);
          if (ImGui::SmallButton("X##rm")) { to_remove = i; }
          ImGui::PopID();
        }
        ImGui::EndTable();
        if (move_from >= 0 && move_to >= 0 && !batch_running) {
          std::swap(batch_queue[move_from], batch_queue[move_to]);
        }
        if (to_remove >= 0 && !batch_running) {
          batch_queue.erase(batch_queue.begin() + to_remove);
        }
      }

      ImGui::Separator();

      if (batch_running) {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Running [%d/%zu]: %s",
          batch_current_task + 1, batch_queue.size(), batch_status.c_str());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.25f, 0.25f, 1.0f));
        if (ImGui::Button(batch_cancel_requested ? "Stopping...##bt" : "Stop##bt")) {
          batch_cancel_requested = true;
        }
        ImGui::PopStyleColor();
      } else {
        const bool can_run = !batch_queue.empty();
        if (!can_run) ImGui::BeginDisabled();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.55f, 0.25f, 1.0f));
        if (ImGui::Button("Run batch")) {
          if (pfd::message("Confirm Batch Run",
              "Execute the queued tasks in order, each with current UI defaults?\n\n"
              "Several of these modify HD frames on disk. Backup first.",
              pfd::choice::ok_cancel, pfd::icon::warning).result() == pfd::button::ok) {
            batch_running = true;
            batch_cancel_requested = false;
            batch_current_task = 0;
            batch_status = "Starting...";
            std::thread([this] {
              for (size_t i = 0; i < batch_queue.size(); i++) {
                if (batch_cancel_requested) break;
                batch_current_task = static_cast<int>(i);
                const BatchTool t = batch_queue[i];
                char buf[96]; std::snprintf(buf, sizeof(buf), "[%zu/%zu] tool=%d", i + 1, batch_queue.size(), static_cast<int>(t));
                batch_status = buf;
                logger->info("[Batch] {}", batch_status);

                switch (t) {
                  case BatchTool::LivoxDelete0:
                  case BatchTool::LivoxMark2ndReturn:
                  case BatchTool::LivoxInterpolate: {
                    const int mode = (t == BatchTool::LivoxDelete0) ? 0 : (t == BatchTool::LivoxMark2ndReturn) ? 1 : 2;
                    const float radius = livox_interp_radius_m;
                    size_t total_zero = 0, total_pts = 0; int frames_touched = 0;
                    for (const auto& submap : submaps) {
                      if (batch_cancel_requested) break;
                      if (!submap) continue;
                      if (hidden_sessions.count(submap->session_id)) continue;
                      std::string shd = hd_frames_path;
                      for (const auto& s : sessions) { if (s.id == submap->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
                      for (const auto& fr : submap->frames) {
                        if (batch_cancel_requested) break;
                        char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                        const std::string fd = shd + "/" + dn;
                        const std::string meta = fd + "/frame_meta.json";
                        if (!boost::filesystem::exists(meta)) continue;
                        std::ifstream ifs(meta);
                        const auto j = nlohmann::json::parse(ifs, nullptr, false);
                        if (j.is_discarded()) continue;
                        const int np = j.value("num_points", 0);
                        if (np == 0) continue;
                        std::vector<float> ints(np, 0.0f);
                        if (!glim::load_bin(fd + "/intensities.bin", ints, np)) continue;
                        if (mode == 0) {
                          std::vector<int> kept; kept.reserve(np);
                          for (int k = 0; k < np; k++) if (ints[k] != 0.0f) kept.push_back(k);
                          const int nc = static_cast<int>(kept.size());
                          total_zero += (np - nc); total_pts += np;
                          if (nc == np) continue;
                          glim::filter_bin_file(fd + "/points.bin",      sizeof(Eigen::Vector3f), np, kept, nc);
                          glim::filter_bin_file(fd + "/intensities.bin", sizeof(float),           np, kept, nc);
                          glim::filter_bin_file(fd + "/range.bin",       sizeof(float),           np, kept, nc);
                          glim::filter_bin_file(fd + "/times.bin",       sizeof(float),           np, kept, nc);
                          glim::filter_bin_file(fd + "/normals.bin",     sizeof(Eigen::Vector3f), np, kept, nc);
                          glim::filter_bin_file(fd + "/aux_ground.bin",  sizeof(float),           np, kept, nc);
                          glim::filter_bin_file(fd + "/aux_rgb.bin",     sizeof(Eigen::Vector3f), np, kept, nc);
                          nlohmann::json jm = j; jm["num_points"] = nc;
                          std::ofstream om(meta); om << jm.dump(2);
                          frames_touched++;
                        } else if (mode == 1) {
                          std::vector<float> sec(np, 0.0f); size_t z = 0;
                          for (int k = 0; k < np; k++) if (ints[k] == 0.0f) { sec[k] = 1.0f; z++; }
                          std::ofstream f(fd + "/aux_second_return.bin", std::ios::binary);
                          if (f) f.write(reinterpret_cast<const char*>(sec.data()), sizeof(float) * np);
                          total_zero += z; total_pts += np; frames_touched++;
                        } else {
                          std::vector<Eigen::Vector3f> pts;
                          if (!glim::load_bin(fd + "/points.bin", pts, np)) continue;
                          const float inv = 1.0f / std::max(0.05f, radius);
                          std::unordered_map<uint64_t, std::vector<int>> vmap;
                          for (int k = 0; k < np; k++) {
                            if (ints[k] == 0.0f) continue;
                            const int vx = static_cast<int>(std::floor(pts[k].x() * inv));
                            const int vy = static_cast<int>(std::floor(pts[k].y() * inv));
                            const int vz = static_cast<int>(std::floor(pts[k].z() * inv));
                            vmap[glim::voxel_key(vx, vy, vz)].push_back(k);
                          }
                          const float r2 = radius * radius; size_t z = 0;
                          for (int k = 0; k < np; k++) {
                            if (ints[k] != 0.0f) continue;
                            z++;
                            const int vx = static_cast<int>(std::floor(pts[k].x() * inv));
                            const int vy = static_cast<int>(std::floor(pts[k].y() * inv));
                            const int vz = static_cast<int>(std::floor(pts[k].z() * inv));
                            double sum = 0; int cnt = 0;
                            for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) for (int dz = -1; dz <= 1; dz++) {
                              auto it2 = vmap.find(glim::voxel_key(vx + dx, vy + dy, vz + dz));
                              if (it2 == vmap.end()) continue;
                              for (int nj : it2->second) {
                                if (ints[nj] == 0.0f) continue;
                                if ((pts[nj] - pts[k]).squaredNorm() > r2) continue;
                                sum += ints[nj]; cnt++;
                              }
                            }
                            if (cnt > 0) ints[k] = static_cast<float>(sum / cnt);
                          }
                          std::ofstream f(fd + "/intensities.bin", std::ios::binary);
                          f.write(reinterpret_cast<const char*>(ints.data()), sizeof(float) * np);
                          total_zero += z; total_pts += np; frames_touched++;
                        }
                        if (frames_touched % 50 == 0) {
                          char b[160]; std::snprintf(b, sizeof(b),
                            "[%zu/%zu] Livox mode %d: %d frames, %zu/%zu zero",
                            i + 1, batch_queue.size(), mode, frames_touched, total_zero, total_pts);
                          batch_status = b;
                        }
                      }
                    }
                    logger->info("[Batch] Livox mode {}: {} frames touched, {}/{} pts zero", mode, frames_touched, total_zero, total_pts);
                    break;
                  }
                  case BatchTool::SOR:
                  case BatchTool::Range:
                  case BatchTool::Dynamic:
                  case BatchTool::Scalar:
                  case BatchTool::Voxelize: {
                    logger->warn("[Batch] Tool {} not yet wired for batch -- skipping. TODO: extract apply-to-HD body into a member function.", static_cast<int>(t));
                    // NOTE for Dynamic when wired: batch runs on raw datasets, so pass
                    // "recompute ground" semantics (not "reuse aux_ground.bin") -- the UI prompt
                    // the user sees interactively should be bypassed and default to recompute.
                    batch_status = std::string(tool_names[static_cast<int>(t)]) + ": NOT WIRED yet, skipped";
                    break;
                  }
                }
              }
              batch_running = false;
              batch_status = batch_cancel_requested ? "Cancelled." : "Batch complete.";
              logger->info("[Batch] {}", batch_status);
            }).detach();
          }
        }
        ImGui::PopStyleColor();
        if (!can_run) ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Clear queue##bt")) batch_queue.clear();
      }

      if (!batch_status.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("Status: %s", batch_status.c_str());
      }

      ImGui::Separator();
      ImGui::TextDisabled(
        "Wired tools: Livox modes (Delete / Mark / Interpolate).\n"
        "Stubbed: SOR / Range / Dynamic / Scalar / Voxelize -- logged as skipped.\n"
        "Extraction of those tools into batch-callable functions is a follow-up.");
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
      ImGui::DragFloat("Min density (pts/m^3)", &df_trail_min_density, 1.0f, 1.0f, 500.0f, "%.0f");
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
        if (ImGui::MenuItem("Auto-calibrate from this camera")) {
          ac_cam_src = src_idx;
          ac_cam_idx = frame_idx;
          ac_show = true;
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
            // Find submap by timestamp -- bracket match, else nearest boundary.
            const double cam_time = frame.timestamp + image_sources[src_idx].time_shift;
            const int best_sm = find_submap_for_timestamp(submaps, cam_time);
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
                // Expand spherical cams to 6 virtual cube-face pinhole cams.
                // No-op for pinhole sources.
                const auto& rc_src = image_sources[src_idx];
                const auto& rc_cp  = rc_src.params;
                auto expanded = expand_source_cams_for_projection(rc_src, {frame}, colorize_mask);
                auto cr = make_colorizer(rc_cp.view_selector_mode)->project(expanded.cams, expanded.intrinsics, world_pts, ints, std::vector<Eigen::Vector3f>{}, std::vector<double>{}, current_blend_params(rc_src));
                logger->info("[Colorize] {} / {} points colored (cams={})", cr.colored, cr.total, expanded.cams.size());
                colorize_last_result = cr;
                { auto vw = guik::LightViewer::instance(); lod_hide_all_submaps = true;
                  push_colorize_preview_drawable(vw, cr, rc_cp); }
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

    // Submap right-click -- add colorize option
    if (type == PickType::FRAME && !image_sources.empty()) {
      const int submap_id = right_clicked_info[3];
      if (submap_id >= 0 && submap_id < static_cast<int>(submaps.size()) && submaps[submap_id]) {
        ImGui::Separator();
        if (ImGui::MenuItem("Colorize submap")) {
          colorize_last_submap = submap_id; colorize_last_cam_src = -1; colorize_last_cam_idx = -1;
          const auto& sm = submaps[submap_id];
          // Select cameras by timestamp -- those within the submap's time range + margin
          const double t_first = sm->frames.front()->stamp;
          const double t_last = sm->frames.back()->stamp;
          const double t_margin = 1.0;  // 1 second before/after submap time range
          std::vector<CameraFrame> nearby_cams;
          // ALWAYS restrict to the active source, regardless of camera_type.
          // project() takes a single intrinsics + camera_type assumption; mixing
          // e.g. a Pinhole source with a Spherical one's cams in the same call
          // re-projects the Spherical cams with pinhole math -> broken output.
          // When we eventually support mixed-type blends we'll route each cam
          // through its source's own expand/intrinsics/mask here.
          {
            auto& src_i = image_sources[colorize_source_idx];
            for (auto& cam : src_i.frames) {
              if (!cam.located || cam.timestamp <= 0.0) continue;
              const double cam_t = cam.timestamp + effective_time_shift(src_i, cam.timestamp);
              if (cam_t >= t_first - t_margin && cam_t <= t_last + t_margin) {
                // Pre-shift .timestamp into LiDAR time so downstream time-slice compares
                // against world_point_times (also LiDAR-frame gps_time) consistently.
                CameraFrame shifted = cam;
                shifted.timestamp = cam_t;
                nearby_cams.push_back(std::move(shifted));
              }
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
              const Eigen::Matrix3d R_wo = T_wo.rotation();
              std::vector<Eigen::Vector3f> world_pts(hd->size());
              std::vector<float> ints(hd->size(), 0.0f);
              std::vector<Eigen::Vector3f> world_normals;
              std::vector<double> world_times;
              if (hd->normals) world_normals.resize(hd->size());
              if (hd->times)   world_times.assign(hd->times, hd->times + hd->size());
              for (size_t i = 0; i < hd->size(); i++) {
                world_pts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
                if (hd->normals)     world_normals[i] = (R_wo * Eigen::Vector3d(hd->normals[i].head<3>())).normalized().cast<float>();
              }
              logger->info("[Colorize] Mask status: empty={}, size={}x{}", colorize_mask.empty(), colorize_mask.cols, colorize_mask.rows);
              // Spherical sources: expand each nearby cam to 6 virtual cube faces.
              const auto& sm_src = image_sources[colorize_source_idx];
              const auto& sm_cp  = sm_src.params;
              auto expanded = expand_source_cams_for_projection(sm_src, nearby_cams, colorize_mask);
              auto cr = make_colorizer(sm_cp.view_selector_mode)->project(expanded.cams, expanded.intrinsics, world_pts, ints, world_normals, world_times, current_blend_params(sm_src));
              logger->info("[Colorize] {} / {} points colored from {} cameras ({} virtual)",
                cr.colored, cr.total, nearby_cams.size(), expanded.cams.size());
              colorize_last_result = cr;
              { auto vw = guik::LightViewer::instance(); lod_hide_all_submaps = true;
                // Grayscale intensity ramp here is a legacy behavior of the
                // right-click per-submap preview -- kept for parity, even
                // though the other previews use intensity_to_color().
                push_colorize_preview_drawable(vw, cr, sm_cp, /*use_grayscale_intensity=*/true); }
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
            const auto timed_traj = timed_traj_snapshot();
            const auto& placement_traj = trajectory_for(src, timed_traj);
            const Eigen::Isometry3d T_world_lidar = Colorizer::interpolate_pose(placement_traj, cam.timestamp + effective_time_shift(src, cam.timestamp));

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

        // Sphere size slider -- update all existing markers when changed
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
      for (size_t si = 0; si < image_sources.size(); si++) {
        for (size_t fi = 0; fi < image_sources[si].frames.size(); fi++) {
          if (!image_sources[si].frames[fi].located) continue;
          render_camera_gizmo(static_cast<int>(si), static_cast<int>(fi));
        }
      }
    }
    prev_cameras = draw_cameras;
  });

  // 3D gizmos for time-shift anchors. Each anchor becomes a cyan cone-pin at
  // the camera's world position at its cam_time (apex down = the anchor's
  // exact moment, base up = ~40 cm above by default). The row the user has
  // selected in the Alignment-check anchor table is scaled x10 on Z so it
  // stands up as a tall pillar -- easy to spot across a long track without
  // digging through the viewer. Placed in a dedicated callback so it refreshes
  // every frame (follows trajectory edits, re-sorts, anchor adds/removes).
  // Alt-trajectory polyline gizmo -- one polyline per source in 'Coords > own
  // path' mode. Drawable name prefixed with "traj_" so the existing Trajectory
  // checkbox hides it alongside the SLAM track. Tropical-blue colour makes it
  // easy to distinguish from the session-coloured SLAM polyline.
  viewer->register_ui_callback("camera_trajectory_gizmos", [this] {
    auto vw = guik::LightViewer::instance();
    static std::unordered_set<std::string> prev_traj_names;
    std::unordered_set<std::string> live_traj_names;
    const Eigen::Vector4f tropical(0.0f, 0.85f, 0.85f, 1.0f);
    for (size_t si = 0; si < image_sources.size(); si++) {
      const auto& s = image_sources[si];
      if (s.params.locate_mode != 2 || s.camera_trajectory.size() < 2) continue;
      std::vector<Eigen::Vector3f> verts;
      std::vector<Eigen::Vector4f> cols;
      verts.reserve(s.camera_trajectory.size());
      cols.reserve(s.camera_trajectory.size());
      for (const auto& tp : s.camera_trajectory) {
        verts.emplace_back(tp.pose.translation().cast<float>());
        cols.push_back(tropical);
      }
      auto line = std::make_shared<glk::ThinLines>(verts, cols, true);
      line->set_line_width(2.0f);
      const std::string name = "traj_cam_src_" + std::to_string(si);
      vw->update_drawable(name, line, guik::VertexColor());
      live_traj_names.insert(name);
    }
    // Prune drawables that no longer correspond to a mode-2 source (e.g. user
    // switched back to Time or removed a source).
    for (const auto& n : prev_traj_names) {
      if (!live_traj_names.count(n)) vw->remove_drawable(n);
    }
    prev_traj_names = std::move(live_traj_names);
  });

  viewer->register_ui_callback("anchor_gizmos", [this] {
    auto vw = guik::LightViewer::instance();
    // Build timed pose vector once (cheap -- avoids recomputing per anchor).
    // If the trajectory isn't built yet, skip; the next frame will retry.
    std::vector<TimedPose> timed_traj;
    if (trajectory_built && !trajectory_data.empty()) {
      timed_traj.reserve(trajectory_data.size());
      for (const auto& rec : trajectory_data) timed_traj.push_back({rec.stamp, rec.pose});
    }
    // Track names currently in use so we can prune stale ones (e.g. after
    // "Clear all" or anchor removal). Using a simple always-prune-then-draw
    // pattern: remember the max count we've seen and remove above the current.
    static std::map<int, int> last_anchor_count;  // src_idx -> count drawn last tick
    for (size_t si = 0; si < image_sources.size(); si++) {
      const auto& s = image_sources[si];
      const int prev_n = last_anchor_count[static_cast<int>(si)];
      for (int k = static_cast<int>(s.anchors.size()); k < prev_n; k++) {
        vw->remove_drawable("anchor_" + std::to_string(si) + "_" + std::to_string(k));
      }
      last_anchor_count[static_cast<int>(si)] = static_cast<int>(s.anchors.size());
      // Route per-source anchors through the alt trajectory when the source
      // is in 'Coords > own path' mode. Mode 0/1 falls through to SLAM.
      const auto& anchor_traj = trajectory_for(s, timed_traj);
      if (anchor_traj.empty()) continue;  // can't place without a trajectory yet
      const Eigen::Isometry3d T_lidar_cam = Colorizer::build_extrinsic(s.lever_arm, s.rotation_rpy);
      for (size_t ai = 0; ai < s.anchors.size(); ai++) {
        const auto& a = s.anchors[ai];
        const double ts = a.cam_time + a.time_shift;
        if (ts < anchor_traj.front().stamp - 2.0 || ts > anchor_traj.back().stamp + 2.0) continue;
        const Eigen::Isometry3d T_world_lidar = Colorizer::interpolate_pose(anchor_traj, ts);
        const Eigen::Isometry3d T_world_cam   = T_world_lidar * T_lidar_cam;
        const Eigen::Vector3f world_pos = T_world_cam.translation().cast<float>();
        // Pin geometry: cone primitive has apex at (0,0,1), base disc at z=0
        // radius 1. We want apex at anchor world position, base extending up.
        //   step 1 (innermost): scale(1,1,-1)   -> apex at z=-1, base at z=0
        //   step 2:            translate(0,0,1) -> apex at z=0,  base at z=1
        //   step 3:            scale(r,r,h)     -> base at z=h, radius r
        //   step 4 (outermost): translate(world_pos)
        const bool is_selected = (align_anchor_selected == static_cast<int>(ai) &&
                                   align_anchor_selected_src == static_cast<int>(si));
        // UNIFORM scaling so selection grows the whole cone in every axis (not
        // just a tall thin pillar) while keeping the tip glued to the anchor's
        // world position. Tip lives at the "origin" of the transform composition
        // below (see math in the initial gizmo implementation) so pure scale
        // here preserves it. Base 2x the previous; selected 5x uniform on top
        // (~20m tall pin, visibly fat across km-scale scenes).
        const float base_r = 1.0f;
        const float base_h = 4.0f;
        const float sel_mul = 5.0f;
        const float pin_r = is_selected ? base_r * sel_mul : base_r;
        const float pin_h = is_selected ? base_h * sel_mul : base_h;
        Eigen::Affine3f tf = Eigen::Affine3f::Identity();
        tf.translate(world_pos);
        tf.scale(Eigen::Vector3f(pin_r, pin_r, pin_h));
        tf.translate(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
        tf.scale(Eigen::Vector3f(1.0f, 1.0f, -1.0f));
        // White, full-alpha for the selected anchor; slightly dimmer off-selected
        // so the selection stands out against non-selected pins at the same size.
        const Eigen::Vector4f color = is_selected
          ? Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f)
          : Eigen::Vector4f(1.0f, 1.0f, 1.0f, 0.85f);
        vw->update_drawable("anchor_" + std::to_string(si) + "_" + std::to_string(ai),
          glk::Primitives::cone(),
          guik::FlatColor(color.x(), color.y(), color.z(), color.w(), tf));
      }
    }
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
    if (ImGui::Begin("Colorize", &show_colorize_window)) {
      if (image_sources.empty()) {
        ImGui::Text("No image sources loaded.\nUse Colorize > Image folder > Add folder...");
      } else {
        // Source selector
        std::vector<const char*> src_names;
        for (const auto& s : image_sources) src_names.push_back(s.name.c_str());
        static int prev_colorize_source_idx = -1;
        ImGui::Combo("Source", &colorize_source_idx, src_names.data(), src_names.size());

        auto& src = image_sources[colorize_source_idx];

        // When the dropdown index changes, reload colorize_mask from the NEW
        // source's mask_path. Without this, the previous source's mask (e.g.
        // a 7680x3840 equirect mask) sticks in the shared cv::Mat and gets
        // applied to the next source (pinhole) -- either failing the size
        // check or getting resized and used wrongly. ImageSource already owns
        // mask_path per-entry; this is just syncing the runtime texture with
        // the active selection.
        if (prev_colorize_source_idx != colorize_source_idx) {
          prev_colorize_source_idx = colorize_source_idx;
          colorize_mask = cv::Mat();
          if (!src.mask_path.empty() && boost::filesystem::exists(src.mask_path)) {
            colorize_mask = cv::imread(src.mask_path, cv::IMREAD_UNCHANGED);
            if (!colorize_mask.empty())
              logger->info("[Colorize] Swapped to source {} ({}), loaded mask {}",
                           colorize_source_idx, camera_type_label(src.camera_type), src.mask_path);
            else
              logger->warn("[Colorize] Swapped to source {} but mask failed to load: {}",
                           colorize_source_idx, src.mask_path);
          } else {
            logger->info("[Colorize] Swapped to source {} ({}), no mask", colorize_source_idx,
                         camera_type_label(src.camera_type));
          }
          colorize_intrinsics_dirty = true;  // invalidate live-preview cache
        }
        ImGui::Text("%zu images (%zu located)", src.frames.size(),
          std::count_if(src.frames.begin(), src.frames.end(), [](const CameraFrame& f) { return f.located; }));

        // Camera type. Pinhole uses Brown-Conrady via PinholeIntrinsics.
        // Spherical = 2:1 equirectangular panorama (Osmo 360 etc); intrinsics
        // degrade to image dimensions only, no fx/fy/cx/cy/distortion. The
        // colorize pipeline branches on this at projection time.
        {
          int ct = static_cast<int>(src.camera_type);
          ImGui::SetNextItemWidth(160);
          if (ImGui::Combo("Camera type", &ct, "Pinhole\0Spherical (equirect)\0")) {
            src.camera_type = static_cast<CameraType>(ct);
            colorize_intrinsics_dirty = true;  // force live-preview refresh
            // Re-render camera gizmos for THIS source so the cube/sphere icon
            // flips immediately.
            for (size_t fi = 0; fi < src.frames.size(); fi++) {
              if (!src.frames[fi].located) continue;
              render_camera_gizmo(colorize_source_idx, static_cast<int>(fi));
            }
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Pinhole: standard perspective lens (Facecam Pro etc). Uses fx/fy/cx/cy\n"
            "         + Brown-Conrady distortion from the Camera Intrinsics block.\n"
            "Spherical: full 360 equirectangular panorama (Osmo 360, Insta360 etc).\n"
            "         Image must be 2:1 aspect ratio. Intrinsics are unused --\n"
            "         focal is derived as f = width/(2*pi).");
          // Badge showing the active type prominently next to the combo.
          ImGui::SameLine();
          if (src.camera_type == CameraType::Spherical) {
            ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.0f, 1.0f), "[Spherical]");
          } else {
            ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.55f, 1.0f), "[Pinhole]");
          }
        }

        ImGui::Separator();

        // Location criteria -- stored per source so Pinhole / Spherical can
        // use different strategies without clobbering each other.
        const int prev_locate_mode = src.params.locate_mode;
        ImGui::Combo("Criteria", &src.params.locate_mode,
                      "Time\0Coords > SLAM\0Coords > own path\0");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "How to place this source's frames in the world.\n"
          "- Time: interpolate the SLAM trajectory at the frame's timestamp + time_shift.\n"
          "  Default for time-synced cameras.\n"
          "- Coords > SLAM: snap each frame's EXIF GPS to the nearest point on the SLAM\n"
          "  trajectory. Useful when camera timestamps aren't GPS-synced (e.g. GoPro\n"
          "  with GPS but independent camera clock).\n"
          "- Coords > own path: build an alternative trajectory from this source's EXIF\n"
          "  GPS track and interpolate THAT by time. Used for a second-pass GPS+camera\n"
          "  sweep (different lighting, etc.) over an existing SLAM map -- the source\n"
          "  rides its own path, not the original capture trajectory.");
        if (prev_locate_mode != src.params.locate_mode &&
            src.params.locate_mode == 2 && src.camera_trajectory.empty()) {
          // Lazy build on first switch to Coords > own path.
          build_camera_trajectory(src, gnss_utm_zone,
                                   gnss_utm_easting_origin, gnss_utm_northing_origin,
                                   gnss_datum_alt);
          if (src.camera_trajectory.empty()) {
            logger->warn("[Colorize] Source '{}' has no valid EXIF GPS timestamps; "
                         "'Coords > own path' will fall back to SLAM until EXIF is present.",
                         src.name);
          } else {
            logger->info("[Colorize] Built alt trajectory for '{}': {} points",
                         src.name, src.camera_trajectory.size());
          }
        }

        // Time shift with +/- step buttons. Bind InputDouble directly to
        // src.time_shift (double) so we don't lose precision through float;
        // 6-decimal display lets the user fine-tune sub-millisecond shifts to
        // chase sub-frame drift on dense sources (Osmo 360 equirects).
        bool time_changed = false;
        ImGui::SetNextItemWidth(150);
        if (ImGui::InputDouble("##ts", &src.time_shift, 0.0, 0.0, "%.6f")) time_changed = true;
        ImGui::SameLine();
        if (ImGui::Button("<")) { src.time_shift -= colorize_time_step; time_changed = true; }
        ImGui::SameLine();
        if (ImGui::Button(">")) { src.time_shift += colorize_time_step; time_changed = true; }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("step##ts_step", &colorize_time_step, 0.0f, 0.0f, "%.6f");
        ImGui::SameLine();
        ImGui::Text("Time shift");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Offset in seconds (6 decimal places visible). Use < > to step.\n"
          "Adjust to align camera timing with LiDAR. For dense sources where\n"
          "frames drift sub-frame over long tracks, sweep at 1e-4 s or finer.");

        // Anchor panel just below the Time shift widget. Resolution priority for
        // "Anchor here" cam_time: (1) alignment-check selected frame if on this
        // source, (2) last live-preview camera on this source, (3) last-colorize
        // submap center time. If none available, button is disabled.
        {
          double anchor_t = 0.0;
          bool   have_t = false;
          if (align_cam_src == colorize_source_idx &&
              align_cam_idx >= 0 &&
              align_cam_idx < static_cast<int>(src.frames.size()) &&
              src.frames[align_cam_idx].timestamp > 0.0) {
            anchor_t = src.frames[align_cam_idx].timestamp; have_t = true;
          } else if (colorize_last_cam_src == colorize_source_idx &&
                     colorize_last_cam_idx >= 0 &&
                     colorize_last_cam_idx < static_cast<int>(src.frames.size()) &&
                     src.frames[colorize_last_cam_idx].timestamp > 0.0) {
            anchor_t = src.frames[colorize_last_cam_idx].timestamp; have_t = true;
          } else if (colorize_last_submap >= 0 &&
                     colorize_last_submap < static_cast<int>(submaps.size()) &&
                     submaps[colorize_last_submap] &&
                     !submaps[colorize_last_submap]->frames.empty()) {
            // Submap preview: snap to the active source's camera CLOSEST in
            // LiDAR-time to the submap's center. Using the camera's source-local
            // cam.timestamp keeps every anchor in the same domain -- lower_bound
            // in effective_time_shift works across the full anchor list even if
            // some were placed from single-cam preview and others from submap
            // preview. Submap LiDAR-frame timestamp itself would be a domain
            // mismatch since cameras are queried by their own timestamps.
            const auto& sm_frames = submaps[colorize_last_submap]->frames;
            const double sm_center = 0.5 * (sm_frames.front()->stamp + sm_frames.back()->stamp);
            double best_dt = std::numeric_limits<double>::max();
            int best_fi = -1;
            for (size_t fi = 0; fi < src.frames.size(); fi++) {
              const auto& c = src.frames[fi];
              if (c.timestamp <= 0.0) continue;
              // Compare on the LiDAR-time side using effective_time_shift so
              // the match is accurate even when anchors are already defined
              // for this source (the scalar may be a scratched value).
              const double ct = c.timestamp + effective_time_shift(src, c.timestamp);
              const double dt = std::abs(ct - sm_center);
              if (dt < best_dt) { best_dt = dt; best_fi = static_cast<int>(fi); }
            }
            if (best_fi >= 0) {
              anchor_t = src.frames[best_fi].timestamp;  // source-local
              have_t = true;
            }
          }
          render_anchor_panel(colorize_source_idx, anchor_t, have_t, "colz");
        }
        // Separator between the time-shift group and the extrinsic (lever + rpy)
        // group -- they're different beasts and benefit from a visual break.
        ImGui::Separator();

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
        // Consume persistent intrinsic-dirty flag from previous frame (the
        // intrinsic input fields live further down in this callback, so their
        // change signal arrives one frame late -- we pick it up here).
        if (colorize_intrinsics_dirty) { time_changed = true; colorize_intrinsics_dirty = false; }

        // Live preview
        ImGui::Checkbox("Live preview", &colorize_live_preview);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Auto-update the 3D colorize preview when time_shift changes.\n"
          "Uses the last right-clicked camera or submap.\n"
          "\n"
          "When anchors are active on this source, the live preview uses the\n"
          "SCALAR time_shift (what the widget shows) so drag gives immediate\n"
          "feedback. 'Anchor here' commits it; Apply / Compute-assignment use\n"
          "anchor-interpolated values as usual.");

        ImGui::Separator();

        // Auto-sync src.time_shift to interpolated value when the live-preview
        // target cam changes (right-click "Colorize from this camera" on a new
        // frame). Mirrors the align-check side's auto-sync, so the widget
        // ALWAYS starts at the correct value for the previewed section before
        // the user drags. Only when anchors exist (otherwise scalar IS baseline).
        {
          static int prev_lp_cam_src = -1;
          static int prev_lp_cam_idx = -1;
          if (!src.anchors.empty() &&
              colorize_last_cam_src == colorize_source_idx &&
              colorize_last_cam_idx >= 0 &&
              colorize_last_cam_idx < static_cast<int>(src.frames.size()) &&
              src.frames[colorize_last_cam_idx].timestamp > 0.0 &&
              (prev_lp_cam_src != colorize_last_cam_src || prev_lp_cam_idx != colorize_last_cam_idx)) {
            src.time_shift = effective_time_shift(src, src.frames[colorize_last_cam_idx].timestamp);
            time_changed = true;  // propagate into live-preview trigger below
          }
          prev_lp_cam_src = colorize_last_cam_src;
          prev_lp_cam_idx = colorize_last_cam_idx;
        }

        // Auto re-colorize on time shift change with live preview
        if (time_changed && colorize_live_preview) {
          if (!trajectory_built) build_trajectory();
          const auto timed_traj = timed_traj_snapshot();

          // Re-locate only the preview cameras (not all)
          if (colorize_last_cam_src >= 0 && colorize_last_cam_idx >= 0 &&
              colorize_last_cam_src < static_cast<int>(image_sources.size())) {
            // Single camera preview -- re-locate just this camera (apply full extrinsic: lever + RPY).
            // Live preview uses the SCALAR src.time_shift (not effective_time_shift)
            // so that dragging the widget gives immediate 3D feedback even when
            // anchors are active. Matches the scratch-override model used in the
            // align-check overlay. Apply / Compute-assignment / non-live paths
            // still use effective_time_shift and honor anchors.
            auto& cam = image_sources[colorize_last_cam_src].frames[colorize_last_cam_idx];
            if (cam.timestamp > 0.0) {
              const double ts = cam.timestamp + src.time_shift;
              const auto& placement_traj = trajectory_for(src, timed_traj);
              const Eigen::Isometry3d T_world_lidar = Colorizer::interpolate_pose(placement_traj, ts);
              const Eigen::Isometry3d T_lidar_cam = Colorizer::build_extrinsic(src.lever_arm, src.rotation_rpy);
              cam.T_world_cam = T_world_lidar * T_lidar_cam;
              cam.located = true;
            }
            // Re-project -- find submap by timestamp (bracket, else nearest).
            const double lp_cam_time = cam.timestamp + src.time_shift;
            const int best_sm = find_submap_for_timestamp(submaps, lp_cam_time);
            if (best_sm >= 0) {
              auto hd = load_hd_for_submap(best_sm, false);
              if (hd && hd->size() > 0) {
                const Eigen::Isometry3d T_wo = submaps[best_sm]->T_world_origin;
                const Eigen::Matrix3d R_wo = T_wo.rotation();
                std::vector<Eigen::Vector3f> wpts(hd->size()); std::vector<float> ints(hd->size(), 0.0f);
                std::vector<Eigen::Vector3f> wnor;
                std::vector<double> wtimes;
                if (hd->normals) wnor.resize(hd->size());
                if (hd->times)   wtimes.assign(hd->times, hd->times + hd->size());
                for (size_t i = 0; i < hd->size(); i++) {
                  wpts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                  if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
                  if (hd->normals)     wnor[i] = (R_wo * Eigen::Vector3d(hd->normals[i].head<3>())).normalized().cast<float>();
                }
                CameraFrame shifted_cam = cam;
                shifted_cam.timestamp = cam.timestamp + src.time_shift;  // LiDAR-time (scratch)
                // Spherical source: expand to 6 cube faces before project().
                auto expanded = expand_source_cams_for_projection(src, {shifted_cam}, colorize_mask);
                const auto& lp_cp = src.params;
                auto cr = make_colorizer(lp_cp.view_selector_mode)->project(expanded.cams, expanded.intrinsics, wpts, ints, wnor, wtimes, current_blend_params(src));
                colorize_last_result = cr;
                { auto vw = guik::LightViewer::instance();
                  push_colorize_preview_drawable(vw, cr, lp_cp); }
              }
            }
          } else if (colorize_last_submap >= 0) {
            // Submap preview. Relocate the ACTIVE source only, using the scalar
            // time_shift (scratch). The previous implementation called
            // Colorizer::locate_by_time for every source, which runs through
            // effective_calib and uses anchor-interpolated time -- that left
            // camera WORLD POSITIONS frozen at the anchor value while the
            // nearby-cams filter below used the scratch value, so dragging
            // time_shift only shifted which cams were SELECTED, not where they
            // project from. Visually: a handful of cams flickering in/out of
            // the frame while the rest stay glued in place. Fixed by keeping
            // everything on the scratch rail for the active source. Other
            // sources are not being edited right now, so their last-located
            // positions stand (cheaper too).
            {
              auto& active_src = image_sources[colorize_source_idx];
              const Eigen::Isometry3d T_lidar_cam_active =
                Colorizer::build_extrinsic(active_src.lever_arm, active_src.rotation_rpy);
              const auto& placement_traj = trajectory_for(active_src, timed_traj);
              for (auto& c : active_src.frames) {
                if (c.timestamp <= 0.0) { c.located = false; continue; }
                const double ts = c.timestamp + active_src.time_shift;
                if (placement_traj.empty() ||
                    ts < placement_traj.front().stamp - 2.0 ||
                    ts > placement_traj.back().stamp + 2.0) {
                  c.located = false; continue;
                }
                const Eigen::Isometry3d T_world_lidar = Colorizer::interpolate_pose(placement_traj, ts);
                c.T_world_cam = T_world_lidar * T_lidar_cam_active;
                c.located = true;
              }
            }
            const auto& sm = submaps[colorize_last_submap];
            const double t_first = sm->frames.front()->stamp;
            const double t_last = sm->frames.back()->stamp;
            const double t_margin = 1.0;
            std::vector<CameraFrame> nearby;
            // ALWAYS restrict to the active source (see note in right-click
            // submap handler above). Mixing sources of different camera_type in
            // one project() call produces wrong projections for the off-type
            // source. Future mixed-type blend = per-cam routing through its own
            // expand/intrinsics/mask.
            {
              const auto& s = image_sources[colorize_source_idx];
              for (const auto& c : s.frames) {
                if (!c.located || c.timestamp <= 0.0) continue;
                // Live-preview scratch: scalar time_shift so drag propagates
                // immediately. Non-live paths still use effective_time_shift.
                const double ct = c.timestamp + s.time_shift;
                if (ct >= t_first - t_margin && ct <= t_last + t_margin) {
                  CameraFrame shifted = c;
                  shifted.timestamp = ct;  // LiDAR-time for time-slice comparisons
                  nearby.push_back(std::move(shifted));
                }
              }
            }
            if (!nearby.empty()) {
              auto hd = load_hd_for_submap(colorize_last_submap, false);
              if (hd && hd->size() > 0) {
                const Eigen::Isometry3d T_wo = sm->T_world_origin;
                const Eigen::Matrix3d R_wo = T_wo.rotation();
                std::vector<Eigen::Vector3f> wpts(hd->size()); std::vector<float> ints(hd->size(), 0.0f);
                std::vector<Eigen::Vector3f> wnor;
                std::vector<double> wtimes;
                if (hd->normals) wnor.resize(hd->size());
                if (hd->times)   wtimes.assign(hd->times, hd->times + hd->size());
                for (size_t i = 0; i < hd->size(); i++) {
                  wpts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                  if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
                  if (hd->normals)     wnor[i] = (R_wo * Eigen::Vector3d(hd->normals[i].head<3>())).normalized().cast<float>();
                }
                // Spherical source: expand nearby equirect cams to 6 cube faces each.
                auto expanded = expand_source_cams_for_projection(src, nearby, colorize_mask);
                const auto& lps_cp = src.params;
                auto cr = make_colorizer(lps_cp.view_selector_mode)->project(expanded.cams, expanded.intrinsics, wpts, ints, wnor, wtimes, current_blend_params(src));
                colorize_last_result = cr;
                { auto vw = guik::LightViewer::instance();
                  push_colorize_preview_drawable(vw, cr, lps_cp); }
              }
            }
          }
          // Update camera gizmo position for the preview camera.
          if (draw_cameras && colorize_last_cam_src >= 0 && colorize_last_cam_idx >= 0) {
            render_camera_gizmo(colorize_last_cam_src, colorize_last_cam_idx);
          }
        }

        // Locate button
        if (ImGui::Button("Locate along path")) {
          // Save colorize config on each locate. Single helper keeps this
          // serializer in lock-step with the "add image folder" serializer below.
          if (!loaded_map_path.empty()) {
            nlohmann::json cfg; cfg["sources"] = nlohmann::json::array();
            for (const auto& s : image_sources) cfg["sources"].push_back(image_source_to_json(s));
            std::ofstream ofs(loaded_map_path + "/colorize_config.json");
            ofs << std::setprecision(10) << cfg.dump(2);
          }
          if (!trajectory_built) build_trajectory();
          // Build timed pose vector from trajectory
          const auto timed_traj = timed_traj_snapshot();

          int count = 0;
          if (src.params.locate_mode == 0) {
            count = Colorizer::locate_by_time(src, timed_traj);
          } else if (src.params.locate_mode == 1) {
            count = Colorizer::locate_by_coordinates(src, timed_traj,
              gnss_utm_zone, gnss_utm_easting_origin, gnss_utm_northing_origin, gnss_datum_alt);
          } else {
            // Mode 2: build (or refresh) the source's own GPS-derived trajectory,
            // then place frames on THAT via the same time-based interp as mode 0.
            build_camera_trajectory(src, gnss_utm_zone,
                                     gnss_utm_easting_origin, gnss_utm_northing_origin,
                                     gnss_datum_alt);
            count = Colorizer::locate_by_time(src, trajectory_for(src, timed_traj));
          }
          logger->info("[Colorize] Located {} / {} cameras (source={} type={})",
            count, src.frames.size(), colorize_source_idx,
            src.camera_type == CameraType::Spherical ? "Spherical" : "Pinhole");
          draw_cameras = true;

          // Render camera gizmos
          if (draw_cameras) {
            auto vw = guik::LightViewer::instance();
            int cam_count = 0;
            for (size_t fi = 0; fi < src.frames.size(); fi++) {
              if (!src.frames[fi].located) continue;
              render_camera_gizmo(colorize_source_idx, static_cast<int>(fi));
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
          // Just turned off -- remove all gizmos
          auto vw = guik::LightViewer::instance();
          for (size_t si = 0; si < image_sources.size(); si++) {
            for (size_t fi = 0; fi < image_sources[si].frames.size(); fi++) {
              vw->remove_drawable("cam_" + std::to_string(si) + "_" + std::to_string(fi));
              vw->remove_drawable("cam_fov_" + std::to_string(si) + "_" + std::to_string(fi));
            }
          }
        } else if (draw_cameras && !prev_draw_cameras) {
          // Just turned on -- re-render all located cameras
          for (size_t si = 0; si < image_sources.size(); si++) {
            for (size_t fi = 0; fi < image_sources[si].frames.size(); fi++) {
              if (!image_sources[si].frames[fi].located) continue;
              render_camera_gizmo(static_cast<int>(si), static_cast<int>(fi));
            }
          }
        }
        prev_draw_cameras = draw_cameras;

        // Intrinsics (compact input fields). Hidden for Spherical sources -- the
        // equirect model needs no fx/fy/cx/cy/distortion; focal is derived from
        // image width (f = w/(2*pi)) and principal point is the image center.
        ImGui::Separator();
        if (src.camera_type == CameraType::Spherical) {
          ImGui::TextDisabled("Camera Intrinsics: not used for Spherical sources.");
          ImGui::TextDisabled("  f = width / (2 pi), principal point = image center.");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Equirectangular cameras have a fixed mathematical projection --\n"
            "no lens parameters to fit. Switch to 'Pinhole' to edit fx/fy/cx/cy.");
        }
        if (src.camera_type == CameraType::Pinhole && ImGui::CollapsingHeader("Camera Intrinsics")) {
          if (ImGui::Button("Import Metashape XML")) {
            auto files = pfd::open_file("Select Metashape calibration XML",
                                        src.path.empty() ? "." : src.path,
                                        {"Metashape calibration", "*.xml", "All files", "*"}).result();
            if (!files.empty()) {
              std::ifstream xf(files[0]);
              const std::string xml((std::istreambuf_iterator<char>(xf)), std::istreambuf_iterator<char>());
              auto pull = [&](const std::string& tag, double& out) {
                std::regex re("<" + tag + ">([^<]+)</" + tag + ">");
                std::smatch m;
                if (std::regex_search(xml, m, re)) { try { out = std::stod(m[1]); return true; } catch (...) {} }
                return false;
              };
              auto pulli = [&](const std::string& tag, int& out) {
                std::regex re("<" + tag + ">([^<]+)</" + tag + ">");
                std::smatch m;
                if (std::regex_search(xml, m, re)) { try { out = std::stoi(m[1]); return true; } catch (...) {} }
                return false;
              };
              int w = src.intrinsics.width, h = src.intrinsics.height;
              double f = 0, cx_off = 0, cy_off = 0;
              double k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
              const bool got_wh = pulli("width", w) && pulli("height", h);
              const bool got_f  = pull("f", f);
              if (!got_wh || !got_f) {
                logger->warn("[Intrinsics import] Missing required tags <width>/<height>/<f> in {}", files[0]);
              } else {
                pull("cx", cx_off); pull("cy", cy_off);  // Metashape: offset from image center
                pull("k1", k1); pull("k2", k2); pull("k3", k3);
                pull("p1", p1); pull("p2", p2);
                src.intrinsics.width  = w;
                src.intrinsics.height = h;
                src.intrinsics.fx = f;
                src.intrinsics.fy = f;   // Metashape frame projection uses single f
                src.intrinsics.cx = 0.5 * w + cx_off;   // convert offset -> absolute
                src.intrinsics.cy = 0.5 * h + cy_off;
                src.intrinsics.k1 = k1; src.intrinsics.k2 = k2; src.intrinsics.k3 = k3;
                src.intrinsics.p1 = p1; src.intrinsics.p2 = p2;
                logger->info("[Intrinsics import] Loaded from {}: {}x{}, f={:.2f}, cx={:.2f}, cy={:.2f}",
                             files[0], w, h, f, src.intrinsics.cx, src.intrinsics.cy);
              }
            }
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Load intrinsics from a Metashape calibration XML (<calibration><f>, <cx>, <cy>, <k1>...</>).\n"
            "Metashape stores cx/cy as OFFSETS from image center -- this converts to\n"
            "absolute pixel coordinates automatically. Single f value is copied to fx and fy.");
          ImGui::Separator();
          bool intrinsics_changed = false;
          float fx = static_cast<float>(src.intrinsics.fx), fy = static_cast<float>(src.intrinsics.fy);
          float cxv = static_cast<float>(src.intrinsics.cx), cyv = static_cast<float>(src.intrinsics.cy);
          // fx/fy/cx/cy: step=1 px, step_fast=10 px (Ctrl+click). Hold +/- to repeat.
          const float IW_FOC = 110.0f;   // field width wide enough to show value after the +/- buttons
          ImGui::SetNextItemWidth(IW_FOC); if (ImGui::InputFloat("fx##i", &fx, 1.0f, 10.0f, "%.0f")) { src.intrinsics.fx = fx; intrinsics_changed = true; }
          ImGui::SameLine(); ImGui::SetNextItemWidth(IW_FOC); if (ImGui::InputFloat("fy##i", &fy, 1.0f, 10.0f, "%.0f")) { src.intrinsics.fy = fy; intrinsics_changed = true; }
          ImGui::SetNextItemWidth(IW_FOC); if (ImGui::InputFloat("cx##i", &cxv, 1.0f, 10.0f, "%.0f")) { src.intrinsics.cx = cxv; intrinsics_changed = true; }
          ImGui::SameLine(); ImGui::SetNextItemWidth(IW_FOC); if (ImGui::InputFloat("cy##i", &cyv, 1.0f, 10.0f, "%.0f")) { src.intrinsics.cy = cyv; intrinsics_changed = true; }
          int iw = src.intrinsics.width, ih = src.intrinsics.height;
          ImGui::SetNextItemWidth(IW_FOC); if (ImGui::InputInt("W##i", &iw)) { src.intrinsics.width = iw; intrinsics_changed = true; }
          ImGui::SameLine(); ImGui::SetNextItemWidth(IW_FOC); if (ImGui::InputInt("H##i", &ih)) { src.intrinsics.height = ih; intrinsics_changed = true; }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Pinhole intrinsics. Default: Elgato Facecam Pro 90 deg FOV.\n+/- buttons hold-to-repeat. Ctrl+click uses fast step (10 px).");
          ImGui::Separator();
          ImGui::Text("Distortion (Brown-Conrady)");
          // User-tunable step for distortion coefficients (defaults 0.0005 -> fine
          // enough for calibration tweaks). Fast step is 10x this for Ctrl+click.
          ImGui::SetNextItemWidth(140);
          ImGui::DragFloat("Dist step##dstep", &intrinsics_dist_step, 0.00001f, 0.00001f, 0.01f, "%.5f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Step size for the +/- buttons on k1/k2/k3/p1/p2.\n"
            "Tweak down to ~1e-5 for fine convergence, up to ~1e-3 for rough search.\n"
            "Ctrl+click the +/- buttons to use 10x this step.");
          const float dstep_fast = intrinsics_dist_step * 10.0f;
          // Wider field so the %.6f value remains readable next to the +/- buttons.
          const float IW_DIST = 130.0f;
          float dk1 = static_cast<float>(src.intrinsics.k1), dk2 = static_cast<float>(src.intrinsics.k2);
          float dp1 = static_cast<float>(src.intrinsics.p1), dp2 = static_cast<float>(src.intrinsics.p2);
          float dk3 = static_cast<float>(src.intrinsics.k3);
          ImGui::SetNextItemWidth(IW_DIST); if (ImGui::InputFloat("k1##d", &dk1, intrinsics_dist_step, dstep_fast, "%.6f")) { src.intrinsics.k1 = dk1; intrinsics_changed = true; }
          ImGui::SameLine(); ImGui::SetNextItemWidth(IW_DIST); if (ImGui::InputFloat("k2##d", &dk2, intrinsics_dist_step, dstep_fast, "%.6f")) { src.intrinsics.k2 = dk2; intrinsics_changed = true; }
          ImGui::SetNextItemWidth(IW_DIST); if (ImGui::InputFloat("p1##d", &dp1, intrinsics_dist_step, dstep_fast, "%.6f")) { src.intrinsics.p1 = dp1; intrinsics_changed = true; }
          ImGui::SameLine(); ImGui::SetNextItemWidth(IW_DIST); if (ImGui::InputFloat("p2##d", &dp2, intrinsics_dist_step, dstep_fast, "%.6f")) { src.intrinsics.p2 = dp2; intrinsics_changed = true; }
          ImGui::SetNextItemWidth(IW_DIST); if (ImGui::InputFloat("k3##d", &dk3, intrinsics_dist_step, dstep_fast, "%.6f")) { src.intrinsics.k3 = dk3; intrinsics_changed = true; }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Radial (k1,k2,k3) and tangential (p1,p2) distortion.\nImport from Metashape or calibration tool.\nLeave zeros for no distortion.");
          // Live preview trigger for intrinsic tweaks: persist a flag so the top
          // of NEXT frame sees it (the time_changed check runs earlier in the
          // callback than this CollapsingHeader). Lets user eyeball calib by
          // dragging fx/cx/k1/etc and watching projection shift in real time.
          if (intrinsics_changed) colorize_intrinsics_dirty = true;
        }

        ImGui::DragFloat("Min range (m)", &src.params.min_range, 0.5f, 0.0f, 50.0f, "%.1f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Skip points closer than this to the camera.\nReduces close-up distortion artifacts.");
        ImGui::DragFloat("Max range (m)", &src.params.max_range, 1.0f, 5.0f, 200.0f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max distance from camera to project points.\nCloser = faster, further = more coverage.");
        // View selector strategy (which camera colors which point). Bind combo
        // to an int temporary so we can round-trip to the enum cleanly.
        {
          const char* vs_labels[] = {
            "Simple nearest",
            "Weighted top-1",
            "Weighted top-K",
            "Matched (WIP)"
          };
          int vs_mode_int = static_cast<int>(src.params.view_selector_mode);
          ImGui::SetNextItemWidth(180);
          if (ImGui::Combo("View selector", &vs_mode_int, vs_labels, IM_ARRAYSIZE(vs_labels))) {
            src.params.view_selector_mode = static_cast<ViewSelectorMode>(vs_mode_int);
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "How to decide which camera's color each point gets.\n"
            "\n"
            "- Simple nearest   : closest-distance camera wins per point. Fast baseline.\n"
            "- Weighted top-1   : per-(point,cam) score = w_range x w_center x w_incidence.\n"
            "                     One winner per point. Best default when filters are on.\n"
            "- Weighted top-K   : softmax over K highest-scoring cameras. Gentler blending.\n"
            "- Matched          : future (LightGlue + per-pixel warp + semantic gating).\n"
            "\n"
            "Time-slice filter below can restrict candidates BEFORE any of these run:\n"
            "Simple nearest + Time-slice hard = the FAST-LIVO2 pairing strategy.");
        }
        // Weighted-mode knobs (visible only when weighted)
        const auto vs_mode_enum = src.params.view_selector_mode;
        if (vs_mode_enum == ViewSelectorMode::WeightedTop1 || vs_mode_enum == ViewSelectorMode::WeightedTopK) {
          ImGui::Indent();
          ImGui::SetNextItemWidth(100); ImGui::DragFloat("range tau (m)", &src.params.range_tau, 0.1f, 0.5f, 100.0f, "%.1f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Range weight: w_range = exp(-dist / tau).\n"
            "Smaller tau -> cameras close to the point dominate more aggressively.\n"
            "Larger tau -> distance matters less. tau=10m is a good default.");
          ImGui::SetNextItemWidth(100); ImGui::DragFloat("center exp", &src.params.center_exp, 0.05f, 0.0f, 8.0f, "%.2f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Image-center weight: w_center = (1 - r^2)^exp, where r is the point's\n"
            "normalized image-plane radius from principal point (0 at center, 1 at edge).\n"
            "Penalizes edges where lens distortion residuals are largest.\n"
            "Set to 0 to disable. Typical 1.5-3.");
          ImGui::SetNextItemWidth(100); ImGui::DragFloat("incidence exp", &src.params.incidence_exp, 0.05f, 0.0f, 8.0f, "%.2f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Incidence weight: w_incidence = max(0, |cos theta|)^exp, where theta is the angle\n"
            "between the surface normal and the ray to the camera. Favors face-on views,\n"
            "demotes grazing. |.| is used because PCA normals have arbitrary sign.\n"
            "Requires per-point normals. Set to 0 to disable. Typical 1-2.");
          if (vs_mode_enum == ViewSelectorMode::WeightedTopK) {
            ImGui::SetNextItemWidth(80); ImGui::SliderInt("top-K", &src.params.topK, 2, 8);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Number of top cameras to softmax-blend per point. 2-3 typical.");
          }
          ImGui::Unindent();
        }

        // Correctness filters -- each one independent so you can A/B the impact.
        ImGui::Separator();
        ImGui::TextDisabled("Correctness filters (independent toggles)");

        // ----- 1. Time-slice per-point camera pairing (FAST-LIVO2 style) -----
        {
          ImGui::Checkbox("Time-slice (hard)", &src.params.time_slice_hard);
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "FAST-LIVO2 pairing. For each LiDAR point (using its per-point GPS time),\n"
            "use ONLY the camera whose timestamp is closest in time to that point.\n"
            "No other camera may contribute -- this eliminates multi-view blending smear.\n"
            "\n"
            "Best default for high-framerate cameras (30 fps = <=17ms drift per point).\n"
            "Requires per-point times; loads from hd->times (gps_time). If unavailable, no-op.\n"
            "Applies to BOTH Simple and Weighted modes.");
          ImGui::SameLine();
          ImGui::Checkbox("Time-slice (soft)", &src.params.time_slice_soft);
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Weighted-only. Adds a w_time = exp(-|dt| / sigma) term into the weighted blend\n"
            "instead of hard-selecting one camera. Gentler than hard mode -- cameras farther\n"
            "in time still contribute but less. Can combine with hard if you want the\n"
            "closest camera to dominate a fallback-capable blend.");
          if (src.params.time_slice_soft) {
            ImGui::Indent();
            ImGui::SetNextItemWidth(100);
            ImGui::DragFloat("time sigma (s)", &src.params.time_slice_sigma, 0.005f, 0.005f, 2.0f, "%.3fs");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Time-decay scale for SOFT mode. Smaller sigma = sharper time preference.\n"
              "Typical: 30fps camera -> sigma ~= 0.03s. 10fps camera -> sigma ~= 0.1s.");
            ImGui::Unindent();
          }
        }

        // ----- 2. Depth-buffer occlusion -----
        {
          ImGui::Checkbox("Depth-buffer occlusion", &src.params.use_occlusion);
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Per-camera z-buffer built before color sampling. Rejects any point whose\n"
            "depth is larger than zbuf[u,v] x (1 + tolerance) -- i.e. hidden behind\n"
            "a closer point at that pixel.\n"
            "\n"
            "This is the single biggest fix for 'color bleeding through thin structures'\n"
            "(fences, poles, tree branches, wires). Applies to BOTH Simple and Weighted.\n"
            "Cost: ~15%% slower projection, one extra z-buffer per camera.");
          if (src.params.use_occlusion) {
            ImGui::Indent();
            ImGui::SetNextItemWidth(100); ImGui::DragFloat("tolerance", &src.params.occlusion_tolerance, 0.005f, 0.005f, 0.5f, "%.3f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Slack as fraction of depth. 0.05 = a point 5%% farther than the\n"
              "z-buffer value is still accepted. Too tight -> valid points get rejected\n"
              "(LiDAR sparsity makes same-surface neighbors land at slightly different depths).\n"
              "Too loose -> occluders bleed through again. 0.03-0.08 is typical.");
            ImGui::SameLine(); ImGui::SetNextItemWidth(60);
            ImGui::DragInt("zbuf /N", &src.params.occlusion_downscale, 1, 1, 16);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Z-buffer resolution divisor. 4 = zbuf is image_w/4 x image_h/4.\n"
              "Larger N = faster + less memory, but less precise at thin structure edges.\n"
              "For 4K images, N=4 gives a ~1000x540 zbuf which is plenty.");
            ImGui::Unindent();
          }
        }

        // ----- 3. Hard incidence gate -----
        {
          ImGui::Checkbox("Hard incidence gate", &src.params.use_incidence_hard);
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Reject (point, camera) pairs where the camera views the surface at a\n"
            "grazing angle -- i.e. angle between surface normal and camera ray exceeds threshold.\n"
            "\n"
            "This does NOT reject close/far points. It rejects viewing DIRECTIONS where\n"
            "each pixel covers a huge area and color gets smeared. FAST-LIVO2 uses a\n"
            "hard 60 deg gate. Example: a road point 30m ahead viewed by a forward-facing\n"
            "camera hits the ground normal at ~85 deg -- grazing, gets dropped.\n"
            "\n"
            "Weighted mode only. Requires per-point normals (loaded from normals.bin).\n"
            "If a point has no covering camera after this gate, it stays uncolored\n"
            "(intensity-gray fallback).");
          if (src.params.use_incidence_hard) {
            ImGui::Indent();
            ImGui::SetNextItemWidth(100);
            ImGui::DragFloat("max angle deg", &src.params.incidence_hard_deg, 1.0f, 10.0f, 89.0f, "%.0f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Views with angle between surface normal and ray larger than this are dropped.\n"
              "60 deg is standard (FAST-LIVO2 default). Lower = stricter + more uncolored points\n"
              "at distance. Higher = more grazing contamination.");
            ImGui::Unindent();
          }
        }

        // ----- 4. NCC cross-check -----
        {
          ImGui::Checkbox("NCC cross-check (top-K only)", &src.params.use_ncc);
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Weighted top-K mode only. After the top-K candidate cameras are picked,\n"
            "patch-match each non-winner against the top-1 winner using normalized\n"
            "cross-correlation; drop non-winners whose NCC < threshold.\n"
            "\n"
            "Catches motion-blurred or time-desynced frames contributing bad color.\n"
            "Typically small quality delta in post-processing (more useful in streaming VO).\n"
            "No effect in Simple or Weighted top-1 modes -- nothing to cross-check.");
          if (src.params.use_ncc) {
            ImGui::Indent();
            ImGui::SetNextItemWidth(100);
            ImGui::DragFloat("NCC thr", &src.params.ncc_threshold, 0.01f, -0.5f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "NCC value below which a non-winner candidate is dropped from the blend.\n"
              "Range [-1, 1]. Typical: 0.3. Higher = stricter (fewer contributors survive).");
            ImGui::SameLine(); ImGui::SetNextItemWidth(60);
            ImGui::DragInt("patch half", &src.params.ncc_half, 1, 1, 10);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Patch size = 2*half + 1. half=3 -> 7x7 patch. Larger = more robust but slower.");
            ImGui::Unindent();
          }
        }

        // ----- 5. Per-image exposure normalization (FAST-LIVO2 trick #1) -----
        {
          ImGui::Checkbox("Exposure normalize", &src.params.use_exposure_norm);
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Per-image brightness gain. Keeps color consistent across frames whose auto-exposure drifts.\n"
            "\n"
            "Two modes (see 'Simple' checkbox):\n"
            "- Simple OFF (surface-pixel):  mean over pixels where LiDAR points project.\n"
            "  Unbiased by sky/ceiling but tends to over-boost sunny outdoor scenes\n"
            "  (surface samples are darker than sky, so gain pushes images too bright).\n"
            "- Simple ON  (image-mean):     mean over the whole downscaled image.\n"
            "  Slightly biased by sky/ceiling, but moderate gain -- visually balanced\n"
            "  outdoors (doesn't burn sunny areas). This is the legacy behavior.\n"
            "\n"
            "Gain is clamped to [0.25, 4.0]. Works with Simple + Weighted selectors.");
          if (src.params.use_exposure_norm) {
            ImGui::Indent();
            ImGui::SetNextItemWidth(100);
            ImGui::DragFloat("target mean", &src.params.exposure_target, 0.01f, 0.1f, 0.9f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Target mean brightness in [0,1] (0.5 = neutral gray). All images get normalized\n"
              "to this. Lower (0.3) = darker final result, higher (0.7) = brighter.");
            ImGui::SameLine();
            ImGui::Checkbox("Simple##expsimp", &src.params.exposure_simple);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "ON:  downscaled image mean (legacy, balanced for sunny outdoor scenes).\n"
              "OFF: surface-pixel mean (unbiased by sky/ceiling, but can over-boost sunny data).\n"
              "\n"
              "If outdoor sunny frames come out burned with 'OFF', turn this ON\n"
              "(or lower target mean to ~0.4).");
            ImGui::Unindent();
          }
        }

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
          // For Spherical sources, intrinsics.width/height aren't used for image
          // dimensions -- cube-face slicing handles sampling regardless of the
          // equirect's pixel count. Skip the mismatch warning in that case; the
          // mask is resized to the image at sample time anyway (see Preview below).
          const bool check_size = (src.camera_type == CameraType::Pinhole);
          const bool mask_size_ok = !check_size ||
            (colorize_mask.cols == src.intrinsics.width && colorize_mask.rows == src.intrinsics.height);
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
          if (ImGui::SmallButton("Clear mask")) { colorize_mask = cv::Mat(); src.mask_path.clear(); }
        } else {
          ImGui::SameLine();
          ImGui::TextDisabled("No mask (place mask.png in image folder)");
        }

        ImGui::Separator();
        bool blend_changed = false;
        if (ImGui::Checkbox("Intensity blend", &src.params.intensity_blend)) blend_changed = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Blend projected RGB with LiDAR intensity.\nUseful for alignment verification using road markings.");
        if (src.params.intensity_blend) {
          if (ImGui::SliderFloat("Mix##intblend", &src.params.intensity_mix, 0.0f, 1.0f, "%.2f")) blend_changed = true;
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("0 = pure RGB, 1 = pure intensity.");
          if (ImGui::Checkbox("Non-linear intensity", &src.params.nonlinear_int)) blend_changed = true;
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Compress intensity range (sqrt) to boost\nroad marking contrast. Useful for Livox scanners.");
        }
        // Re-render preview with new blend without re-projecting.
        if (blend_changed && !colorize_last_result.points.empty()) {
          auto vw = guik::LightViewer::instance();
          push_colorize_preview_drawable(vw, colorize_last_result, src.params);
        }

        // Cube-face cache cap (Spherical only uses it). FIFO-evicts when total
        // bytes exceed the cap. 8 GB default keeps long-session previews from
        // eating all RAM -- bump if you have headroom and want bigger caches.
        {
          float cap_gb = static_cast<float>(preview_cache_cap_gb);
          ImGui::SetNextItemWidth(100);
          if (ImGui::DragFloat("Cache cap (GB)##cap", &cap_gb, 0.5f, 0.0f, 128.0f, "%.1f")) {
            preview_cache_cap_gb = static_cast<double>(std::max(0.0f, cap_gb));
            g_cube_face_cache_cap_bytes.store(
              static_cast<size_t>(preview_cache_cap_gb * 1024.0 * 1024.0 * 1024.0),
              std::memory_order_relaxed);
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "RAM cap for the spherical cube-face cache (not VRAM).\n"
            "Each equirect frame caches as ~66 MB of cube faces; the cache grows\n"
            "as Preview / Apply touch more frames. When total exceeds this cap,\n"
            "oldest entries are evicted FIFO.\n"
            "\n"
            "0 = no cap (previous behavior, grow unbounded).");
          // Live stats: total bytes + frame count, so user sees current pressure.
          size_t bytes = 0, frames = 0;
          get_cube_face_cache_stats(bytes, frames);
          ImGui::SameLine();
          ImGui::TextDisabled("%.2f GB, %zu frames cached",
            static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), frames);
        }

        // Colorize all submaps + clear
        static bool colorize_all_running = false;
        static std::string colorize_all_status;
        if (colorize_all_running) {
          ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", colorize_all_status.c_str());
          ImGui::SameLine();
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.25f, 0.25f, 1.0f));
          if (ImGui::Button(colorize_all_cancel_requested ? "Stopping...##cap" : "Stop##cap")) {
            colorize_all_cancel_requested = true;
          }
          ImGui::PopStyleColor();
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Stop after the current submap. Submaps already colorized stay visible.\nClear preview to remove them.");
        } else {
          if (ImGui::Button("Colorize all submaps (preview)")) {
            colorize_all_running = true;
            colorize_all_cancel_requested = false;
            colorize_all_status = "Starting...";
            const auto& isrc = image_sources[colorize_source_idx];
            // Snapshot the 3D viewer's camera world position so the worker can
            // order submaps nearest-first. Must be taken on the UI thread.
            // The camera-control exposes view_matrix() (world->cam); the world
            // position sits at -R^T * t when decomposed, or equivalently the
            // inverse's translation block.
            Eigen::Vector3f view_pos(0.0f, 0.0f, 0.0f);
            {
              auto vw = guik::LightViewer::instance();
              if (auto cam = vw->get_camera_control()) {
                const Eigen::Matrix4f vm = cam->view_matrix();
                view_pos = vm.inverse().block<3, 1>(0, 3);
              }
            }
            std::thread([this, &isrc, view_pos] {
              if (!trajectory_built) build_trajectory();
              const auto timed_traj = timed_traj_snapshot();

              // Per-submap preview: emit one drawable per submap instead of accumulating.
              // Memory scales with a single submap (~500k pts) not the full map, so this
              // handles 100km+ trajectories without a single giant VBO blowing up.
              // Side effects: intensity-blend slider re-render is skipped (would need to
              // store per-submap cr copies, which defeats the point). Use single-submap
              // colorize if you need blend interactivity.
              size_t total_colored = 0;
              size_t total_rendered_pts = 0;
              size_t sm_drawables_made = 0;

              // Set up one-time preview state on the main thread.
              guik::LightViewer::instance()->invoke([this] {
                lod_hide_all_submaps = true;
                colorize_last_submap = -1;
                colorize_last_cam_src = -1;
                colorize_last_cam_idx = -1;
                // Clear any previous per-submap preview drawables
                auto vw = guik::LightViewer::instance();
                for (int sid : colorize_preview_sm_ids) vw->remove_drawable("colorize_preview_sm_" + std::to_string(sid));
                colorize_preview_sm_ids.clear();
                // Also clear the legacy single drawable in case it exists from an older run
                vw->remove_drawable("colorize_preview");
                // Intensity blend slider has nothing to re-render in per-submap mode
                colorize_last_result = ColorizeResult{};
              });

              // Submap visit order: nearest-to-view-camera first, then follow the
              // trajectory forward from there, wrapping around. Rationale: a
              // user tuning a specific section is usually looking at it in the
              // 3D view; if the cache fills or they cancel, the region they can
              // actually see gets colorized before far-away sections. On cold
              // start (no cache pressure) the perceived responsiveness is the
              // same -- the first colored submap lands right where they're
              // looking instead of at submap 0.
              const int nsm = static_cast<int>(submaps.size());
              int start_si = 0;
              {
                double best_dsq = std::numeric_limits<double>::max();
                for (int si = 0; si < nsm; si++) {
                  const auto& sm = submaps[si];
                  if (!sm || sm->frames.empty()) continue;
                  const Eigen::Vector3f c = sm->T_world_origin.translation().cast<float>();
                  const double dsq = (c - view_pos).squaredNorm();
                  if (dsq < best_dsq) { best_dsq = dsq; start_si = si; }
                }
              }
              // Build visit order: [start, start+1, ..., nsm-1, 0, 1, ..., start-1].
              std::vector<int> visit_order;
              visit_order.reserve(nsm);
              for (int k = 0; k < nsm; k++) visit_order.push_back((start_si + k) % nsm);

              for (int vi = 0; vi < static_cast<int>(visit_order.size()); vi++) {
                if (colorize_all_cancel_requested) break;  // Stop requested
                const int si = visit_order[vi];
                const auto& sm = submaps[si];
                if (!sm || sm->frames.empty()) continue;
                if (hidden_sessions.count(sm->session_id)) continue;

                char buf[96]; std::snprintf(buf, sizeof(buf), "Submap %d (%d/%zu visited, starting from nearest #%d)...",
                  si, vi + 1, visit_order.size(), start_si);
                colorize_all_status = buf;

                // Find cameras for this submap by timestamp. ALWAYS restrict to
                // the active source (isrc) -- same reasoning as right-click and
                // live-preview submap handlers: mixing camera_types in one
                // project() call breaks the off-type cams. Previous version
                // looped over all sources and passed them raw through project()
                // with isrc.intrinsics -- Spherical cams in a Pinhole run
                // projected as if pinhole (bogus), and Spherical active never
                // got cube-face expansion because expand_source_cams_for_projection
                // was not called here at all.
                const double t0 = sm->frames.front()->stamp, t1 = sm->frames.back()->stamp;
                std::vector<CameraFrame> cams;
                for (const auto& cam : isrc.frames) {
                  if (!cam.located || cam.timestamp <= 0) continue;
                  const double ct = cam.timestamp + effective_time_shift(isrc, cam.timestamp);
                  if (ct >= t0 - 1.0 && ct <= t1 + 1.0) {
                    CameraFrame shifted = cam;
                    shifted.timestamp = ct;  // LiDAR-time for time-slice comparisons
                    cams.push_back(std::move(shifted));
                  }
                }
                if (cams.empty()) continue;

                // Load HD points (+ normals & times when available)
                auto hd = load_hd_for_submap(si, false);
                if (!hd || hd->size() == 0) continue;
                const Eigen::Isometry3d T_wo = sm->T_world_origin;
                const Eigen::Matrix3d R_wo = T_wo.rotation();
                std::vector<Eigen::Vector3f> wpts(hd->size());
                std::vector<float> ints(hd->size(), 0.0f);
                std::vector<Eigen::Vector3f> wnor;
                std::vector<double> wtimes;
                if (hd->normals) wnor.resize(hd->size());
                if (hd->times)   wtimes.assign(hd->times, hd->times + hd->size());
                for (size_t i = 0; i < hd->size(); i++) {
                  wpts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                  if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
                  if (hd->normals)     wnor[i] = (R_wo * Eigen::Vector3d(hd->normals[i].head<3>())).normalized().cast<float>();
                }

                // Cube-face expand for Spherical (no-op for Pinhole). Without
                // this, a Spherical active source projects as if pinhole and
                // produces near-zero colored points.
                auto expanded = expand_source_cams_for_projection(isrc, cams, colorize_mask);
                auto cr = make_colorizer(isrc.params.view_selector_mode)->project(
                  expanded.cams, expanded.intrinsics, wpts, ints, wnor, wtimes, current_blend_params(isrc));
                total_colored += cr.colored;

                // Build this submap's drawable data (only colored, non-gray points)
                std::vector<Eigen::Vector4d> sm_pts; sm_pts.reserve(cr.points.size());
                std::vector<Eigen::Vector4f> sm_cols; sm_cols.reserve(cr.points.size());
                for (size_t i = 0; i < cr.points.size(); i++) {
                  if (cr.colors[i].x() == 0.5f && cr.colors[i].y() == 0.5f && cr.colors[i].z() == 0.5f) continue;
                  sm_pts.emplace_back(cr.points[i].x(), cr.points[i].y(), cr.points[i].z(), 1.0);
                  sm_cols.emplace_back(cr.colors[i].x(), cr.colors[i].y(), cr.colors[i].z(), 1.0f);
                }
                if (sm_pts.empty()) continue;
                total_rendered_pts += sm_pts.size();
                sm_drawables_made++;

                // Upload immediately on main thread, by submap id.
                const int submap_id = sm->id;
                guik::LightViewer::instance()->invoke([this, submap_id, pts = std::move(sm_pts), cols = std::move(sm_cols)]() mutable {
                  auto vw = guik::LightViewer::instance();
                  auto cb = std::make_shared<glk::PointCloudBuffer>(pts.data(), pts.size());
                  cb->add_color(cols);
                  const std::string name = "colorize_preview_sm_" + std::to_string(submap_id);
                  vw->update_drawable(name, cb, guik::Rainbow().set_color_mode(guik::ColorMode::VERTEX_COLOR));
                  colorize_preview_sm_ids.push_back(submap_id);
                });
              }

              // Final status
              guik::LightViewer::instance()->invoke([this, total_colored, total_rendered_pts, sm_drawables_made] {
                char buf[192]; std::snprintf(buf, sizeof(buf),
                  "Done: %zu colored / %zu rendered pts across %zu submap drawables",
                  total_colored, total_rendered_pts, sm_drawables_made);
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
          for (int sid : colorize_preview_sm_ids) vw->remove_drawable("colorize_preview_sm_" + std::to_string(sid));
          colorize_preview_sm_ids.clear();
          lod_hide_all_submaps = false;
        }

        // Apply colorize to HD (write aux_rgb.bin per frame)
        static bool apply_rgb_running = false;
        static std::string apply_rgb_status;
        {
          const char* methods[] = {"Per-submap (legacy)", "Chunk-based"};
          ImGui::SetNextItemWidth(180);
          ImGui::Combo("Method##apply", &apply_method, methods, IM_ARRAYSIZE(methods));
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Per-submap: iterate submaps, project all cameras in each submap's time range.\n"
            "  Fast, but submap boundaries can cut off cameras that would have seen edge points.\n"
            "\n"
            "Chunk-based: iterate spatial chunks along the trajectory (like Voxelize HD / Data Filter).\n"
            "  For each chunk, load frames + cameras whose bbox/position overlap, with edge margin.\n"
            "  Eliminates submap-boundary cut-offs -- a point near a submap edge sees EVERY camera\n"
            "  that could see it, not just those in its own submap's time window.");
          if (apply_method == 1) {
            ImGui::Indent();
            ImGui::SetNextItemWidth(100); ImGui::DragFloat("chunk (m)##apply", &apply_chunk_size_m, 1.0f, 5.0f, 200.0f, "%.1f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Core chunk size. Points in this area are the ones WRITTEN per chunk.");
            ImGui::SameLine(); ImGui::SetNextItemWidth(100);
            ImGui::DragFloat("margin (m)##apply", &apply_chunk_margin_m, 1.0f, 0.0f, 200.0f, "%.1f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Edge overlap: frames+cameras within (core + margin) are LOADED so points near the\n"
              "chunk boundary see all cameras that might contribute. Set >= colorize max_range.");
            ImGui::Unindent();
          }
        }
        if (apply_rgb_running) {
          ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", apply_rgb_status.c_str());
          ImGui::SameLine();
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.25f, 0.25f, 1.0f));
          if (ImGui::Button(apply_cancel_requested ? "Stopping...##app" : "Stop##app")) apply_cancel_requested = true;
          ImGui::PopStyleColor();
        } else {
          if (ImGui::Button(apply_method == 0 ? "Apply colorize to HD (per-submap)" : "Apply colorize to HD (chunked)")) {
            apply_rgb_running = true;
            apply_cancel_requested = false;
            apply_rgb_status = "Starting...";
            const auto mask_copy = colorize_mask.clone();
            if (apply_method == 0) std::thread([this, mask_copy] {
              if (!trajectory_built) build_trajectory();
              const auto start_time = std::chrono::steady_clock::now();
              const auto& isrc = image_sources[colorize_source_idx];
              int frames_written = 0;

              for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
                if (apply_cancel_requested) break;
                const auto& sm = submaps[si];
                if (!sm || sm->frames.empty() || hidden_sessions.count(sm->session_id)) continue;

                char buf[64]; std::snprintf(buf, sizeof(buf), "Submap %d/%zu...", si + 1, submaps.size());
                apply_rgb_status = buf;

                // Find cameras by timestamp. Restricted to active source +
                // expanded via cube-face faces (no-op for Pinhole). Same reason
                // as the preview paths: single intrinsics/camera_type per
                // project() call.
                const double t0 = sm->frames.front()->stamp, t1 = sm->frames.back()->stamp;
                std::vector<CameraFrame> cams;
                for (const auto& cam : isrc.frames) {
                  if (!cam.located || cam.timestamp <= 0) continue;
                  const double ct = cam.timestamp + effective_time_shift(isrc, cam.timestamp);
                  if (ct >= t0 - 1.0 && ct <= t1 + 1.0) {
                    CameraFrame shifted = cam;
                    shifted.timestamp = ct;  // LiDAR-time for time-slice comparisons
                    cams.push_back(std::move(shifted));
                  }
                }
                if (cams.empty()) continue;

                // Load full submap (same as preview) -- project once, split by frame
                auto hd = load_hd_for_submap(si, false);
                if (!hd || hd->size() == 0) continue;
                const Eigen::Isometry3d T_wo = sm->T_world_origin;
                const Eigen::Matrix3d R_wo = T_wo.rotation();
                std::vector<Eigen::Vector3f> wpts(hd->size());
                std::vector<float> ints(hd->size(), 0.0f);
                std::vector<Eigen::Vector3f> wnor;
                if (hd->normals) wnor.resize(hd->size());
                for (size_t i = 0; i < hd->size(); i++) {
                  wpts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
                  if (hd->intensities) ints[i] = static_cast<float>(hd->intensities[i]);
                  if (hd->normals)     wnor[i] = (R_wo * Eigen::Vector3d(hd->normals[i].head<3>())).normalized().cast<float>();
                }

                // Gather per-point times from hd (gps_time = frame_stamp + per-point offset)
                std::vector<double> ptimes;
                if (hd->times) { ptimes.assign(hd->times, hd->times + hd->size()); }
                // Apply path runs in a worker thread that snapshotted mask_copy
                // beforehand -- override the live mask on the base params.
                BlendParams bp_apply = current_blend_params(isrc);
                bp_apply.mask = mask_copy;
                auto expanded = expand_source_cams_for_projection(isrc, cams, mask_copy);
                auto cr = make_colorizer(isrc.params.view_selector_mode)->project(
                  expanded.cams, expanded.intrinsics, wpts, ints, wnor, ptimes, bp_apply);
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

                  // Write aux_rgb.bin -- map merged submap indices back to frame indices
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
                      // Point filtered by range -- use intensity fallback
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
            // end per-submap branch
            else std::thread([this, mask_copy] {
              // ---- Chunk-based Apply ----
              if (!trajectory_built) build_trajectory();
              const auto start_time = std::chrono::steady_clock::now();
              const auto& isrc = image_sources[colorize_source_idx];

              // Build chunks along trajectory
              auto chunks = glim::build_chunks(trajectory_data, trajectory_total_dist,
                apply_chunk_size_m, apply_chunk_size_m * 0.5 + apply_chunk_margin_m);
              logger->info("[Apply/chunk] {} chunks along {:.0f}m (core={}m, margin={}m)",
                chunks.size(), trajectory_total_dist, apply_chunk_size_m, apply_chunk_margin_m);

              // Index all frames (world bbox + frame_info from meta)
              std::vector<glim::FrameInfo> all_frames;
              for (const auto& sm : submaps) {
                if (!sm) continue;
                if (hidden_sessions.count(sm->session_id)) continue;
                std::string shd = hd_frames_path;
                for (const auto& s : sessions) { if (s.id == sm->session_id && !s.hd_frames_path.empty()) { shd = s.hd_frames_path; break; } }
                const Eigen::Isometry3d T0 = sm->frames.front()->T_world_imu;
                for (const auto& fr : sm->frames) {
                  char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
                  auto fi = glim::frame_info_from_meta(shd + "/" + dn,
                    glim::compute_frame_world_pose(sm->T_world_origin, sm->T_origin_endpoint_L, T0, fr->T_world_imu, fr->T_lidar_imu),
                    sm->id, sm->session_id);
                  if (fi.num_points > 0) all_frames.push_back(std::move(fi));
                }
              }

              // Frame-keyed RGB accumulator. Each entry holds a full vector
              // sized to the frame's num_points (~12 B/pt). Without bounded
              // lifetime this used to grow until end-of-Apply -- 5K frames at
              // 64K pts each = ~3.7 GB just sitting around. Now flushed +
              // evicted as soon as no future chunk needs the frame, so RAM
              // stays bounded to "frames in-flight across the chunk margin".
              struct FrameRgb {
                int num_points = 0;
                std::vector<Eigen::Vector3f> rgb;      // default (0.5, 0.5, 0.5) = uncolored gray
                std::vector<uint8_t> has_color;        // parallel
              };
              std::unordered_map<std::string, FrameRgb> frame_rgb;  // key = frame.dir

              // Pre-compute per-frame-dir "last chunk index that still needs it"
              // by spatial AABB intersection. After processing chunk ci, any
              // frame with last_chunk[fdir] <= ci can be flushed and dropped
              // from frame_rgb -- no more chunks will touch it. This bounds the
              // accumulator's RAM regardless of session length.
              std::unordered_map<std::string, int> frame_last_chunk;
              for (size_t ci = 0; ci < chunks.size(); ci++) {
                const auto chunk_aabb = chunks[ci].world_aabb();
                for (const auto& fi : all_frames) {
                  if (fi.num_points == 0 || !chunk_aabb.intersects(fi.world_bbox)) continue;
                  frame_last_chunk[fi.dir] = static_cast<int>(ci);  // overwrites with later ci automatically
                }
              }

              // Loaded frame cache (sliding). Holds world-space points per frame directory.
              struct CachedFrame {
                std::vector<Eigen::Vector3f> world_pts;
                std::vector<float> intensities;
                std::vector<Eigen::Vector3f> world_normals;
                std::vector<double> times;
                std::vector<int> orig_idx;              // parallel; original index in frame's on-disk order
              };
              std::unordered_map<std::string, std::shared_ptr<CachedFrame>> frame_cache;

              size_t total_points_written = 0;
              size_t total_core_points_processed = 0;
              int frames_written = 0;  // promoted out of the final loop so incremental flush can update it

              for (size_t ci = 0; ci < chunks.size(); ci++) {
                if (apply_cancel_requested) break;
                const auto& chunk = chunks[ci];           // already built with margin half-size
                const auto chunk_aabb = chunk.world_aabb();
                glim::Chunk core = chunk;
                core.half_size = apply_chunk_size_m * 0.5;

                char buf[192]; std::snprintf(buf, sizeof(buf), "Chunk %zu/%zu loading (cache: %zu frames)",
                  ci + 1, chunks.size(), frame_cache.size());
                apply_rgb_status = buf;

                // Find frames overlapping the LOADED area (chunk with margin)
                std::vector<const glim::FrameInfo*> chunk_frame_infos;
                std::unordered_set<std::string> needed_dirs;
                for (const auto& fi : all_frames) {
                  if (fi.num_points == 0 || !chunk_aabb.intersects(fi.world_bbox)) continue;
                  chunk_frame_infos.push_back(&fi);
                  needed_dirs.insert(fi.dir);
                }

                // Evict frames no longer needed
                for (auto it = frame_cache.begin(); it != frame_cache.end();) {
                  if (!needed_dirs.count(it->first)) it = frame_cache.erase(it);
                  else ++it;
                }

                // Load missing frames into cache (+ normals, times when available)
                for (const auto* fi : chunk_frame_infos) {
                  if (frame_cache.count(fi->dir)) continue;
                  std::vector<Eigen::Vector3f> pts; std::vector<float> rng, ints(fi->num_points, 0.0f);
                  if (!glim::load_bin(fi->dir + "/points.bin", pts, fi->num_points)) continue;
                  glim::load_bin(fi->dir + "/range.bin", rng, fi->num_points);
                  glim::load_bin(fi->dir + "/intensities.bin", ints, fi->num_points);
                  std::vector<Eigen::Vector3f> fnor;
                  std::ifstream fnorm(fi->dir + "/normals.bin", std::ios::binary);
                  if (fnorm) { fnor.resize(fi->num_points); fnorm.read(reinterpret_cast<char*>(fnor.data()), sizeof(Eigen::Vector3f) * fi->num_points); }
                  std::vector<float> ft(fi->num_points, 0.0f);
                  glim::load_bin(fi->dir + "/times.bin", ft, fi->num_points);
                  const Eigen::Matrix3f R = fi->T_world_lidar.rotation().cast<float>();
                  const Eigen::Vector3f t = fi->T_world_lidar.translation().cast<float>();
                  const Eigen::Matrix3f R_for_normals = R;  // normals rotate same as points
                  auto cf = std::make_shared<CachedFrame>();
                  cf->world_pts.reserve(fi->num_points); cf->intensities.reserve(fi->num_points);
                  cf->orig_idx.reserve(fi->num_points);
                  if (!fnor.empty()) cf->world_normals.reserve(fi->num_points);
                  cf->times.reserve(fi->num_points);
                  for (int i = 0; i < fi->num_points; i++) {
                    if (!rng.empty() && rng[i] < 1.5f) continue;
                    cf->world_pts.push_back(R * pts[i] + t);
                    cf->intensities.push_back(ints[i]);
                    cf->orig_idx.push_back(i);
                    if (!fnor.empty()) cf->world_normals.push_back((R_for_normals * fnor[i]).normalized());
                    cf->times.push_back(static_cast<double>(fi->stamp) + static_cast<double>(ft[i]));
                  }
                  // Prepare persistent FrameRgb on first touch
                  auto& pf = frame_rgb[fi->dir];
                  if (pf.rgb.empty()) {
                    pf.num_points = fi->num_points;
                    pf.rgb.assign(fi->num_points, Eigen::Vector3f(0.5f, 0.5f, 0.5f));
                    pf.has_color.assign(fi->num_points, 0);
                  }
                  frame_cache[fi->dir] = cf;
                }

                // Find cameras whose POSITION is within the loaded chunk AABB, AND
                // whose forward axis points toward the core (rejects cameras driving past
                // the chunk and looking the other way -- otherwise they project nothing
                // useful but pay full NCC/occlusion cost). Restricted to the active
                // source only (one camera_type per project() call).
                const Eigen::Vector3d core_center = core.center;
                std::vector<CameraFrame> cams;
                int rejected_backward = 0, rejected_outside = 0;
                // For Spherical the forward-dot-to-core gate is meaningless
                // (360 cams see in every direction) -- skip it when equirect.
                const bool is_spherical_src = (isrc.camera_type == CameraType::Spherical);
                for (const auto& cam : isrc.frames) {
                  if (!cam.located || cam.timestamp <= 0) continue;
                  const Eigen::Vector3f cp = cam.T_world_cam.translation().cast<float>();
                  if (!chunk.contains(cp)) { rejected_outside++; continue; }
                  if (!is_spherical_src) {
                    const Eigen::Vector3d cam_fwd = cam.T_world_cam.rotation() * Eigen::Vector3d::UnitZ();
                    const Eigen::Vector3d to_core = (core_center - cam.T_world_cam.translation()).normalized();
                    if (cam_fwd.dot(to_core) < 0.2) { rejected_backward++; continue; }
                  }
                  CameraFrame shifted = cam;
                  shifted.timestamp = cam.timestamp + effective_time_shift(isrc, cam.timestamp);
                  cams.push_back(std::move(shifted));
                }
                if (cams.empty()) continue;

                // Assemble points in the CORE area (only these get written), keeping back-references
                std::vector<Eigen::Vector3f> chunk_pts;
                std::vector<float> chunk_ints;
                std::vector<Eigen::Vector3f> chunk_nors;
                std::vector<double> chunk_times;
                std::vector<std::pair<std::string, int>> chunk_src;  // (frame.dir, orig_idx)
                for (const auto* fi : chunk_frame_infos) {
                  auto it = frame_cache.find(fi->dir);
                  if (it == frame_cache.end()) continue;
                  const auto& cf = it->second;
                  const bool have_normals = !cf->world_normals.empty();
                  for (size_t i = 0; i < cf->world_pts.size(); i++) {
                    if (!core.contains(cf->world_pts[i])) continue;
                    chunk_pts.push_back(cf->world_pts[i]);
                    chunk_ints.push_back(cf->intensities[i]);
                    if (have_normals) chunk_nors.push_back(cf->world_normals[i]);
                    chunk_times.push_back(cf->times[i]);
                    chunk_src.emplace_back(fi->dir, cf->orig_idx[i]);
                  }
                }
                if (chunk_pts.empty()) continue;

                {
                  char sbuf[192]; std::snprintf(sbuf, sizeof(sbuf),
                    "Chunk %zu/%zu projecting: %zu cams, %zu pts (skipped %d back-facing, %d outside)",
                    ci + 1, chunks.size(), cams.size(), chunk_pts.size(), rejected_backward, rejected_outside);
                  apply_rgb_status = sbuf;
                }

                BlendParams bp_chunk = current_blend_params(isrc);
                bp_chunk.mask = mask_copy;
                auto expanded = expand_source_cams_for_projection(isrc, cams, mask_copy);
                auto cr = make_colorizer(isrc.params.view_selector_mode)->project(
                  expanded.cams, expanded.intrinsics, chunk_pts, chunk_ints,
                  chunk_nors.size() == chunk_pts.size() ? chunk_nors : std::vector<Eigen::Vector3f>{},
                  chunk_times, bp_chunk);

                // Scatter results back to per-frame RGB arrays
                for (size_t i = 0; i < cr.points.size(); i++) {
                  // Skip uncolored (gray sentinel) points
                  if (cr.colors[i].x() == 0.5f && cr.colors[i].y() == 0.5f && cr.colors[i].z() == 0.5f) continue;
                  const auto& [fdir, oidx] = chunk_src[i];
                  auto& pf = frame_rgb[fdir];
                  if (oidx >= 0 && oidx < pf.num_points) {
                    pf.rgb[oidx] = cr.colors[i];
                    pf.has_color[oidx] = 1;
                    total_points_written++;
                  }
                }
                total_core_points_processed += chunk_pts.size();

                // Incremental flush: any frame that no later chunk will touch
                // gets its aux_rgb.bin written and dropped from frame_rgb. This
                // is the bound that keeps RAM linear in chunk-margin frame
                // count, not total session frame count.
                for (auto it = frame_rgb.begin(); it != frame_rgb.end();) {
                  auto last_it = frame_last_chunk.find(it->first);
                  const bool no_more_chunks =
                    (last_it == frame_last_chunk.end()) ||
                    (last_it->second <= static_cast<int>(ci));
                  if (!no_more_chunks) { ++it; continue; }
                  const auto& fdir = it->first;
                  const auto& pf = it->second;
                  bool any = false;
                  for (uint8_t b : pf.has_color) if (b) { any = true; break; }
                  if (any) {
                    std::ofstream f(fdir + "/aux_rgb.bin", std::ios::binary);
                    if (f) {
                      f.write(reinterpret_cast<const char*>(pf.rgb.data()), sizeof(Eigen::Vector3f) * pf.num_points);
                      frames_written++;
                    }
                  }
                  it = frame_rgb.erase(it);
                }
              }

              // Final flush -- safety net for anything not caught by the
              // per-chunk pass (e.g. a frame whose last_chunk index wasn't
              // computed because its bbox didn't overlap any chunk).
              apply_rgb_status = "Writing remaining aux_rgb.bin...";
              for (const auto& [fdir, pf] : frame_rgb) {
                bool any = false;
                for (uint8_t b : pf.has_color) if (b) { any = true; break; }
                if (!any) continue;
                std::ofstream f(fdir + "/aux_rgb.bin", std::ios::binary);
                if (!f) continue;
                f.write(reinterpret_cast<const char*>(pf.rgb.data()), sizeof(Eigen::Vector3f) * pf.num_points);
                frames_written++;
              }
              frame_rgb.clear();

              const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
              char buf[192]; std::snprintf(buf, sizeof(buf),
                "Chunked done: %d frames written, %zu core pts processed, %zu colored (%.1f sec)",
                frames_written, total_core_points_processed, total_points_written, elapsed);
              apply_rgb_status = buf;
              apply_rgb_running = false;
              logger->info("[Apply/chunk] {}", apply_rgb_status);
            }).detach();
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("WRITE aux_rgb.bin to every HD frame.\nPersistent -- appears in color dropdown after HD reload.");
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

  // Alignment check window -- image + projected LiDAR overlay, scale-aware
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
      // Camera-type badge -- same wording as the Colorize window so the user
      // sees the model this overlay is branching on (pinhole proj + distortion
      // vs equirectangular proj with no distortion).
      ImGui::SameLine();
      if (src.camera_type == CameraType::Spherical) {
        ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.0f, 1.0f), "[%s]", camera_type_label(src.camera_type));
      } else {
        ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.55f, 1.0f), "[%s]", camera_type_label(src.camera_type));
      }

      ImGui::SetNextItemWidth(120); ImGui::SliderFloat("View scale", &align_display_scale, 0.05f, 10.0f, "%.2fx");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Display scale vs native pixels. Math always runs at native resolution.\n"
        "Mouse wheel on the canvas zooms around the cursor; +/- buttons step.");
      ImGui::SameLine(); if (ImGui::Button("Fit")) {
        // Will be computed below once we know window size
        align_display_scale = -1.0f;  // sentinel
      }
      ImGui::SameLine();
      if (ImGui::Button("-##zoom")) align_display_scale = std::max(0.05f, align_display_scale / 1.25f);
      ImGui::SameLine();
      if (ImGui::Button("+##zoom")) align_display_scale = std::min(10.0f, align_display_scale * 1.25f);
      ImGui::SameLine(); ImGui::SetNextItemWidth(100);
      ImGui::SliderFloat("Pt size", &align_point_size, 0.5f, 6.0f, "%.1f");
      ImGui::SameLine(); ImGui::SetNextItemWidth(110);
      ImGui::Combo("Color", &align_point_color_mode, "Intensity\0Range\0Depth\0Winner-mask\0Weight\0");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "How to color projected dots.\n"
        "- Winner-mask: only points where this camera is the top-1 winner (requires Compute assignment)\n"
        "- Weight: heatmap of per-point winning weight (requires Compute assignment)");
      // Color scale dropdown: applies to any scalar mode (Intensity/Range/Depth/Weight).
      // Winner-mask is categorical (this-cam vs others) and ignores the scale.
      if (align_point_color_mode != 3) {
        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
        static const auto _cmap_names = glk::colormap_names();
        ImGui::Combo("Scale##al", &align_colormap_sel, _cmap_names.data(), static_cast<int>(_cmap_names.size()));
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Color scale applied to the active scalar field (Intensity / Range / Depth / Weight).\n"
          "Winner-mask is categorical and ignores this.\n"
          "Turbo (default) is perceptually strong; Cividis is colour-blind safe; Ocean / Turbo\n"
          "are great for LiDAR intensity. Try different scales to surface contrast in dim data.");
      }
      // Colorize-from-this-camera: runs the full colorize pipeline (view-selector
      // + all vs_* params from the Colorize window) on the current submap cloud
      // using ONLY this one frame as the camera source. Shows what this single
      // frame actually contributes to the overall colorize, so misalignment
      // between image + LiDAR jumps out immediately. Lives through param tweaks
      // when live preview is on.
      ImGui::SameLine();
      if (ImGui::Checkbox("Colorize", &align_colorize_auto)) {
        // Click just flipped the value; act on the new state.
        if (align_colorize_auto) {
          align_colorize_dirty = true;
          align_colorize_valid = true;
        } else {
          align_colorize_valid = false;
          align_colorize_dirty = false;
          align_colorize_rgb.clear();
        }
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "TOGGLE. When ON, the RGB cache auto-recomputes:\n"
        "  - every time you step to a different camera frame\n"
        "  - on every intrinsic / extrinsic / time-shift change (with Live preview ON)\n"
        "\n"
        "Runs the full Colorize pipeline on this single camera (view selector,\n"
        "range gates, exposure, NCC, incidence, occlusion, time-slice). Overrides\n"
        "the scalar color mode while active. Cost is ~50-200 ms per compute --\n"
        "fast enough to feel instant while scrubbing frames.");
      if (align_colorize_valid) {
        ImGui::SameLine();
        if (ImGui::Button("Clear RGB")) {
          align_colorize_valid = false;
          align_colorize_dirty = false;
          align_colorize_auto = false;   // turn toggle off so frame-change won't re-fire
          align_colorize_rgb.clear();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Drop the colorize cache AND turn off the auto-colorize toggle.\n"
          "Back to scalar color mode. Saves navigating away and back to reset.");
        ImGui::SameLine();
        ImGui::Checkbox("Hide uncol", &align_colorize_hide_uncolored);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Skip drawing points the colorizer couldn't colour (gray-sentinel).\n"
          "These are points rejected by time-slice / incidence / NCC / occlusion\n"
          "gates. Hiding them shows EXACTLY what this camera contributes --\n"
          "matches the scene-view colorize output 1:1.");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f), "RGB active");
      }
      // Frame change: either drop the cache (one-shot mode) or request a
      // recompute for the new frame (auto mode). Auto mode keeps the "RGB
      // active" label lit so the user sees continuous coloring while scrubbing.
      if (align_colorize_valid &&
          (align_colorize_cam_src != align_cam_src || align_colorize_cam_idx != align_cam_idx)) {
        if (align_colorize_auto) {
          align_colorize_dirty = true;  // recompute for the new frame
        } else {
          align_colorize_valid = false;
          align_colorize_rgb.clear();
        }
      }
      // Auto-refresh while live preview is ON and anything the colorizer depends
      // on changed since last compute (intrinsics, lever arm, rpy, time shift).
      if (align_colorize_valid && align_live_preview) {
        const auto& lk = align_last_intrinsics; const auto& ck = src.intrinsics;
        const bool extr_changed =
          align_last_rpy != src.rotation_rpy || align_last_lever != src.lever_arm ||
          align_last_time_shift != src.time_shift;
        const bool intr_changed =
          lk.fx != ck.fx || lk.fy != ck.fy || lk.cx != ck.cx || lk.cy != ck.cy ||
          lk.k1 != ck.k1 || lk.k2 != ck.k2 || lk.k3 != ck.k3 ||
          lk.p1 != ck.p1 || lk.p2 != ck.p2;
        if (extr_changed || intr_changed) align_colorize_dirty = true;
      }
      ImGui::SameLine(); ImGui::SetNextItemWidth(100);
      ImGui::SliderFloat("Min bright", &align_bright_threshold, 0.0f, 1.0f, "%.2f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Only show points with intensity above this threshold.\nUseful for comparing road markings.");

      ImGui::SetNextItemWidth(120); ImGui::SliderFloat("Max range", &align_max_range, 2.0f, 200.0f, "%.1fm");
      ImGui::SameLine(); ImGui::SetNextItemWidth(120);
      ImGui::SliderFloat("Min range", &align_min_range, 0.1f, 10.0f, "%.1fm");
      ImGui::SameLine(); ImGui::SetNextItemWidth(120);
      ImGui::SliderFloat("Alpha", &align_point_alpha, 0.05f, 1.0f, "%.2f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overlay point transparency -- blend dots with image.");
      ImGui::SameLine();
      if (ImGui::Checkbox("Gray BG", &align_image_grayscale)) align_loaded_path.clear();  // force reload
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Render the background image as grayscale. Boosts visual contrast between\n"
        "the colored LiDAR dots and the photo, making small offsets easier to spot.");
      ImGui::SameLine();
      ImGui::Checkbox("Hide BG", &align_image_hidden);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Fully hide the background image -- inspect coloring quality on the raw\n"
        "dot overlay. Useful for judging colorize (RGB) output without photo bias.");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(90);
      ImGui::InputInt("+/- frames", &align_nearest_frames, 1, 5);
      align_nearest_frames = std::clamp(align_nearest_frames, 0, 60);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Temporal filter on displayed points:\n"
        "  0 = whole submap (may include ~1-2s of LiDAR frames)\n"
        "  N = keep only points from +/- N LiDAR frames around this camera's time\n"
        "\n"
        "Set to 1-2 to mimic what colorize-with-time-slice actually covers --\n"
        "the alignment check then shows the same subset the scene view colors,\n"
        "making offset diagnostics 1:1 with the final output.");
      ImGui::SameLine();
      ImGui::Checkbox("Grid##al", &align_grid_show);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Overlay a reference grid of straight horizontal + vertical lines.\n"
        "Real-world straight features (building edges, road lanes, sign posts)\n"
        "should project to STRAIGHT lines on a well-calibrated image. Distortion\n"
        "bends them. Eyeball the bend against a grid line to spot residual k1/k2.");
      if (align_grid_show) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90);
        ImGui::InputInt("+/- lines", &align_grid_lines, 1, 5);
        align_grid_lines = std::clamp(align_grid_lines, 1, 60);
      }
      // User-placed reference lines: click a feature that should be straight in
      // the world -> a distorted straight line is drawn through it. Store in
      // IDEAL pinhole coords (distortion-invariant) so the line re-bends correctly
      // as the user tweaks k1/k2/p1/p2 live.
      ImGui::SameLine();
      if (ImGui::Button(align_add_line_mode == 1 ? "Click V...##al" : "+V##al")) {
        align_add_line_mode = (align_add_line_mode == 1) ? 0 : 1;
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Arm 'add vertical reference line' -- next click on the image places a\n"
        "vertical line passing through that point. The line bends with current\n"
        "distortion so it always represents a world-straight feature.");
      ImGui::SameLine();
      if (ImGui::Button(align_add_line_mode == 2 ? "Click H...##al" : "+H##al")) {
        align_add_line_mode = (align_add_line_mode == 2) ? 0 : 2;
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Arm 'add horizontal reference line' -- next click on the image places\n"
        "a horizontal line through that point, bent by current distortion.");
      if (!align_user_lines.empty()) {
        ImGui::SameLine();
        if (ImGui::Button("Clear lines##al")) {
          align_user_lines.clear();
          align_add_line_mode = 0;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu)", align_user_lines.size());
      }
      ImGui::SameLine();
      // Rectified only makes sense for Brown-Conrady pinhole. Greyed out +
      // forced off for Spherical (equirect has no k1/k2/p1/p2 to undo).
      const bool rectify_applicable = camera_type_has_brown_conrady(src.camera_type);
      ImGui::BeginDisabled(!rectify_applicable);
      if (ImGui::Checkbox("Rectified", &align_rectified)) align_loaded_path.clear();  // force reload
      ImGui::EndDisabled();
      if (!rectify_applicable && align_rectified) { align_rectified = false; align_loaded_path.clear(); }
      if (ImGui::IsItemHovered()) {
        if (rectify_applicable) {
          ImGui::SetTooltip("OFF: raw image + distorted projection (matches colorize phase).\nON: undistorted image + pinhole projection (isolates extrinsic error).");
        } else {
          ImGui::SetTooltip("Not applicable for %s sources -- equirectangular has no\nBrown-Conrady distortion parameters to undo.",
            camera_type_label(src.camera_type));
        }
      }
      ImGui::SameLine();
      ImGui::Checkbox("Live preview##al", &align_live_preview);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Re-render this overlay whenever intrinsics / extrinsics / time-shift change.\n"
        "Cheaper than the Colorize window's live preview (one image, not a full HD\n"
        "cloud re-project), so it's the right place to iterate on distortion + pose.\n"
        "\n"
        "When ON: the Colorize window's 'Live preview' is overridden OFF for perf --\n"
        "you get instant overlay feedback here while tuning the params there.");
      // Both live-preview toggles coexist now. They operate on independent
      // state (align overlays a single image; Colorize re-projects a cloud),
      // and the user benefits from seeing BOTH update while dragging time_shift
      // -- align gives 2D offset feedback, Colorize 3D gives section-wide effect.
      // If you hit a perf wall, turn one off manually.
      if (src.camera_type == CameraType::Spherical) {
        ImGui::TextDisabled("Mode: equirectangular image vs spherical projection -- residual = extrinsic + time shift.");
      } else if (align_rectified) {
        ImGui::TextDisabled("Mode: rectified image vs pinhole projection -- residual = extrinsic only.");
      } else {
        ImGui::TextDisabled("Mode: raw image vs distorted projection -- residual = extrinsic + distortion model.");
      }

      // --- Load image if needed ---
      // Triggers: new camera, rectified toggle changed, OR live-preview mode
      // with rectified ON and intrinsics changed since last load (cheap check
      // on raw doubles -- one exact compare per field).
      const auto& cam = src.frames[align_cam_idx];
      const auto& lk = align_last_intrinsics; const auto& ck = src.intrinsics;
      // Re-undistort whenever intrinsics drift from the snapshot the current
      // texture was built from, regardless of the Live preview toggle. Otherwise
      // tweaking k1/cx with Live preview OFF leaves the image stale while the
      // grid / user lines update, making them appear mis-registered.
      const bool intrinsics_changed_for_rect =
        align_rectified && (
          lk.fx != ck.fx || lk.fy != ck.fy || lk.cx != ck.cx || lk.cy != ck.cy ||
          lk.k1 != ck.k1 || lk.k2 != ck.k2 || lk.k3 != ck.k3 ||
          lk.p1 != ck.p1 || lk.p2 != ck.p2);
      if (cam.filepath != align_loaded_path || align_rect_applied != align_rectified
          || intrinsics_changed_for_rect) {
        cv::Mat img = cv::imread(cam.filepath);
        if (!img.empty()) {
          align_img_w = img.cols; align_img_h = img.rows;
          // Optional rectification: only meaningful for Brown-Conrady pinhole.
          // Spherical/Fisheye ignore the toggle -- no k1/k2/p1/p2 to undo.
          if (align_rectified && camera_type_has_brown_conrady(src.camera_type)) {
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
          if (align_image_grayscale) {
            // Grayscale then broadcast back to 3 channels -- texture stays RGB,
            // the extra contrast makes colored dots pop against the photo.
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, img, cv::COLOR_GRAY2RGB);
          } else {
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
          }
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
          align_last_intrinsics = src.intrinsics;  // snapshot for change detection next frame
        }
      }

      // --- Locate camera using current colorize extrinsic (live-linked) ---
      if (!trajectory_built) build_trajectory();
      const auto timed_traj = timed_traj_snapshot();
      // Auto-sync scalar time_shift to the interpolated value when the user
      // scrubs to a different frame (anchors active only). Two effects:
      //   1. The time_shift widget in the Colorize window shows a MEANINGFUL
      //      number for this specific section of track -- not a frozen global.
      //   2. The user can drag that widget to preview a BETTER calib for this
      //      frame; the align-check overlay updates live (scratch override).
      //      "Anchor here" then commits the dragged value as an anchor.
      {
        static int prev_align_cam_src = -1;
        static int prev_align_cam_idx = -1;
        if (!src.anchors.empty() && cam.timestamp > 0.0 &&
            (prev_align_cam_src != align_cam_src || prev_align_cam_idx != align_cam_idx)) {
          src.time_shift = effective_time_shift(src, cam.timestamp);
        }
        prev_align_cam_src = align_cam_src;
        prev_align_cam_idx = align_cam_idx;
      }

      Eigen::Isometry3d T_world_cam = Eigen::Isometry3d::Identity();
      bool cam_ok = false;
      // Route through the per-source alt trajectory when the active source is
      // in 'Coords > own path' mode; otherwise use SLAM (modes 0/1 unchanged).
      const auto& align_traj = trajectory_for(src, timed_traj);
      if (cam.timestamp > 0.0 && !align_traj.empty()) {
        // Align-check uses the scalar src.time_shift DIRECTLY (not
        // effective_time_shift) so dragging the widget updates this overlay
        // live. Every other pipeline still goes through effective_time_shift
        // and honors anchors as usual.
        const double ts = cam.timestamp + src.time_shift;
        if (ts >= align_traj.front().stamp - 2.0 && ts <= align_traj.back().stamp + 2.0) {
          const Eigen::Isometry3d T_world_lidar = Colorizer::interpolate_pose(align_traj, ts);
          const Eigen::Isometry3d T_lidar_cam = Colorizer::build_extrinsic(src.lever_arm, src.rotation_rpy);
          T_world_cam = T_world_lidar * T_lidar_cam;
          cam_ok = true;
        }
      }

      // --- Find submap at camera timestamp; cache its world points ---
      int best_sm = -1;
      if (cam_ok) {
        best_sm = find_submap_for_timestamp(submaps, cam.timestamp + src.time_shift);
      }
      if (best_sm >= 0 && best_sm != align_last_submap_id) {
        align_submap_world_pts.clear(); align_submap_ints.clear(); align_submap_world_normals.clear(); align_submap_world_times.clear();
        auto hd = load_hd_for_submap(best_sm, false);
        if (hd && hd->size() > 0) {
          const Eigen::Isometry3d T_wo = submaps[best_sm]->T_world_origin;
          const Eigen::Matrix3d R_wo = T_wo.rotation();
          align_submap_world_pts.resize(hd->size());
          align_submap_ints.assign(hd->size(), 0.0f);
          const bool have_normals = (hd->normals != nullptr);
          const bool have_times   = (hd->times   != nullptr);
          if (have_normals) align_submap_world_normals.resize(hd->size());
          if (have_times)   align_submap_world_times.resize(hd->size());
          for (size_t i = 0; i < hd->size(); i++) {
            align_submap_world_pts[i] = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
            if (hd->intensities) align_submap_ints[i] = static_cast<float>(hd->intensities[i]);
            if (have_normals) {
              align_submap_world_normals[i] = (R_wo * Eigen::Vector3d(hd->normals[i].head<3>())).normalized().cast<float>();
            }
            if (have_times)  align_submap_world_times[i] = hd->times[i];
          }
        }
        // Percentile-based intensity range: 5% / 95% keeps a few bright outliers
        // from flattening the whole scale to "dark". Recomputed per submap load.
        if (!align_submap_ints.empty()) {
          std::vector<float> ints_copy(align_submap_ints);
          const size_t n = ints_copy.size();
          const size_t lo = static_cast<size_t>(0.05 * n);
          const size_t hi = static_cast<size_t>(0.95 * n);
          std::nth_element(ints_copy.begin(), ints_copy.begin() + lo, ints_copy.end());
          const float v_lo = ints_copy[lo];
          std::nth_element(ints_copy.begin() + lo, ints_copy.begin() + hi, ints_copy.end());
          const float v_hi = ints_copy[hi];
          align_intensity_range = (v_hi > v_lo + 1e-3f)
            ? Eigen::Vector2f(v_lo, v_hi)
            : Eigen::Vector2f(0.0f, 255.0f);
        } else {
          align_intensity_range = Eigen::Vector2f(0.0f, 255.0f);
        }
        // Cache average LiDAR-frame interval for the "+/- N frames" temporal
        // filter: allows translating a frame count into a wall-clock window.
        align_frame_interval_s = 0.1;  // sensible default
        if (submaps[best_sm] && submaps[best_sm]->frames.size() > 1) {
          const double t0 = submaps[best_sm]->frames.front()->stamp;
          const double t1 = submaps[best_sm]->frames.back()->stamp;
          align_frame_interval_s = (t1 - t0) / std::max<double>(1, submaps[best_sm]->frames.size() - 1);
          if (align_frame_interval_s < 1e-3 || align_frame_interval_s > 1.0) align_frame_interval_s = 0.1;
        }
        align_last_submap_id = best_sm;
      }

      // --- Identify nearest LiDAR frame to this camera (for frame-assignment check) ---
      long nearest_frame_id = -1;
      double nearest_frame_stamp = 0.0;
      double nearest_dt = 0.0;
      if (cam_ok && best_sm >= 0 && !submaps[best_sm]->frames.empty()) {
        const double ct = cam.timestamp + src.time_shift;  // scratch override; see note above
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

      // --- Time-shift anchors (shared widget; see render_anchor_panel) -------
      ImGui::Separator();
      render_anchor_panel(align_cam_src, cam.timestamp, cam.timestamp > 0.0, "align");

      // --- Assignment compute controls (Winner-mask / Weight viz) ---
      const bool need_assignment = (align_point_color_mode == 3 || align_point_color_mode == 4);
      const bool cache_valid_for_current_view =
        align_winner_sm == best_sm && align_winner_src == align_cam_src &&
        align_winner_frame_idx.size() == align_submap_world_pts.size();
      if (need_assignment) {
        ImGui::PushStyleColor(ImGuiCol_Button, cache_valid_for_current_view ? ImVec4(0.2f, 0.4f, 0.2f, 1.0f) : ImVec4(0.5f, 0.3f, 0.1f, 1.0f));
        const bool do_compute = ImGui::Button("Compute assignment");
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Run weighted colorizer on this submap using cameras in its time window.\n"
          "Caches per-point winner camera + weight. Required for Winner-mask / Weight viz.");
        ImGui::SameLine();
        if (cache_valid_for_current_view) {
          ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "cached (sm=%d, src=%d)", align_winner_sm, align_winner_src);
        } else {
          ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.3f, 1.0f), "STALE -- click to recompute");
        }
        if (do_compute && cam_ok && best_sm >= 0 && !align_submap_world_pts.empty()) {
          const double t0 = submaps[best_sm]->frames.front()->stamp;
          const double t1 = submaps[best_sm]->frames.back()->stamp;
          const double t_margin = 2.0;
          // Build camera list for current source within submap time range, remember frame indices
          std::vector<CameraFrame> cams;
          std::vector<int> frame_idx_map;
          for (int fi = 0; fi < static_cast<int>(src.frames.size()); fi++) {
            const auto& c = src.frames[fi];
            if (!c.located || c.timestamp <= 0.0) continue;
            const double ct2 = c.timestamp + effective_time_shift(src, c.timestamp);
            if (ct2 >= t0 - t_margin && ct2 <= t1 + t_margin) {
              CameraFrame shifted = c;
              shifted.timestamp = ct2;  // LiDAR-time for time-slice comparisons
              cams.push_back(std::move(shifted));
              frame_idx_map.push_back(fi);
            }
          }
          if (!cams.empty()) {
            // Assignment compute forces Weighted-Top-1 (deterministic winner per
            // point) but otherwise honors every other tuning from the source's
            // ColorizeParams. topK is pinned to 1 since the viz is winner-based.
            BlendParams p = current_blend_params(src);
            p.topK = 1;
            auto impl = make_colorizer(ViewSelectorMode::WeightedTop1);
            // For Spherical, each equirect cam becomes 6 virtual cube-face cams.
            // winner_cam then indexes into the expanded list (0..6*N-1); recover
            // the parent equirect index by dividing by 6 before mapping to the
            // source's original frame list via frame_idx_map.
            const bool is_sph = (src.camera_type == CameraType::Spherical);
            auto expanded = expand_source_cams_for_projection(src, cams, colorize_mask);
            auto res = impl->project(expanded.cams, expanded.intrinsics, align_submap_world_pts, align_submap_ints,
                                     align_submap_world_normals, align_submap_world_times, p);
            align_winner_frame_idx.assign(res.winner_cam.size(), -1);
            align_winner_weight_vec.assign(res.winner_weight.size(), 0.0f);
            align_weight_max_cached = 0.0f;
            for (size_t i = 0; i < res.winner_cam.size(); i++) {
              const int ci = res.winner_cam[i];
              if (ci >= 0) {
                const int parent = is_sph ? (ci / 6) : ci;
                if (parent >= 0 && parent < static_cast<int>(frame_idx_map.size())) {
                  align_winner_frame_idx[i] = frame_idx_map[parent];
                }
              }
              align_winner_weight_vec[i] = res.winner_weight[i];
              if (res.winner_weight[i] > align_weight_max_cached) align_weight_max_cached = res.winner_weight[i];
            }
            align_winner_sm = best_sm;
            align_winner_src = align_cam_src;
            logger->info("[Align] Computed assignment: sm={} src={} type={} cams_used={} (expanded={}) pts={} max_w={:.4f}",
              best_sm, align_cam_src, camera_type_label(src.camera_type),
              cams.size(), expanded.cams.size(), align_submap_world_pts.size(), align_weight_max_cached);
          } else {
            logger->warn("[Align] No cameras in submap time window -- cannot compute assignment");
          }
        }
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
      // Reserve space for the canvas via Dummy when hiding the image -- keeps
      // the scroll region and cursor position math identical to the image path.
      if (align_texture && align_img_w > 0 && !align_image_hidden) {
        ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(align_texture)), ImVec2(disp_w, disp_h));
      } else {
        ImGui::Dummy(ImVec2(disp_w > 0 ? disp_w : 400, disp_h > 0 ? disp_h : 300));
      }
      // Click handling for user-placed reference lines. We need this BEFORE the
      // wheel handler so the mouse-wheel hover check doesn't swallow clicks.
      if (align_add_line_mode != 0 && ImGui::IsWindowHovered() &&
          ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        const ImVec2 mpos = ImGui::GetMousePos();
        const double s = std::max(1e-4f, align_display_scale);
        // Click position in distorted image pixels (what we actually see on screen).
        const double u_click = (mpos.x - cur.x) / s;
        const double v_click = (mpos.y - cur.y) / s;
        // Convert to ideal pinhole pixels via iterative inverse Brown-Conrady
        // (Pinhole only). For Spherical the click coord IS the equirect pixel
        // and we draw a straight axis-aligned line -- no inverse needed.
        double u_ideal = u_click, v_ideal = v_click;
        const auto& k = src.intrinsics;
        const bool have_dist = camera_type_has_brown_conrady(src.camera_type)
          && !align_rectified && (k.k1 != 0 || k.k2 != 0 || k.k3 != 0 || k.p1 != 0 || k.p2 != 0);
        if (have_dist) {
          const double xd = (u_click - k.cx) / k.fx;
          const double yd = (v_click - k.cy) / k.fy;
          double xn = xd, yn = yd;
          for (int it = 0; it < 10; it++) {
            const double r2 = xn*xn + yn*yn, r4 = r2*r2, r6 = r4*r2;
            const double radial = 1.0 + k.k1*r2 + k.k2*r4 + k.k3*r6;
            const double dx = 2.0*k.p1*xn*yn + k.p2*(r2 + 2.0*xn*xn);
            const double dy = k.p1*(r2 + 2.0*yn*yn) + 2.0*k.p2*xn*yn;
            xn = (xd - dx) / radial;
            yn = (yd - dy) / radial;
          }
          u_ideal = k.fx * xn + k.cx;
          v_ideal = k.fy * yn + k.cy;
        }
        if (u_ideal >= 0 && u_ideal < align_img_w && v_ideal >= 0 && v_ideal < align_img_h) {
          if (align_add_line_mode == 1)      align_user_lines.emplace_back(0, u_ideal);   // V: fixed u
          else if (align_add_line_mode == 2) align_user_lines.emplace_back(1, v_ideal);   // H: fixed v
        }
        align_add_line_mode = 0;  // consume the click, back to passive
      }

      // Mouse-wheel zoom centred on the cursor -- scroll is adjusted so the
      // image pixel under the mouse stays under the mouse after zoom change.
      // Runs only while hovering the canvas; safe vs. other ImGui wheel users.
      if (ImGui::IsWindowHovered() && !ImGui::GetIO().KeyCtrl) {
        const float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
          const ImVec2 mpos = ImGui::GetMousePos();
          const float old_scale = std::max(1e-4f, align_display_scale);
          const float ix = (mpos.x - cur.x) / old_scale;
          const float iy = (mpos.y - cur.y) / old_scale;
          const float factor = (wheel > 0.0f) ? 1.15f : (1.0f / 1.15f);
          align_display_scale = std::clamp(old_scale * factor, 0.05f, 10.0f);
          const float desired_cur_x = mpos.x - ix * align_display_scale;
          const float desired_cur_y = mpos.y - iy * align_display_scale;
          ImGui::SetScrollX(ImGui::GetScrollX() + (cur.x - desired_cur_x));
          ImGui::SetScrollY(ImGui::GetScrollY() + (cur.y - desired_cur_y));
        }
      }

      // --- Colorize-from-this-camera: (re)compute on demand ---
      // Runs the actual colorize pipeline with the single current frame as
      // the camera list. Returned per-point RGB is cached and consumed in the
      // draw loop below. Runs synchronously in the UI thread -- acceptable at
      // ~50-200 ms for typical submap point counts; fires only on the click or
      // on a single param change (not every ImGui frame).
      if (align_colorize_dirty && cam_ok && !align_submap_world_pts.empty()) {
        CameraFrame single = cam;
        single.T_world_cam = T_world_cam;
        single.timestamp = cam.timestamp + src.time_shift;  // scratch override; live-tune via scalar widget
        single.located = true;
        // Spherical source -> expand single equirect cam to 6 virtual cube faces.
        auto expanded = expand_source_cams_for_projection(src, {single}, colorize_mask);
        auto cr = make_colorizer(src.params.view_selector_mode)->project(
          expanded.cams, expanded.intrinsics, align_submap_world_pts, align_submap_ints,
          align_submap_world_normals, align_submap_world_times, current_blend_params(src));
        align_colorize_rgb = std::move(cr.colors);
        align_colorize_cam_src = align_cam_src;
        align_colorize_cam_idx = align_cam_idx;
        align_last_intrinsics = src.intrinsics;
        align_last_rpy = src.rotation_rpy;
        align_last_lever = src.lever_arm;
        align_last_time_shift = src.time_shift;
        align_colorize_dirty = false;
        // Ensure the cache is marked live. Without this, the auto-invalidate
        // above (cached cam_src == -1 at init, current == 0) flipped valid off
        // right after the button set it on, requiring a second click.
        align_colorize_valid = true;
      }

      // --- Reference grid overlay (drawn BEFORE dots so dots stay on top) ---
      // Lines represent world-straight features projected through the SAME
      // model as the LiDAR dots: on Rectified = ON (pinhole) they're literally
      // straight; on Rectified = OFF (raw image) they get warped by the lens
      // distortion model so the overlay matches the image's geometry. Any bend
      // you see in a real straight feature vs the grid is residual miscalibration.
      if (align_grid_show && align_img_w > 0 && align_img_h > 0) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const int N = align_grid_lines;
        const float img_w = static_cast<float>(align_img_w);
        const float img_h = static_cast<float>(align_img_h);
        const float s = align_display_scale;
        const ImU32 black = IM_COL32(0, 0, 0, 200);
        const ImU32 white = IM_COL32(255, 255, 255, 220);
        const double fx = src.intrinsics.fx, fy = src.intrinsics.fy;
        const double cx_d = src.intrinsics.cx, cy_d = src.intrinsics.cy;
        const double k1 = src.intrinsics.k1, k2 = src.intrinsics.k2, k3 = src.intrinsics.k3;
        const double p1 = src.intrinsics.p1, p2 = src.intrinsics.p2;
        // Brown-Conrady distortion is Pinhole-only. For Spherical, lines are
        // drawn straight in equirect pixel space -- constant-u corresponds to
        // a meridian (world-vertical great circle) so verticals ARE straight
        // in the image; constant-v is only straight for the equator, but the
        // grid is still useful as a visual reference.
        const bool do_warp = camera_type_has_brown_conrady(src.camera_type)
          && !align_rectified && (k1 != 0.0 || k2 != 0.0 || k3 != 0.0 || p1 != 0.0 || p2 != 0.0);
        // Forward Brown-Conrady: takes ideal-pinhole pixel -> distorted pixel.
        auto distort_px = [&](double u, double v) {
          if (!do_warp) return ImVec2(cur.x + static_cast<float>(u) * s, cur.y + static_cast<float>(v) * s);
          const double xn = (u - cx_d) / fx;
          const double yn = (v - cy_d) / fy;
          const double r2 = xn * xn + yn * yn, r4 = r2 * r2, r6 = r4 * r2;
          const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
          const double xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
          const double yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
          const double ud = fx * xd + cx_d;
          const double vd = fy * yd + cy_d;
          return ImVec2(cur.x + static_cast<float>(ud) * s, cur.y + static_cast<float>(vd) * s);
        };
        // Sample each line as 64 segments so the curve is smooth at any zoom.
        const int segments = 64;
        std::vector<ImVec2> pts; pts.reserve(segments + 1);
        auto draw_polyline = [&](const std::vector<ImVec2>& poly) {
          // Black halo first, white on top -- visible on any background.
          dl->AddPolyline(poly.data(), static_cast<int>(poly.size()), black, 0, 3.0f);
          dl->AddPolyline(poly.data(), static_cast<int>(poly.size()), white, 0, 1.0f);
        };
        for (int i = 1; i <= N; i++) {
          const double x_img = (img_w * i) / (N + 1);
          const double y_img = (img_h * i) / (N + 1);
          // Vertical line (constant u)
          pts.clear();
          for (int k = 0; k <= segments; k++) {
            const double v = (img_h * k) / segments;
            pts.push_back(distort_px(x_img, v));
          }
          draw_polyline(pts);
          // Horizontal line (constant v)
          pts.clear();
          for (int k = 0; k <= segments; k++) {
            const double u = (img_w * k) / segments;
            pts.push_back(distort_px(u, y_img));
          }
          draw_polyline(pts);
        }
        // Image frame (outline). Warp the four edges the same way so the frame
        // stays true to the distorted image boundary.
        auto warp_edge = [&](double u0, double v0, double u1, double v1) {
          pts.clear();
          for (int k = 0; k <= segments; k++) {
            const double t = static_cast<double>(k) / segments;
            pts.push_back(distort_px(u0 + t * (u1 - u0), v0 + t * (v1 - v0)));
          }
          dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), white, 0, 1.5f);
        };
        warp_edge(0,     0,     img_w, 0);
        warp_edge(img_w, 0,     img_w, img_h);
        warp_edge(img_w, img_h, 0,     img_h);
        warp_edge(0,     img_h, 0,     0);
        // Principal point marker -- handy reference for cx/cy. Apply warp so
        // it sits exactly on the image's actual principal point location.
        const ImVec2 pp = distort_px(cx_d, cy_d);
        dl->AddCircleFilled(pp, 4.0f, black);
        dl->AddCircleFilled(pp, 2.5f, IM_COL32(255, 230, 80, 255));
      }

      // --- User-placed reference lines ---
      // Drawn independently of the grid toggle. Each line is a world-straight
      // line stored in ideal pinhole coords; forward Brown-Conrady warps it to
      // match the current raw image's distortion. Bright green for contrast vs
      // the white grid.
      if (!align_user_lines.empty() && align_img_w > 0 && align_img_h > 0) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float img_w = static_cast<float>(align_img_w);
        const float img_h = static_cast<float>(align_img_h);
        const float s = align_display_scale;
        const double fx = src.intrinsics.fx, fy = src.intrinsics.fy;
        const double cx_d = src.intrinsics.cx, cy_d = src.intrinsics.cy;
        const double k1 = src.intrinsics.k1, k2 = src.intrinsics.k2, k3 = src.intrinsics.k3;
        const double p1 = src.intrinsics.p1, p2 = src.intrinsics.p2;
        const bool do_warp = camera_type_has_brown_conrady(src.camera_type)
          && !align_rectified && (k1 != 0.0 || k2 != 0.0 || k3 != 0.0 || p1 != 0.0 || p2 != 0.0);
        auto distort_px = [&](double u, double v) {
          if (!do_warp) return ImVec2(cur.x + static_cast<float>(u) * s, cur.y + static_cast<float>(v) * s);
          const double xn = (u - cx_d) / fx;
          const double yn = (v - cy_d) / fy;
          const double r2 = xn * xn + yn * yn, r4 = r2 * r2, r6 = r4 * r2;
          const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
          const double xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
          const double yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
          return ImVec2(cur.x + static_cast<float>(fx * xd + cx_d) * s,
                        cur.y + static_cast<float>(fy * yd + cy_d) * s);
        };
        const int segments = 96;
        const ImU32 halo  = IM_COL32(0, 0, 0, 220);
        const ImU32 green = IM_COL32(100, 255, 120, 240);
        std::vector<ImVec2> pts; pts.reserve(segments + 1);
        for (const auto& [type, coord] : align_user_lines) {
          pts.clear();
          if (type == 0) {
            // Vertical reference: constant u = coord (in ideal pinhole px).
            for (int i = 0; i <= segments; i++) {
              const double v = (img_h * i) / segments;
              pts.push_back(distort_px(coord, v));
            }
          } else {
            // Horizontal reference: constant v = coord.
            for (int i = 0; i <= segments; i++) {
              const double u = (img_w * i) / segments;
              pts.push_back(distort_px(u, coord));
            }
          }
          dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), halo,  0, 3.5f);
          dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), green, 0, 1.5f);
        }
      }

      // --- Project and draw points ---
      if (cam_ok && !align_submap_world_pts.empty()) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const Eigen::Isometry3d T_cw = T_world_cam.inverse();
        const Eigen::Matrix3d R_cam = T_cw.rotation();
        const Eigen::Vector3d t_cam = T_cw.translation();
        const double fx = src.intrinsics.fx, fy = src.intrinsics.fy;
        const double cx_d = src.intrinsics.cx, cy_d = src.intrinsics.cy;
        const bool is_spherical = (src.camera_type == CameraType::Spherical);
        const bool has_dist = camera_type_has_brown_conrady(src.camera_type)
          && !align_rectified && (src.intrinsics.k1 != 0 || src.intrinsics.k2 != 0 || src.intrinsics.p1 != 0 || src.intrinsics.p2 != 0);
        const Eigen::Vector3f cam_pos = T_world_cam.translation().cast<float>();
        const float max_r_sq = align_max_range * align_max_range;
        const float min_r_sq = align_min_range * align_min_range;
        // Spherical equirect mapping: cam-frame (X fwd, Y left, Z up) ->
        //   lon = atan2(-py, px)  (range [-pi, pi])
        //   lat = atan2(pz, sqrt(px^2+py^2))  (range [-pi/2, pi/2])
        //   u = (lon + pi)/(2*pi) * W   (0..W)
        //   v = (0.5 - lat/pi) * H      (0..H)
        // No distortion, no "behind camera" rejection (any finite-range ray is visible).

        // Temporal filter: keep only points within +/- N LiDAR frames of the
        // camera's timestamp. Matches the subset the real colorize would touch
        // under time-slice, so the alignment check becomes 1:1 with scene view.
        // N = 0 disables the filter (whole submap).
        const bool use_time_filter = (align_nearest_frames > 0) && !align_submap_world_times.empty();
        const double cam_time_for_filter = cam.timestamp + src.time_shift;  // scratch override
        const double time_half_window = align_nearest_frames * align_frame_interval_s;
        auto passes_time_filter = [&](size_t pi) {
          if (!use_time_filter) return true;
          if (pi >= align_submap_world_times.size()) return true;
          return std::abs(align_submap_world_times[pi] - cam_time_for_filter) <= time_half_window;
        };

        // --- Scalar normalization range (per frame, only over VISIBLE points) ---
        // Without this, when most points cluster in a narrow sub-band of the
        // user's max-range the colormap collapses into one colour (the "blob").
        // Cheap (~1-2 ms per frame for 100k points); adapts to camera movement.
        float scalar_lo = std::numeric_limits<float>::max();
        float scalar_hi = std::numeric_limits<float>::lowest();
        {
          const int mode = align_point_color_mode;  // 0=Int 1=Range 2=Depth 4=Weight
          const bool have_ints = !align_submap_ints.empty();
          const bool cache_ok_for_weight = (align_winner_sm == best_sm && align_winner_src == align_cam_src
                                            && align_winner_weight_vec.size() == align_submap_world_pts.size());
          for (size_t pi = 0; pi < align_submap_world_pts.size(); pi++) {
            if (!passes_time_filter(pi)) continue;
            const float dsq = (align_submap_world_pts[pi] - cam_pos).squaredNorm();
            if (dsq > max_r_sq || dsq < min_r_sq) continue;
            const Eigen::Vector3d p_cam = R_cam * align_submap_world_pts[pi].cast<double>() + t_cam;
            // For Pinhole, depth = +X (forward); points behind camera get culled.
            // For Spherical, "depth" degrades to radial distance -- no hemisphere
            // cull, since the 360 panorama sees in every direction.
            const double depth = is_spherical ? p_cam.norm() : p_cam.x();
            if (!is_spherical && depth <= 0.1) continue;
            float s = 0.0f;
            if (mode == 0)      s = have_ints ? align_submap_ints[pi] : 0.0f;
            else if (mode == 1) s = std::sqrt(dsq);
            else if (mode == 2) s = static_cast<float>(depth);
            else if (mode == 4) s = cache_ok_for_weight ? align_winner_weight_vec[pi] : 0.0f;
            else continue;
            scalar_lo = std::min(scalar_lo, s);
            scalar_hi = std::max(scalar_hi, s);
          }
          if (!(scalar_hi > scalar_lo)) {
            // All values equal or no visible points -- guard against div-by-zero.
            scalar_lo = 0.0f; scalar_hi = 1.0f;
          }
        }
        // Intensity range for the brightness threshold gate (keeps working even
        // when the color mode is Range/Depth/Weight). Uses the submap-level
        // 5%/95% percentile cache populated when the submap was loaded.
        const float ithresh_lo = align_intensity_range.x();
        const float ithresh_hi = std::max(ithresh_lo + 1e-4f, align_intensity_range.y());
        int drawn = 0;
        for (size_t pi = 0; pi < align_submap_world_pts.size(); pi++) {
          if (!passes_time_filter(pi)) continue;
          const float dsq = (align_submap_world_pts[pi] - cam_pos).squaredNorm();
          if (dsq > max_r_sq || dsq < min_r_sq) continue;
          const Eigen::Vector3d p_cam = R_cam * align_submap_world_pts[pi].cast<double>() + t_cam;
          double u, v;
          double depth;  // used below for Depth color mode
          if (is_spherical) {
            // Equirectangular projection: every direction is visible.
            const double r = p_cam.norm();
            if (r < 1e-6) continue;  // point at camera origin -- undefined direction
            depth = r;
            const double lon = std::atan2(-p_cam.y(), p_cam.x());          // [-pi, pi]
            const double lat = std::atan2(p_cam.z(), std::sqrt(p_cam.x()*p_cam.x() + p_cam.y()*p_cam.y()));  // [-pi/2, pi/2]
            u = (lon + M_PI) / (2.0 * M_PI) * static_cast<double>(align_img_w);
            v = (0.5 - lat / M_PI) * static_cast<double>(align_img_h);
          } else {
            depth = p_cam.x();
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
            u = fx * xn + cx_d;
            v = fy * yn + cy_d;
          }
          if (u < 0 || u >= align_img_w || v < 0 || v >= align_img_h) continue;
          // Intensity gate -- always against the submap intensity range so the
          // threshold slider means the same thing regardless of active color mode.
          const float in_norm_gate = align_submap_ints.empty() ? 0.0f
            : std::clamp((align_submap_ints[pi] - ithresh_lo) / (ithresh_hi - ithresh_lo), 0.0f, 1.0f);
          if (align_bright_threshold > 0.0f && in_norm_gate < align_bright_threshold) continue;
          // Winner/Weight viz requires fresh cache for this submap+source
          const bool cache_usable = (align_winner_sm == best_sm && align_winner_src == align_cam_src
                                     && pi < align_winner_frame_idx.size());
          if (align_point_color_mode == 3) {  // Winner-mask: only points where THIS camera won
            if (!cache_usable) continue;
            if (align_winner_frame_idx[pi] != align_cam_idx) continue;
          }
          // Color -- scalar modes go through the selected colormap; Winner-mask stays categorical.
          // When the Colorize-from-this-camera cache is active, use the projected
          // RGB instead (falls back to scalar for gray-sentinel uncolored points).
          const int a8 = std::clamp(static_cast<int>(align_point_alpha * 255.0f), 0, 255);
          ImU32 col;
          bool used_rgb = false;
          bool skip_uncolored = false;
          if (align_colorize_valid && pi < align_colorize_rgb.size()) {
            const auto& c = align_colorize_rgb[pi];
            const bool is_sentinel =
              std::abs(c.x() - 0.5f) < 1e-3f && std::abs(c.y() - 0.5f) < 1e-3f && std::abs(c.z() - 0.5f) < 1e-3f;
            if (!is_sentinel) {
              const int rr = std::clamp(static_cast<int>(c.x() * 255.0f), 0, 255);
              const int gg = std::clamp(static_cast<int>(c.y() * 255.0f), 0, 255);
              const int bb = std::clamp(static_cast<int>(c.z() * 255.0f), 0, 255);
              col = IM_COL32(rr, gg, bb, a8);
              used_rgb = true;
            } else if (align_colorize_hide_uncolored) {
              // Colorize cache active + user wants ground-truth view: skip
              // points this camera didn't actually colour (time-slice / NCC /
              // incidence / occlusion rejects). Matches scene-view coverage.
              skip_uncolored = true;
            }
          }
          if (skip_uncolored) continue;
          // Normalize the active scalar against the per-frame min/max we gathered
          // in the pre-pass. This is what spreads the full colormap across the
          // actual visible data range instead of against the slider (which would
          // give a "blob" when points cluster in a narrow band of the slider).
          const float inv_scalar_span = 1.0f / (scalar_hi - scalar_lo);
          auto norm = [&](float s) { return std::clamp((s - scalar_lo) * inv_scalar_span, 0.0f, 1.0f); };
          if (used_rgb) {
            // Fall through to the draw below -- col is already set.
          } else if (align_point_color_mode == 0) {     // Intensity
            const float t = align_submap_ints.empty() ? 0.0f : norm(align_submap_ints[pi]);
            col = scalar_to_imu32(align_colormap_sel, t, a8);
          } else if (align_point_color_mode == 1) {     // Range
            col = scalar_to_imu32(align_colormap_sel, norm(std::sqrt(dsq)), a8);
          } else if (align_point_color_mode == 2) {     // Depth
            col = scalar_to_imu32(align_colormap_sel, norm(static_cast<float>(depth)), a8);
          } else if (align_point_color_mode == 3) {     // Winner-mask (categorical)
            col = IM_COL32(80, 255, 120, a8);
          } else {                                      // Weight
            if (!cache_usable || align_weight_max_cached <= 0.0f) { col = IM_COL32(120, 120, 120, a8); }
            else {
              col = scalar_to_imu32(align_colormap_sel, norm(align_winner_weight_vec[pi]), a8);
            }
          }
          const ImVec2 sp(cur.x + static_cast<float>(u) * align_display_scale,
                          cur.y + static_cast<float>(v) * align_display_scale);
          dl->AddCircleFilled(sp, align_point_size, col);
          drawn++;
        }
        // Status line on top of image
        const char* mode_label = "?";
        switch (align_point_color_mode) {
          case 0: mode_label = "Int";    break;
          case 1: mode_label = "Range";  break;
          case 2: mode_label = "Depth";  break;
          case 3: mode_label = "Winner"; break;
          case 4: mode_label = "Weight"; break;
        }
        char buf[224]; std::snprintf(buf, sizeof(buf),
          "sm=%d  pts_drawn=%d  native=%dx%d  scale=%.2fx  shift=%.2fs  RPY=(%.2f,%.2f,%.2f)  %s[%.2f..%.2f]",
          best_sm, drawn, align_img_w, align_img_h, align_display_scale,
          src.time_shift, src.rotation_rpy.x(), src.rotation_rpy.y(), src.rotation_rpy.z(),
          mode_label, scalar_lo, scalar_hi);
        dl->AddText(ImVec2(cur.x + 6, cur.y + 6), IM_COL32(255, 255, 0, 255), buf);
      } else if (!cam_ok) {
        ImGui::GetWindowDrawList()->AddText(ImVec2(cur.x + 6, cur.y + 6),
          IM_COL32(255, 80, 80, 255), "Camera not locatable (timestamp out of trajectory range).");
      }
      ImGui::EndChild();
    }
    ImGui::End();
  });

  // Virtual LiDAR cameras: walks the trajectory at a user interval, renders
  // 6-face cube (or a subset) from the LiDAR data at each anchor with locked
  // pose + zero-distortion pinhole intrinsics. Output is a set of JPGs that
  // Metashape imports as control-anchor cameras; real cameras BA-refine
  // against them. Automation of the "hand-placed marker" workflow at scale.
  viewer->register_ui_callback("virtual_cameras_window", [this] {
    if (!show_virtual_cameras_window) return;
    ImGui::SetNextWindowSize(ImVec2(540, 560), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Virtual LiDAR Cameras", &show_virtual_cameras_window)) { ImGui::End(); return; }

    // Output dir
    ImGui::TextDisabled("Output folder");
    char dir_buf[512];
    std::snprintf(dir_buf, sizeof(dir_buf), "%s", vc_output_dir.c_str());
    ImGui::SetNextItemWidth(-110);
    if (ImGui::InputText("##vc_dir", dir_buf, sizeof(dir_buf))) vc_output_dir = dir_buf;
    ImGui::SameLine();
    if (ImGui::Button("Browse##vc")) {
      const std::string chosen = pfd::select_folder("Choose output dir for virtual cameras",
        vc_output_dir.empty() ? (loaded_map_path.empty() ? "." : loaded_map_path) : vc_output_dir).result();
      if (!chosen.empty()) vc_output_dir = chosen;
    }
    ImGui::Separator();

    // Placement-mode radio. "Waypoints" walks the trajectory dropping anchors
    // every N metres (legacy). "Per RGB camera" places a virtual LiDAR camera
    // at each real RGB frame's estimated world pose -- the 1:1 co-located mode
    // intended for SFM anchoring; cheap to match because the virtual twin
    // shares the same viewpoint as the real image (within a few cm).
    ImGui::TextDisabled("Placement mode");
    ImGui::RadioButton("Waypoints along trajectory##vcmode", &vc_placement_mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Per RGB camera (co-located)##vcmode", &vc_placement_mode, 1);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Emits one virtual LiDAR-intensity image per real RGB frame at the frame's\n"
      "estimated world pose (from the Colorize extrinsic). Virtual + real sit in\n"
      "the same bundle with ~10 cm offset -- SFM matching is nearly guaranteed\n"
      "to fire, giving BA a locked LiDAR-frame anchor for every real photo.\n"
      "Requires the source to have been 'Located' in the Colorize window.");
    ImGui::Separator();

    if (vc_placement_mode == 0) {

    // Placement
    ImGui::TextDisabled("Placement");
    ImGui::SetNextItemWidth(160);
    ImGui::DragFloat("Interval (m)##vc", &vc_interval_m, 0.5f, 1.0f, 500.0f, "%.1f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Distance between consecutive virtual cameras along the trajectory.\n"
      "5-10 m = dense BA constraints, 20-50 m = lighter. Denser is usually\n"
      "safer for feature-sparse LiDARs (Livox Horizon).");
    ImGui::SetNextItemWidth(160);
    ImGui::DragFloat("Context radius (m)##vc", &vc_context_radius_m, 1.0f, 5.0f, 500.0f, "%.0f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "LiDAR points within this radius of an anchor are included in its render.\n"
      "Smaller = faster render, tighter depth. Larger = more context / background.");
    ImGui::Separator();

    // Face selection
    ImGui::TextDisabled("Cube faces to render");
    const char* face_labels[6] = { "+X (fwd)", "-X (back)", "+Y (left)", "-Y (right)", "+Z (up/sky)", "-Z (down/gnd)" };
    for (int f = 0; f < 6; f++) {
      if (f) ImGui::SameLine();
      ImGui::Checkbox((std::string(face_labels[f]) + "##vcf" + std::to_string(f)).c_str(), &vc_face_enabled[f]);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "+Z (sky) is off by default -- rarely has useful features.\n"
      "-Z (down/ground) is critical on Livox Horizon: road markings are the best\n"
      "matchable content. Keep ON unless you have a reason to drop it.");
    ImGui::Separator();

    // Resolution
    ImGui::TextDisabled("Face resolution");
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt("pixels/side##vc", &vc_face_size, 64, 256);
    vc_face_size = std::clamp(vc_face_size, 256, 8192);
    ImGui::SameLine();
    if (ImGui::SmallButton(" /2##vcres")) vc_face_size = std::max(256, vc_face_size / 2);
    ImGui::SameLine();
    if (ImGui::SmallButton(" x2##vcres")) vc_face_size = std::min(8192, vc_face_size * 2);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Pixels per cube-face side. 1920 is a good balance; 3840 for Pandar 128\n"
      "density to extract many features; 1024 for quick test runs.");
    ImGui::Separator();

    // Filters
    ImGui::TextDisabled("Filters");
    ImGui::Checkbox("Ground only (requires aux_ground.bin)##vc", &vc_ground_only);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Render only points tagged as ground by PatchWork++. On Livox Horizon this\n"
      "concentrates the feature-rich regions (asphalt, road markings, kerbs) so\n"
      "the matcher has a better shot. Requires a prior PatchWork++ classify run\n"
      "that wrote aux_ground.bin per frame.");
    ImGui::Checkbox("Also render RGB (requires aux_rgb.bin)##vc", &vc_render_rgb);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "In addition to intensity, render an RGB image per face using colors from\n"
      "a prior Colorize Apply run. Doubles the render time + disk but gives the\n"
      "matcher both modalities. SIFT works on intensity so this is optional.");
    ImGui::Checkbox("Embed UTM in EXIF##vc", &vc_embed_exif_gps);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Write virtual cameras' world positions into EXIF GPS tags (using UTM from\n"
      "gnss_datum.json + our session-local offset). Lets Metashape/RealityScan\n"
      "seed pose priors from the EXIF even without the BlocksExchange import.");
    ImGui::Separator();

    // Action buttons
    ImGui::BeginDisabled(vc_running);
    if (ImGui::Button("Preview placement (no render)##vc")) {
      // Fill vc_preview_anchors from the trajectory at current interval.
      if (!trajectory_built) build_trajectory();
      std::vector<TrajRecord> tr;
      tr.reserve(trajectory_data.size());
      for (const auto& rec : trajectory_data) {
        tr.push_back({rec.stamp, rec.pose, rec.cumulative_dist});
      }
      auto anchors = place_virtual_cameras(tr, vc_interval_m);
      vc_preview_anchors.clear();
      vc_preview_orient.clear();
      vc_preview_anchors.reserve(anchors.size());
      vc_preview_orient.reserve(anchors.size());
      for (const auto& a : anchors) {
        vc_preview_anchors.push_back(a.T_world_cam.translation().cast<float>());
        vc_preview_orient.push_back(a.T_world_cam.rotation().cast<float>());
      }
      vc_anchors_placed_last = anchors.size();
      vc_status = "Preview: " + std::to_string(anchors.size()) + " anchors placed along "
                  + std::to_string(static_cast<int>(tr.empty() ? 0 : tr.back().cumulative_dist)) + " m of trajectory";
      logger->info("[VirtualCams] {}", vc_status);
    }
    ImGui::SameLine();
    const bool can_render = !vc_output_dir.empty() &&
                            std::any_of(std::begin(vc_face_enabled), std::end(vc_face_enabled), [](bool b){ return b; });
    ImGui::BeginDisabled(!can_render);
    if (ImGui::Button("Render all##vc")) {
      vc_running = true;
      vc_status = "TODO: render worker not yet implemented; placement + UI + preview are in. "
                  "Files would land in: " + vc_output_dir;
      // NOTE: the render worker goes here next session. It will:
      //  1. Walk trajectory via place_virtual_cameras.
      //  2. Build a per-anchor CalibrationContext by filtering world_points
      //     within vc_context_radius_m (optionally ground-only).
      //  3. For each enabled face, call render_intensity_image with
      //     cube_face_intrinsics(vc_face_size) and T_world_cam * cube_face_rotation(f).
      //  4. Save JPG + optional RGB render, populate EXIF if enabled.
      //  5. Emit a manifest.json with per-anchor poses for BlocksExchange.
      vc_running = false;
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();

    if (!vc_status.empty()) {
      ImGui::Separator();
      ImGui::TextWrapped("%s", vc_status.c_str());
    }

    } else {
      // ---------------- Per RGB camera mode ----------------
      // Lazy-seed factory scanner presets on first enter.
      vc_pcam_seed_factory_presets();

      // --- Presets dropdown (scanner-tuned defaults).
      {
        std::vector<const char*> names;
        for (const auto& p : vc_pcam_presets) names.push_back(p.name.c_str());
        vc_pcam_preset_idx = std::clamp(vc_pcam_preset_idx, 0,
                                         static_cast<int>(vc_pcam_presets.size()) - 1);
        ImGui::SetNextItemWidth(220);
        ImGui::Combo("Preset##vcp", &vc_pcam_preset_idx, names.data(),
                      static_cast<int>(names.size()));
        ImGui::SameLine();
        if (ImGui::Button("Apply##vcp_preset") && !vc_pcam_presets.empty()) {
          vc_pcam_apply_preset(vc_pcam_presets[vc_pcam_preset_idx]);
          vc_status = std::string("Applied preset: ") + vc_pcam_presets[vc_pcam_preset_idx].name;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Overwrites the context + render + face + output settings below with a\n"
          "scanner-specific defaults bundle. Currently factory-only; save/rename\n"
          "and persistence to disk land in a later pass.");
        ImGui::Separator();
      }

      if (image_sources.empty()) {
        ImGui::TextDisabled("No image sources loaded. Open Colorize > Image folder > Add folder...");
      } else {
        // Single source dropdown drives everything: preview, test matches, and
        // batch export. The earlier multi-select checkboxes were redundant with
        // this dropdown for the common case, so they've been removed.
        std::vector<std::string> src_labels;
        for (size_t i = 0; i < image_sources.size(); i++) {
          src_labels.push_back(std::string(image_sources[i].name) + " [" +
            camera_type_label(image_sources[i].camera_type) + ", " +
            std::to_string(image_sources[i].frames.size()) + " frames]");
        }
        std::vector<const char*> lptrs;
        for (auto& s : src_labels) lptrs.push_back(s.c_str());
        vc_pcam_active_src = std::clamp(vc_pcam_active_src, 0,
                                         static_cast<int>(image_sources.size()) - 1);
        {
          ImGui::SetNextItemWidth(360);
          if (ImGui::Combo("Source##vcp", &vc_pcam_active_src, lptrs.data(), static_cast<int>(lptrs.size()))) {
            vc_pcam_preview_frame = 0;
            vc_pcam_preview_dirty = true;
          }
          auto& src = image_sources[vc_pcam_active_src];
          const int nframes = static_cast<int>(src.frames.size());
          if (nframes > 0) {
            int f = std::clamp(vc_pcam_preview_frame, 0, nframes - 1);
            ImGui::SetNextItemWidth(260);
            // Slider scrubs without triggering a render -- the context rebuild
            // is expensive enough that live-rendering every drag step feels
            // jerky and makes it hard to land on a specific frame. We only
            // trigger the preview on release (or on the arrow / Jump paths).
            ImGui::SetNextItemWidth(220);
            if (ImGui::SliderInt("##vcp_frame_slider", &f, 0, nframes - 1, "%d")) {
              vc_pcam_preview_frame = f;
            }
            if (ImGui::IsItemDeactivatedAfterEdit()) {
              vc_pcam_preview_dirty = true;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Drag to scrub without rendering. Releases trigger the preview.\n"
              "Ctrl+click to type an exact frame, or use the Jump box beside it.");
            ImGui::SameLine();
            if (ImGui::ArrowButton("##vcp_prev", ImGuiDir_Left)) {
              vc_pcam_preview_frame = std::max(0, f - 1);
              vc_pcam_preview_dirty = true;
            }
            ImGui::SameLine();
            if (ImGui::ArrowButton("##vcp_next", ImGuiDir_Right)) {
              vc_pcam_preview_frame = std::min(nframes - 1, f + 1);
              vc_pcam_preview_dirty = true;
            }
            // Explicit "jump to frame" input: type a number + Enter. Renders
            // only when the user presses Enter (EnterReturnsTrue), so typing
            // intermediate digits doesn't thrash the preview.
            ImGui::SameLine(); ImGui::SetNextItemWidth(90);
            int jump = vc_pcam_preview_frame;
            if (ImGui::InputInt("Jump##vcp_jump", &jump, 0, 0,
                                 ImGuiInputTextFlags_EnterReturnsTrue)) {
              vc_pcam_preview_frame = std::clamp(jump, 0, nframes - 1);
              vc_pcam_preview_dirty = true;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Type an exact frame number and press Enter to jump there.");
            if (!src.frames[f].located) {
              ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1),
                "Frame not located -- run Colorize > Locate first.");
            }
          }
        }
        ImGui::Separator();

        // --- Context window knobs (shared struct; set per-type defaults below).
        ImGui::TextDisabled("Context window");
        if (ImGui::Button("Pinhole defaults##vcp")) {
          vc_pcam_ctx_opts.use_time_window = false;
          vc_pcam_ctx_opts.n_frames_before = 15;
          vc_pcam_ctx_opts.n_frames_after  = 15;
          vc_pcam_ctx_opts.directional_filter = true;
          vc_pcam_ctx_opts.directional_threshold_deg = 60.0f;
          vc_pcam_ctx_opts.min_range = 0.5f;
          vc_pcam_ctx_opts.max_range = 80.0f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Spherical (360) defaults##vcp")) {
          vc_pcam_ctx_opts.use_time_window = true;
          vc_pcam_ctx_opts.time_before_s   = 20.0;
          vc_pcam_ctx_opts.time_after_s    = 20.0;
          vc_pcam_ctx_opts.directional_filter = false;
          vc_pcam_ctx_opts.min_range = 0.5f;
          vc_pcam_ctx_opts.max_range = 80.0f;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Spherical presets use a wider time window and disable the directional\n"
          "filter so the faces looking backward/sideways have LiDAR context to\n"
          "render against, not just the instantaneous forward-scanning frames.");

        ImGui::Checkbox("Use time window##vcp", &vc_pcam_ctx_opts.use_time_window);
        if (vc_pcam_ctx_opts.use_time_window) {
          float tb = static_cast<float>(vc_pcam_ctx_opts.time_before_s);
          float ta = static_cast<float>(vc_pcam_ctx_opts.time_after_s);
          ImGui::SetNextItemWidth(100); ImGui::DragFloat("before (s)##vcp", &tb, 0.1f, 0.1f, 120.0f, "%.1f");
          ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::DragFloat("after (s)##vcp", &ta, 0.1f, 0.1f, 120.0f, "%.1f");
          vc_pcam_ctx_opts.time_before_s = tb; vc_pcam_ctx_opts.time_after_s = ta;
        } else {
          ImGui::SetNextItemWidth(100); ImGui::DragInt("N before##vcp", &vc_pcam_ctx_opts.n_frames_before, 1, 0, 1000);
          ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::DragInt("N after##vcp", &vc_pcam_ctx_opts.n_frames_after, 1, 0, 1000);
        }
        ImGui::Checkbox("Directional filter##vcp", &vc_pcam_ctx_opts.directional_filter);
        if (vc_pcam_ctx_opts.directional_filter) {
          ImGui::SameLine(); ImGui::SetNextItemWidth(100);
          ImGui::DragFloat("deg##vcp", &vc_pcam_ctx_opts.directional_threshold_deg, 1.0f, 5.0f, 180.0f, "%.0f");
        }
        ImGui::SetNextItemWidth(100); ImGui::DragFloat("min range (m)##vcp", &vc_pcam_ctx_opts.min_range, 0.05f, 0.1f, 50.0f, "%.1f");
        ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::DragFloat("max range (m)##vcp", &vc_pcam_ctx_opts.max_range, 1.0f, 5.0f, 500.0f, "%.0f");
        ImGui::Separator();

        // --- Render knobs (splat + intensity + colormap).
        ImGui::TextDisabled("Render");

        // Splat mode + its parameters. Three modes: Formula (1/depth),
        // Fixed (one size for everything), Stepped (user-defined bins).
        const char* splat_modes[] = { "Formula (1/depth)", "Fixed", "Linear ramp" };
        int splat_mode_idx = static_cast<int>(vc_pcam_render_opts.splat_mode);
        ImGui::SetNextItemWidth(160);
        if (ImGui::Combo("Splat mode##vcp", &splat_mode_idx, splat_modes, IM_ARRAYSIZE(splat_modes))) {
          vc_pcam_render_opts.splat_mode = static_cast<IntensityRenderOptions::SplatMode>(splat_mode_idx);
          vc_pcam_preview_dirty = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Formula: splat = clamp(round(3 * ref / depth), min, max). Smooth 1/depth taper.\n"
          "Fixed: one radius for every point, regardless of distance.\n"
          "Linear ramp: user-defined (depth, size) knots. Size lerps between knots\n"
          "             so each knot meets the next without a step. Past the last\n"
          "             knot the size stays flat.");

        switch (vc_pcam_render_opts.splat_mode) {
          case IntensityRenderOptions::SplatMode::Formula: {
            ImGui::SetNextItemWidth(120);
            float nd = static_cast<float>(vc_pcam_render_opts.near_depth_m);
            if (ImGui::DragFloat("Ref depth (m)##vcp", &nd, 0.1f, 0.5f, 20.0f, "%.2f")) {
              vc_pcam_render_opts.near_depth_m = nd; vc_pcam_preview_dirty = true;
            }
            ImGui::SameLine(); ImGui::SetNextItemWidth(80);
            if (ImGui::DragInt("Min splat##vcp", &vc_pcam_render_opts.min_splat_px, 1, 0, 10)) vc_pcam_preview_dirty = true;
            ImGui::SameLine(); ImGui::SetNextItemWidth(80);
            if (ImGui::DragInt("Max splat##vcp", &vc_pcam_render_opts.max_splat_px, 1, 1, 20)) vc_pcam_preview_dirty = true;
            break;
          }
          case IntensityRenderOptions::SplatMode::Fixed: {
            ImGui::SetNextItemWidth(120);
            if (ImGui::DragInt("Splat size (px)##vcp", &vc_pcam_render_opts.fixed_splat_px, 1, 0, 20)) vc_pcam_preview_dirty = true;
            break;
          }
          case IntensityRenderOptions::SplatMode::LinearRamp: {
            auto& ranges = vc_pcam_render_opts.splat_ranges;
            // Seed a first knot the first time the user opens this mode so the
            // table isn't empty and confusing.
            if (ranges.empty()) ranges.push_back({0.0, 5});
            int remove_idx = -1;
            for (size_t i = 0; i < ranges.size(); i++) {
              ImGui::PushID(static_cast<int>(i));
              // First knot is locked to 0 m so the function always covers depth 0+.
              ImGui::SetNextItemWidth(100);
              if (i == 0) {
                ImGui::BeginDisabled();
                float d0 = 0.0f;
                ImGui::DragFloat("at (m)", &d0, 0.0f, 0.0f, 0.0f, "%.2f");
                ImGui::EndDisabled();
                ranges[0].start_depth_m = 0.0;
              } else {
                float d = static_cast<float>(ranges[i].start_depth_m);
                if (ImGui::DragFloat("at (m)", &d, 0.1f, 0.01f, 500.0f, "%.2f")) {
                  ranges[i].start_depth_m = d; vc_pcam_preview_dirty = true;
                }
              }
              ImGui::SameLine(); ImGui::SetNextItemWidth(80);
              if (ImGui::DragInt("size (px)", &ranges[i].splat_px, 1, 0, 20)) vc_pcam_preview_dirty = true;
              if (i > 0) {
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) remove_idx = static_cast<int>(i);
              }
              ImGui::PopID();
            }
            if (remove_idx >= 0) {
              ranges.erase(ranges.begin() + remove_idx);
              vc_pcam_preview_dirty = true;
            }
            if (ImGui::SmallButton("+ Add knot##vcp_sr")) {
              // New knot 2 m past the last one, size one smaller (typical taper).
              const auto& last = ranges.back();
              ranges.push_back({last.start_depth_m + 2.0, std::max(1, last.splat_px - 1)});
              vc_pcam_preview_dirty = true;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
              "Adds a (depth, size) knot. Between consecutive knots the size\n"
              "interpolates linearly. Past the last knot the size stays flat.");
            break;
          }
        }

        if (ImGui::Checkbox("Round splats##vcp_round", &vc_pcam_render_opts.round_splats)) vc_pcam_preview_dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Off = square (dy/dx AABB stamp, cheapest).\n"
          "On = disk (dx*dx+dy*dy <= r*r gate). Matches the 3D viewer's rounded\n"
          "points and avoids the faint axis-aligned edges a square stamp imprints\n"
          "at coarse splat sizes.");

        ImGui::SameLine();
        if (ImGui::Checkbox("Non-linear intensity##vcp", &vc_pcam_render_opts.non_linear_intensity)) vc_pcam_preview_dirty = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Gamma-lift + top-5%% clamp so retroreflective markings spike to white.\n"
          "Off = linear (2nd-99th percentile) stretch -- softer, may match better\n"
          "on some scenes. A/B against Metashape matching to decide.");
        ImGui::SameLine(); ImGui::SetNextItemWidth(140);
        const char* cmap_names[] = { "Grayscale", "Inverted", "Turbo", "Viridis", "Cividis" };
        int cmap_idx = static_cast<int>(vc_pcam_render_opts.colormap);
        if (ImGui::Combo("Scale##vcp", &cmap_idx, cmap_names, IM_ARRAYSIZE(cmap_names))) {
          vc_pcam_render_opts.colormap = static_cast<IntensityRenderOptions::Colormap>(cmap_idx);
          vc_pcam_preview_dirty = true;
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Grayscale##vcp_gs", &vc_pcam_render_opts.return_to_grayscale_after_colormap)) {
          vc_pcam_preview_dirty = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Force the final output to single-channel grayscale regardless of the\n"
          "Scale choice. Scale = Grayscale or Inverted are already single-channel\n"
          "so the flag is a no-op there; Turbo/Viridis/Cividis normally produce\n"
          "RGB -- with this flag ON they pass through cv::applyColorMap and then\n"
          "get BT.601-converted back to luminance. The colormap's non-linear hue\n"
          "sweep acts as a contrast remap on the way, pulling out faint features\n"
          "(road markings, sign glyphs) a flat linear grayscale washes out.\n"
          "LightGlue / SIFT get single-channel input either way.");

        // Intensity range lock. Per-frame percentile drift makes the same scene
        // look like it's being re-exposed between renders; locking the 2nd /
        // 95th / 99th percentiles from a representative preview frame freezes
        // the synthetic-exposure curve for everything that follows.
        if (vc_pcam_render_opts.intensity_locked) {
          ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1),
            "Intensity locked: %.1f / %.1f / %.1f",
            vc_pcam_render_opts.intensity_locked_imin,
            vc_pcam_render_opts.intensity_locked_ibulk,
            vc_pcam_render_opts.intensity_locked_imax);
          ImGui::SameLine();
          if (ImGui::SmallButton("Clear lock##vcp_lock")) {
            vc_pcam_render_opts.intensity_locked = false;
            vc_pcam_preview_dirty = true;
          }
        } else {
          ImGui::BeginDisabled(!vc_pcam_have_last_percentiles);
          if (ImGui::SmallButton("Lock range from preview##vcp_lock")) {
            vc_pcam_render_opts.intensity_locked = true;
            vc_pcam_render_opts.intensity_locked_imin  = vc_pcam_last_imin;
            vc_pcam_render_opts.intensity_locked_ibulk = vc_pcam_last_ibulk;
            vc_pcam_render_opts.intensity_locked_imax  = vc_pcam_last_imax;
            vc_pcam_preview_dirty = true;
          }
          ImGui::EndDisabled();
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            vc_pcam_have_last_percentiles
              ? "Capture (2nd / 95th / 99th) intensity percentiles from the last\n"
                "preview's context and use them for every subsequent render. Stops\n"
                "contrast from drifting frame to frame."
              : "Run a preview first -- the lock grabs percentiles from its context.");
        }
        ImGui::Separator();

        // --- Cube faces (Spherical only).
        const bool any_spherical =
          (vc_pcam_active_src >= 0 && vc_pcam_active_src < static_cast<int>(image_sources.size()) &&
           image_sources[vc_pcam_active_src].camera_type == CameraType::Spherical);
        if (any_spherical) {
          ImGui::TextDisabled("Cube faces (Spherical sources)");
          const char* face_labels[6] = { "Front", "Back", "Left", "Right", "Up", "Down" };
          for (int f = 0; f < 6; f++) {
            if (f) ImGui::SameLine();
            ImGui::Checkbox((std::string(face_labels[f]) + "##vcpf" + std::to_string(f)).c_str(), &vc_face_enabled[f]);
          }
          ImGui::SetNextItemWidth(120);
          ImGui::DragInt("Face size (px)##vcp", &vc_face_size, 64, 256, 8192);
          // Face-size buttons also rescale the pixel-valued splat knobs so a
          // 4096->8192 doubling keeps the effective angular splat size.
          auto scale_splats_fs = [this](double mul) {
            auto& ro = vc_pcam_render_opts;
            auto si = [mul](int v) { return std::max(0, static_cast<int>(std::round(v * mul))); };
            ro.min_splat_px   = si(ro.min_splat_px);
            ro.max_splat_px   = std::max(ro.min_splat_px, si(ro.max_splat_px));
            ro.fixed_splat_px = si(ro.fixed_splat_px);
            for (auto& r : ro.splat_ranges) r.splat_px = si(r.splat_px);
            vc_pcam_preview_dirty = true;
          };
          ImGui::SameLine();
          if (ImGui::SmallButton(" /2##vcp_fs")) {
            vc_face_size = std::max(256, vc_face_size / 2);
            scale_splats_fs(0.5);
          }
          ImGui::SameLine();
          if (ImGui::SmallButton(" x2##vcp_fs")) {
            vc_face_size = std::min(8192, vc_face_size * 2);
            scale_splats_fs(2.0);
          }
          ImGui::Separator();
        }

        // --- Pinhole render size (shown when any enabled source is Pinhole).
        //     Default 0 = native intrinsics; override here to super-sample the
        //     rasterizer (more pixels, sharper per-splat placement) when the
        //     LiDAR density can support it. x2 pushes a 3840x2160 cam to
        //     7680x4320; /2 halves. Splat sizes don't auto-scale, so if you
        //     double the canvas you may want to bump splat px too.
        const bool any_pinhole =
          (vc_pcam_active_src >= 0 && vc_pcam_active_src < static_cast<int>(image_sources.size()) &&
           image_sources[vc_pcam_active_src].camera_type == CameraType::Pinhole);
        if (any_pinhole) {
          ImGui::TextDisabled("Pinhole render size");
          // Eagerly resolve 0 -> native on display so the drag shows the
          // actual pixel count (e.g. 3840) instead of a meaningless "0".
          // The batch / preview paths still accept 0 as a sentinel but it
          // never leaves this block unset now.
          const auto& psrc_here = image_sources[vc_pcam_active_src];
          if (vc_pcam_render_w <= 0) vc_pcam_render_w = psrc_here.intrinsics.width;
          if (vc_pcam_render_h <= 0) vc_pcam_render_h = psrc_here.intrinsics.height;
          ImGui::SetNextItemWidth(100);
          ImGui::DragInt("W##vcp_pw", &vc_pcam_render_w, 8, 256, 16000);
          ImGui::SameLine(); ImGui::SetNextItemWidth(100);
          ImGui::DragInt("H##vcp_ph", &vc_pcam_render_h, 8, 144, 12000);
          // Helper: scale every pixel-valued splat knob by `mul` so a canvas
          // resize preserves the effective angular splat size. Depth-valued
          // fields (near_depth_m, LinearRamp start_depth_m) are NOT scaled --
          // those live in world metres and are independent of canvas pixels.
          auto scale_splats = [this](double mul) {
            auto& ro = vc_pcam_render_opts;
            auto scale_int = [mul](int v) {
              return std::max(0, static_cast<int>(std::round(v * mul)));
            };
            ro.min_splat_px   = scale_int(ro.min_splat_px);
            ro.max_splat_px   = std::max(ro.min_splat_px, scale_int(ro.max_splat_px));
            ro.fixed_splat_px = scale_int(ro.fixed_splat_px);
            for (auto& r : ro.splat_ranges) r.splat_px = scale_int(r.splat_px);
            vc_pcam_preview_dirty = true;
          };

          ImGui::SameLine();
          if (ImGui::SmallButton(" /2##vcp_ph_half")) {
            // Resolve native-if-zero, then halve. Splat sizes halve too.
            const auto& asrc = image_sources[vc_pcam_active_src];
            if (vc_pcam_render_w <= 0) vc_pcam_render_w = asrc.intrinsics.width;
            if (vc_pcam_render_h <= 0) vc_pcam_render_h = asrc.intrinsics.height;
            vc_pcam_render_w = std::max(256, vc_pcam_render_w / 2);
            vc_pcam_render_h = std::max(144, vc_pcam_render_h / 2);
            scale_splats(0.5);
          }
          ImGui::SameLine();
          if (ImGui::SmallButton(" x2##vcp_ph_dbl")) {
            const auto& asrc = image_sources[vc_pcam_active_src];
            if (vc_pcam_render_w <= 0) vc_pcam_render_w = asrc.intrinsics.width;
            if (vc_pcam_render_h <= 0) vc_pcam_render_h = asrc.intrinsics.height;
            vc_pcam_render_w = std::min(16000, vc_pcam_render_w * 2);
            vc_pcam_render_h = std::min(12000, vc_pcam_render_h * 2);
            scale_splats(2.0);
          }
          ImGui::SameLine();
          if (ImGui::SmallButton("Native##vcp_ph_nat")) {
            vc_pcam_render_w = 0;
            vc_pcam_render_h = 0;
          }
          ImGui::Separator();
        }

        // --- Output format.
        ImGui::TextDisabled("Output");
        ImGui::RadioButton("PNG##vcp", &vc_pcam_format, 0);
        ImGui::SameLine();
        ImGui::RadioButton("JPG##vcp", &vc_pcam_format, 1);
        if (vc_pcam_format == 1) {
          ImGui::SameLine(); ImGui::SetNextItemWidth(100);
          ImGui::SliderInt("quality##vcp", &vc_pcam_jpg_quality, 60, 100);
        }
        ImGui::Separator();

        // --- Action buttons.
        // "Have source" is implicit now -- the dropdown picks one.
        const bool have_source = !image_sources.empty();
        ImGui::BeginDisabled(vc_running || !have_source);
        const bool preview_clicked = ImGui::Button("Refresh preview##vcp");
        ImGui::SameLine();
        const bool test_matches_clicked = ImGui::Button("Test matches##vcp");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Runs the auto-calibrate LightGlue script between the active source's\n"
          "real image and the currently rendered preview. Reports total matches\n"
          "plus high/mid/low quality buckets so you can tune the rasterization\n"
          "knobs for the best match yield.");
        ImGui::SameLine();
        const bool can_batch = have_source && !vc_output_dir.empty();
        ImGui::BeginDisabled(!can_batch);
        const bool batch_clicked = ImGui::Button("Export cameras##vcp");
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (!can_batch && have_source && vc_output_dir.empty()) {
          ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "Pick an output folder first.");
        }
        // Tuning knobs for the match tester (kept separate from ac_* so each
        // tool keeps its own feel).
        ImGui::SetNextItemWidth(120);
        ImGui::SliderFloat("LG min score##vcp", &vc_pcam_lg_min_score, 0.05f, 0.95f, "%.2f");
        ImGui::SameLine(); ImGui::SetNextItemWidth(100);
        ImGui::DragInt("LG max kp##vcp", &vc_pcam_lg_max_kp, 64, 128, 8192);

        // --- Preview render (blocking; cheap enough for a single frame).
        if ((preview_clicked || vc_pcam_preview_dirty) && have_source) {
          vc_pcam_preview_dirty = false;
          // Drop old textures so we don't leak GL.
          for (auto& t : vc_pcam_preview_textures) if (t.tex) glDeleteTextures(1, &t.tex);
          vc_pcam_preview_textures.clear();

          auto& psrc = image_sources[vc_pcam_active_src];
          if (!psrc.frames.empty()) {
            const int fidx = std::clamp(vc_pcam_preview_frame, 0, static_cast<int>(psrc.frames.size()) - 1);
            const auto& pcam = psrc.frames[fidx];
            if (!pcam.located) {
              vc_status = "Preview: frame not located; run Colorize > Locate on this source first.";
            } else {
              if (!trajectory_built) build_trajectory();
              const auto timed_traj = timed_traj_snapshot();
              const Eigen::Vector3f anchor_pos = pcam.T_world_cam.translation().cast<float>();
              const Eigen::Vector3f anchor_fwd = pcam.T_world_cam.rotation().col(0).cast<float>().normalized();
              CalibrationContext ctx = build_calibration_context(
                submaps, timed_traj, pcam.timestamp,
                anchor_pos, anchor_fwd, vc_pcam_ctx_opts,
                [this](int si) { return load_hd_for_submap(si, false); });

              // Snapshot intensity percentiles so "Lock range from preview"
              // has a ready value to clamp to.
              if (!ctx.intensities.empty()) {
                compute_intensity_percentiles(ctx.intensities,
                                              vc_pcam_last_imin,
                                              vc_pcam_last_ibulk,
                                              vc_pcam_last_imax);
                vc_pcam_have_last_percentiles = true;
              }

              auto upload_tex = [](const cv::Mat& img, const std::string& label,
                                    std::vector<VcamPreviewTex>& out) {
                VcamPreviewTex t; t.label = label; t.w = img.cols; t.h = img.rows;
                glGenTextures(1, &t.tex);
                glBindTexture(GL_TEXTURE_2D, t.tex);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                if (img.channels() == 1) {
                  // Broadcast single-channel luminance to RGB -- simpler than
                  // juggling GL_LUMINANCE across core-profile drivers.
                  cv::Mat rgb; cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
                  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
                } else {
                  cv::Mat rgb; cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
                  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
                }
                glBindTexture(GL_TEXTURE_2D, 0);
                out.push_back(t);
              };

              if (psrc.camera_type == CameraType::Spherical) {
                const int fs = vc_face_size;
                const PinholeIntrinsics K = cube_face_intrinsics(fs);
                for (int fi = 0; fi < 6; fi++) {
                  if (!vc_face_enabled[fi]) continue;
                  Eigen::Isometry3d T_face_in_cam = Eigen::Isometry3d::Identity();
                  T_face_in_cam.linear() = cube_face_rotation(fi);
                  const Eigen::Isometry3d T_world_face = pcam.T_world_cam * T_face_in_cam;
                  auto r = render_intensity_image(ctx, T_world_face, K, fs, fs, vc_pcam_render_opts);
                  upload_tex(r.image, vc_face_label(fi), vc_pcam_preview_textures);
                }
              } else {
                const int W = (vc_pcam_render_w > 0) ? vc_pcam_render_w : psrc.intrinsics.width;
                const int H = (vc_pcam_render_h > 0) ? vc_pcam_render_h : psrc.intrinsics.height;
                // Scale intrinsics when W/H override differs from native so FOV
                // stays constant -- without this, x2 would double the sensor
                // extent while keeping fx/fy fixed, producing a 2x-wider FOV.
                PinholeIntrinsics Kp = psrc.intrinsics;
                const double sx = static_cast<double>(W) / std::max(1, psrc.intrinsics.width);
                const double sy = static_cast<double>(H) / std::max(1, psrc.intrinsics.height);
                Kp.fx *= sx; Kp.cx *= sx;
                Kp.fy *= sy; Kp.cy *= sy;
                Kp.width = W; Kp.height = H;
                auto r = render_intensity_image(ctx, pcam.T_world_cam, Kp, W, H, vc_pcam_render_opts);
                upload_tex(r.image, psrc.name, vc_pcam_preview_textures);
              }
              vc_status = "Preview: " + std::to_string(vc_pcam_preview_textures.size()) +
                          " face(s), context " + std::to_string(ctx.world_points.size()) + " points";
            }
          }
        }

        // --- Test matches. Reuses auto-calibrate's LightGlue pipeline: dumps
        //     the real image + the rendered preview to temp files, shells out
        //     to lightglue_match.py, loads confidences, buckets them. Gives a
        //     fast "is this rasterization good enough for BA matches?" signal
        //     without running the full extrinsic / intrinsics refinement.
        if (test_matches_clicked && have_source) {
          // Drop any previously allocated match viz textures before rebuilding.
          for (auto& r : vc_pcam_match_results) {
            if (r.real_tex) glDeleteTextures(1, &r.real_tex);
            if (r.rend_tex) glDeleteTextures(1, &r.rend_tex);
          }
          vc_pcam_match_results.clear();
          vc_pcam_match_viz_idx = 0;
          vc_pcam_match_log.clear();
          auto& psrc = image_sources[vc_pcam_active_src];
          if (psrc.frames.empty()) {
            vc_pcam_match_log = "No frames on active source.";
          } else {
            const int fidx = std::clamp(vc_pcam_preview_frame, 0, static_cast<int>(psrc.frames.size()) - 1);
            const auto& pcam = psrc.frames[fidx];
            if (pcam.filepath.empty() || !boost::filesystem::exists(pcam.filepath)) {
              vc_pcam_match_log = "Real image not found: " + pcam.filepath;
            } else if (vc_pcam_preview_textures.empty()) {
              vc_pcam_match_log = "Run 'Preview renders' first -- match tester needs a rendered image on disk.";
            } else {
              // Resolve LightGlue script like auto-calibrate does.
              std::string script = ac_python_script_path;
              if (script.empty() && boost::filesystem::exists("/ros2_ws/src/glim/scripts/lightglue_match.py")) {
                script = "/ros2_ws/src/glim/scripts/lightglue_match.py";
              }
              if (script.empty()) {
                vc_pcam_match_log = "lightglue_match.py not found (set ac_python_script_path in Auto-calibrate window).";
              } else {
                const std::string py = ac_python_interpreter.empty() ? std::string("python3") : ac_python_interpreter;
                boost::filesystem::path tmp_dir = boost::filesystem::temp_directory_path() / "glim_vcam_match";
                boost::filesystem::create_directories(tmp_dir);

                // Re-render each face into a temp PNG and dump the paired real
                // slice. For Spherical we slice the real equirect into the same
                // 6 faces the rendered side just used, so the pairing is direct.
                cv::Mat real_full = cv::imread(pcam.filepath);
                if (real_full.empty()) {
                  vc_pcam_match_log = "Failed to load real image from " + pcam.filepath;
                } else {
                  // Pre-slice the real equirect for Spherical sources.
                  std::array<std::shared_ptr<cv::Mat>, 6> real_faces;
                  if (psrc.camera_type == CameraType::Spherical) {
                    real_faces = slice_equirect_cubemap(real_full, vc_face_size);
                  }
                  // Upload a cv::Mat to a GL texture, handling 1-channel and 3-channel
                  // inputs. Returns 0 on empty Mat. Used for the match-viz side-by-side.
                  auto upload_tex_mat = [](const cv::Mat& img) -> unsigned int {
                    if (img.empty()) return 0;
                    unsigned int tex = 0;
                    glGenTextures(1, &tex);
                    glBindTexture(GL_TEXTURE_2D, tex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    cv::Mat rgb;
                    if (img.channels() == 1) cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
                    else                      cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0,
                                 GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
                    glBindTexture(GL_TEXTURE_2D, 0);
                    return tex;
                  };

                  auto run_match_one = [&](const cv::Mat& real_img,
                                            const cv::Mat& rendered_img,
                                            const std::string& label) {
                    const std::string rp = (tmp_dir / (std::string("real_") + label + ".png")).string();
                    const std::string gp = (tmp_dir / (std::string("rend_") + label + ".png")).string();
                    const std::string jp = (tmp_dir / (std::string("matches_") + label + ".json")).string();
                    // lightglue_match.py expects both images sized identically.
                    // Render is already at face_size (spherical) or native intrinsics (pinhole).
                    cv::Mat real_resized;
                    cv::resize(real_img, real_resized, cv::Size(rendered_img.cols, rendered_img.rows));
                    cv::imwrite(rp, real_resized);
                    // Convert 8UC3 colormapped render to 8UC3 BGR, or keep single-channel grayscale.
                    cv::imwrite(gp, rendered_img);
                    char args[128];
                    std::snprintf(args, sizeof(args), " --max-kp %d --min-score %.3f",
                                  vc_pcam_lg_max_kp, vc_pcam_lg_min_score);
                    const std::string cmd = py + " " + script + " " + rp + " " + gp + " " + jp + args + " 2>&1";
                    logger->info("[VirtualCams] {}", cmd);
                    const int rc = std::system(cmd.c_str());
                    VcamMatchResult res; res.label = label;
                    if (rc == 0) {
                      std::vector<float> confidences;
                      auto raw = load_lightglue_matches(jp, &confidences);
                      res.stats = compute_match_quality(confidences);
                      res.match_pairs.reserve(raw.size());
                      res.match_scores.reserve(raw.size());
                      for (size_t i = 0; i < raw.size(); i++) {
                        res.match_pairs.emplace_back(raw[i].first.cast<float>(), raw[i].second.cast<float>());
                        res.match_scores.push_back(i < confidences.size() ? confidences[i] : 1.0f);
                      }
                    } else {
                      logger->warn("[VirtualCams] lightglue_match.py returned {} for {}", rc, label);
                    }
                    // Texture uploads for the viz window. Tied to render-space so
                    // match UVs index into them directly.
                    res.real_tex = upload_tex_mat(real_resized);
                    res.rend_tex = upload_tex_mat(rendered_img);
                    res.real_w = real_resized.cols; res.real_h = real_resized.rows;
                    res.rend_w = rendered_img.cols; res.rend_h = rendered_img.rows;
                    vc_pcam_match_results.push_back(std::move(res));
                  };

                  // Re-render face-by-face (or just once for pinhole) so the
                  // rendered side is an on-disk PNG the script can eat.
                  if (!trajectory_built) build_trajectory();
                  const auto t_traj = timed_traj_snapshot();
                  const Eigen::Vector3f apos = pcam.T_world_cam.translation().cast<float>();
                  const Eigen::Vector3f afwd = pcam.T_world_cam.rotation().col(0).cast<float>().normalized();
                  CalibrationContext tctx = build_calibration_context(
                    submaps, t_traj, pcam.timestamp, apos, afwd, vc_pcam_ctx_opts,
                    [this](int si) { return load_hd_for_submap(si, false); });

                  if (psrc.camera_type == CameraType::Spherical) {
                    const int fs = vc_face_size;
                    const PinholeIntrinsics K = cube_face_intrinsics(fs);
                    for (int fi = 0; fi < 6; fi++) {
                      if (!vc_face_enabled[fi]) continue;
                      if (!real_faces[fi]) continue;
                      Eigen::Isometry3d T_face_in_cam = Eigen::Isometry3d::Identity();
                      T_face_in_cam.linear() = cube_face_rotation(fi);
                      const Eigen::Isometry3d T_world_face = pcam.T_world_cam * T_face_in_cam;
                      cv::Mat rendered = render_intensity_image(
                        tctx, T_world_face, K, fs, fs, vc_pcam_render_opts).image;
                      run_match_one(*real_faces[fi], rendered, vc_face_label(fi));
                    }
                  } else {
                    const int W = (vc_pcam_render_w > 0) ? vc_pcam_render_w : psrc.intrinsics.width;
                    const int H = (vc_pcam_render_h > 0) ? vc_pcam_render_h : psrc.intrinsics.height;
                    PinholeIntrinsics Kt = psrc.intrinsics;
                    const double sx = static_cast<double>(W) / std::max(1, psrc.intrinsics.width);
                    const double sy = static_cast<double>(H) / std::max(1, psrc.intrinsics.height);
                    Kt.fx *= sx; Kt.cx *= sx;
                    Kt.fy *= sy; Kt.cy *= sy;
                    Kt.width = W; Kt.height = H;
                    cv::Mat rendered = render_intensity_image(
                      tctx, pcam.T_world_cam, Kt, W, H, vc_pcam_render_opts).image;
                    run_match_one(real_full, rendered, psrc.name);
                  }
                  // Log which labels were actually matched (disabled faces are
                  // skipped earlier) -- visible in the status line so a mismatch
                  // between the face checkboxes and the runner is easy to spot.
                  std::string labels_tested;
                  for (const auto& r : vc_pcam_match_results) {
                    if (!labels_tested.empty()) labels_tested += ", ";
                    labels_tested += r.label;
                  }
                  vc_pcam_match_log = "Test matches: " +
                    std::to_string(vc_pcam_match_results.size()) + " face(s) evaluated (" +
                    labels_tested + ").";
                  logger->info("[VirtualCams] {}", vc_pcam_match_log);
                  // Auto-open the side-by-side viz so the user sees matches
                  // immediately rather than hunting for a button.
                  vc_pcam_match_viz_show = !vc_pcam_match_results.empty();
                  vc_pcam_match_viz_idx  = 0;
                }
              }
            }
          }
        }

        // --- Match-test results panel.
        if (!vc_pcam_match_results.empty()) {
          ImGui::Separator();
          ImGui::TextDisabled("Match test (LightGlue, min-score %.2f, max-kp %d)",
                               vc_pcam_lg_min_score, vc_pcam_lg_max_kp);
          ImGui::SameLine();
          if (ImGui::SmallButton("View matches##vcp_viz")) {
            vc_pcam_match_viz_show = true;
          }
          int sum_total = 0, sum_high = 0, sum_mid = 0, sum_low = 0;
          for (const auto& r : vc_pcam_match_results) {
            sum_total += r.stats.total;
            sum_high  += r.stats.high;
            sum_mid   += r.stats.mid;
            sum_low   += r.stats.low;
            ImGui::Text("  %-7s  total %4d", r.label.c_str(), r.stats.total);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.35f, 1.0f), "  %d high", r.stats.high);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f), "  %d mid", r.stats.mid);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "  %d low", r.stats.low);
          }
          if (vc_pcam_match_results.size() > 1) {
            ImGui::Text("  %-7s  total %4d", "TOTAL", sum_total);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.35f, 1.0f), "  %d high", sum_high);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f), "  %d mid", sum_mid);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "  %d low", sum_low);
          }
        }
        if (!vc_pcam_match_log.empty()) {
          ImGui::TextDisabled("%s", vc_pcam_match_log.c_str());
        }

        // --- Batch export (blocking -- user sees a frozen UI during the run,
        // progress goes to the status line. Threading is a later optimisation).
        if (batch_clicked) {
          vc_pcam_cancel = false;
          vc_pcam_progress_cur = 0;
          // Total = located_frames * faces_for_active_source. The dropdown is
          // the single source of truth now; batch runs on the active source only.
          int total = 0;
          {
            const auto& s = image_sources[vc_pcam_active_src];
            int per_cam = 1;
            if (s.camera_type == CameraType::Spherical) {
              per_cam = 0;
              for (int f = 0; f < 6; f++) if (vc_face_enabled[f]) per_cam++;
            }
            if (per_cam > 0) {
              for (const auto& c : s.frames) if (c.located) total += per_cam;
            }
          }
          vc_pcam_progress_total = total;
          const std::string ext = (vc_pcam_format == 0) ? ".png" : ".jpg";
          boost::filesystem::create_directories(vc_output_dir);

          if (!trajectory_built) build_trajectory();
          const auto batch_timed_traj = timed_traj_snapshot();

          auto sanitise = [](std::string s) {
            for (auto& ch : s) if (ch == '/' || ch == '\\' || ch == ' ' || ch == ':' || ch == '?') ch = '_';
            return s;
          };

          std::vector<int> jpg_params;
          if (vc_pcam_format == 1) {
            jpg_params = { cv::IMWRITE_JPEG_QUALITY, vc_pcam_jpg_quality };
          }

          int written = 0;
          {
            auto& s = image_sources[vc_pcam_active_src];
            const std::string sname = sanitise(s.name);
            const bool is_sph = (s.camera_type == CameraType::Spherical);
            std::vector<int> faces_to_render;
            if (is_sph) { for (int f = 0; f < 6; f++) if (vc_face_enabled[f]) faces_to_render.push_back(f); }
            else        { faces_to_render.push_back(-1); }  // sentinel = pinhole, no face label

            for (size_t fi = 0; fi < s.frames.size(); fi++) {
              if (vc_pcam_cancel) break;
              const auto& c = s.frames[fi];
              if (!c.located) continue;

              // Build context once per frame (same context reused across faces).
              const Eigen::Vector3f apos = c.T_world_cam.translation().cast<float>();
              const Eigen::Vector3f afwd = c.T_world_cam.rotation().col(0).cast<float>().normalized();
              CalibrationContext ctx = build_calibration_context(
                submaps, batch_timed_traj, c.timestamp, apos, afwd,
                vc_pcam_ctx_opts, [this](int sm) { return load_hd_for_submap(sm, false); });

              for (int face : faces_to_render) {
                if (vc_pcam_cancel) break;
                cv::Mat img;
                if (face < 0) {
                  const int W = (vc_pcam_render_w > 0) ? vc_pcam_render_w : s.intrinsics.width;
                  const int H = (vc_pcam_render_h > 0) ? vc_pcam_render_h : s.intrinsics.height;
                  PinholeIntrinsics Kb = s.intrinsics;
                  const double sx = static_cast<double>(W) / std::max(1, s.intrinsics.width);
                  const double sy = static_cast<double>(H) / std::max(1, s.intrinsics.height);
                  Kb.fx *= sx; Kb.cx *= sx;
                  Kb.fy *= sy; Kb.cy *= sy;
                  Kb.width = W; Kb.height = H;
                  img = render_intensity_image(ctx, c.T_world_cam, Kb, W, H, vc_pcam_render_opts).image;
                } else {
                  const int fs = vc_face_size;
                  const PinholeIntrinsics K = cube_face_intrinsics(fs);
                  Eigen::Isometry3d T_face_in_cam = Eigen::Isometry3d::Identity();
                  T_face_in_cam.linear() = cube_face_rotation(face);
                  const Eigen::Isometry3d T_world_face = c.T_world_cam * T_face_in_cam;
                  img = render_intensity_image(ctx, T_world_face, K, fs, fs, vc_pcam_render_opts).image;
                }
                char frame_buf[16]; std::snprintf(frame_buf, sizeof(frame_buf), "%04zu", fi);
                std::string path = vc_output_dir + "/LidarCam_" + sname;
                if (face >= 0) path += std::string("_") + vc_face_label(face);
                path += std::string("_") + frame_buf + ext;
                cv::imwrite(path, img, jpg_params);
                written++;
                vc_pcam_progress_cur = written;
              }
            }
          }
          vc_status = (vc_pcam_cancel ? "Cancelled after " : "Done: ") +
                      std::to_string(written) + "/" + std::to_string(total) + " images written to " + vc_output_dir;
          logger->info("[VirtualCams/per-cam] {}", vc_status);
        }

        // --- Thumbnail row. Click any thumbnail to open a large preview window.
        if (!vc_pcam_preview_textures.empty()) {
          ImGui::Separator();
          ImGui::TextDisabled("Preview (click a thumbnail to enlarge)");
          const float thumb_w = 180.0f;
          for (size_t ti = 0; ti < vc_pcam_preview_textures.size(); ti++) {
            const auto& t = vc_pcam_preview_textures[ti];
            const float thumb_h = (t.w > 0) ? (thumb_w * static_cast<float>(t.h) / static_cast<float>(t.w)) : thumb_w;
            ImGui::BeginGroup();
            ImGui::PushID(static_cast<int>(ti));
            if (ImGui::ImageButton("thumb",
                                    reinterpret_cast<ImTextureID>(static_cast<intptr_t>(t.tex)),
                                    ImVec2(thumb_w, thumb_h))) {
              vc_pcam_focused_tex = static_cast<int>(ti);
            }
            ImGui::PopID();
            ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), "%s", t.label.c_str());
            ImGui::EndGroup();
            ImGui::SameLine();
          }
          ImGui::NewLine();
        }

        if (!vc_status.empty()) {
          ImGui::Separator();
          ImGui::TextWrapped("%s", vc_status.c_str());
        }
      }
    }

    // 3D-viewer preview: show anchors as small spheres so the user can eyeball
    // the spacing before committing to a render.
    if (!vc_preview_anchors.empty()) {
      auto vw = guik::LightViewer::instance();
      for (size_t i = 0; i < vc_preview_anchors.size(); i++) {
        Eigen::Affine3f tf = Eigen::Affine3f::Identity();
        tf.translate(vc_preview_anchors[i]);
        tf.linear() = vc_preview_orient[i];
        tf = tf * Eigen::Scaling(Eigen::Vector3f(0.20f, 0.20f, 0.20f));
        // Cyan so anchors stand out against the green trajectory line.
        vw->update_drawable("vc_anchor_" + std::to_string(i),
          glk::Primitives::sphere(),
          guik::FlatColor(0.1f, 0.85f, 0.95f, 0.95f, tf));
        // Forward-direction line, same cyan.
        const Eigen::Vector3f pos = vc_preview_anchors[i];
        const Eigen::Vector3f fwd = vc_preview_orient[i].col(0).normalized();
        std::vector<Eigen::Vector3f> verts = { pos, pos + fwd * 0.8f };
        vw->update_drawable("vc_anchor_fwd_" + std::to_string(i),
          std::make_shared<glk::ThinLines>(verts.data(), static_cast<int>(verts.size()), false),
          guik::FlatColor(0.1f, 0.85f, 0.95f, 0.9f));
      }
    }

    ImGui::End();

    // --- Enlarged preview window. Opens when the user clicks a thumbnail;
    //     closable via the X so a stale focus index doesn't linger after the
    //     textures it pointed into are recreated.
    if (vc_pcam_focused_tex >= 0 &&
        vc_pcam_focused_tex < static_cast<int>(vc_pcam_preview_textures.size())) {
      const auto& t = vc_pcam_preview_textures[vc_pcam_focused_tex];
      bool open = true;
      ImGui::SetNextWindowSize(ImVec2(2100, 2100), ImGuiCond_FirstUseEver);
      const std::string title = std::string("Preview: ") + t.label + "###vc_pcam_enlarged";
      if (ImGui::Begin(title.c_str(), &open)) {
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float scale = std::min(avail.x / std::max(1.0f, static_cast<float>(t.w)),
                                      avail.y / std::max(1.0f, static_cast<float>(t.h)));
        const ImVec2 sz(t.w * scale, t.h * scale);
        ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(t.tex)), sz);
      }
      ImGui::End();
      if (!open) vc_pcam_focused_tex = -1;
    } else if (vc_pcam_focused_tex >= static_cast<int>(vc_pcam_preview_textures.size())) {
      // Textures got recreated or cleared -- drop the stale focus.
      vc_pcam_focused_tex = -1;
    }

    // --- Match viz window. Side-by-side real / rendered images with LightGlue
    //     match lines drawn between them. Colour-coded by confidence: green =
    //     high, amber = mid, red = low -- same bucket thresholds the results
    //     panel uses. For Spherical sources the face selector picks which pair
    //     to show. Mirrors the auto-calibrate visualisation so the cognitive
    //     overhead is zero for anyone who's tuned extrinsics before.
    if (vc_pcam_match_viz_show && !vc_pcam_match_results.empty()) {
      vc_pcam_match_viz_idx = std::clamp(
        vc_pcam_match_viz_idx, 0, static_cast<int>(vc_pcam_match_results.size()) - 1);
      bool open = true;
      ImGui::SetNextWindowSize(ImVec2(1400, 900), ImGuiCond_FirstUseEver);
      if (ImGui::Begin("VC Match viz###vc_pcam_match_viz", &open)) {
        // Face selector when multiple results are in.
        if (vc_pcam_match_results.size() > 1) {
          std::vector<std::string> labels;
          for (const auto& r : vc_pcam_match_results) {
            labels.push_back(r.label + " (" + std::to_string(r.stats.total) + ")");
          }
          std::vector<const char*> lptrs;
          for (auto& s : labels) lptrs.push_back(s.c_str());
          ImGui::SetNextItemWidth(240);
          ImGui::Combo("Face##vc_viz", &vc_pcam_match_viz_idx, lptrs.data(),
                        static_cast<int>(lptrs.size()));
          ImGui::SameLine();
        }
        const auto& r = vc_pcam_match_results[vc_pcam_match_viz_idx];
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.35f, 1.0f), "%d high", r.stats.high);
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f), "| %d mid", r.stats.mid);
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "| %d low", r.stats.low);
        ImGui::SameLine();
        ImGui::TextDisabled("(total %d)", r.stats.total);
        ImGui::Separator();

        if (!r.real_tex || !r.rend_tex) {
          ImGui::TextDisabled("Match viz unavailable -- textures not uploaded.");
        } else {
          // Layout: two images side-by-side, each taking half the content width
          // minus a small gutter. Scale preserves aspect ratio.
          const ImVec2 avail = ImGui::GetContentRegionAvail();
          const float gutter = 12.0f;
          const float pair_w = std::max(50.0f, (avail.x - gutter) * 0.5f);
          const float s_real = pair_w / static_cast<float>(std::max(1, r.real_w));
          const float s_rend = pair_w / static_cast<float>(std::max(1, r.rend_w));
          const float h_real = s_real * r.real_h;
          const float h_rend = s_rend * r.rend_h;
          const float row_h = std::max(h_real, h_rend);
          const ImVec2 real_pos = ImGui::GetCursorScreenPos();
          ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(r.real_tex)),
                       ImVec2(pair_w, h_real));
          ImGui::SameLine(0.0f, gutter);
          const ImVec2 rend_pos = ImGui::GetCursorScreenPos();
          ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(r.rend_tex)),
                       ImVec2(pair_w, h_rend));

          // Match lines. UVs are in render-space (both sides resized to the
          // same W/H) so left-side scale == right-side scale.
          ImDrawList* dl = ImGui::GetWindowDrawList();
          for (size_t i = 0; i < r.match_pairs.size(); i++) {
            const auto& m = r.match_pairs[i];
            const float score = i < r.match_scores.size() ? r.match_scores[i] : 1.0f;
            ImU32 col;
            if (score >= 0.8f)       col = IM_COL32( 90, 240,  90, 200);  // high -- green
            else if (score >= 0.5f)  col = IM_COL32(255, 190,  60, 200);  // mid  -- amber
            else                      col = IM_COL32(240,  90,  90, 180); // low  -- red
            const ImVec2 a(real_pos.x + m.first.x()  * s_real, real_pos.y + m.first.y()  * s_real);
            const ImVec2 b(rend_pos.x + m.second.x() * s_rend, rend_pos.y + m.second.y() * s_rend);
            dl->AddCircleFilled(a, 2.5f, col);
            dl->AddCircleFilled(b, 2.5f, col);
            dl->AddLine(a, b, col, 1.0f);
          }
          ImGui::Dummy(ImVec2(1.0f, row_h - std::min(h_real, h_rend)));
          ImGui::TextDisabled(
            "Left: real (resized to render). Right: LiDAR-rendered virtual. "
            "Lines: green >= 0.80, amber 0.50-0.80, red < 0.50.");
        }
      }
      ImGui::End();
      if (!open) vc_pcam_match_viz_show = false;
    }
  });

  // Camera Time Matcher: side-by-side scrub to align a dumb-frames source
  // (no timestamps, e.g. Osmo 360 video frames) to a time-stamped reference
  // source. User anchors one or two matching moments, enters FPS, clicks
  // Apply -> right-side frames get synthetic timestamps back-filled then get
  // located along the trajectory in one shot.
  viewer->register_ui_callback("time_matcher_window", [this] {
    if (!show_time_matcher) return;
    ImGui::SetNextWindowSize(ImVec2(1100, 720), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Camera Time Matcher", &show_time_matcher)) { ImGui::End(); return; }
    if (image_sources.size() < 2) {
      ImGui::TextDisabled("Need at least two image sources loaded (one time-stamped, one to be matched).");
      ImGui::End(); return;
    }

    // --- Source selectors ---
    std::vector<std::string> labels;
    for (size_t i = 0; i < image_sources.size(); i++) labels.push_back("src " + std::to_string(i) + " [" + image_sources[i].name + "]");
    std::vector<const char*> lptrs; for (auto& s : labels) lptrs.push_back(s.c_str());
    ImGui::SetNextItemWidth(260);
    ImGui::Combo("Left (time-stamped)##tm", &tm_left_src,  lptrs.data(), static_cast<int>(lptrs.size()));
    ImGui::SameLine(560);
    ImGui::SetNextItemWidth(260);
    ImGui::Combo("Right (dumb frames)##tm", &tm_right_src, lptrs.data(), static_cast<int>(lptrs.size()));

    tm_left_src  = std::clamp(tm_left_src,  0, static_cast<int>(image_sources.size()) - 1);
    tm_right_src = std::clamp(tm_right_src, 0, static_cast<int>(image_sources.size()) - 1);
    auto& srcL = image_sources[tm_left_src];
    auto& srcR = image_sources[tm_right_src];
    if (srcL.frames.empty() || srcR.frames.empty()) {
      ImGui::TextDisabled("One of the selected sources has no frames.");
      ImGui::End(); return;
    }
    tm_left_idx  = std::clamp(tm_left_idx,  0, static_cast<int>(srcL.frames.size()) - 1);
    tm_right_idx = std::clamp(tm_right_idx, 0, static_cast<int>(srcR.frames.size()) - 1);
    const auto& camL = srcL.frames[tm_left_idx];
    const auto& camR = srcR.frames[tm_right_idx];

    // --- Image loader helper (downscaled texture for responsiveness) ---
    auto load_texture = [](const std::string& path, std::string& loaded_path,
                           unsigned int& tex, int& tw, int& th) {
      if (path == loaded_path) return;
      cv::Mat img = cv::imread(path);
      if (img.empty()) return;
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
      const int max_tex = 1200;
      cv::Mat tex_img = img;
      if (img.cols > max_tex) {
        const double s = static_cast<double>(max_tex) / img.cols;
        cv::resize(img, tex_img, cv::Size(), s, s);
      }
      if (tex) { glDeleteTextures(1, &tex); tex = 0; }
      glGenTextures(1, &tex);
      glBindTexture(GL_TEXTURE_2D, tex);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_img.cols, tex_img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_img.data);
      glBindTexture(GL_TEXTURE_2D, 0);
      tw = tex_img.cols; th = tex_img.rows;
      loaded_path = path;
    };
    const std::string prev_left_path  = tm_left_loaded_path;
    const std::string prev_right_path = tm_right_loaded_path;
    load_texture(camL.filepath, tm_left_loaded_path,  tm_left_tex,  tm_left_tex_w,  tm_left_tex_h);
    load_texture(camR.filepath, tm_right_loaded_path, tm_right_tex, tm_right_tex_w, tm_right_tex_h);
    // Reset scale to fit when the frame changed -- different resolutions would
    // otherwise show at the previous scale and misframe. Auto-fit flag stays
    // as the user last set it (so if they were zoomed in on the left, stepping
    // to the next left frame stays zoomed in).
    if (tm_left_loaded_path  != prev_left_path)  tm_left_scale  = -1.0f;
    if (tm_right_loaded_path != prev_right_path) tm_right_scale = -1.0f;

    // --- Two-column layout: left preview / right preview ---
    // Both dimensions scale with window size. Reserve ~260 px at the bottom for
    // the anchor + rate + apply controls. col_h minimum 240 keeps the image
    // useful even in a cramped window; above that it grows with the window.
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float col_w = (avail.x - 16.0f) * 0.5f;
    const float col_h = std::max(240.0f, avail.y - 260.0f);

    auto draw_column = [&](const char* tag, int& idx, int max_idx,
                           unsigned int tex, int tw, int th, double display_time, bool is_right,
                           float& scale, bool& auto_fit) {
      ImGui::BeginChild((std::string("tmcol_") + tag).c_str(), ImVec2(col_w, col_h + 170.0f), true);
      // Inner scrollable viewport for the image so zoom-in can push past the
      // visible area without overflowing the outer child.
      const ImVec2 viewport_size(col_w - 18.0f, col_h);
      ImGui::BeginChild((std::string("tmview_") + tag).c_str(), viewport_size,
                        false, ImGuiWindowFlags_HorizontalScrollbar);
      const ImVec2 canvas_origin = ImGui::GetCursorScreenPos();
      if (tex && tw > 0 && th > 0) {
        // Auto-fit: scale tracks current viewport size until the user zooms
        // manually. Lets the image grow when the window is resized larger.
        if (auto_fit || scale <= 0.0f) {
          scale = std::min(viewport_size.x / tw, viewport_size.y / th);
        }
        ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(tex)),
                     ImVec2(tw * scale, th * scale));
      } else {
        ImGui::Dummy(viewport_size);
      }
      if (apply_wheel_zoom_around_cursor(scale, canvas_origin, 0.05f, 10.0f)) {
        auto_fit = false;  // user zoomed -- stop auto-fitting until they click Reset.
      }
      ImGui::EndChild();

      // Zoom controls
      if (ImGui::SmallButton((std::string("-##") + tag + "zoom").c_str())) { scale = std::max(0.05f, scale / 1.25f); auto_fit = false; }
      ImGui::SameLine();
      if (ImGui::SmallButton((std::string("+##") + tag + "zoom").c_str())) { scale = std::min(10.0f,  scale * 1.25f); auto_fit = false; }
      ImGui::SameLine();
      if (ImGui::SmallButton((std::string("Reset##") + tag + "zoom").c_str()) && tex && tw > 0 && th > 0) {
        auto_fit = true;  // resume auto-fit; scale will be recomputed next frame.
      }
      ImGui::SameLine();
      ImGui::TextDisabled("%s%.2fx", auto_fit ? "fit " : "", scale);

      // Scrub: narrower slider + prominent step buttons for fine tuning.
      ImGui::SetNextItemWidth(col_w * 0.55f);
      ImGui::SliderInt((std::string("Frame##") + tag).c_str(), &idx, 0, max_idx);
      ImGui::SameLine();
      if (ImGui::ArrowButton((std::string("##") + tag + "prev").c_str(), ImGuiDir_Left))  idx = std::max(0, idx - 1);
      ImGui::SameLine();
      if (ImGui::ArrowButton((std::string("##") + tag + "next").c_str(), ImGuiDir_Right)) idx = std::min(max_idx, idx + 1);
      ImGui::SameLine();
      ImGui::Text("%d / %d", idx, max_idx);
      if (is_right && display_time == 0.0) {
        ImGui::TextDisabled("time: (unstamped)");
      } else {
        ImGui::Text("time: %.3f s", display_time);
      }
      ImGui::EndChild();
    };

    const double left_time  = camL.timestamp + effective_time_shift(srcL, camL.timestamp);
    const double right_time = camR.timestamp;  // dumb source: raw timestamp (often 0)
    draw_column("L", tm_left_idx,  static_cast<int>(srcL.frames.size()) - 1,
                tm_left_tex,  tm_left_tex_w,  tm_left_tex_h,  left_time,  false,
                tm_left_scale,  tm_left_auto_fit);
    ImGui::SameLine();
    draw_column("R", tm_right_idx, static_cast<int>(srcR.frames.size()) - 1,
                tm_right_tex, tm_right_tex_w, tm_right_tex_h, right_time, true,
                tm_right_scale, tm_right_auto_fit);

    ImGui::Separator();

    // --- Anchor + apply controls ---
    if (ImGui::Button("Set Anchor 1")) {
      tm_anchor1_right_idx = tm_right_idx;
      tm_anchor1_left_time = left_time;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Record the current LEFT time as the anchor for the current RIGHT frame.\n"
      "With 'Two-anchor mode' OFF: Apply back-fills using anchor1 + FPS.\n"
      "With 'Two-anchor mode' ON: set a second anchor at a DIFFERENT moment to\n"
      "solve the actual rate.");
    ImGui::SameLine();
    if (tm_anchor1_right_idx >= 0) {
      ImGui::Text("Anchor1: right[%d] <-> left_t=%.3f", tm_anchor1_right_idx, tm_anchor1_left_time);
    } else {
      ImGui::TextDisabled("Anchor 1 not set");
    }

    ImGui::Checkbox("Two-anchor mode (solve rate)##tm", &tm_two_anchor_mode);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
      "Two anchors let BA solve the actual frame interval instead of trusting FPS.\n"
      "Pick the two anchors as far apart as possible (e.g. near-start and near-end\n"
      "of the dumb source) for best numerical conditioning.");

    if (tm_two_anchor_mode) {
      if (ImGui::Button("Set Anchor 2")) {
        tm_anchor2_right_idx = tm_right_idx;
        tm_anchor2_left_time = left_time;
      }
      ImGui::SameLine();
      if (tm_anchor2_right_idx >= 0) {
        ImGui::Text("Anchor2: right[%d] <-> left_t=%.3f", tm_anchor2_right_idx, tm_anchor2_left_time);
      } else {
        ImGui::TextDisabled("Anchor 2 not set");
      }
    } else {
      ImGui::SetNextItemWidth(140);
      ImGui::DragFloat("Right FPS##tm", &tm_right_fps, 0.1f, 0.1f, 240.0f, "%.2f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Frame rate of the dumb source as exported. Exact-rate assumption -- if\n"
        "the export dropped frames unevenly, use Two-anchor mode instead.");
    }

    // Show solved rate when in two-anchor mode and both anchors set.
    double effective_interval = 0.0;
    bool ready_to_apply = false;
    if (tm_two_anchor_mode) {
      if (tm_anchor1_right_idx >= 0 && tm_anchor2_right_idx >= 0 &&
          tm_anchor1_right_idx != tm_anchor2_right_idx) {
        effective_interval = (tm_anchor2_left_time - tm_anchor1_left_time) /
                             static_cast<double>(tm_anchor2_right_idx - tm_anchor1_right_idx);
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f),
          "Solved rate: %.4f FPS (interval %.5f s)",
          effective_interval > 1e-9 ? 1.0 / effective_interval : 0.0,
          effective_interval);
        ready_to_apply = true;
      } else {
        ImGui::TextDisabled("Set both anchors to solve the rate.");
      }
    } else {
      if (tm_anchor1_right_idx >= 0 && tm_right_fps > 0.01f) {
        effective_interval = 1.0 / static_cast<double>(tm_right_fps);
        ready_to_apply = true;
      }
    }

    ImGui::Separator();

    ImGui::BeginDisabled(!ready_to_apply);
    if (ImGui::Button("Apply to all frames + Locate##tm")) {
      // Back-fill timestamps: t[i] = anchor1_left_time + (i - anchor1_right_idx) * effective_interval.
      for (size_t i = 0; i < srcR.frames.size(); i++) {
        const double dt = (static_cast<int>(i) - tm_anchor1_right_idx) * effective_interval;
        srcR.frames[i].timestamp = tm_anchor1_left_time + dt;
      }
      // Persist the anchor state on the source so the next session reload
      // reapplies the back-fill automatically (dumb frames have no EXIF time
      // to fall back to). User needs to Save config after this for it to stick.
      srcR.tm_anchor1_idx  = tm_anchor1_right_idx;
      srcR.tm_anchor1_time = tm_anchor1_left_time;
      srcR.tm_anchor2_idx  = tm_two_anchor_mode ? tm_anchor2_right_idx : -1;
      srcR.tm_anchor2_time = tm_two_anchor_mode ? tm_anchor2_left_time : 0.0;
      srcR.tm_fps          = tm_right_fps;
      // Back-filled timestamps are in LiDAR-time base already (anchor was the
      // LEFT camera's effective time, which is left EXIF + left time_shift).
      // Downstream colorize does `effective_t = frame.timestamp + src.time_shift`,
      // so zeroing out time_shift here prevents a double-shift. Leaves the knob
      // free for later fine-tune nudges.
      srcR.time_shift = 0.0;
      // Locate the right source along the trajectory using the new timestamps.
      if (!trajectory_built) build_trajectory();
      const auto timed_traj = timed_traj_snapshot();
      // Honor srcR's locate_mode -- Time Matcher back-fills timestamps, then
      // the source gets placed via whichever trajectory the user picked.
      const int located = Colorizer::locate_by_time(srcR, trajectory_for(srcR, timed_traj));
      logger->info("[TimeMatcher] Back-filled {} timestamps, located {}/{} frames",
                   srcR.frames.size(), located, srcR.frames.size());
    }
    ImGui::EndDisabled();
    if (!ready_to_apply) {
      ImGui::SameLine();
      ImGui::TextDisabled("(set anchors + rate first)");
    }

    ImGui::End();
  });

  // Auto-calibration window (LightGlue-assisted)
  viewer->register_ui_callback("auto_calibrate_window", [this] {
    if (!ac_show) return;
    ImGui::SetNextWindowSize(ImVec2(460, 520), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Auto-calibrate (LightGlue)", &ac_show)) {
      if (image_sources.empty()) { ImGui::TextDisabled("No image sources loaded."); ImGui::End(); return; }
      ac_cam_src = std::clamp(ac_cam_src, 0, static_cast<int>(image_sources.size()) - 1);
      auto& src = image_sources[ac_cam_src];
      if (src.frames.empty()) { ImGui::TextDisabled("Selected source has no frames."); ImGui::End(); return; }
      ac_cam_idx = std::clamp(ac_cam_idx, 0, static_cast<int>(src.frames.size()) - 1);
      const auto& cam = src.frames[ac_cam_idx];

      // --- Anchor ---
      ImGui::TextDisabled("Anchor camera");
      if (image_sources.size() > 1) {
        std::vector<std::string> labels;
        for (size_t i = 0; i < image_sources.size(); i++) labels.push_back("src " + std::to_string(i));
        std::vector<const char*> lptrs; for (auto& s : labels) lptrs.push_back(s.c_str());
        ImGui::SetNextItemWidth(100); ImGui::Combo("Source##ac", &ac_cam_src, lptrs.data(), lptrs.size());
        ImGui::SameLine();
      }
      ImGui::SetNextItemWidth(200);
      ImGui::SliderInt("Image##ac", &ac_cam_idx, 0, static_cast<int>(src.frames.size()) - 1);
      ImGui::SameLine();
      if (ImGui::ArrowButton("##ac_prev", ImGuiDir_Left)) ac_cam_idx = std::max(0, ac_cam_idx - 1);
      ImGui::SameLine();
      if (ImGui::ArrowButton("##ac_next", ImGuiDir_Right)) ac_cam_idx = std::min(static_cast<int>(src.frames.size()) - 1, ac_cam_idx + 1);
      ImGui::TextDisabled("%s  t=%.3f", boost::filesystem::path(cam.filepath).filename().string().c_str(), cam.timestamp);
      ImGui::Separator();

      // --- Accumulation ---
      ImGui::TextDisabled("LiDAR context accumulation");
      ImGui::Checkbox("Use time window (instead of N frames)##ac", &ac_use_time_window);
      if (ac_use_time_window) {
        ImGui::SetNextItemWidth(100); ImGui::DragFloat("before (s)", &ac_time_before_s, 0.1f, 0.1f, 30.0f, "%.1f");
        ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::DragFloat("after (s)", &ac_time_after_s, 0.1f, 0.1f, 30.0f, "%.1f");
      } else {
        ImGui::SetNextItemWidth(100); ImGui::DragInt("N before", &ac_n_frames_before, 1, 0, 500);
        ImGui::SameLine(); ImGui::SetNextItemWidth(100); ImGui::DragInt("N after", &ac_n_frames_after, 1, 0, 500);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Frames before/after the anchor camera's nearest LiDAR frame.\n"
          "Set small for Livox Horizon single-direction; larger for Pandar 128.");
      }
      ImGui::Checkbox("Directional filter##ac", &ac_directional_filter);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Skip frames whose heading disagrees with the anchor's -- avoids mixing outbound & return pass.");
      if (ac_directional_filter) {
        ImGui::SameLine(); ImGui::SetNextItemWidth(80);
        ImGui::DragFloat(" deg threshold##ac", &ac_directional_threshold_deg, 1.0f, 5.0f, 180.0f, "%.0f");
      }
      ImGui::SetNextItemWidth(100); ImGui::DragFloat("min range##ac", &ac_min_range, 0.05f, 0.1f, 50.0f, "%.1f");
      ImGui::SameLine(); ImGui::SetNextItemWidth(100);
      ImGui::DragFloat("max range##ac", &ac_max_range, 1.0f, 5.0f, 200.0f, "%.1f");

      ImGui::Separator();
      // --- Render ---
      ImGui::TextDisabled("Intensity render (LightGlue input)");
      // Auto-populate render W/H from source intrinsics on first open or if still at sentinel 0
      if (ac_render_width <= 0 || ac_render_height <= 0) {
        ac_render_width  = src.intrinsics.width;
        ac_render_height = src.intrinsics.height;
      }
      ImGui::SetNextItemWidth(80); ImGui::DragInt("W##ac", &ac_render_width, 8, 320, 8000);
      ImGui::SameLine(); ImGui::SetNextItemWidth(80);
      ImGui::DragInt("H##ac", &ac_render_height, 8, 240, 5000);
      ImGui::SameLine();
      if (ImGui::Button("Native##ac")) { ac_render_width = src.intrinsics.width; ac_render_height = src.intrinsics.height; }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Snap to source camera's native resolution (%dx%d).", src.intrinsics.width, src.intrinsics.height);
      ImGui::TextDisabled("Default: native (one-time calib). Lower values only to save LightGlue time on CPU.");

      ImGui::Separator();
      // --- LightGlue tuning ---
      ImGui::TextDisabled("LightGlue tuning");
      ImGui::SetNextItemWidth(100); ImGui::DragInt("max keypoints", &ac_max_kp, 32, 256, 8192);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Max keypoints per image extracted by SuperPoint.\n"
        "Increase (e.g. 4096) if the rendered intensity image has sparse features (few LiDAR returns, sparse context).");
      ImGui::SameLine(); ImGui::SetNextItemWidth(100);
      ImGui::DragFloat("min score", &ac_min_score, 0.01f, 0.0f, 1.0f, "%.2f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "LightGlue match score threshold. Lower (e.g. 0.1) if you see few matches.\n"
        "PnP-RANSAC will still filter outliers -- a noisier match pool with RANSAC is often better than too few.");

      ImGui::Separator();
      // --- Python interpreter override ---
      {
        char pybuf[512];
        std::snprintf(pybuf, sizeof(pybuf), "%s", ac_python_interpreter.c_str());
        ImGui::SetNextItemWidth(300);
        if (ImGui::InputText("Python##ac", pybuf, sizeof(pybuf))) ac_python_interpreter = pybuf;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Python interpreter to run lightglue_match.py. Default 'python3'.\n"
          "Set to a full path (e.g. /home/you/venv/bin/python) if LightGlue is in a venv\n"
          "or a different Python than what 'python3' points to.\n"
          "Tip: installing with 'python3 -m pip install ...' ensures the default works.");
      }

      ImGui::Separator();
      // --- Intrinsics optimization ---
      ImGui::Checkbox("Refine intrinsics too (second pass)##ac", &ac_optimize_intrinsics);
      if (ac_optimize_intrinsics) {
        ImGui::SameLine();
        ImGui::Checkbox("Lock extrinsic##ac", &ac_lock_extrinsic_for_intr);
      }

      // Time-shift sweep
      ImGui::Separator();
      ImGui::Checkbox("Fine-tune time shift (sweep)##ac", &ac_sweep_on);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Run the calibration multiple times, varying time_shift from (current - neg) to (current + pos)\n"
        "in the given step. Each iteration is recorded; nothing is applied until you click Apply\n"
        "on one of the result rows. Best row highlighted by inliers / (residual + 1).");
      if (ac_sweep_on) {
        ImGui::Indent();
        ImGui::SetNextItemWidth(90); ImGui::DragFloat("neg (s)##acsw", &ac_sweep_neg_range_s, 0.005f, 0.0f, 1.0f, "%.3f");
        ImGui::SameLine(); ImGui::SetNextItemWidth(90);
        ImGui::DragFloat("pos (s)##acsw", &ac_sweep_pos_range_s, 0.005f, 0.0f, 1.0f, "%.3f");
        ImGui::SameLine(); ImGui::SetNextItemWidth(90);
        ImGui::DragFloat("step (s)##acsw", &ac_sweep_step_s, 0.001f, 0.001f, 0.5f, "%.3f");
        const int est = 1 + static_cast<int>((ac_sweep_neg_range_s + ac_sweep_pos_range_s) / std::max(0.001f, ac_sweep_step_s));
        ImGui::TextDisabled("Will run %d iterations per Run click.", est);
        ImGui::Unindent();
      }

      ImGui::Separator();
      // --- Run ---
      if (ac_running) {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Running: %s", ac_status.c_str());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.25f, 0.25f, 1.0f));
        if (ImGui::Button(ac_cancel_requested ? "Stopping..." : "Stop")) {
          ac_cancel_requested = true;
        }
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Stop after the current iteration.\nSweep results already produced are kept and shown in the table.");
      } else {
        if (ImGui::Button("Run auto-calibrate")) {
          ac_running = true;
          ac_cancel_requested = false;
          ac_status = "Building context...";
          // Capture values by copy for thread-safety
          const int cam_src_i = ac_cam_src;
          const int cam_i = ac_cam_idx;
          std::thread([this, cam_src_i, cam_i]() {
            try {
              auto& src2 = image_sources[cam_src_i];
              const auto& cam2 = src2.frames[cam_i];
              // Backup for revert AND for time-shift sweep restore
              ac_backup_lever = src2.lever_arm;
              ac_backup_rpy = src2.rotation_rpy;
              ac_backup_intrinsics = src2.intrinsics;
              ac_have_backup = true;
              const double saved_time_shift = src2.time_shift;

              // Body wrapped in an inner lambda so sweep mode can invoke it per iteration.
              // Early-exits become `return;` -- they exit the current iteration only.
              auto run_once = [&]() {

              if (!trajectory_built) build_trajectory();
              const auto timed_traj = timed_traj_snapshot();

              // Current T_world_cam (using current extrinsic guess). Anchor-aware
              // so autocalibrator sweeps respect per-moment drift when anchors are set.
              const auto cc2 = effective_calib(src2, cam2.timestamp);
              const double ts = cam2.timestamp + cc2.time_shift;
              const Eigen::Isometry3d T_world_lidar_anchor = Colorizer::interpolate_pose(timed_traj, ts);
              const Eigen::Isometry3d T_lidar_cam = Colorizer::build_extrinsic(cc2.lever_arm, cc2.rotation_rpy);
              const Eigen::Isometry3d T_world_cam_init = T_world_lidar_anchor * T_lidar_cam;
              const Eigen::Vector3f anchor_forward = T_world_lidar_anchor.rotation().col(0).cast<float>().normalized();
              const Eigen::Vector3f anchor_pos = T_world_cam_init.translation().cast<float>();

              // --- Build calibration context (shared helper in auto_calibrate) ---
              ac_status = "Gathering LiDAR frames...";
              CalibContextOptions ctx_opts;
              ctx_opts.n_frames_before = ac_n_frames_before;
              ctx_opts.n_frames_after  = ac_n_frames_after;
              ctx_opts.use_time_window = ac_use_time_window;
              ctx_opts.time_before_s   = ac_time_before_s;
              ctx_opts.time_after_s    = ac_time_after_s;
              ctx_opts.directional_filter = ac_directional_filter;
              ctx_opts.directional_threshold_deg = ac_directional_threshold_deg;
              ctx_opts.min_range = ac_min_range;
              ctx_opts.max_range = ac_max_range;
              CalibrationContext ctx = build_calibration_context(
                submaps, timed_traj, ts, anchor_pos, anchor_forward, ctx_opts,
                [this](int sm_idx) { return load_hd_for_submap(sm_idx, false); });
              logger->info("[AutoCalib] Context: {} points (frames window {} directional={})",
                ctx.world_points.size(),
                ac_use_time_window
                  ? fmt::format("time +/-{:.1f}s", std::max(ac_time_before_s, ac_time_after_s))
                  : fmt::format("N {}/{} frames", ac_n_frames_before, ac_n_frames_after),
                ac_directional_filter);
              {
                const auto& p = T_world_cam_init.translation();
                logger->info("[AutoCalib] Anchor cam world pos: ({:.3f}, {:.3f}, {:.3f})  anchor_pos used for filter: ({:.3f}, {:.3f}, {:.3f})",
                  p.x(), p.y(), p.z(), anchor_pos.x(), anchor_pos.y(), anchor_pos.z());
                logger->info("[AutoCalib] Src lever: ({:.4f}, {:.4f}, {:.4f}) rpy: ({:.3f}, {:.3f}, {:.3f}) time_shift: {:.3f}",
                  src2.lever_arm.x(), src2.lever_arm.y(), src2.lever_arm.z(),
                  src2.rotation_rpy.x(), src2.rotation_rpy.y(), src2.rotation_rpy.z(),
                  src2.time_shift);
                logger->info("[AutoCalib] Cam timestamp: {:.3f} (trajectory range [{:.3f}..{:.3f}])",
                  cam2.timestamp, timed_traj.empty() ? 0.0 : timed_traj.front().stamp, timed_traj.empty() ? 0.0 : timed_traj.back().stamp);
              }

              // --- Render intensity image always (so the user can SEE the sparse rendering
              //     even when we bail out below). The render is cheap -- the expensive parts
              //     are LightGlue + PnP, which we still gate behind the sparsity check.
              ac_status = "Rendering LiDAR intensity...";
              PinholeIntrinsics K_render = src2.intrinsics;
              const double sx = static_cast<double>(ac_render_width) / src2.intrinsics.width;
              const double sy = static_cast<double>(ac_render_height) / src2.intrinsics.height;
              K_render.fx *= sx; K_render.fy *= sy;
              K_render.cx *= sx; K_render.cy *= sy;
              K_render.width = ac_render_width; K_render.height = ac_render_height;
              auto render = render_intensity_image(ctx, T_world_cam_init, K_render, ac_render_width, ac_render_height);
              {
                // Count non-zero pixels in the rendered image to detect empty renders
                long nonzero = 0;
                for (int yy = 0; yy < render.image.rows; yy++) {
                  const uint8_t* row = render.image.ptr<uint8_t>(yy);
                  for (int xx = 0; xx < render.image.cols; xx++) if (row[xx] > 0) nonzero++;
                }
                logger->info("[AutoCalib] Rendered {}x{}, {} non-zero pixels ({:.2f}%% fill)",
                  render.image.cols, render.image.rows, nonzero,
                  100.0 * nonzero / (render.image.cols * render.image.rows));
              }

              // --- Prepare the real image at the same render resolution (downscaled) ---
              cv::Mat real = cv::imread(cam2.filepath);
              if (real.empty()) { ac_status = "Failed: could not read camera image"; return; }
              cv::Mat real_resized;
              cv::resize(real, real_resized, cv::Size(ac_render_width, ac_render_height));
              cv::Mat real_gray;
              cv::cvtColor(real_resized, real_gray, cv::COLOR_BGR2GRAY);

              // --- Write both images immediately so the match viz has something to display even on failure
              if (ac_work_dir.empty()) {
                ac_work_dir = (boost::filesystem::temp_directory_path() / boost::filesystem::unique_path("glim-autocalib-%%%%")).string();
              }
              boost::filesystem::create_directories(ac_work_dir);
              const std::string real_path = ac_work_dir + "/real.png";
              const std::string rend_path = ac_work_dir + "/rendered.png";
              const std::string json_path = ac_work_dir + "/matches.json";
              cv::imwrite(real_path, real_gray);
              cv::imwrite(rend_path, render.image);
              // Signal the UI thread to (re)load these textures for the viz
              ac_match_pairs.clear();
              ac_match_scores.clear();
              ac_match_render_w = ac_render_width;
              ac_match_render_h = ac_render_height;
              ac_match_viz_needs_reload = true;

              // NOW bail if the context was too sparse for a meaningful calibration.
              if (ctx.world_points.size() < 1000) {
                ac_status = "Failed: context too sparse (" + std::to_string(ctx.world_points.size()) + " pts). See match viz below to inspect the render.";
                return;
              }

              // --- Run Python LightGlue ---
              ac_status = "Running LightGlue...";

              // Resolve script path relative to loaded_map_path or fall back to install share dir.
              std::string script = ac_python_script_path;
              if (script.empty()) {
                // Try a few defaults
                const char* candidates[] = {
                  "/ros2_ws/src/glim/scripts/lightglue_match.py",
                };
                for (const char* c : candidates) { if (boost::filesystem::exists(c)) { script = c; break; } }
              }
              if (script.empty()) { ac_status = "Failed: lightglue_match.py not found (set ac_python_script_path)"; return; }
              const std::string py = ac_python_interpreter.empty() ? std::string("python3") : ac_python_interpreter;
              char lg_args[128];
              std::snprintf(lg_args, sizeof(lg_args), " --max-kp %d --min-score %.3f", ac_max_kp, ac_min_score);
              const std::string cmd = py + " " + script + " " + real_path + " " + rend_path + " " + json_path + lg_args + " 2>&1";
              logger->info("[AutoCalib] {}", cmd);
              const int rc = std::system(cmd.c_str());
              if (rc != 0) { ac_status = "Failed: lightglue_match.py returned " + std::to_string(rc); return; }

              // --- Load matches and convert to 2D↔3D ---
              std::vector<float> confidences;
              auto raw_matches = load_lightglue_matches(json_path, &confidences);

              // Capture match pairs in RENDER-SPACE for the visualization (both images are at render res).
              ac_match_pairs.clear();
              ac_match_scores.clear();
              ac_match_pairs.reserve(raw_matches.size());
              ac_match_scores.reserve(raw_matches.size());
              for (size_t i = 0; i < raw_matches.size(); i++) {
                ac_match_pairs.emplace_back(raw_matches[i].first.cast<float>(), raw_matches[i].second.cast<float>());
                ac_match_scores.push_back(i < confidences.size() ? confidences[i] : 1.0f);
              }
              ac_match_render_w = ac_render_width;
              ac_match_render_h = ac_render_height;
              // Signal the UI thread to free & recreate textures. We CANNOT touch GL
              // from here -- GL calls only on the main thread (the GL context owner).
              ac_match_viz_needs_reload = true;

              if (raw_matches.empty()) { ac_status = "Failed: 0 matches returned (try lowering min-score or raising max-kp)"; return; }
              // The real matches are in RESIZED coords; scale back to full image to match intrinsics' native
              const double sx_back = static_cast<double>(src2.intrinsics.width) / ac_render_width;
              const double sy_back = static_cast<double>(src2.intrinsics.height) / ac_render_height;
              for (auto& p : raw_matches) { p.first.x() *= sx_back; p.first.y() *= sy_back; }
              auto corrs = matches_to_correspondences(raw_matches, confidences, render, ctx);
              ac_last_matches = static_cast<int>(corrs.size());
              if (corrs.size() < 12) { ac_status = "Failed: only " + std::to_string(corrs.size()) + " usable correspondences (see Match viz below)"; return; }

              // --- PnP refinement ---
              ac_status = "Solving PnP...";
              Eigen::Isometry3d T_world_cam_new;
              int inliers = 0; double residual = 0.0;
              if (!refine_extrinsic_pnp(corrs, src2.intrinsics, T_world_cam_init, T_world_cam_new, inliers, residual)) {
                ac_status = "Failed: PnP did not converge"; return;
              }
              ac_last_inliers = inliers;
              ac_residual_before = 0.0;  // not computed for now
              ac_residual_after = residual;

              // Compute proposed new extrinsic BUT don't apply yet -- sanity-check first.
              const Eigen::Isometry3d T_lidar_cam_new = T_world_lidar_anchor.inverse() * T_world_cam_new;
              const Eigen::Vector3d proposed_lever = T_lidar_cam_new.translation();
              const Eigen::Matrix3d Rn = T_lidar_cam_new.rotation();
              const double p_pitch = std::asin(-std::clamp(Rn(2, 0), -1.0, 1.0));
              const double r_roll  = std::atan2(Rn(2, 1), Rn(2, 2));
              const double y_yaw   = std::atan2(Rn(1, 0), Rn(0, 0));
              const Eigen::Vector3d proposed_rpy = Eigen::Vector3d(r_roll, p_pitch, y_yaw) * (180.0 / M_PI);

              // Sanity checks: reject absurd results so we never write km-offset lever arms or wild rotations.
              const double lever_shift = (proposed_lever - ac_backup_lever).norm();
              // Per-axis RPY shift, wrapped for +/-180 deg equivalence.
              auto wrap_deg = [](double d) { while (d >  180.0) d -= 360.0; while (d < -180.0) d += 360.0; return std::abs(d); };
              const double rpy_shift = std::max({wrap_deg(proposed_rpy.x() - ac_backup_rpy.x()),
                                                 wrap_deg(proposed_rpy.y() - ac_backup_rpy.y()),
                                                 wrap_deg(proposed_rpy.z() - ac_backup_rpy.z())});
              const bool residual_ok = residual <= ac_max_residual_px;
              const bool lever_ok    = lever_shift <= ac_max_lever_shift_m;
              const bool rpy_ok      = rpy_shift <= ac_max_rotation_shift_deg;

              if (!residual_ok || !lever_ok || !rpy_ok) {
                char buf[512]; std::snprintf(buf, sizeof(buf),
                  "REJECTED: residual=%.1fpx (max %.1f) lever_shift=%.3fm (max %.2f) rpy_shift=%.2f deg (max %.1f). Values NOT applied.",
                  residual, ac_max_residual_px, lever_shift, ac_max_lever_shift_m, rpy_shift, ac_max_rotation_shift_deg);
                ac_status = buf;
                logger->warn("[AutoCalib] {}", ac_status);
                logger->warn("[AutoCalib] proposed lever=({:.3f},{:.3f},{:.3f}) rpy=({:.2f},{:.2f},{:.2f})",
                  proposed_lever.x(), proposed_lever.y(), proposed_lever.z(),
                  proposed_rpy.x(), proposed_rpy.y(), proposed_rpy.z());
                return;  // leave src2.lever_arm / rotation_rpy unchanged
              }

              // Store as proposed -- DO NOT write to src yet.
              ac_proposed_lever = proposed_lever;
              ac_proposed_rpy = proposed_rpy;
              ac_proposed_has_intrinsics = false;

              // --- Optional intrinsics refinement ---
              // Refinement runs on a tentative PinholeIntrinsics copy; user still has to Apply.
              if (ac_optimize_intrinsics) {
                ac_status = "Refining intrinsics...";
                Eigen::Isometry3d T_tmp = T_world_cam_new;
                PinholeIntrinsics intr_tmp = src2.intrinsics;
                double rms_intr = 0.0;
                if (refine_intrinsics_lm(corrs, intr_tmp, T_tmp, ac_lock_extrinsic_for_intr, rms_intr)) {
                  ac_residual_after = rms_intr;
                  ac_proposed_intrinsics = intr_tmp;
                  ac_proposed_has_intrinsics = true;
                  if (!ac_lock_extrinsic_for_intr) {
                    const Eigen::Isometry3d T_lidar_cam_n2 = T_world_lidar_anchor.inverse() * T_tmp;
                    ac_proposed_lever = T_lidar_cam_n2.translation();
                    const Eigen::Matrix3d R2 = T_lidar_cam_n2.rotation();
                    const double p2 = std::asin(-std::clamp(R2(2, 0), -1.0, 1.0));
                    const double r2 = std::atan2(R2(2, 1), R2(2, 2));
                    const double y2 = std::atan2(R2(1, 0), R2(0, 0));
                    ac_proposed_rpy = Eigen::Vector3d(r2, p2, y2) * (180.0 / M_PI);
                  }
                } else {
                  logger->warn("[AutoCalib] Intrinsics LM refine failed, keeping extrinsic-only result");
                }
              }
              ac_has_proposed = true;

              char buf[256];
              std::snprintf(buf, sizeof(buf), "OK: %d matches, %d inliers, residual=%.2fpx",
                ac_last_matches, inliers, ac_residual_after);
              ac_status = buf;
              logger->info("[AutoCalib] {}", ac_status);
              };  // end run_once lambda

              // Build list of time_shifts to sweep (single entry if sweep is off)
              std::vector<float> time_shifts;
              if (ac_sweep_on) {
                const float base = static_cast<float>(saved_time_shift);
                const float step = std::max(0.001f, ac_sweep_step_s);
                for (float dt = -ac_sweep_neg_range_s; dt <= ac_sweep_pos_range_s + 1e-4f; dt += step) {
                  time_shifts.push_back(base + dt);
                }
              } else {
                time_shifts.push_back(static_cast<float>(saved_time_shift));
              }
              ac_sweep_total = static_cast<int>(time_shifts.size());
              ac_sweep_progress = 0;
              ac_sweep_results.clear();

              for (float ts : time_shifts) {
                if (ac_cancel_requested) break;  // user clicked Stop
                src2.time_shift = ts;
                ac_sweep_progress++;
                if (ac_sweep_on) {
                  char sbuf[96]; std::snprintf(sbuf, sizeof(sbuf), "Sweep %d/%d: time_shift=%+.3fs",
                    ac_sweep_progress, ac_sweep_total, ts);
                  ac_status = sbuf;
                }
                ac_has_proposed = false;
                ac_last_matches = 0; ac_last_inliers = 0; ac_residual_after = 0.0;
                run_once();
                AcSweepResult r;
                r.time_shift = ts;
                r.matches = ac_last_matches;
                r.inliers = ac_last_inliers;
                r.residual = static_cast<float>(ac_residual_after);
                r.success = ac_has_proposed;
                r.has_intrinsics = ac_proposed_has_intrinsics;
                if (ac_has_proposed) {
                  r.lever = ac_proposed_lever;
                  r.rpy = ac_proposed_rpy;
                  if (ac_proposed_has_intrinsics) r.intrinsics = ac_proposed_intrinsics;
                } else {
                  r.reject_reason = ac_status;
                }
                ac_sweep_results.push_back(r);
              }

              // Restore time_shift (none of the sweep values should stick unless user applies one)
              src2.time_shift = saved_time_shift;

              if (ac_sweep_on) {
                // Pick best by inliers / (residual + 1)
                int best_i = -1; float best_score = -1.0f;
                for (int i = 0; i < static_cast<int>(ac_sweep_results.size()); i++) {
                  const auto& r = ac_sweep_results[i];
                  if (!r.success) continue;
                  const float score = r.inliers / (r.residual + 1.0f);
                  if (score > best_score) { best_score = score; best_i = i; }
                }
                // Clear single-proposed since sweep produces a list
                ac_has_proposed = false;
                if (best_i >= 0) {
                  char sbuf[192]; std::snprintf(sbuf, sizeof(sbuf),
                    "Sweep done: %d/%d succeeded. Best = row %d (ts=%+.3fs, %d inliers, %.2fpx). Click Apply on the row you trust.",
                    static_cast<int>(std::count_if(ac_sweep_results.begin(), ac_sweep_results.end(),
                      [](const AcSweepResult& x){ return x.success; })),
                    ac_sweep_total, best_i + 1,
                    ac_sweep_results[best_i].time_shift, ac_sweep_results[best_i].inliers, ac_sweep_results[best_i].residual);
                  ac_status = sbuf;
                } else {
                  ac_status = "Sweep done: no successful runs. Try a different anchor, wider time window, or lower min-score.";
                }
                logger->info("[AutoCalib] {}", ac_status);
              } else if (ac_has_proposed) {
                ac_status = std::string(ac_status) + " -- review & click Apply to commit.";
              }
            } catch (const std::exception& e) {
              ac_status = std::string("Failed: ") + e.what();
              logger->error("[AutoCalib] Exception: {}", e.what());
            }
            ac_running = false;
          }).detach();
        }
        ImGui::SameLine();
        if (ac_have_backup && ImGui::Button("Revert last run")) {
          auto& s = image_sources[ac_cam_src];
          s.lever_arm = ac_backup_lever;
          s.rotation_rpy = ac_backup_rpy;
          s.intrinsics = ac_backup_intrinsics;
          ac_have_backup = false;
          ac_status = "Reverted to pre-run extrinsic + intrinsics.";
        }
      }
      if (!ac_status.empty()) ImGui::TextDisabled("%s", ac_status.c_str());
      if (ac_last_matches > 0) {
        ImGui::Separator();
        ImGui::TextDisabled("Last run");
        ImGui::Text("matches: %d   inliers: %d   residual: %.2fpx", ac_last_matches, ac_last_inliers, ac_residual_after);
        if (!ac_match_scores.empty()) {
          const auto q = compute_match_quality(ac_match_scores);
          ImGui::Text("  quality:  "); ImGui::SameLine();
          ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.35f, 1.0f), "%d high", q.high); ImGui::SameLine();
          ImGui::TextDisabled(" | "); ImGui::SameLine();
          ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f), "%d mid", q.mid); ImGui::SameLine();
          ImGui::TextDisabled(" | "); ImGui::SameLine();
          ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "%d low", q.low);
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "LightGlue confidence buckets: high >= %.2f, mid %.2f..%.2f, low < %.2f.",
            q.high_thresh, q.mid_thresh, q.high_thresh, q.mid_thresh);
        }

        // Proposed vs current, side-by-side. User must click Apply to commit.
        if (ac_has_proposed) {
          const auto& s2 = image_sources[ac_cam_src];
          ImGui::TextDisabled("                    current                        proposed");
          ImGui::Text("lever: (%7.4f, %7.4f, %7.4f)   (%7.4f, %7.4f, %7.4f)",
            s2.lever_arm.x(), s2.lever_arm.y(), s2.lever_arm.z(),
            ac_proposed_lever.x(), ac_proposed_lever.y(), ac_proposed_lever.z());
          ImGui::Text("RPY  : (%7.3f, %7.3f, %7.3f)   (%7.3f, %7.3f, %7.3f)",
            s2.rotation_rpy.x(), s2.rotation_rpy.y(), s2.rotation_rpy.z(),
            ac_proposed_rpy.x(), ac_proposed_rpy.y(), ac_proposed_rpy.z());
          if (ac_proposed_has_intrinsics) {
            ImGui::Text("f    : (%.2f, %.2f) -> (%.2f, %.2f)",
              s2.intrinsics.fx, s2.intrinsics.fy,
              ac_proposed_intrinsics.fx, ac_proposed_intrinsics.fy);
            ImGui::Text("c    : (%.2f, %.2f) -> (%.2f, %.2f)",
              s2.intrinsics.cx, s2.intrinsics.cy,
              ac_proposed_intrinsics.cx, ac_proposed_intrinsics.cy);
            ImGui::Text("k1   : %.4f -> %.4f   k2: %.4f -> %.4f   k3: %.4f -> %.4f",
              s2.intrinsics.k1, ac_proposed_intrinsics.k1,
              s2.intrinsics.k2, ac_proposed_intrinsics.k2,
              s2.intrinsics.k3, ac_proposed_intrinsics.k3);
          }
          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.55f, 0.25f, 1.0f));
          if (ImGui::Button("Apply to source")) {
            auto& s = image_sources[ac_cam_src];
            s.lever_arm = ac_proposed_lever;
            s.rotation_rpy = ac_proposed_rpy;
            if (ac_proposed_has_intrinsics) s.intrinsics = ac_proposed_intrinsics;
            ac_has_proposed = false;
            ac_status = "Applied to source.";
            logger->info("[AutoCalib] Proposed values applied to src[{}]", ac_cam_src);
          }
          ImGui::PopStyleColor();
          ImGui::SameLine();
          if (ImGui::Button("Reject##acprop")) {
            ac_has_proposed = false;
            ac_status = "Rejected. No changes applied.";
          }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Discard proposed values without applying.");
        } else {
          const auto& s2 = image_sources[ac_cam_src];
          ImGui::TextDisabled("lever: (%.4f, %.4f, %.4f)", s2.lever_arm.x(), s2.lever_arm.y(), s2.lever_arm.z());
          ImGui::TextDisabled("RPY : (%.3f, %.3f, %.3f)", s2.rotation_rpy.x(), s2.rotation_rpy.y(), s2.rotation_rpy.z());
          ImGui::TextDisabled("f   : (%.2f, %.2f)  c: (%.2f, %.2f)", s2.intrinsics.fx, s2.intrinsics.fy, s2.intrinsics.cx, s2.intrinsics.cy);
          ImGui::TextDisabled("dist: k1=%.4f k2=%.4f p1=%.4f p2=%.4f k3=%.4f",
            s2.intrinsics.k1, s2.intrinsics.k2, s2.intrinsics.p1, s2.intrinsics.p2, s2.intrinsics.k3);
        }
      }

      // Sanity-check threshold controls (fold so they don't clutter the window)
      // Sweep results table
      if (!ac_sweep_results.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("Sweep results (%zu runs)  -- green = best by inliers/(residual+1)", ac_sweep_results.size());
        // Find best row (successful only)
        int best_i = -1; float best_score = -1.0f;
        for (int i = 0; i < static_cast<int>(ac_sweep_results.size()); i++) {
          const auto& r = ac_sweep_results[i];
          if (!r.success) continue;
          const float score = r.inliers / (r.residual + 1.0f);
          if (score > best_score) { best_score = score; best_i = i; }
        }
        if (ImGui::BeginTable("##sweeptable", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
          ImGui::TableSetupColumn("ts (s)");
          ImGui::TableSetupColumn("matches");
          ImGui::TableSetupColumn("inliers");
          ImGui::TableSetupColumn("resid (px)");
          ImGui::TableSetupColumn("status");
          ImGui::TableSetupColumn("");
          ImGui::TableHeadersRow();
          for (int i = 0; i < static_cast<int>(ac_sweep_results.size()); i++) {
            const auto& r = ac_sweep_results[i];
            ImGui::TableNextRow();
            if (i == best_i) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(40, 90, 40, 180));
            ImGui::TableSetColumnIndex(0); ImGui::Text("%+.3f", r.time_shift);
            ImGui::TableSetColumnIndex(1); ImGui::Text("%d", r.matches);
            ImGui::TableSetColumnIndex(2); ImGui::Text("%d", r.inliers);
            ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f", r.residual);
            ImGui::TableSetColumnIndex(4); ImGui::Text("%s", r.success ? "OK" : "fail");
            ImGui::TableSetColumnIndex(5);
            if (r.success) {
              ImGui::PushID(i);
              if (ImGui::SmallButton("Apply")) {
                auto& s = image_sources[ac_cam_src];
                s.lever_arm = r.lever;
                s.rotation_rpy = r.rpy;
                s.time_shift = r.time_shift;
                if (r.has_intrinsics) s.intrinsics = r.intrinsics;
                char sbuf[128]; std::snprintf(sbuf, sizeof(sbuf),
                  "Applied sweep row %d (ts=%+.3f, %d inliers, %.2fpx)",
                  i + 1, r.time_shift, r.inliers, r.residual);
                ac_status = sbuf;
                logger->info("[AutoCalib] {}", ac_status);
              }
              ImGui::PopID();
            }
          }
          ImGui::EndTable();
        }
        if (ImGui::Button("Clear sweep results##acsw")) ac_sweep_results.clear();
      }

      if (ImGui::CollapsingHeader("Sanity thresholds##ac")) {
        ImGui::SetNextItemWidth(100); ImGui::DragFloat("max residual (px)", &ac_max_residual_px, 0.5f, 1.0f, 200.0f, "%.1f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reject result if PnP reprojection residual exceeds this.\n20px is a loose default for LightGlue on LiDAR intensity.");
        ImGui::SetNextItemWidth(100); ImGui::DragFloat("max lever shift (m)", &ac_max_lever_shift_m, 0.05f, 0.01f, 10.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reject if the proposed lever differs from pre-run by more than this distance.\n1m is conservative for small-rig setups.");
        ImGui::SetNextItemWidth(100); ImGui::DragFloat("max RPY shift ( deg)", &ac_max_rotation_shift_deg, 0.5f, 0.1f, 90.0f, "%.1f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reject if any axis of RPY differs from pre-run by more than this.\n15 deg is conservative; a well-tuned manual cal should shift <5 deg.");
      }

      // --- Match visualization (side-by-side real | rendered with lines between matches) ---
      // Show viz whenever a render exists (matches can be empty on context-too-sparse / LightGlue fail)
      if (ac_match_render_w > 0 && ac_match_render_h > 0 && !ac_work_dir.empty()) {
        ImGui::Separator();
        ImGui::Checkbox("Show match viz##ac", &ac_show_match_viz);
        // Consume a pending reload from the worker thread (GL calls MUST run here)
        if (ac_match_viz_needs_reload) {
          if (ac_real_tex) { glDeleteTextures(1, &ac_real_tex); ac_real_tex = 0; }
          if (ac_rend_tex) { glDeleteTextures(1, &ac_rend_tex); ac_rend_tex = 0; }
          ac_match_viz_needs_reload = false;
        }
        if (ac_show_match_viz) {
          // Lazy texture load on the UI thread (GL calls MUST run here)
          if (ac_real_tex == 0) {
            const std::string rp = ac_work_dir + "/real.png";
            cv::Mat img = cv::imread(rp);
            if (!img.empty()) {
              cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
              glGenTextures(1, &ac_real_tex);
              glBindTexture(GL_TEXTURE_2D, ac_real_tex);
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);
              glBindTexture(GL_TEXTURE_2D, 0);
              ac_real_tex_w = img.cols; ac_real_tex_h = img.rows;
            }
          }
          if (ac_rend_tex == 0) {
            const std::string rp = ac_work_dir + "/rendered.png";
            cv::Mat img = cv::imread(rp, cv::IMREAD_GRAYSCALE);
            if (!img.empty()) {
              cv::Mat rgb; cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
              glGenTextures(1, &ac_rend_tex);
              glBindTexture(GL_TEXTURE_2D, ac_rend_tex);
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
              glBindTexture(GL_TEXTURE_2D, 0);
              ac_rend_tex_w = rgb.cols; ac_rend_tex_h = rgb.rows;
            }
          }

          if (ac_real_tex && ac_rend_tex) {
            // Display at capped width (half the ImGui window, each image)
            const float avail_w = ImGui::GetContentRegionAvail().x;
            const float pair_w = (avail_w - 8.0f) * 0.5f;  // 8px gap between the two
            // Aspect-preserving height
            const float rh = pair_w * static_cast<float>(ac_match_render_h) / static_cast<float>(ac_match_render_w);
            const ImVec2 real_pos = ImGui::GetCursorScreenPos();
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(ac_real_tex)), ImVec2(pair_w, rh));
            ImGui::SameLine(0.0f, 8.0f);
            const ImVec2 rend_pos = ImGui::GetCursorScreenPos();
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(ac_rend_tex)), ImVec2(pair_w, rh));

            // Match lines
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float sx = pair_w / static_cast<float>(ac_match_render_w);
            const float sy = rh     / static_cast<float>(ac_match_render_h);
            // Find score range for coloring
            float smin = 1e9f, smax = -1e9f;
            for (float s : ac_match_scores) { smin = std::min(smin, s); smax = std::max(smax, s); }
            if (smax <= smin) { smin = 0.0f; smax = 1.0f; }
            for (size_t i = 0; i < ac_match_pairs.size(); i++) {
              const auto& m = ac_match_pairs[i];
              const ImVec2 a(real_pos.x + m.first.x()  * sx, real_pos.y + m.first.y()  * sy);
              const ImVec2 b(rend_pos.x + m.second.x() * sx, rend_pos.y + m.second.y() * sy);
              const float sn = (ac_match_scores[i] - smin) / (smax - smin);
              const int r = static_cast<int>((1.0f - sn) * 255);  // low score = red
              const int g = static_cast<int>(sn * 255);           // high score = green
              const ImU32 col = IM_COL32(r, g, 0, 180);
              dl->AddCircleFilled(a, 2.5f, col);
              dl->AddCircleFilled(b, 2.5f, col);
              dl->AddLine(a, b, col, 1.0f);
            }
            ImGui::TextDisabled("matches: %zu  (render %dx%d displayed at %.0fx%.0f each; red=low score, green=high)",
              ac_match_pairs.size(), ac_match_render_w, ac_match_render_h, pair_w, rh);
          } else {
            ImGui::TextDisabled("(loading textures...)");
          }
        }
      }
    }
    ImGui::End();
  });

  // COLMAP exporter: click-to-place handler (runs only when in "place" mode)
  viewer->register_ui_callback("colmap_place_3d", [this] {
    if (!ce_placing) return;
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    static bool mouse_was_down = false;
    static double mouse_down_time = 0.0;
    if (ImGui::IsMouseDown(0) && !mouse_was_down) { mouse_was_down = true; mouse_down_time = ImGui::GetTime(); }
    if (!ImGui::IsMouseReleased(0)) { if (!ImGui::IsMouseDown(0)) mouse_was_down = false; return; }
    mouse_was_down = false;
    if (ImGui::GetTime() - mouse_down_time > 0.25) return;  // was a drag

    auto vw = guik::LightViewer::instance();
    const auto mouse = ImGui::GetMousePos();
    const Eigen::Vector2i mpos(static_cast<int>(mouse.x), static_cast<int>(mouse.y));
    const float depth = vw->pick_depth(mpos);
    if (depth >= 1.0f) return;  // background
    ce_center = vw->unproject(mpos, depth);
    ce_region_placed = true;
    ce_placing = false;
    ce_status = "Region placed. Adjust size/pos then Export.";
    logger->info("[COLMAP] Region placed at ({:.2f}, {:.2f}, {:.2f})", ce_center.x(), ce_center.y(), ce_center.z());
  });

  // COLMAP exporter: cube rendering (wire frame + transparent fill)
  viewer->register_ui_callback("colmap_cube_render", [this] {
    auto vw = guik::LightViewer::instance();
    if (!ce_region_placed) {
      vw->remove_drawable("colmap_cube_wire");
      vw->remove_drawable("colmap_cube_fill");
      return;
    }
    Eigen::Affine3f tf = Eigen::Affine3f::Identity();
    tf.translate(ce_center);
    // Yaw rotation around world Z so the cube visually aligns with the region
    // (road / building edge) the user wants to export.
    tf.rotate(Eigen::AngleAxisf(ce_yaw_deg * static_cast<float>(M_PI) / 180.0f,
                                Eigen::Vector3f::UnitZ()));
    // Primitives::cube() is a [-1,1]^3 cube, so scale by half-size
    tf.scale(Eigen::Vector3f(0.5f * ce_size.x(), 0.5f * ce_size.y(), 0.5f * ce_size.z()));
    vw->update_drawable("colmap_cube_wire", glk::Primitives::wire_cube(),
      guik::FlatColor(0.3f, 0.9f, 0.4f, 1.0f, tf));
    vw->update_drawable("colmap_cube_fill", glk::Primitives::cube(),
      guik::FlatColor(0.2f, 0.8f, 0.3f, 0.15f, tf).make_transparent());
  });

  // COLMAP exporter window
  viewer->register_ui_callback("colmap_export_window", [this] {
    if (!ce_show) return;
    ImGui::SetNextWindowSize(ImVec2(420, 520), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("COLMAP Exporter (dev, single-chunk)", &ce_show)) {
      ImGui::TextDisabled("Places a 3D cube; 2D top-view footprint is used for trimming.\nZ range is ignored (full column exported).");
      // Shortcut to the virtual-cameras tool -- commonly run alongside a COLMAP
      // export so the anchor virtuals land in the same package that goes to
      // Metashape / LichtFeld.
      if (ImGui::Button("Open Virtual Cameras...##ce")) show_virtual_cameras_window = true;
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Open the Virtual Cameras window. Render LiDAR-synthesized locked-pose\n"
        "anchors along the trajectory -- Metashape uses them as rigid BA anchors\n"
        "to align real cameras against. Typical workflow: tune here first, then\n"
        "run the main export below with real cameras + their intrinsics.");
      ImGui::Separator();

      // Placement
      if (!ce_region_placed) {
        ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.2f, 1.0f), "No region placed yet.");
      } else {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "Region placed.");
      }
      if (ImGui::Button(ce_placing ? "Click on map to place..." : "Place region")) { ce_placing = !ce_placing; }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click anywhere on the 3D map to position the cube there (size stays at current XYZ).");
      ImGui::SameLine();
      if (ImGui::Button("Clear")) {
        ce_region_placed = false;
        auto vw = guik::LightViewer::instance();
        vw->remove_drawable("colmap_cube_wire");
        vw->remove_drawable("colmap_cube_fill");
      }

      ImGui::Separator();
      ImGui::TextDisabled("Position (world, m)");
      ImGui::SetNextItemWidth(110); ImGui::DragFloat("X##cep", &ce_center.x(), 0.5f, -1e6f, 1e6f, "%.3f");
      ImGui::SameLine(); ImGui::SetNextItemWidth(110); ImGui::DragFloat("Y##cep", &ce_center.y(), 0.5f, -1e6f, 1e6f, "%.3f");
      ImGui::SameLine(); ImGui::SetNextItemWidth(110); ImGui::DragFloat("Z##cep", &ce_center.z(), 0.5f, -1e6f, 1e6f, "%.3f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Drag to move; double-click to type an exact value.");

      ImGui::TextDisabled("Size (m)");
      ImGui::SetNextItemWidth(110); ImGui::DragFloat("Xs##ces", &ce_size.x(), 1.0f, 1.0f, 2000.0f, "%.1f");
      ImGui::SameLine(); ImGui::SetNextItemWidth(110); ImGui::DragFloat("Ys##ces", &ce_size.y(), 1.0f, 1.0f, 2000.0f, "%.1f");
      ImGui::SameLine(); ImGui::SetNextItemWidth(110); ImGui::DragFloat("Zs##ces", &ce_size.z(), 1.0f, 1.0f, 2000.0f, "%.1f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Z size is only for the cube preview -- export trimming uses only XY.");

      ImGui::TextDisabled("Yaw (deg, around world Z)");
      ImGui::SetNextItemWidth(160);
      ImGui::DragFloat("Yaw##ceyaw", &ce_yaw_deg, 0.5f, -180.0f, 180.0f, "%.1f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Rotate the export region around world Z (top-view yaw). Useful to align\n"
        "the rectangle with a road / building edge that runs diagonally.\n"
        "Cube updates live. On export, the world is also rotated so the region's\n"
        "local X/Y become the output's X/Y (tile comes out axis-aligned).");
      ImGui::SameLine();
      if (ImGui::SmallButton("0##yawreset")) ce_yaw_deg = 0.0f;
      ImGui::SameLine();
      if (ImGui::SmallButton("+90##yaw")) ce_yaw_deg = std::fmod(ce_yaw_deg + 90.0f + 180.0f, 360.0f) - 180.0f;
      ImGui::SameLine();
      if (ImGui::SmallButton("-90##yaw")) ce_yaw_deg = std::fmod(ce_yaw_deg - 90.0f + 180.0f, 360.0f) - 180.0f;

      ImGui::Separator();
      ImGui::TextDisabled("Export options");
      // Copy/symlink <-> Undistort are mutually constrained:
      //   undistort ON  => copy ON    (can't symlink to freshly-rectified JPEGs)
      //   copy OFF      => undistort OFF  (can't rectify if we're symlinking)
      // Show both checkboxes always; when the invariant would be violated the
      // offending box is disabled and synced to the forced value so the UI
      // reflects exactly what the export will do.
      {
        const bool force_copy = ce_undistort_images;
        if (force_copy) ce_copy_images = true;
        ImGui::BeginDisabled(force_copy);
        ImGui::Checkbox("Copy images (else symlink)##ce", &ce_copy_images);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) {
          if (force_copy)
            ImGui::SetTooltip(
              "Forced ON while 'Undistort images' is ON:\n"
              "the exported JPEGs are freshly rectified -- a symlink would\n"
              "point at the raw distorted originals and bypass the rectification.");
          else
            ImGui::SetTooltip(
              "Copy is slower + doubles disk use but the export stays valid if\n"
              "you move/delete the source images. Symlink is cheap but the\n"
              "export breaks if the sources move.");
        }
      }
      ImGui::SetNextItemWidth(120); ImGui::DragFloat("Overlap margin (m)", &ce_overlap_margin_m, 0.5f, 0.0f, 50.0f, "%.1f");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cameras outside the trim bounds but within this distance are still included.\nHelps 3DGS get context at the tile boundary.");
      ImGui::Checkbox("Voxelized HD only##ce", &ce_voxelized_only);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Require voxelized HD (hd_frames_voxelized/). Massive raw HD would produce an unusable points3D.txt.");
      ImGui::Checkbox("Rotate to Y-up (3DGS)##ce", &ce_rotate_to_y_up);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Rotate the exported world from our Z-up frame to a 3DGS-style Y-up frame.\n"
        "LichtFeld/gsplat/nerfstudio viewers orbit around Y-up by default -- without\n"
        "this the scene loads rotated 90 deg. Training math is identical either way,\n"
        "this is purely viewer ergonomics.");
      {
        const bool block_undistort = !ce_copy_images;
        if (block_undistort) ce_undistort_images = false;
        ImGui::BeginDisabled(block_undistort);
        ImGui::Checkbox("Undistort images (PINHOLE)##ce", &ce_undistort_images);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) {
          if (block_undistort)
            ImGui::SetTooltip(
              "Disabled while 'Copy images' is OFF. Undistorting produces fresh\n"
              "JPEGs on disk -- we can't symlink to them. Enable Copy to unlock.");
          else
            ImGui::SetTooltip(
              "Who applies the k1/k2/k3/p1/p2 distortion -- we do it at export, or\n"
              "the downstream tool does it at projection. Same physics either way,\n"
              "just split labour differently.\n"
              "\n"
              "ON  (default): we apply distortion NOW via cv::remap using the full\n"
              "                k1/k2/k3/p1/p2 source lens model. Output images are\n"
              "                rectified (pinhole-like). cameras.txt declares PINHOLE\n"
              "                (fx fy cx cy) since there's nothing left to undistort.\n"
              "                Required by LichtFeld unless 3DGUT is enabled. Forces\n"
              "                image COPY (symlinks would point at the raw distorted\n"
              "                sources and undo the rectification).\n"
              "\n"
              "OFF: images are copied / symlinked RAW, and cameras.txt carries the\n"
              "     OPENCV model (fx fy cx cy k1 k2 p1 p2) so the downstream tool\n"
              "     applies distortion at projection time. Note: COLMAP's OPENCV\n"
              "     model has no slot for k3 -- if your lens needs k3, pair ON with\n"
              "     BlocksExchange (which carries the full k1/k2/k3/p1/p2 set).");
        }
      }
      ImGui::Checkbox("Emit Bundler (bundle.out)##ce", &ce_write_bundler);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Additionally write bundle.out + bundle.out.list.txt at the dataset root.\n"
        "Metashape: File -> Import Cameras... -> Bundler to pick them up\n"
        "(COLMAP import isn't built in to Metashape). Sparse cloud is included\n"
        "with 0-length tracks -- enough for camera alignment + color sampling.\n"
        "Bundler can only represent k1/k2 radial distortion (k3/p1/p2 are lost),\n"
        "so pair this with 'Undistort images' when possible.");
      ImGui::Checkbox("Emit BlocksExchange (xml)##ce", &ce_write_blocks_exchange);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Additionally write blocks_exchange.xml at the dataset root.\n"
        "ContextCapture / RealityCapture / Metashape all import this format.\n"
        "Supports the FULL Brown-Conrady distortion model (k1/k2/k3/p1/p2),\n"
        "so it's the better pick when emitting RAW images + OPENCV intrinsics.\n"
        "Note: 360/equirectangular cameras are NOT handled here yet -- those\n"
        "would need a cube-face splitter pre-step (planned).");

      // Pose priors only apply to BlocksExchange (Bundler has no accuracy slot).
      if (ce_write_blocks_exchange) {
        ImGui::Indent();
        ImGui::Checkbox("Pose priors (pos/rot accuracy)##ce", &ce_use_pose_priors);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Emit per-photo accuracy hints in blocks_exchange.xml so Metashape's\n"
          "Optimize Cameras (BA) treats our poses as CONSTRAINED priors instead\n"
          "of loose initial guesses. Keeps scale + origin nailed to our session\n"
          "frame during refinement -- critical for multi-tile coherence and\n"
          "aerial-LiDAR fusion.\n"
          "\n"
          "OFF: no accuracy tags emitted. Metashape BA can freely apply a\n"
          "     similarity transform to the whole block (scale / drift / rotate).\n"
          "ON:  emit <Accuracy> tags with the sigmas below. BA refines poses\n"
          "     locally within those bounds.");
        if (ce_use_pose_priors) {
          ImGui::BeginDisabled(!ce_use_pose_priors);
          ImGui::SetNextItemWidth(140);
          ImGui::DragFloat("Position sigma (m)##cepp", &ce_pose_pos_sigma_m, 0.005f, 0.001f, 10.0f, "%.3f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Position accuracy per photo (metres). Sensible range 0.02-0.10 m\n"
            "for LiDAR-derived poses. Tighter = less freedom for BA drift,\n"
            "looser = BA allowed to relocate cameras more.");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(140);
          ImGui::DragFloat("Rotation sigma (deg)##cepp", &ce_pose_rot_sigma_deg, 0.1f, 0.01f, 30.0f, "%.2f");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Rotation accuracy per photo (degrees). Sensible range 0.5-3 deg\n"
            "for LiDAR-derived poses. Tighter = BA trusts the orientation,\n"
            "looser = BA free to rotate cameras to fit features.");
          ImGui::EndDisabled();
        }
        ImGui::Unindent();
      }

      // Check voxelized availability
      const bool have_voxelized = !hd_frames_path.empty() &&
        boost::filesystem::is_directory(hd_frames_path + "_voxelized");
      if (ce_voxelized_only && !have_voxelized) {
        ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f), "No voxelized HD found at %s_voxelized",
          hd_frames_path.empty() ? "(no hd_frames_path)" : hd_frames_path.c_str());
        ImGui::TextDisabled("Run Tools -> Utils -> Voxelize HD data first.");
      }

      ImGui::Separator();
      // Output dir
      {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s", ce_output_dir.c_str());
        ImGui::SetNextItemWidth(280);
        if (ImGui::InputText("Output dir##ce", buf, sizeof(buf))) ce_output_dir = buf;
        ImGui::SameLine();
        if (ImGui::Button("...##ceout")) {
          const std::string chosen = pfd::select_folder("Choose output dir for COLMAP export", ce_output_dir).result();
          if (!chosen.empty()) ce_output_dir = chosen;
        }
      }

      // Export button
      const bool can_export = ce_region_placed && !ce_output_dir.empty() &&
        (!ce_voxelized_only || have_voxelized) && !ce_running;
      if (!can_export) ImGui::BeginDisabled();
      if (ImGui::Button("Export")) {
        ce_running = true;
        ce_status = "Gathering points and cameras...";
        const Eigen::Vector3f center = ce_center;
        const Eigen::Vector3f size = ce_size;
        const std::string out_dir = ce_output_dir;
        const bool copy_img = ce_copy_images;
        const float overlap = ce_overlap_margin_m;
        const bool rot_yup = ce_rotate_to_y_up;
        const float yaw_deg = ce_yaw_deg;
        const bool undistort = ce_undistort_images;
        const bool write_bundler = ce_write_bundler;
        const bool write_blocks_exchange = ce_write_blocks_exchange;
        const bool use_priors = ce_use_pose_priors && ce_write_blocks_exchange;
        const float pos_sigma = ce_pose_pos_sigma_m;
        const float rot_sigma = ce_pose_rot_sigma_deg;
        std::thread([this, center, size, out_dir, copy_img, overlap, rot_yup, yaw_deg, undistort, write_bundler, write_blocks_exchange, use_priors, pos_sigma, rot_sigma]() {
          try {
            // 1. Gather points from hd_frames_voxelized/ (or from loaded voxelized state -- simplest to load here)
            ExportBounds2D bounds;
            bounds.x_min = center.x() - 0.5f * size.x();
            bounds.x_max = center.x() + 0.5f * size.x();
            bounds.y_min = center.y() - 0.5f * size.y();
            bounds.y_max = center.y() + 0.5f * size.y();
            bounds.z_min = center.z() - 0.5f * size.z();
            bounds.z_max = center.z() + 0.5f * size.z();
            bounds.yaw_deg = yaw_deg;

            std::vector<ColoredPoint> pts;
            const std::string vox_dir = hd_frames_path + "_voxelized";
            if (!boost::filesystem::is_directory(vox_dir)) {
              ce_status = "Failed: voxelized dir not found: " + vox_dir;
              ce_running = false;
              return;
            }
            // Walk voxelized frames: each subdir has points.bin and intensities.bin;
            // aux_rgb.bin may be present (from Apply colorize). We iterate once.
            if (!trajectory_built) build_trajectory();
            size_t total_scanned = 0;
            for (boost::filesystem::directory_iterator it(vox_dir), end; it != end; ++it) {
              if (!boost::filesystem::is_directory(it->path())) continue;
              const std::string fdir = it->path().string();
              const std::string meta = fdir + "/frame_meta.json";
              if (!boost::filesystem::exists(meta)) continue;
              std::ifstream ifs(meta);
              const auto j = nlohmann::json::parse(ifs, nullptr, false);
              if (j.is_discarded()) continue;
              const int np = j.value("num_points", 0);
              if (np == 0) continue;
              // Frame pose: use frame_meta.json's T_world_lidar if present, else skip
              // (for voxelized frames we rely on per-frame stored world-frame transforms;
              //  if not available, we fall back to per-point world coords stored directly)
              std::vector<Eigen::Vector3f> frame_pts(np);
              std::vector<float> frame_int(np, 0.0f);
              std::vector<Eigen::Vector3f> frame_rgb(np, Eigen::Vector3f(0.5f, 0.5f, 0.5f));
              bool frame_has_rgb = false;
              { std::ifstream f(fdir + "/points.bin", std::ios::binary);
                if (!f) continue;
                f.read(reinterpret_cast<char*>(frame_pts.data()), sizeof(Eigen::Vector3f) * np); }
              { std::ifstream f(fdir + "/intensities.bin", std::ios::binary);
                if (f) f.read(reinterpret_cast<char*>(frame_int.data()), sizeof(float) * np); }
              { std::ifstream f(fdir + "/aux_rgb.bin", std::ios::binary);
                if (f) {
                  f.read(reinterpret_cast<char*>(frame_rgb.data()), sizeof(Eigen::Vector3f) * np);
                  frame_has_rgb = static_cast<bool>(f);
                }
              }
              // If aux_rgb.bin wasn't there, fall back to intensity mapped to
              // grayscale so the points3D.ply still carries meaningful colour
              // instead of uniform 0.5 grey. Intensity is scanner-dependent, so
              // we normalise per-frame by the 99th percentile (same clip the
              // virtual-camera rasterizer uses) to keep highlights from
              // saturating a single retroreflective-marking cluster.
              if (!frame_has_rgb) {
                std::vector<float> sorted_int(frame_int);
                std::sort(sorted_int.begin(), sorted_int.end());
                const float imax = sorted_int.empty()
                  ? 255.0f
                  : sorted_int[std::min<size_t>(sorted_int.size() - 1,
                                                 static_cast<size_t>(0.99 * sorted_int.size()))];
                const float inv = (imax > 1e-6f) ? 1.0f / imax : 1.0f / 255.0f;
                for (int i = 0; i < np; i++) {
                  const float g = std::clamp(frame_int[i] * inv, 0.0f, 1.0f);
                  frame_rgb[i] = Eigen::Vector3f(g, g, g);
                }
              }
              // Points are expected to already be in world frame (voxelized step saves world coords).
              for (int i = 0; i < np; i++) {
                if (!bounds.contains_xy(frame_pts[i])) { total_scanned++; continue; }
                ColoredPoint cp;
                cp.xyz = frame_pts[i];
                cp.rgb = frame_rgb[i];
                pts.push_back(cp);
                total_scanned++;
              }
            }
            logger->info("[COLMAP] Filtered {} points within bounds (scanned {})", pts.size(), total_scanned);
            if (pts.empty()) { ce_status = "Failed: 0 points in region"; ce_running = false; return; }

            // 2. Gather cameras from all image_sources
            const auto timed_traj = timed_traj_snapshot();
            std::vector<ExportCameraFrame> cams;
            std::vector<PinholeIntrinsics> intrs;
            std::vector<CameraType> cam_types;
            for (size_t si = 0; si < image_sources.size(); si++) {
              intrs.push_back(image_sources[si].intrinsics);
              cam_types.push_back(image_sources[si].camera_type);
              for (size_t fi = 0; fi < image_sources[si].frames.size(); fi++) {
                const auto& cf = image_sources[si].frames[fi];
                if (!cf.located || cf.timestamp <= 0.0) continue;
                ExportCameraFrame e;
                e.source_image_path = cf.filepath;
                e.source_mask_path  = image_sources[si].mask_path;
                e.source_idx = static_cast<int>(si);
                const std::string stem = boost::filesystem::path(cf.filepath).stem().string();
                const std::string ext  = boost::filesystem::path(cf.filepath).extension().string();
                e.export_name = "src" + std::to_string(si) + "_" + stem + ext;
                e.T_world_cam = cf.T_world_cam;
                cams.push_back(std::move(e));
              }
            }
            logger->info("[COLMAP] {} candidate cameras across {} sources", cams.size(), image_sources.size());

            // 3. Write export
            ExportOptions opt;
            opt.output_dir = out_dir;
            // Copy is mandatory when undistorting: the exported images are
            // freshly-encoded, a symlink would only round back to the raw input.
            opt.copy_images = copy_img || undistort;
            // Per-tile recenter DISABLED: every tile exported from the same
            // session shares the same session-local-UTM frame, so aerial LiDAR
            // and other sessions can fuse with a single constant offset.
            opt.re_origin = false;
            opt.overlap_margin_m = overlap;
            opt.rotate_to_y_up = rot_yup;
            opt.export_undistorted = undistort;
            opt.export_bundler = write_bundler;
            opt.export_blocks_exchange = write_blocks_exchange;
            opt.emit_pose_priors = use_priors;
            opt.pose_pos_sigma_m = pos_sigma;
            opt.pose_rot_sigma_deg = rot_sigma;
            std::string err;
            auto stats = write_colmap_export(bounds, pts, cams, intrs, cam_types, opt, &err);
            ce_last_points = stats.points_written;
            ce_last_cameras = stats.cameras_written;
            ce_last_images = stats.images_copied;
            ce_last_masks = stats.masks_copied;

            // --- SHIFT.txt: explain the coord frame + how to recover real UTM ---
            // All tiles from this session land in the SAME session-local frame, so
            // aerial LiDAR / other sessions can be fused with a single constant
            // offset. SHIFT.txt is for user reference only, not read by any tool.
            {
              std::ofstream sf(out_dir + "/SHIFT.txt");
              if (sf) {
                sf << "# GLIM COLMAP export -- coordinate frame notes\n\n";
                sf << "Exported coordinate frame: GLIM session-local.\n";
                sf << "Per-tile recentering is OFF, so every tile from this dataset\n";
                sf << "shares this exact frame. Aerial LiDAR / other sessions fuse\n";
                sf << "by applying a single constant offset.\n\n";
                if (rot_yup) {
                  sf << "Axis mapping (Rotate to Y-up was ON):\n";
                  sf << "  exported_x =  -session_y   (our Y-left -> X-right)\n";
                  sf << "  exported_y =  +session_z   (our Z-up   -> Y-up)\n";
                  sf << "  exported_z =  -session_x   (our X-fwd  -> Z-back)\n";
                  sf << "  Undo this rotation BEFORE applying UTM offsets below.\n\n";
                } else {
                  sf << "Axis mapping: identity (Rotate to Y-up was OFF). Exported\n";
                  sf << "  axes equal session axes: X fwd, Y left, Z up.\n\n";
                }
                if (gnss_datum_available) {
                  sf << "UTM datum (from gnss_datum.json):\n";
                  sf << "  zone             = " << gnss_utm_zone << "\n";
                  sf << "  easting_origin   = " << std::setprecision(12) << gnss_utm_easting_origin << "\n";
                  sf << "  northing_origin  = " << std::setprecision(12) << gnss_utm_northing_origin << "\n";
                  sf << "  alt_origin       = " << std::setprecision(6)  << gnss_datum_alt << "\n";
                  sf << "  datum_lat (WGS84)= " << std::setprecision(10) << gnss_datum_lat << "\n";
                  sf << "  datum_lon (WGS84)= " << std::setprecision(10) << gnss_datum_lon << "\n\n";
                  sf << "To recover real UTM coordinates from session-local coords:\n";
                  sf << "  UTM_easting  = session_x + easting_origin\n";
                  sf << "  UTM_northing = session_y + northing_origin\n";
                  sf << "  UTM_alt      = session_z + alt_origin\n";
                } else {
                  sf << "No GNSS datum loaded for this session -- the session-local\n";
                  sf << "frame has no recorded UTM anchor. Coordinates are whatever\n";
                  sf << "GLIM produced at SLAM time (typically trajectory start = 0).\n";
                }
              }
              // Copy gnss_datum.json straight through for downstream use.
              const std::string datum_src = glim::GlobalConfig::get_config_path("gnss_datum");
              if (!datum_src.empty() && boost::filesystem::is_regular_file(datum_src)) {
                try {
                  boost::filesystem::copy_file(datum_src, out_dir + "/gnss_datum.json",
                    boost::filesystem::copy_option::overwrite_if_exists);
                } catch (const boost::filesystem::filesystem_error& e) {
                  logger->warn("[COLMAP] gnss_datum.json copy failed: {}", e.what());
                }
              }
            }
            if (!err.empty()) {
              ce_status = "Partial: " + err + " (pts=" + std::to_string(stats.points_written) +
                " cams=" + std::to_string(stats.cameras_written) + ")";
            } else {
              char buf[256];
              std::snprintf(buf, sizeof(buf), "Done: %zu points, %zu cameras, %zu images, %zu masks",
                stats.points_written, stats.cameras_written, stats.images_copied, stats.masks_copied);
              ce_status = buf;
            }
            logger->info("[COLMAP] {}", ce_status);
          } catch (const std::exception& e) {
            ce_status = std::string("Failed: ") + e.what();
            logger->error("[COLMAP] Exception: {}", e.what());
          }
          ce_running = false;
        }).detach();
      }
      if (!can_export) ImGui::EndDisabled();
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        if (!ce_region_placed) ImGui::SetTooltip("Place a region first.");
        else if (ce_output_dir.empty()) ImGui::SetTooltip("Choose an output directory.");
        else if (ce_voxelized_only && !have_voxelized) ImGui::SetTooltip("Voxelized HD not available -- run Voxelize HD first or uncheck the requirement.");
      }

      if (!ce_status.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("Status");
        ImGui::TextWrapped("%s", ce_status.c_str());
      }
      if (ce_last_points > 0) {
        ImGui::Separator();
        ImGui::TextDisabled("Last export");
        ImGui::Text("points: %zu", ce_last_points);
        ImGui::Text("cameras: %zu", ce_last_cameras);
        ImGui::Text("images copied: %zu", ce_last_images);
        ImGui::Text("masks copied: %zu", ce_last_masks);
      }
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
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Spatial grid cell size (XY) for grouping points.\nZ extent = voxel size x Height mult.");
        ImGui::SameLine(); ImGui::SetNextItemWidth(90);
        ImGui::SliderFloat("Height x##rfz", &rf_voxel_height_mult, 0.5f, 5.0f, "%.2fx");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Z-extent multiplier on the voxel. 1.0 = cubic.\n"
          "2.0 = voxel is 2x taller in Z without enlarging XY --\n"
          "catches larger vertical pass-to-pass misalignment (GPS Time mode)\n"
          "without blurring XY spatial resolution.");

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
          ImGui::Combo("Keep", &rf_gps_keep, "Dominant\0Newest\0Oldest\0Newest-Dominant\0Oldest-Dominant\0");
          if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Strategy for voxels covered by multiple passes (>1 time cluster):\n"
            "- Dominant: keep cluster with most points.\n"
            "- Newest:   keep latest temporal cluster.\n"
            "- Oldest:   keep earliest temporal cluster.\n"
            "- Newest-Dominant: prefer newest cluster; fall back to overall dominant\n"
            "  when the newest has < 20%% of voxel points (drops sparse stray newer\n"
            "  points that otherwise patch the dense older pass).\n"
            "- Oldest-Dominant: prefer oldest cluster; fall back to overall dominant\n"
            "  when the oldest has < 20%% of voxel points (lets a denser later pass\n"
            "  fill adjacent-street voxels without stray older points inside them).\n"
            "\n"
            "Voxels with only 1 time cluster are always kept (no holes).");
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
                  // Might be an accumulation neighbor not in chunk_frame_indices -- find from all_frames
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
        // Scalar visibility -- same pattern as range highlight but for any scalar field
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
            // Range mode supports a Z-extent multiplier (rf_voxel_height_mult); Dynamic mode uses cubic voxels.
            const float inv_voxel_z = (df_mode == 2)
              ? (1.0f / (active_voxel_size * std::max(0.5f, rf_voxel_height_mult)))
              : inv_voxel;
            auto vkey = [inv_voxel, inv_voxel_z](const Eigen::Vector3f& p) {
              return glim::voxel_key(
                static_cast<int>(std::floor(p.x() * inv_voxel)),
                static_cast<int>(std::floor(p.y() * inv_voxel)),
                static_cast<int>(std::floor(p.z() * inv_voxel_z)));
            };
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

                // PatchWork++ ground classification for this frame (cached -> scalar file -> recompute)
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
                  const uint64_t key = vkey(wp);
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
              const float time_gap = 5.0f;  // seconds -- points within this gap are same cluster
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
                  // Only one cluster -- keep all
                  for (const auto& e : entries) { kept_points.push_back(e.world_pos); kept_intensities.push_back(e.intensity); kept_ranges.push_back(e.range); preview_kept++; }
                  continue;
                }

                // Select cluster to keep
                int best_cluster = 0;
                const size_t total_in_voxel = entries.size();
                const size_t dominant_threshold = (total_in_voxel * 20 + 99) / 100;  // 20%, round up
                auto dominant = [&]() {
                  int b = 0;
                  for (int ci = 1; ci < static_cast<int>(clusters.size()); ci++) {
                    if (clusters[ci].size() > clusters[b].size()) b = ci;
                  }
                  return b;
                };
                if (rf_gps_keep == 0) { best_cluster = dominant(); }
                else if (rf_gps_keep == 1) { best_cluster = static_cast<int>(clusters.size()) - 1; }
                else if (rf_gps_keep == 2) { best_cluster = 0; }
                else if (rf_gps_keep == 3) {
                  best_cluster = static_cast<int>(clusters.size()) - 1;  // prefer newest
                  if (clusters[best_cluster].size() < dominant_threshold) best_cluster = dominant();
                }
                else if (rf_gps_keep == 4) {
                  best_cluster = 0;  // prefer oldest
                  if (clusters[best_cluster].size() < dominant_threshold) best_cluster = dominant();
                }

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
                  // Get scalar value -- use aux attribute from the submap frame
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
                    // DO NOT clear chunk_normal_z or chunk_ground_pw -- gap-fill needs them for protection
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
              const float inv_voxel_z = 1.0f / (rf_voxel_size * std::max(0.5f, rf_voxel_height_mult));
              auto vkey = [inv_voxel, inv_voxel_z](const Eigen::Vector3f& p) {
                return glim::voxel_key(
                  static_cast<int>(std::floor(p.x() * inv_voxel)),
                  static_cast<int>(std::floor(p.y() * inv_voxel)),
                  static_cast<int>(std::floor(p.z() * inv_voxel_z)));
              };
              struct VoxEntry { size_t idx; float range; float gps_time; };
              std::unordered_map<uint64_t, std::vector<VoxEntry>> voxels;
              for (size_t i = 0; i < chunk_pts.size(); i++) {
                if (!core_chunk.contains(chunk_pts[i])) continue;
                if (range_ground_only && (i >= chunk_is_ground_scalar.size() || !chunk_is_ground_scalar[i])) continue;
                const uint64_t key = vkey(chunk_pts[i]);
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
                // GPS Time criteria -- keep dominant temporal cluster per voxel
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
                  const size_t total_in_voxel = entries.size();
                  const size_t dom_thresh = (total_in_voxel * 20 + 99) / 100;
                  auto dominant = [&]() {
                    int b = 0;
                    for (int tci = 1; tci < static_cast<int>(clusters.size()); tci++) {
                      if (clusters[tci].size() > clusters[b].size()) b = tci;
                    }
                    return b;
                  };
                  if (rf_gps_keep == 0) { best = dominant(); }
                  else if (rf_gps_keep == 1) { best = static_cast<int>(clusters.size()) - 1; }
                  else if (rf_gps_keep == 2) { best = 0; }
                  else if (rf_gps_keep == 3) {
                    best = static_cast<int>(clusters.size()) - 1;
                    if (clusters[best].size() < dom_thresh) best = dominant();
                  }
                  else if (rf_gps_keep == 4) {
                    best = 0;
                    if (clusters[best].size() < dom_thresh) best = dominant();
                  }
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

              // Render (same pattern as regular preview -- with intensity buffer)
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
          if (ImGui::Button("Yes -- reuse aux_ground.bin")) { reuse_ground = true; launch = true; ImGui::CloseCurrentPopup(); }
          if (ImGui::IsItemHovered()) ImGui::SetTooltip("Uses ground from last Classify ground to scalar.\nFaster, recommended if ground is already tuned.");
          if (ImGui::Button("No -- recompute PatchWork++")) { reuse_ground = false; launch = true; ImGui::CloseCurrentPopup(); }
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

              // Core area (for writing removals -- no overlap)
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
                  if (chunk_ground[i]) { result.is_dynamic[i] = false; continue; }  // ground -> force static
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
                // Evaluate clusters -- keep only trail-shaped ones
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

            // Write filtered frames -- with final ground safety check
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
            std::unordered_map<std::string, std::unordered_set<int>> frame_removals;  // frame_dir -> set of point indices to remove

            // Step 4: Process each chunk
            const float inv_voxel = 1.0f / rf_voxel_size;
            const float inv_voxel_z = 1.0f / (rf_voxel_size * std::max(0.5f, rf_voxel_height_mult));
            auto vkey = [inv_voxel, inv_voxel_z](const Eigen::Vector3f& p) {
              return glim::voxel_key(
                static_cast<int>(std::floor(p.x() * inv_voxel)),
                static_cast<int>(std::floor(p.y() * inv_voxel)),
                static_cast<int>(std::floor(p.z() * inv_voxel_z)));
            };
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
                  const uint64_t key = vkey(cf.world_points[pi]);
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
                // GPS time criteria -- keep dominant temporal cluster per voxel
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
                  const size_t total_in_voxel = entries.size();
                  const size_t dom_thresh = (total_in_voxel * 20 + 99) / 100;
                  auto dominant = [&]() {
                    int b = 0;
                    for (int ci = 1; ci < static_cast<int>(clusters.size()); ci++) {
                      if (clusters[ci].size() > clusters[b].size()) b = ci;
                    }
                    return b;
                  };
                  if (rf_gps_keep == 0) { best = dominant(); }
                  else if (rf_gps_keep == 1) { best = static_cast<int>(clusters.size()) - 1; }
                  else if (rf_gps_keep == 2) { best = 0; }
                  else if (rf_gps_keep == 3) {
                    best = static_cast<int>(clusters.size()) - 1;
                    if (clusters[best].size() < dom_thresh) best = dominant();
                  }
                  else if (rf_gps_keep == 4) {
                    best = 0;
                    if (clusters[best].size() < dom_thresh) best = dominant();
                  }
                  std::unordered_set<int> keep_set(clusters[best].begin(), clusters[best].end());
                  for (int ei = 0; ei < static_cast<int>(entries.size()); ei++) {
                    if (!keep_set.count(ei)) { frame_removals[chunk_frames[entries[ei].cf_idx].dir].insert(chunk_frames[entries[ei].cf_idx].original_indices[entries[ei].pt_idx]); }
                  }
                }
              }
            }

            // Step 5: Apply removals -- rewrite each affected frame
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

      ImGui::Separator();
      if (ImGui::MenuItem("Export COLMAP...", nullptr, ce_show)) { ce_show = !ce_show; }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Open the COLMAP exporter (single-chunk, dev mode).\n"
        "Place a region cube and export points + cameras for 3DGS training.");

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
                    submaps[si].reset();  // null out -- all iterators check for null
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
      if (ImGui::MenuItem("Batch process...", nullptr, show_batch_window)) {
        show_batch_window = !show_batch_window;
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Queue multiple apply-to-HD filter/tool runs sequentially using current UI defaults.");
      ImGui::Separator();

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
            follow_speed_kmh = 30.0f;  // reset to default cruising speed on every mode entry
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
        if (ImGui::MenuItem("Axis gizmo", nullptr, show_axis_gizmo)) {
          show_axis_gizmo = !show_axis_gizmo;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Bottom-left world-axis indicator (X red, Y green, Z blue).\n"
          "Updates as the camera rotates -- shows the world frame's orientation\n"
          "relative to your current view.");
        ImGui::Separator();
        // Projection -- flip both the camera control AND the projection matrix.
        // FPV mode replaces projection_control with an FPSCameraControl (which
        // is a ProjectionControl but NOT a BasicProjectionControl), so we must
        // always install a fresh BasicProjectionControl here before configuring.
        // The canvas resize code will overwrite set_size() with the real viewport
        // on the next frame, so the initial size value is irrelevant.
        auto install_basic_proj = [](int mode, double ortho_w) {
          auto vw = guik::LightViewer::instance();
          auto proj = std::make_shared<guik::BasicProjectionControl>(Eigen::Vector2i(1920, 1080));
          proj->set_projection_mode(mode);
          if (mode == 1) proj->set_ortho_width(ortho_w);
          vw->set_projection_control(proj);
        };
        if (ImGui::MenuItem("Perspective")) {
          auto vw = guik::LightViewer::instance();
          vw->use_orbit_camera_control();
          install_basic_proj(0, 0.0);
          camera_mode_sel = 0;
        }
        if (ImGui::MenuItem("Orthographic")) {
          auto vw = guik::LightViewer::instance();
          vw->use_topdown_camera_control(200.0, 0.0);
          install_basic_proj(1, 200.0);   // ~200m footprint default
          camera_mode_sel = 0;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Camera Time Matcher", nullptr, show_time_matcher)) {
          show_time_matcher = !show_time_matcher;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Side-by-side time matcher for dumb-frames sources (Osmo 360 video etc).\n"
          "Pick a time-stamped source on the left, the dumb source on the right,\n"
          "scrub both to a matching moment, set an anchor, enter FPS, Apply.\n"
          "Use Two-anchor mode to solve the actual frame rate from the data.");
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
          ImGui::SliderFloat("Height (m)", &follow_height_offset_m, -5.0f, 50.0f, "%.1f");
          if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
              "Vertical shift above the trajectory spline.\n"
              "0 = car/operator view. 5-15m = drone-ish chase view.\n"
              "30m+ = top-down map view.\n"
              "Heading/pitch still track the path -- combine with mouse-drag\n"
              "pitch offset to tilt the view downward for a true drone angle.");
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

        // Livox intensity-0 filter
        if (!has_hd) ImGui::BeginDisabled();
        if (ImGui::MenuItem("Livox", nullptr, show_livox_tool)) {
          show_livox_tool = !show_livox_tool;
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          if (has_hd) ImGui::SetTooltip("Livox intensity-0 filter: delete / mark-as-2nd-return / interpolate.");
          else ImGui::SetTooltip("No HD frames available.");
        }
        if (!has_hd) ImGui::EndDisabled();

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
            // Auto-detect camera_type + intrinsics width/height from the first
            // image. A 2:1 aspect ratio (within 2%) is the equirectangular
            // signature (Osmo 360, Insta360, Ricoh Theta, etc.) -> Spherical.
            // Anything else stays Pinhole (user can still override in the UI).
            if (!source.frames.empty()) {
              cv::Mat sample = cv::imread(source.frames[0].filepath, cv::IMREAD_REDUCED_COLOR_4);
              if (!sample.empty()) {
                const int w = sample.cols * 4, h = sample.rows * 4;  // undo the reduction
                source.intrinsics.width  = w;
                source.intrinsics.height = h;
                const double aspect = static_cast<double>(w) / std::max(1, h);
                if (std::abs(aspect - 2.0) < 0.04) {
                  source.camera_type = CameraType::Spherical;
                  logger->info("[Colorize] Auto-detected Spherical (equirect) from {}x{} first image", w, h);
                } else {
                  logger->info("[Colorize] Assuming Pinhole from {}x{} first image (aspect {:.3f})", w, h, aspect);
                }
              }
            }
            int with_gps = 0, with_time = 0;
            for (const auto& f : source.frames) {
              if (f.lat != 0.0 || f.lon != 0.0) with_gps++;
              if (f.timestamp > 0.0) with_time++;
            }
            // Seed ColorizeParams with camera-type-aware defaults so Spherical
            // sources open with loose time-slice + Weighted Top-1 out of the box.
            source.params = default_colorize_params_for(source.camera_type);
            logger->info("[Colorize] Loaded {} images ({}) ({} with GPS, {} with timestamp)",
                         source.frames.size(), camera_type_label(source.camera_type), with_gps, with_time);
            image_sources.push_back(std::move(source));
            colorize_source_idx = static_cast<int>(image_sources.size()) - 1;
            // Build the alt (GPS-derived) trajectory eagerly so 'Coords > own
            // path' is selectable with no extra click. Skipped silently when
            // the source has no EXIF GPS/timestamps.
            if (with_gps > 0 && with_time > 0) {
              build_camera_trajectory(image_sources.back(), gnss_utm_zone,
                                       gnss_utm_easting_origin, gnss_utm_northing_origin,
                                       gnss_datum_alt);
            }
            // Save colorize config to dump (shared helper; see image_source_to_json).
            if (!loaded_map_path.empty()) {
              nlohmann::json cfg;
              cfg["sources"] = nlohmann::json::array();
              for (const auto& s : image_sources) cfg["sources"].push_back(image_source_to_json(s));
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
      if (ImGui::MenuItem("Colorize", nullptr, show_colorize_window)) {
        show_colorize_window = !show_colorize_window;
      }
      if (ImGui::MenuItem("Alignment check", nullptr, align_show)) {
        align_show = !align_show;
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overlay image + projected LiDAR points to assess calibration.");
      if (ImGui::MenuItem("Auto-calibrate", nullptr, ac_show)) {
        ac_show = !ac_show;
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("LightGlue-assisted extrinsic + intrinsics refinement.");
      ImGui::Separator();
      // Utils submenu lives at the bottom of Colorize so anything that's a
      // follow-on of a Colorize calibration (e.g. virtual cameras keyed off
      // the Located extrinsic) sits near its upstream step.
      if (ImGui::BeginMenu("Utils")) {
        if (ImGui::MenuItem("Virtual Cameras", nullptr, show_virtual_cameras_window)) {
          show_virtual_cameras_window = !show_virtual_cameras_window;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
          "Render LiDAR-synthesized intensity images at real-camera poses (per-RGB mode)\n"
          "or along the trajectory (waypoints mode). Drop into Metashape as locked\n"
          "anchors for SFM/BA -- automates the hand-placed marker workflow for km of\n"
          "trajectory.");
        ImGui::EndMenu();
      }
      ImGui::EndMenu();
    }

    // ----- About menu: acknowledgements, grouped by module + third-party libs -----
    if (ImGui::BeginMenu("About")) {
      // Helper lambda: a pair of leaf entries (Author, License). The MenuItem text
      // IS the info; hovering shows expanded context.
      auto entry = [](const char* author_text, const char* author_tooltip,
                      const char* license_text, const char* license_tooltip) {
        if (ImGui::BeginMenu("Author")) {
          ImGui::TextUnformatted(author_text);
          if (author_tooltip && *author_tooltip) {
            ImGui::Separator();
            ImGui::TextDisabled("%s", author_tooltip);
          }
          ImGui::EndMenu();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", author_text);
        if (ImGui::BeginMenu("License")) {
          ImGui::TextUnformatted(license_text);
          if (license_tooltip && *license_tooltip) {
            ImGui::Separator();
            ImGui::TextDisabled("%s", license_tooltip);
          }
          ImGui::EndMenu();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", license_text);
      };

      // --- GLIM core (Kenji Koide) ---
      if (ImGui::BeginMenu("GLIM (core SLAM)")) {
        entry(
          "Kenji Koide (AIST)",
          "Creator of GLIM: versatile 3D LiDAR-IMU mapping framework with interactive map editing.",
          "MIT",
          "https://github.com/koide3/glim");
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("gtsam_points")) {
        entry("Kenji Koide (AIST)", "Point-cloud-based GTSAM factors, CPU + GPU variants.",
              "BSD-2-Clause", "https://github.com/koide3/gtsam_points");
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Iridescence (viewer)")) {
        entry("Kenji Koide", "ImGui + OpenGL 3D viewer backing GLIM's UI.",
              "MIT", "https://github.com/koide3/iridescence");
        ImGui::EndMenu();
      }

      ImGui::Separator();

      // --- Mobile Mapper (umbrella claim; anything not explicitly attributed below is ours) ---
      if (ImGui::BeginMenu("Mobile Mapper (this fork)")) {
        entry("Pablo Vidaurre Sanz + Claude (Anthropic)",
              "Post-processing pipeline built on top of GLIM: colorize, auto-calibrate, data filters,\n"
              "voxelize HD, COLMAP export, camera modes, GNSS / coordinates tooling, Livox filters,\n"
              "alignment check, and all associated UI.\n\n"
              "Rule of thumb: anything NOT listed elsewhere in About is part of this fork.",
              "Mobile Mapper EULA v1 (source-available, non-commercial modification only)",
              "Software: may be USED for any purpose (including commercial) to produce output data.\n"
              "Modification is permitted for non-commercial purposes (tinkering, academic, evaluation).\n"
              "Modification FOR COMMERCIAL USE -- direct (reselling a modified version) or indirect\n"
              "(derivative software that is sold/licensed/commercialized) -- requires written permission.\n"
              "Output data (point clouds, 3DGS exports, COLMAP datasets): UNRESTRICTED -- yours.\n"
              "GLIM core files retain their original MIT license.\n"
              "See LICENSE.mobile-mapper at the repo root for full text.");
        ImGui::EndMenu();
      }

      ImGui::Separator();

      // --- Features authored by others (called out individually) ---
      if (ImGui::BeginMenu("Auto-calibrate -> LightGlue")) {
        entry("Philipp Lindenberger, Paul-Edouard Sarlin, Marc Pollefeys",
              "LightGlue: Local Feature Matching at Light Speed. ICCV 2023.\n"
              "Used for 2D feature matching between the real camera image and the rendered LiDAR\n"
              "intensity image during auto-calibration. Feature engineering + pipeline integration\n"
              "by Mobile Mapper; LightGlue itself is attributed here.",
              "Apache-2.0",
              "https://github.com/cvg/LightGlue\n"
              "Note: default SuperPoint weights are CC-BY-NC-SA-4.0 (non-commercial).\n"
              "For commercial output, swap to ALIKED in scripts/lightglue_match.py.");
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Ground segmentation -> PatchWork++")) {
        entry("Hyungtae Lim et al. (URL team, KAIST)",
              "Patchwork++: Fast and Robust Ground Segmentation Solving Partial Under-Segmentation\n"
              "Using 3D Point Cloud. IROS 2022.\n"
              "Used by the Dynamic filter's ground protection step. Integration + extensions\n"
              "(frame accumulation, Z-column refinement) by Mobile Mapper.",
              "BSD-2-Clause",
              "https://github.com/url-kaist/patchwork-plusplus");
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Dynamic filter -> MapCleaner (inspiration)")) {
        entry("H. Fu et al. -- clean-room reimpl by Mobile Mapper",
              "MapCleaner: Efficiently Removing Moving Objects from Point Cloud Maps in Autonomous\n"
              "Driving Scenarios. IEEE RA-L 2024. Conceptual inspiration only; this implementation\n"
              "is independent with differences in voting, clustering, and chunk handling.",
              "Algorithm: not 1:1. Our code: MIT.",
              "Not a fork of original code.");
        ImGui::EndMenu();
      }

      ImGui::Separator();

      // --- Third-party libs ---
      if (ImGui::BeginMenu("Third-party libraries")) {
        if (ImGui::BeginMenu("GTSAM")) {
          entry("Frank Dellaert et al. (Georgia Tech)",
                "Factor graph optimization backend.",
                "BSD-3-Clause",
                "https://github.com/borglab/gtsam");
          ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Eigen")) {
          entry("Benoit Jacob, Gael Guennebaud, et al.",
                "C++ linear algebra template library.",
                "MPL-2.0", "https://eigen.tuxfamily.org/");
          ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("OpenCV")) {
          entry("OpenCV Team", "Computer vision library.",
                "Apache-2.0", "https://opencv.org/");
          ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Boost")) {
          entry("Boost community", "Filesystem, program_options, etc.",
                "Boost Software License 1.0", "https://www.boost.org/");
          ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Dear ImGui")) {
          entry("Omar Cornut et al.", "Immediate-mode GUI.",
                "MIT", "https://github.com/ocornut/imgui");
          ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("LightGlue (optional, auto-calib)")) {
          entry("Philipp Lindenberger, Paul-Edouard Sarlin, Marc Pollefeys",
                "LightGlue: Local Feature Matching at Light Speed. ICCV 2023.",
                "Apache-2.0", "https://github.com/cvg/LightGlue");
          ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("libexif")) {
          entry("libexif contributors", "EXIF metadata parsing.",
                "LGPL-2.1", "https://libexif.github.io/");
          ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("nlohmann/json")) {
          entry("Niels Lohmann", "JSON for Modern C++.",
                "MIT", "https://github.com/nlohmann/json");
          ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("portable-file-dialogs")) {
          entry("Sam Hocevar", "Native file dialogs.",
                "WTFPL / BSL-1.0", "https://github.com/samhocevar/portable-file-dialogs");
          ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("FAST-LIVO2 (conceptual influence)")) {
          entry("Chunran Zheng, Wei Xu, Yunfan Ren, Fu Zhang (HKU-MARS)",
                "FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry. IEEE TRO 2024.\n"
                "Not a dependency -- inspired the Weighted view selector, photometric exposure, "
                "and depth-buffer occlusion approach.",
                "N/A -- inspiration only, no code reuse",
                "https://github.com/hku-mars/FAST-LIVO2");
          ImGui::EndMenu();
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
                         "Cross-zone alignment is not supported -- coordinates may be incorrect.",
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
          // First map with datum -- store as reference
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
          image_source_apply_json(source, sj);
          // If the saved config didn't carry camera_type (pre-Spherical sessions
          // or manually-edited json), auto-detect from first image's aspect.
          if (!sj.contains("camera_type") && !source.frames.empty()) {
            cv::Mat sample = cv::imread(source.frames[0].filepath, cv::IMREAD_REDUCED_COLOR_4);
            if (!sample.empty()) {
              const double aspect = static_cast<double>(sample.cols) / std::max(1, sample.rows);
              if (std::abs(aspect - 2.0) < 0.04) {
                source.camera_type = CameraType::Spherical;
                source.params = default_colorize_params_for(CameraType::Spherical);
                logger->info("[Colorize] Auto-detected Spherical for {} (config had no camera_type)", source.path);
              }
            }
          }
          // Re-apply Time-Matcher back-fill if the source had been anchored in a
          // previous session (dumb-frames source lost its timestamps on reload).
          if (source.tm_anchor1_idx >= 0) {
            double interval = 0.0;
            if (source.tm_anchor2_idx >= 0 && source.tm_anchor2_idx != source.tm_anchor1_idx) {
              interval = (source.tm_anchor2_time - source.tm_anchor1_time) /
                         static_cast<double>(source.tm_anchor2_idx - source.tm_anchor1_idx);
            } else if (source.tm_fps > 0.01f) {
              interval = 1.0 / static_cast<double>(source.tm_fps);
            }
            if (interval > 0.0) {
              for (size_t i = 0; i < source.frames.size(); i++) {
                const double dt = (static_cast<int>(i) - source.tm_anchor1_idx) * interval;
                source.frames[i].timestamp = source.tm_anchor1_time + dt;
              }
              // Back-filled timestamps are already in LiDAR-time base; the
              // persisted source.time_shift (if any) was typically 0 from the
              // Apply step but we don't touch it here -- user may have added a
              // deliberate nudge after the fact which we should preserve.
              logger->info("[Colorize] Re-applied Time Matcher back-fill: anchor1=({},{:.3f}) interval={:.5f}",
                           source.tm_anchor1_idx, source.tm_anchor1_time, interval);
            }
          }
          // Load the mask for the first source into the runtime cache so the
          // active-source selection on startup has the right mask resident.
          if (!source.mask_path.empty() && boost::filesystem::exists(source.mask_path)) {
            colorize_mask = cv::imread(source.mask_path, cv::IMREAD_UNCHANGED);
            if (!colorize_mask.empty()) logger->info("[Colorize] Auto-loaded mask from {}", source.mask_path);
          }
          logger->info("[Colorize] Restored: {} images ({}), time_shift={:.3f}, lever=[{:.3f},{:.3f},{:.3f}]",
            source.frames.size(), camera_type_label(source.camera_type),
            source.time_shift, source.lever_arm.x(), source.lever_arm.y(), source.lever_arm.z());
          image_sources.push_back(std::move(source));
          // Rebuild the alt (GPS-derived) trajectory after reload -- not
          // serialised, so it's always rebuilt from EXIF on the way back in.
          build_camera_trajectory(image_sources.back(), gnss_utm_zone,
                                   gnss_utm_easting_origin, gnss_utm_northing_origin,
                                   gnss_datum_alt);
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
    // No session tracking (single map) -- export all
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
  // HD export path -- load frames from disk, transform, write
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
  // "property double" in the PLY header -- float32 loses ~128 s precision on GPS epoch values.
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
    // populated -- these would corrupt MIN-blend colorisation by pulling the range to zero).
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
    // Written as "property double <name>" -- full 64-bit precision in the PLY file.
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

      // Phase 1: inverse UTM -> lat/lon for all points (zone-independent, cached)
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
  // Save -- either single file or per-tile split
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

      // Double-precision x, y, z -- may be re-projected for JGD2011 per-tile zones
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
    // No datum -- local coordinates
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

  // Build voxel grid: map voxel key -> list of point indices
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
      // No safe-range anchor -- apply secondary (far) delta from min range
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
