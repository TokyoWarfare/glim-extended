#include <glim/util/auto_calibrate.hpp>
#include <glim/util/map_cleaner.hpp>  // classify_ground_patchwork

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace glim {

CalibrationContext build_calibration_context(
  const std::vector<SubMap::ConstPtr>& submaps,
  const std::vector<TimedPose>& timed_traj,
  double anchor_stamp,
  const Eigen::Vector3f& anchor_pos,
  const Eigen::Vector3f& anchor_forward,
  const CalibContextOptions& opts,
  std::function<gtsam_points::PointCloudCPU::Ptr(int)> load_hd_for_submap,
  std::function<std::vector<Eigen::Vector3f>(int)> load_hd_aux_rgb_for_submap) {

  CalibrationContext ctx;
  ctx.anchor_stamp = anchor_stamp;
  ctx.anchor_forward = anchor_forward;

  // Flatten submaps into a single chronological list pointing to (submap_idx,
  // frame_idx). The window logic below runs over this flat list so the N-frames
  // and time-window options both resolve deterministically regardless of how
  // frames are distributed across submaps.
  struct FRef { int si; int fi; double stamp; };
  std::vector<FRef> all;
  for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
    if (!submaps[si]) continue;
    for (int fi = 0; fi < static_cast<int>(submaps[si]->frames.size()); fi++) {
      all.push_back({si, fi, submaps[si]->frames[fi]->stamp});
    }
  }
  if (all.empty()) return ctx;
  std::sort(all.begin(), all.end(), [](const FRef& a, const FRef& b){ return a.stamp < b.stamp; });

  // Anchor index inside the flat list (nearest by |stamp - anchor_stamp|).
  int anchor_global = 0;
  {
    double bd = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(all.size()); i++) {
      const double dd = std::abs(all[i].stamp - anchor_stamp);
      if (dd < bd) { bd = dd; anchor_global = i; }
    }
  }

  // Frame range: either a time window or a fixed number of frames around anchor.
  int idx_start = 0;
  int idx_end = static_cast<int>(all.size()) - 1;
  if (opts.use_time_window) {
    for (int i = anchor_global; i >= 0; i--) {
      if (all[i].stamp < anchor_stamp - opts.time_before_s) { idx_start = i + 1; break; }
    }
    for (int i = anchor_global; i < static_cast<int>(all.size()); i++) {
      if (all[i].stamp > anchor_stamp + opts.time_after_s) { idx_end = i - 1; break; }
    }
  } else {
    idx_start = std::max(0, anchor_global - opts.n_frames_before);
    idx_end   = std::min(static_cast<int>(all.size()) - 1, anchor_global + opts.n_frames_after);
  }

  // Collect the touched submaps. load_hd_for_submap loads an entire submap, so
  // we accept slight redundancy at the window edges -- any submap whose frame
  // falls in [idx_start..idx_end] is loaded in full.
  std::set<int> sm_needed;
  for (int i = idx_start; i <= idx_end; i++) sm_needed.insert(all[i].si);

  const float cos_thresh = std::cos(opts.directional_threshold_deg * static_cast<float>(M_PI) / 180.0f);

  for (int si : sm_needed) {
    auto hd = load_hd_for_submap(si);
    if (!hd || hd->size() == 0) continue;
    const Eigen::Isometry3d T_wo = submaps[si]->T_world_origin;
    const Eigen::Matrix3d R_wo = T_wo.rotation();
    const bool have_normals = (hd->normals != nullptr);

    if (opts.directional_filter && !timed_traj.empty()) {
      // Submap-level forward at its MIDDLE frame's time. Matches the convention
      // used to pick anchor_forward -- interpolating the trajectory at the
      // frame's stamp keeps the two directions directly comparable.
      const auto& mid_frame = submaps[si]->frames[submaps[si]->frames.size() / 2];
      const Eigen::Isometry3d T_world_lidar_mid = Colorizer::interpolate_pose(timed_traj, mid_frame->stamp);
      const Eigen::Vector3f sm_forward = T_world_lidar_mid.rotation().col(0).cast<float>().normalized();
      if (sm_forward.dot(anchor_forward) < cos_thresh) continue;
    }

    // Per-frame ground flag (preferred over PatchWork). load_hd_for_submap
    // attaches "aux_ground" when ANY HD frame in the submap had its
    // aux_ground.bin saved (Data Cleaner > Dynamic > Save ground to HD).
    // Stored as float 0/1 per cloud point, aligned to hd->points order.
    const float* gnd_attr = nullptr;
    {
      const auto it = hd->aux_attributes.find("aux_ground");
      if (it != hd->aux_attributes.end() && it->second.first == sizeof(float)) {
        gnd_attr = static_cast<const float*>(it->second.second);
      }
    }
    // Native lidar RGB (Ouster colour sensor) or Colorize > Apply output.
    // Loaded from disk by an optional caller-supplied loader -- one entry
    // per HD point in the SAME order load_hd_for_submap returned the cloud.
    // Empty => no RGB for this submap; ctx.colors_rgb stays empty for these
    // points and the rasterizer falls back to intensity.
    std::vector<Eigen::Vector3f> sm_rgb;
    if (load_hd_aux_rgb_for_submap) {
      sm_rgb = load_hd_aux_rgb_for_submap(si);
      if (!sm_rgb.empty() && sm_rgb.size() != hd->size()) {
        std::cerr << "[VC] aux_rgb size mismatch for submap " << si
                  << " (rgb=" << sm_rgb.size() << " pts=" << hd->size()
                  << ") -- skipping RGB for this submap" << std::endl;
        sm_rgb.clear();
      }
    }

    for (size_t i = 0; i < hd->size(); i++) {
      const Eigen::Vector3f wp = (T_wo * Eigen::Vector3d(hd->points[i].head<3>().cast<double>())).cast<float>();
      const float dsq = (wp - anchor_pos).squaredNorm();
      if (dsq < opts.min_range * opts.min_range || dsq > opts.max_range * opts.max_range) continue;
      ctx.world_points.push_back(wp);
      ctx.intensities.push_back(hd->intensities ? static_cast<float>(hd->intensities[i]) : 0.0f);
      if (have_normals) {
        ctx.world_normals.push_back((R_wo * Eigen::Vector3d(hd->normals[i].head<3>())).normalized().cast<float>());
      }
      // Push ground flag *iff* at least one submap so far has carried the
      // attribute. Mixed sessions (some submaps with aux_ground, some
      // without) get treated as "no per-frame data" via the post-loop guard.
      if (gnd_attr != nullptr) {
        ctx.is_ground.push_back(gnd_attr[i] >= 0.5f ? 1 : 0);
      } else if (!ctx.is_ground.empty()) {
        // This submap missing aux_ground while a previous one had it ->
        // length mismatch incoming. Mark with 0 (safest fallback: include
        // the point if the consumer was hoping for ground filtering).
        ctx.is_ground.push_back(0);
      }
      // Same parallel-accumulation pattern for native RGB. Pushed only when
      // a submap has provided RGB via the caller's loader. Sentinel
      // (-1,0,0) marks points from a submap whose loader returned empty so
      // the rasterizer (use_rgb sentinel check) skips those splats cleanly.
      if (!sm_rgb.empty()) {
        ctx.colors_rgb.push_back(sm_rgb[i]);
      } else if (!ctx.colors_rgb.empty()) {
        ctx.colors_rgb.push_back(Eigen::Vector3f(-1.0f, 0.0f, 0.0f));
      }
    }
  }

  // Per-frame ground flags from aux_ground.bin take precedence -- they're
  // the original PatchWork verdict in the correct sensor-local frame, so
  // they classify each point in the context the way PatchWork actually
  // intends. The fallback below (PatchWork on the accumulated cloud) is
  // a last-resort estimate that's known to be erratic on aggregated data
  // because the polar discretization assumes a single sensor centre.
  //
  // If the per-point array we just accumulated covers the whole context
  // (one entry per kept point) and at least one entry is ground, we're
  // done -- skip the recomputation entirely.
  bool have_per_frame_ground = (ctx.is_ground.size() == ctx.world_points.size())
                                 && !ctx.is_ground.empty();
  if (have_per_frame_ground) {
    int n_g = 0; for (auto b : ctx.is_ground) if (b) n_g++;
    std::cerr << "[VC] Using per-frame aux_ground.bin: "
              << n_g << " / " << ctx.is_ground.size()
              << " points classified as ground" << std::endl;
  } else {
    // Length mismatch (e.g. some submaps had aux_ground.bin and some didn't)
    // would corrupt the rasterizer's per-point lookup -- safer to wipe and
    // either fall back to PatchWork or leave it empty.
    ctx.is_ground.clear();
  }

  // Optional ground classification fallback. Runs PatchWork++ on the
  // accumulated points; output is per-point uint8 (1 = ground, 0 = otherwise).
  // Skipped when not requested, when per-frame data already filled is_ground,
  // or when the context is empty.
  //
  // PatchWork wants points in *sensor-local* frame -- it discretizes the
  // ground plane at z = 0 and uses sensor_height as the assumed sensor
  // mounting height. Passing world points directly lands the ground at
  // arbitrary world-Z (wherever the road happens to be), PatchWork finds
  // no plane, and every point gets non-ground -> entire context filtered
  // out at the rasterizer -> blank render. Translate so the anchor sits at
  // (0, 0, sensor_height) before classifying, then map results back 1:1.
  if (opts.classify_ground && !have_per_frame_ground && !ctx.world_points.empty()) {
    std::vector<Eigen::Vector3f> local_pts;
    local_pts.reserve(ctx.world_points.size());
    const Eigen::Vector3f shift(-anchor_pos.x(), -anchor_pos.y(),
                                 -anchor_pos.z() + opts.patchwork_sensor_height);
    for (const auto& p : ctx.world_points) local_pts.push_back(p + shift);

    auto pw = glim::MapCleanerFilter::classify_ground_patchwork(
      local_pts,
      static_cast<int>(local_pts.size()),
      opts.patchwork_sensor_height,
      ctx.intensities);
    ctx.is_ground.assign(pw.begin(), pw.end());

    // Diagnostic: counts feed back into the VC status / log so the user can
    // see whether classification actually returned ground (a 0/N count is
    // the symptom of the frame-of-reference bug above).
    int n_g = 0; for (auto b : pw) if (b) n_g++;
    std::cerr << "[VC] PatchWork ground classification: "
              << n_g << " / " << pw.size()
              << " points classified as ground (sensor_h="
              << opts.patchwork_sensor_height << ")" << std::endl;
  }

  return ctx;
}

MatchQualityStats compute_match_quality(
  const std::vector<float>& scores,
  float high_thresh,
  float mid_thresh) {
  MatchQualityStats out;
  out.high_thresh = high_thresh;
  out.mid_thresh  = mid_thresh;
  out.total = static_cast<int>(scores.size());
  for (float s : scores) {
    if (s >= high_thresh) out.high++;
    else if (s >= mid_thresh) out.mid++;
    else out.low++;
  }
  return out;
}

void compute_intensity_percentiles(
  const std::vector<float>& intensities,
  float& imin, float& ibulk, float& imax) {

  imin = 0.0f; ibulk = 230.0f; imax = 250.0f;
  if (intensities.empty()) return;
  std::vector<float> sorted(intensities);
  std::sort(sorted.begin(), sorted.end());
  const size_t n = sorted.size();
  imin  = sorted[static_cast<size_t>(0.02 * n)];
  ibulk = sorted[std::min(n - 1, static_cast<size_t>(0.95 * n))];
  // 99th percentile clip -- this was live when the 120-match baseline was
  // observed, so restoring it here to keep that number the reference point.
  // Outliers above p99 clamp to 255 via std::clamp in the renderer's final
  // mapping; the lost dynamic range at the very top end is the trade-off.
  imax  = sorted[std::min(n - 1, static_cast<size_t>(0.99 * n))];
  if (ibulk <= imin)  ibulk = imin + 1.0f;
  if (imax  <= ibulk) imax  = ibulk + 1.0f;
  if (imax  <= imin)  imax  = imin + 1.0f;
}

// -----------------------------------------------------------------------------
// Intensity image renderer
// -----------------------------------------------------------------------------
RenderedIntensity render_intensity_image(
  const CalibrationContext& ctx,
  const Eigen::Isometry3d& T_world_cam,
  const PinholeIntrinsics& intrinsics,
  int width, int height,
  const IntensityRenderOptions& opts) {

  RenderedIntensity out;
  out.image = cv::Mat::zeros(height, width, CV_8UC1);
  out.depth = cv::Mat::zeros(height, width, CV_32F);
  out.pixel_to_point.assign(static_cast<size_t>(width) * height, -1);

  const Eigen::Isometry3d T_cam_world = T_world_cam.inverse();
  const Eigen::Matrix3d R = T_cam_world.rotation();
  const Eigen::Vector3d t = T_cam_world.translation();
  const double fx = intrinsics.fx, fy = intrinsics.fy;
  const double cx = intrinsics.cx, cy = intrinsics.cy;
  const bool has_distortion = (intrinsics.k1 != 0 || intrinsics.k2 != 0 || intrinsics.p1 != 0 || intrinsics.p2 != 0);

  // Intensity remap. Non-linear mode: two-piece (bottom 95% gamma-lifted, top 5%
  // clamped to 250-255 so retroreflective markings pop). Linear mode: straight
  // stretch from (2nd..99th percentile) to 0-255 -- lets the user A/B how much
  // the non-linear shaping helps LightGlue matching.
  //
  // When `intensity_locked` is on we skip the per-frame percentile compute and
  // use the values the UI captured from a reference frame -- keeps contrast
  // stable across a batch of renders (same synthetic "exposure" everywhere).
  float imin = 0.0f, ibulk = 255.0f, imax = 255.0f;
  if (opts.intensity_locked) {
    imin  = opts.intensity_locked_imin;
    ibulk = opts.intensity_locked_ibulk;
    imax  = opts.intensity_locked_imax;
    if (ibulk <= imin)  ibulk = imin + 1.0f;
    if (imax  <= ibulk) imax  = ibulk + 1.0f;
    if (imax  <= imin)  imax  = imin + 1.0f;
  } else if (!ctx.intensities.empty()) {
    compute_intensity_percentiles(ctx.intensities, imin, ibulk, imax);
  }
  const float inv_bulk_range = 1.0f / (ibulk - imin);
  const float inv_top_range  = 1.0f / (imax  - ibulk);
  const float inv_lin_range  = 1.0f / (imax  - imin);
  const float gamma = 0.5f;

  // Splat-size dispatch. Three modes:
  //  - Formula:    1/depth curve clamped to [min, max].
  //  - Fixed:      same radius at every depth.
  //  - LinearRamp: user-defined knots. Between two consecutive knots the size
  //                linearly interpolates; beyond the last knot stays flat.
  // The LinearRamp path pre-sorts a local copy so the hot loop can do a simple
  // scan without assuming the caller's vector is ordered.
  const double near_depth = opts.near_depth_m;
  const int min_splat = std::max(0, opts.min_splat_px);
  const int max_splat = std::max(min_splat, opts.max_splat_px);
  std::vector<IntensityRenderOptions::SplatRange> ranges;
  if (opts.splat_mode == IntensityRenderOptions::SplatMode::LinearRamp) {
    ranges = opts.splat_ranges;
    std::sort(ranges.begin(), ranges.end(),
              [](const auto& a, const auto& b) { return a.start_depth_m < b.start_depth_m; });
  }
  auto splat_for = [&](double depth) -> int {
    switch (opts.splat_mode) {
      case IntensityRenderOptions::SplatMode::Fixed:
        return std::max(0, opts.fixed_splat_px);
      case IntensityRenderOptions::SplatMode::LinearRamp: {
        if (ranges.empty()) return std::max(0, opts.fixed_splat_px);
        // Below / at first knot -> first knot's size (flat clamp).
        if (depth <= ranges.front().start_depth_m) return std::max(0, ranges.front().splat_px);
        // At or past last knot -> last knot's size (flat tail).
        if (depth >= ranges.back().start_depth_m)  return std::max(0, ranges.back().splat_px);
        // In between: find the bracketing pair and lerp.
        for (size_t k = 0; k + 1 < ranges.size(); k++) {
          const auto& a = ranges[k];
          const auto& b = ranges[k + 1];
          if (depth >= a.start_depth_m && depth <= b.start_depth_m) {
            const double span = b.start_depth_m - a.start_depth_m;
            const double t = span > 1e-9 ? (depth - a.start_depth_m) / span : 0.0;
            const double lerped = static_cast<double>(a.splat_px) +
                                   t * (static_cast<double>(b.splat_px) - static_cast<double>(a.splat_px));
            return std::max(0, static_cast<int>(std::round(lerped)));
          }
        }
        return std::max(0, ranges.back().splat_px);  // unreachable but keeps compiler happy
      }
      case IntensityRenderOptions::SplatMode::Formula:
      default:
        return std::clamp(static_cast<int>(std::round(3.0 * near_depth / depth)), min_splat, max_splat);
    }
  };

  // Ground-only filter -- when on (and is_ground was populated by the
  // context builder), points classified as non-ground are skipped before
  // projection. Falls through cleanly when is_ground is empty (no filter).
  const bool ground_filter_active = opts.ground_only && !ctx.is_ground.empty()
                                     && ctx.is_ground.size() == ctx.world_points.size();

  // Whether per-point RGB drives the splat colour. Requires the source mode
  // to ask for it AND colors_rgb to be populated by the caller. Falls back
  // to intensity when missing -- safe even if the user toggles a mode that
  // requires data the context doesn't carry.
  const bool use_rgb = (opts.source == RenderSource::BootstrapColorizedSplat ||
                        opts.source == RenderSource::BootstrapColorizedDepth ||
                        opts.source == RenderSource::NativeRGB) &&
                       !ctx.colors_rgb.empty() &&
                       ctx.colors_rgb.size() == ctx.world_points.size();
  if (use_rgb) {
    out.image = cv::Mat::zeros(height, width, CV_8UC3);
  }

  for (size_t pi = 0; pi < ctx.world_points.size(); pi++) {
    if (ground_filter_active && !ctx.is_ground[pi]) continue;
    // Skip points whose RGB sampling failed (off-image, masked, behind cam).
    // Sentinel: x() < 0. Avoids leaking black splats into uncovered areas.
    if (use_rgb && ctx.colors_rgb[pi].x() < 0.0f) continue;
    const Eigen::Vector3d p_cam = R * ctx.world_points[pi].cast<double>() + t;
    const double depth = p_cam.x();
    if (depth <= 0.2) continue;

    double xn = -p_cam.y() / depth;
    double yn = -p_cam.z() / depth;

    if (has_distortion) {
      const double r2 = xn * xn + yn * yn, r4 = r2 * r2, r6 = r4 * r2;
      const double radial = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r4 + intrinsics.k3 * r6;
      const double xd = xn * radial + 2.0 * intrinsics.p1 * xn * yn + intrinsics.p2 * (r2 + 2.0 * xn * xn);
      const double yd = yn * radial + intrinsics.p1 * (r2 + 2.0 * yn * yn) + 2.0 * intrinsics.p2 * xn * yn;
      xn = xd; yn = yd;
    }

    const double u = fx * xn + cx;
    const double v = fy * yn + cy;
    const int iu = static_cast<int>(std::round(u));
    const int iv = static_cast<int>(std::round(v));
    if (iu < 0 || iu >= width || iv < 0 || iv >= height) continue;

    // Pick the splat colour. RGB path stores BGR (OpenCV convention) so the
    // resulting image is directly viewable / writable as a colour image
    // without channel swaps. Intensity path keeps the existing scalar logic.
    uint8_t val_i = 0;
    cv::Vec3b val_bgr(0, 0, 0);
    if (use_rgb) {
      const auto& c = ctx.colors_rgb[pi];
      val_bgr = cv::Vec3b(
        static_cast<uint8_t>(std::clamp(c.z(), 0.0f, 255.0f)),  // B
        static_cast<uint8_t>(std::clamp(c.y(), 0.0f, 255.0f)),  // G
        static_cast<uint8_t>(std::clamp(c.x(), 0.0f, 255.0f))); // R
    } else {
      const float iv_raw = ctx.intensities[pi];
      if (opts.non_linear_intensity) {
        if (iv_raw >= ibulk) {
          const float ti = std::clamp((iv_raw - ibulk) * inv_top_range, 0.0f, 1.0f);
          val_i = static_cast<uint8_t>(250.0f + ti * 5.0f);
        } else {
          const float lin = std::clamp((iv_raw - imin) * inv_bulk_range, 0.0f, 1.0f);
          val_i = static_cast<uint8_t>(std::pow(lin, gamma) * 250.0f);
        }
      } else {
        const float lin = std::clamp((iv_raw - imin) * inv_lin_range, 0.0f, 1.0f);
        val_i = static_cast<uint8_t>(lin * 255.0f);
      }
    }

    const int splat = splat_for(depth);
    const int r2_max = splat * splat;  // inclusive, disk boundary
    for (int dy = -splat; dy <= splat; dy++) {
      for (int dx = -splat; dx <= splat; dx++) {
        if (opts.round_splats && dx * dx + dy * dy > r2_max) continue;
        const int x2 = iu + dx, y2 = iv + dy;
        if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height) continue;
        float& d_prev = out.depth.at<float>(y2, x2);
        if (d_prev == 0.0f || depth < d_prev) {
          d_prev = static_cast<float>(depth);
          if (use_rgb) out.image.at<cv::Vec3b>(y2, x2) = val_bgr;
          else         out.image.at<uint8_t>(y2, x2) = val_i;
          out.pixel_to_point[static_cast<size_t>(y2) * width + x2] = static_cast<int>(pi);
        }
      }
    }
  }

  // Depth-aware gap fill. For each empty pixel, look at neighbours within
  // fill_max_gap_px radius and average those whose depth falls within
  // fill_max_depth_jump_m of the local mean. Skipped when fill is disabled
  // (gap_px <= 0). Single in-place pass: we read the input image+depth into
  // separate buffers so writes don't influence subsequent reads (would
  // otherwise propagate fills outward unboundedly in one pass).
  if (opts.fill_max_gap_px > 0) {
    const cv::Mat src_img   = out.image.clone();
    const cv::Mat src_depth = out.depth.clone();
    const int R_fill = std::max(1, opts.fill_max_gap_px);
    const float dz_max = std::max(0.0f, opts.fill_max_depth_jump_m);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (src_depth.at<float>(y, x) > 0.0f) continue;  // already filled
        // Two-pass neighbourhood scan: first pick a candidate depth (the
        // nearest non-empty pixel within the radius), then average all
        // neighbours within dz_max of that depth. This anchors the fill on
        // the nearest *foreground* surface so we don't spuriously bridge to
        // a distant background just because both are within range.
        float best_d = 0.0f; int best_d2 = INT32_MAX;
        for (int dy = -R_fill; dy <= R_fill; dy++) {
          const int yy = y + dy;
          if (yy < 0 || yy >= height) continue;
          for (int dx = -R_fill; dx <= R_fill; dx++) {
            const int xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            const float d = src_depth.at<float>(yy, xx);
            if (d <= 0.0f) continue;
            const int d2 = dx * dx + dy * dy;
            if (d2 < best_d2) { best_d2 = d2; best_d = d; }
          }
        }
        if (best_d <= 0.0f) continue;  // no neighbour found

        // Now sum colour over neighbours within dz_max of the anchor depth.
        // Weight is 1/distance^2 so closer neighbours dominate; gives a soft
        // anti-alias on splat boundaries.
        double w_sum = 0.0;
        double br = 0.0, bg = 0.0, bb = 0.0;
        double i_sum = 0.0;
        for (int dy = -R_fill; dy <= R_fill; dy++) {
          const int yy = y + dy;
          if (yy < 0 || yy >= height) continue;
          for (int dx = -R_fill; dx <= R_fill; dx++) {
            const int xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            const float d = src_depth.at<float>(yy, xx);
            if (d <= 0.0f) continue;
            if (std::abs(d - best_d) > dz_max) continue;  // depth jump kills this neighbour
            const int d2 = dx * dx + dy * dy;
            const double w = 1.0 / std::max(1, d2);
            w_sum += w;
            if (use_rgb) {
              const cv::Vec3b& c = src_img.at<cv::Vec3b>(yy, xx);
              bb += w * c[0]; bg += w * c[1]; br += w * c[2];
            } else {
              i_sum += w * src_img.at<uint8_t>(yy, xx);
            }
          }
        }
        if (w_sum <= 0.0) continue;
        const double inv = 1.0 / w_sum;
        out.depth.at<float>(y, x) = best_d;  // record the (estimated) depth too
        if (use_rgb) {
          out.image.at<cv::Vec3b>(y, x) = cv::Vec3b(
            static_cast<uint8_t>(std::clamp(bb * inv, 0.0, 255.0)),
            static_cast<uint8_t>(std::clamp(bg * inv, 0.0, 255.0)),
            static_cast<uint8_t>(std::clamp(br * inv, 0.0, 255.0)));
        } else {
          out.image.at<uint8_t>(y, x) = static_cast<uint8_t>(std::clamp(i_sum * inv, 0.0, 255.0));
        }
      }
    }
  }

  // Optional colormap post-pass. Grayscale stays 8UC1; Inverted stays 8UC1
  // (bit-not); Turbo/Viridis/Cividis go through cv::applyColorMap and come
  // back 8UC3 BGR. If `return_to_grayscale_after_colormap` is on, we then
  // convert the colormapped RGB back to luminance so LightGlue / SIFT still
  // get a single-channel image, but with a contrast curve reshaped by the
  // colormap's non-linear hue sweep.
  // Skipped entirely when the rasterizer already produced a colour image
  // (RGB sources): colormaps are an intensity-shaping tool, meaningless on
  // already-coloured pixels and would corrupt them.
  if (!use_rgb && opts.colormap != IntensityRenderOptions::Colormap::Grayscale) {
    cv::Mat gray = out.image;  // keep reference for potential inversion
    cv::Mat colored;
    switch (opts.colormap) {
      case IntensityRenderOptions::Colormap::Inverted: {
        cv::Mat inv; cv::bitwise_not(gray, inv);
        out.image = inv;
        break;
      }
      case IntensityRenderOptions::Colormap::Turbo:
        cv::applyColorMap(gray, colored, cv::COLORMAP_TURBO); out.image = colored; break;
      case IntensityRenderOptions::Colormap::Viridis:
        cv::applyColorMap(gray, colored, cv::COLORMAP_VIRIDIS); out.image = colored; break;
      case IntensityRenderOptions::Colormap::Cividis:
        cv::applyColorMap(gray, colored, cv::COLORMAP_CIVIDIS); out.image = colored; break;
      default: break;
    }
    if (opts.return_to_grayscale_after_colormap && out.image.channels() == 3) {
      cv::Mat g; cv::cvtColor(out.image, g, cv::COLOR_BGR2GRAY);
      out.image = g;
    }
  }

  return out;
}

// -----------------------------------------------------------------------------
// LightGlue match loader
// -----------------------------------------------------------------------------
std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> load_lightglue_matches(
  const std::string& json_path, std::vector<float>* confidences) {

  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> matches;
  if (confidences) confidences->clear();

  std::ifstream ifs(json_path);
  if (!ifs) return matches;
  nlohmann::json j;
  try { ifs >> j; }
  catch (...) { return matches; }

  if (!j.contains("matches") || !j["matches"].is_array()) return matches;
  for (const auto& m : j["matches"]) {
    if (!m.contains("real") || !m.contains("rendered")) continue;
    const auto& r = m["real"];
    const auto& q = m["rendered"];
    if (!r.is_array() || !q.is_array() || r.size() < 2 || q.size() < 2) continue;
    matches.emplace_back(
      Eigen::Vector2d(r[0].get<double>(), r[1].get<double>()),
      Eigen::Vector2d(q[0].get<double>(), q[1].get<double>()));
    if (confidences) confidences->push_back(m.value("score", 1.0f));
  }
  return matches;
}

// -----------------------------------------------------------------------------
// 2D↔2D → 2D↔3D correspondence mapping
// -----------------------------------------------------------------------------
std::vector<CalibCorrespondence> matches_to_correspondences(
  const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matches_real_rendered,
  const std::vector<float>& confidences,
  const RenderedIntensity& render,
  const CalibrationContext& ctx) {

  std::vector<CalibCorrespondence> out;
  out.reserve(matches_real_rendered.size());
  const int W = render.image.cols, H = render.image.rows;
  for (size_t mi = 0; mi < matches_real_rendered.size(); mi++) {
    const auto& m = matches_real_rendered[mi];
    const int qu = static_cast<int>(std::round(m.second.x()));
    const int qv = static_cast<int>(std::round(m.second.y()));
    if (qu < 0 || qu >= W || qv < 0 || qv >= H) continue;
    const int pi = render.pixel_to_point[static_cast<size_t>(qv) * W + qu];
    if (pi < 0 || pi >= static_cast<int>(ctx.world_points.size())) continue;
    CalibCorrespondence c;
    c.uv_image = m.first;
    c.xyz_world = ctx.world_points[pi].cast<double>();
    c.confidence = (mi < confidences.size()) ? confidences[mi] : 1.0f;
    out.push_back(c);
  }
  return out;
}

// -----------------------------------------------------------------------------
// PnP extrinsic refinement
// -----------------------------------------------------------------------------
// Convention notes:
// Our LiDAR convention: X fwd, Y left, Z up.
// OpenCV camera:        Z fwd, X right, Y down.
// To use OpenCV solvePnP we project world points into OpenCV-style camera frame.
// The rotation result from solvePnP is world→cv-camera. We convert back to world→our-camera.
static const Eigen::Matrix3d R_our_to_cv = (Eigen::Matrix3d() <<
   0, -1,  0,
   0,  0, -1,
   1,  0,  0).finished();
static const Eigen::Matrix3d R_cv_to_our = R_our_to_cv.transpose();

bool refine_extrinsic_pnp(
  const std::vector<CalibCorrespondence>& corrs,
  const PinholeIntrinsics& intrinsics,
  const Eigen::Isometry3d& T_world_cam_init,
  Eigen::Isometry3d& T_world_cam_refined,
  int& n_inliers, double& residual_px) {

  n_inliers = 0; residual_px = 0.0;
  if (corrs.size() < 6) return false;

  std::vector<cv::Point3d> object_points;
  std::vector<cv::Point2d> image_points;
  object_points.reserve(corrs.size());
  image_points.reserve(corrs.size());
  for (const auto& c : corrs) {
    object_points.emplace_back(c.xyz_world.x(), c.xyz_world.y(), c.xyz_world.z());
    image_points.emplace_back(c.uv_image.x(), c.uv_image.y());
  }

  cv::Mat K = (cv::Mat_<double>(3, 3) <<
    intrinsics.fx, 0, intrinsics.cx,
    0, intrinsics.fy, intrinsics.cy,
    0, 0, 1);
  cv::Mat D = (cv::Mat_<double>(1, 5) <<
    intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, intrinsics.k3);

  // Initial guess: convert T_world_cam_init (our frame) → rvec, tvec (cv frame).
  // OpenCV wants T_cv_cam_world: world → OpenCV camera frame.
  const Eigen::Isometry3d T_our_world = T_world_cam_init.inverse();
  const Eigen::Matrix3d R_cv_world = R_our_to_cv * T_our_world.rotation();
  const Eigen::Vector3d t_cv_world = R_our_to_cv * T_our_world.translation();

  cv::Mat rvec, tvec;
  {
    cv::Mat R_init(3, 3, CV_64F);
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R_init.at<double>(i, j) = R_cv_world(i, j);
    cv::Rodrigues(R_init, rvec);
    tvec = (cv::Mat_<double>(3, 1) << t_cv_world.x(), t_cv_world.y(), t_cv_world.z());
  }

  std::vector<int> inliers;
  const double ransac_thresh_px = 8.0;  // loose — LightGlue on rendered LiDAR intensity produces noisy matches
  const int iters = 500;
  const double confidence = 0.99;
  bool ok = cv::solvePnPRansac(object_points, image_points, K, D,
    rvec, tvec, true /*use extrinsic guess*/, iters, ransac_thresh_px, confidence,
    inliers, cv::SOLVEPNP_ITERATIVE);
  std::cerr << "[AutoCalib] PnP (ITERATIVE, thresh=" << ransac_thresh_px << "px): ok=" << ok
            << " inliers=" << inliers.size() << " / " << corrs.size() << std::endl;
  if (!ok || inliers.size() < 6) {
    // Retry without extrinsic guess using EPNP — robust to bad initial pose and handles
    // noisier correspondences better than ITERATIVE.
    inliers.clear();
    rvec.release(); tvec.release();
    ok = cv::solvePnPRansac(object_points, image_points, K, D,
      rvec, tvec, false, iters, ransac_thresh_px, confidence,
      inliers, cv::SOLVEPNP_EPNP);
    std::cerr << "[AutoCalib] PnP retry (EPNP, no guess): ok=" << ok
              << " inliers=" << inliers.size() << " / " << corrs.size() << std::endl;
    if (!ok || inliers.size() < 6) return false;
  }

  // Convert rvec/tvec back to our T_world_cam.
  cv::Mat R_mat;
  cv::Rodrigues(rvec, R_mat);
  Eigen::Matrix3d R_cv;
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R_cv(i, j) = R_mat.at<double>(i, j);
  const Eigen::Vector3d t_cv(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

  Eigen::Matrix3d R_our = R_cv_to_our * R_cv;
  Eigen::Vector3d t_our = R_cv_to_our * t_cv;
  Eigen::Isometry3d T_our_world_new = Eigen::Isometry3d::Identity();
  T_our_world_new.linear() = R_our;
  T_our_world_new.translation() = t_our;
  T_world_cam_refined = T_our_world_new.inverse();

  // Residual on inliers
  std::vector<cv::Point3d> obj_in; std::vector<cv::Point2d> img_in;
  obj_in.reserve(inliers.size()); img_in.reserve(inliers.size());
  for (int idx : inliers) { obj_in.push_back(object_points[idx]); img_in.push_back(image_points[idx]); }
  std::vector<cv::Point2d> reproj;
  cv::projectPoints(obj_in, rvec, tvec, K, D, reproj);
  double sum = 0.0;
  for (size_t i = 0; i < reproj.size(); i++) {
    const double du = reproj[i].x - img_in[i].x;
    const double dv = reproj[i].y - img_in[i].y;
    sum += std::sqrt(du * du + dv * dv);
  }
  residual_px = sum / static_cast<double>(reproj.size());
  n_inliers = static_cast<int>(inliers.size());
  return true;
}

// -----------------------------------------------------------------------------
// Intrinsics refinement with OPTIONAL locked extrinsic
// -----------------------------------------------------------------------------
// If !lock_extrinsic: uses cv::calibrateCamera (single-view) which co-estimates
// pose + intrinsics jointly.
// If  lock_extrinsic: runs a manual Levenberg–Marquardt on 9 intrinsic parameters
// (fx, fy, cx, cy, k1, k2, p1, p2, k3) with T_world_cam FIXED. Numerical Jacobian.
// Returned intrinsics are thus valid ONLY under the provided pose.
static bool refine_intrinsics_fixed_pose(
  const std::vector<CalibCorrespondence>& corrs,
  const Eigen::Isometry3d& T_world_cam,
  PinholeIntrinsics& intrinsics,
  double& residual_px,
  int max_iters = 50) {

  if (corrs.size() < 12) return false;

  // Pre-compute rvec/tvec in OpenCV convention from the fixed pose.
  const Eigen::Isometry3d T_our_world = T_world_cam.inverse();
  const Eigen::Matrix3d R_cv_world = R_our_to_cv * T_our_world.rotation();
  const Eigen::Vector3d t_cv_world = R_our_to_cv * T_our_world.translation();
  cv::Mat rvec;
  {
    cv::Mat R_init(3, 3, CV_64F);
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R_init.at<double>(i, j) = R_cv_world(i, j);
    cv::Rodrigues(R_init, rvec);
  }
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << t_cv_world.x(), t_cv_world.y(), t_cv_world.z());

  std::vector<cv::Point3d> obj_pts;
  std::vector<cv::Point2d> img_pts;
  obj_pts.reserve(corrs.size()); img_pts.reserve(corrs.size());
  for (const auto& c : corrs) {
    obj_pts.emplace_back(c.xyz_world.x(), c.xyz_world.y(), c.xyz_world.z());
    img_pts.emplace_back(c.uv_image.x(), c.uv_image.y());
  }

  // State: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
  Eigen::VectorXd x(9);
  x << intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
       intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, intrinsics.k3;

  auto residuals_fn = [&](const Eigen::VectorXd& xc, Eigen::VectorXd& r) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << xc[0], 0, xc[2], 0, xc[1], xc[3], 0, 0, 1);
    cv::Mat D = (cv::Mat_<double>(1, 5) << xc[4], xc[5], xc[6], xc[7], xc[8]);
    std::vector<cv::Point2d> proj;
    cv::projectPoints(obj_pts, rvec, tvec, K, D, proj);
    r.resize(obj_pts.size() * 2);
    for (size_t i = 0; i < obj_pts.size(); i++) {
      r[2 * i]     = proj[i].x - img_pts[i].x;
      r[2 * i + 1] = proj[i].y - img_pts[i].y;
    }
  };

  Eigen::VectorXd res;
  residuals_fn(x, res);
  double prev_sq = res.squaredNorm();
  double lambda = 1e-3;

  for (int iter = 0; iter < max_iters; iter++) {
    // Numerical Jacobian (central difference)
    Eigen::MatrixXd J(res.size(), 9);
    Eigen::VectorXd res_p(res.size()), res_m(res.size());
    for (int p = 0; p < 9; p++) {
      const double h = std::max(1e-5, std::abs(x[p]) * 1e-5);
      Eigen::VectorXd xp = x; xp[p] += h;
      Eigen::VectorXd xm = x; xm[p] -= h;
      residuals_fn(xp, res_p);
      residuals_fn(xm, res_m);
      J.col(p) = (res_p - res_m) / (2.0 * h);
    }

    Eigen::MatrixXd JtJ = J.transpose() * J;
    Eigen::VectorXd Jtr = J.transpose() * res;
    JtJ += lambda * Eigen::MatrixXd::Identity(9, 9);
    Eigen::VectorXd dx = JtJ.ldlt().solve(-Jtr);
    Eigen::VectorXd x_new = x + dx;
    Eigen::VectorXd res_new;
    residuals_fn(x_new, res_new);
    const double new_sq = res_new.squaredNorm();

    if (new_sq < prev_sq) {
      x = x_new;
      res = res_new;
      const double old_prev = prev_sq;
      prev_sq = new_sq;
      lambda *= 0.5;
      if (dx.norm() < 1e-6 || (old_prev - new_sq) / old_prev < 1e-6) break;
    } else {
      lambda *= 2.0;
      if (lambda > 1e6) break;
    }
  }

  intrinsics.fx = x[0]; intrinsics.fy = x[1];
  intrinsics.cx = x[2]; intrinsics.cy = x[3];
  intrinsics.k1 = x[4]; intrinsics.k2 = x[5];
  intrinsics.p1 = x[6]; intrinsics.p2 = x[7];
  intrinsics.k3 = x[8];
  residual_px = std::sqrt(prev_sq / static_cast<double>(obj_pts.size()));
  return true;
}

bool refine_intrinsics_lm(
  const std::vector<CalibCorrespondence>& corrs,
  PinholeIntrinsics& intrinsics,
  Eigen::Isometry3d& T_world_cam,
  bool lock_extrinsic,
  double& residual_px) {

  residual_px = 0.0;
  if (corrs.size() < 12) return false;

  // Locked path: intrinsics-only LM with fixed pose. Returns intrinsics that
  // are actually valid under the caller's extrinsic.
  if (lock_extrinsic) {
    return refine_intrinsics_fixed_pose(corrs, T_world_cam, intrinsics, residual_px);
  }

  // Unlocked path: joint pose + intrinsics via cv::calibrateCamera.

  std::vector<std::vector<cv::Point3f>> object_pts(1);
  std::vector<std::vector<cv::Point2f>> image_pts(1);
  for (const auto& c : corrs) {
    object_pts[0].emplace_back(static_cast<float>(c.xyz_world.x()),
                               static_cast<float>(c.xyz_world.y()),
                               static_cast<float>(c.xyz_world.z()));
    image_pts[0].emplace_back(static_cast<float>(c.uv_image.x()),
                              static_cast<float>(c.uv_image.y()));
  }

  cv::Mat K = (cv::Mat_<double>(3, 3) <<
    intrinsics.fx, 0, intrinsics.cx,
    0, intrinsics.fy, intrinsics.cy,
    0, 0, 1);
  cv::Mat D = (cv::Mat_<double>(1, 5) <<
    intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, intrinsics.k3);

  std::vector<cv::Mat> rvecs, tvecs;
  const cv::Size img_size(intrinsics.width, intrinsics.height);
  const int flags = cv::CALIB_USE_INTRINSIC_GUESS
                  | cv::CALIB_RATIONAL_MODEL * 0   // stick with Brown-Conrady (k1,k2,k3,p1,p2)
                  | cv::CALIB_FIX_ASPECT_RATIO * 0;
  double rms;
  try {
    rms = cv::calibrateCamera(object_pts, image_pts, img_size, K, D, rvecs, tvecs, flags);
  } catch (const cv::Exception& e) {
    std::cerr << "[AutoCalib] calibrateCamera failed: " << e.what() << std::endl;
    return false;
  }

  // Write back intrinsics
  intrinsics.fx = K.at<double>(0, 0);
  intrinsics.fy = K.at<double>(1, 1);
  intrinsics.cx = K.at<double>(0, 2);
  intrinsics.cy = K.at<double>(1, 2);
  intrinsics.k1 = D.at<double>(0, 0);
  intrinsics.k2 = D.at<double>(0, 1);
  intrinsics.p1 = D.at<double>(0, 2);
  intrinsics.p2 = D.at<double>(0, 3);
  intrinsics.k3 = D.at<double>(0, 4);

  // Write back extrinsic from the single view
  cv::Mat R_mat;
  cv::Rodrigues(rvecs[0], R_mat);
  Eigen::Matrix3d R_cv;
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R_cv(i, j) = R_mat.at<double>(i, j);
  const Eigen::Vector3d t_cv(tvecs[0].at<double>(0), tvecs[0].at<double>(1), tvecs[0].at<double>(2));
  Eigen::Matrix3d R_our = R_cv_to_our * R_cv;
  Eigen::Vector3d t_our = R_cv_to_our * t_cv;
  Eigen::Isometry3d T_our_world_new = Eigen::Isometry3d::Identity();
  T_our_world_new.linear() = R_our;
  T_our_world_new.translation() = t_our;
  T_world_cam = T_our_world_new.inverse();

  residual_px = rms;
  return true;
}

}  // namespace glim
