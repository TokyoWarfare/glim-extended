#include <glim/util/lidar_colorizer.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace glim {

/// Simple view selector: keep the nearest camera's color per point.
/// Supports time-slice + depth-occlusion filters from BlendParams.
class LidarColorizerSimple : public ILidarColorizer {
public:
  ColorizeResult project(
    const std::vector<CameraFrame>& cameras,
    const PinholeIntrinsics& intrinsics,
    const std::vector<Eigen::Vector3f>& world_points,
    const std::vector<float>& intensities,
    const std::vector<Eigen::Vector3f>& /* world_normals — unused */,
    const std::vector<double>& world_point_times,
    const BlendParams& params) override {

    const double max_range = params.max_range;
    const double min_range = params.min_range;
    const cv::Mat& mask = params.mask;

    // Time-slice hard filter: pre-compute closest-in-time camera per point.
    // If world_point_times is empty or time_slice_hard is off, this stays empty and no filtering happens.
    const bool use_time_slice =
      params.time_slice_hard && !world_point_times.empty() &&
      world_point_times.size() == world_points.size();
    std::vector<int> closest_cam_per_point;
    if (use_time_slice) {
      closest_cam_per_point.assign(world_points.size(), -1);
      for (size_t pi = 0; pi < world_points.size(); pi++) {
        const double tp = world_point_times[pi];
        double best_dt = std::numeric_limits<double>::max();
        int best_ci = -1;
        for (int ci = 0; ci < static_cast<int>(cameras.size()); ci++) {
          if (!cameras[ci].located || cameras[ci].timestamp <= 0.0) continue;
          const double dt = std::abs(cameras[ci].timestamp - tp);
          if (dt < best_dt) { best_dt = dt; best_ci = ci; }
        }
        closest_cam_per_point[pi] = best_ci;
      }
    }

    ColorizeResult result;
    result.total = world_points.size();
    result.points = world_points;
    result.intensities = intensities;
    result.colors.resize(world_points.size(), Eigen::Vector3f::Zero());

    // Nearest-camera bookkeeping (no blend/average; one color per point).
    std::vector<float> closest_dist(world_points.size(), std::numeric_limits<float>::max());
    std::vector<Eigen::Vector3f> closest_color(world_points.size(), Eigen::Vector3f::Zero());
    std::vector<uint8_t> has_color(world_points.size(), 0);

    const double max_range_sq = max_range * max_range;
    const double min_range_sq = min_range * min_range;

    // Point-cloud centroid for quick camera rejection
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto& p : world_points) centroid += p;
    if (!world_points.empty()) centroid /= static_cast<float>(world_points.size());

    const bool use_occlusion = params.use_depth_occlusion;
    const int zbuf_ds = std::max(1, params.occlusion_z_downscale);
    const float occl_tol = std::max(0.001f, params.occlusion_tolerance);

    for (int ci = 0; ci < static_cast<int>(cameras.size()); ci++) {
      const auto& cam = cameras[ci];
      if (!cam.located) continue;

      const Eigen::Vector3f cam_pos = cam.T_world_cam.translation().cast<float>();
      const float dist_to_centroid_sq = (centroid - cam_pos).squaredNorm();
      if (dist_to_centroid_sq > max_range_sq * 9.0) continue;  // 3x margin, skip if way too far

      // Prefer caller-provided image (e.g. cube-face slice from a spherical source);
      // fall back to reading from disk for ordinary pinhole frames.
      cv::Mat img = cam.image_override ? *cam.image_override : cv::imread(cam.filepath);
      if (img.empty()) continue;

      // Per-cam mask override -- used when the caller pre-sliced an equirect
      // mask into per-face pieces (expand_source_cams_for_projection). When
      // present we sample THIS instead of params.mask, skipping the broken
      // linear-rescale path that produced X-axis tiling for spherical sources.
      const cv::Mat& cam_mask = cam.mask_override ? *cam.mask_override : mask;

      // Camera transform: world → camera local
      // T_world_cam already includes extrinsic (T_world_lidar * T_lidar_cam)
      const Eigen::Isometry3d T_cam_world_d = cam.T_world_cam.inverse();
      const Eigen::Matrix3d R_cam = T_cam_world_d.rotation();
      const Eigen::Vector3d t_cam = T_cam_world_d.translation();

      const double fx = intrinsics.fx, fy = intrinsics.fy;
      const double cx_d = intrinsics.cx, cy_d = intrinsics.cy;
      const bool has_distortion = (intrinsics.k1 != 0 || intrinsics.k2 != 0 || intrinsics.p1 != 0 || intrinsics.p2 != 0);

      // Photometric exposure gain. Two modes:
      //   simple (image-mean): downscaled full-image mean. Includes sky/ceiling so
      //     outdoor scenes register brighter and the gain stays moderate -- less
      //     boost, visually balanced.
      //   surface-pixel: mean only over pixels where LiDAR points project. Unbiased
      //     by sky/ceiling but can over-boost sunny outdoor scenes (surface pixels
      //     are darker than sky/glare, so the gain pushes the whole image bright).
      float exposure_gain = 1.0f;
      if (params.exposure_normalize) {
        float cur_mean;
        if (params.exposure_simple) {
          cv::Mat small; cv::resize(img, small, cv::Size(200, 200));
          cv::Mat g; cv::cvtColor(small, g, cv::COLOR_BGR2GRAY);
          cur_mean = static_cast<float>(std::max(1.0, cv::mean(g)[0]));
        } else {
          double surf_sum = 0.0; int surf_cnt = 0;
          for (size_t pi = 0; pi < world_points.size(); pi++) {
            const float dist_sq = (world_points[pi] - cam_pos).squaredNorm();
            if (dist_sq > max_range_sq || dist_sq < min_range_sq) continue;
            const Eigen::Vector3d p_cam = R_cam * world_points[pi].cast<double>() + t_cam;
            const double depth = p_cam.x();
            if (depth <= 0.1) continue;
            double xn = -p_cam.y() / depth, yn = -p_cam.z() / depth;
            if (has_distortion) {
              const double r2 = xn * xn + yn * yn, r4 = r2 * r2, r6 = r4 * r2;
              const double radial = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r4 + intrinsics.k3 * r6;
              const double xd = xn * radial + 2.0 * intrinsics.p1 * xn * yn + intrinsics.p2 * (r2 + 2.0 * xn * xn);
              const double yd = yn * radial + intrinsics.p1 * (r2 + 2.0 * yn * yn) + 2.0 * intrinsics.p2 * xn * yn;
              xn = xd; yn = yd;
            }
            const int iu = static_cast<int>(fx * xn + cx_d);
            const int iv = static_cast<int>(fy * yn + cy_d);
            if (iu < 0 || iu >= img.cols || iv < 0 || iv >= img.rows) continue;
            const cv::Vec3b bgr = img.at<cv::Vec3b>(iv, iu);
            surf_sum += 0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0];  // ITU-R BT.601 luma
            surf_cnt++;
          }
          if (surf_cnt >= 100) {
            cur_mean = static_cast<float>(std::max(1.0, surf_sum / surf_cnt));
          } else {
            cv::Mat small; cv::resize(img, small, cv::Size(200, 200));
            cv::Mat g; cv::cvtColor(small, g, cv::COLOR_BGR2GRAY);
            cur_mean = static_cast<float>(std::max(1.0, cv::mean(g)[0]));
          }
        }
        exposure_gain = std::clamp(
          (params.exposure_target * 255.0f) / cur_mean, 0.25f, 4.0f);
      }

      // --- Build depth buffer for this camera (optional) ---
      // Downscaled for speed + memory. Stores nearest-depth per (zu, zv).
      const int zW = (img.cols + zbuf_ds - 1) / zbuf_ds;
      const int zH = (img.rows + zbuf_ds - 1) / zbuf_ds;
      std::vector<float> zbuf;
      if (use_occlusion) {
        zbuf.assign(static_cast<size_t>(zW) * zH, std::numeric_limits<float>::max());
        for (size_t pi = 0; pi < world_points.size(); pi++) {
          const float dist_sq = (world_points[pi] - cam_pos).squaredNorm();
          if (dist_sq > max_range_sq || dist_sq < min_range_sq) continue;
          const Eigen::Vector3d p_cam = R_cam * world_points[pi].cast<double>() + t_cam;
          const double depth = p_cam.x();
          if (depth <= 0.1) continue;
          double xn = -p_cam.y() / depth, yn = -p_cam.z() / depth;
          if (has_distortion) {
            const double r2 = xn*xn + yn*yn, r4 = r2*r2, r6 = r4*r2;
            const double radial = 1.0 + intrinsics.k1*r2 + intrinsics.k2*r4 + intrinsics.k3*r6;
            const double xd = xn*radial + 2.0*intrinsics.p1*xn*yn + intrinsics.p2*(r2 + 2.0*xn*xn);
            const double yd = yn*radial + intrinsics.p1*(r2 + 2.0*yn*yn) + 2.0*intrinsics.p2*xn*yn;
            xn = xd; yn = yd;
          }
          const double u = fx*xn + cx_d, v = fy*yn + cy_d;
          const int zu = static_cast<int>(u) / zbuf_ds;
          const int zv = static_cast<int>(v) / zbuf_ds;
          if (zu < 0 || zu >= zW || zv < 0 || zv >= zH) continue;
          float& d = zbuf[static_cast<size_t>(zv) * zW + zu];
          if (static_cast<float>(depth) < d) d = static_cast<float>(depth);
        }
      }

      int projected = 0;
      int occluded = 0;
      for (size_t pi = 0; pi < world_points.size(); pi++) {
        // Time-slice hard filter: only the time-closest camera may colorize this point
        if (use_time_slice && closest_cam_per_point[pi] != ci) continue;

        const float dist_sq = (world_points[pi] - cam_pos).squaredNorm();
        if (dist_sq > max_range_sq || dist_sq < min_range_sq) continue;

        // Transform to camera local frame
        // Convention: camera looks along +X (lidar frame), Y=left, Z=up
        const Eigen::Vector3d p_cam = R_cam * world_points[pi].cast<double>() + t_cam;

        const double depth = p_cam.x();
        if (depth <= 0.1) continue;  // behind camera

        double xn = -p_cam.y() / depth;  // negate Y: LiDAR Y=left → camera right
        double yn = -p_cam.z() / depth;  // negate Z: LiDAR Z=up → camera down

        if (has_distortion) {
          const double r2 = xn * xn + yn * yn;
          const double r4 = r2 * r2;
          const double r6 = r4 * r2;
          const double radial = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r4 + intrinsics.k3 * r6;
          const double xd = xn * radial + 2.0 * intrinsics.p1 * xn * yn + intrinsics.p2 * (r2 + 2.0 * xn * xn);
          const double yd = yn * radial + intrinsics.p1 * (r2 + 2.0 * yn * yn) + 2.0 * intrinsics.p2 * xn * yn;
          xn = xd;
          yn = yd;
        }

        const double u = fx * xn + cx_d;
        const double v = fy * yn + cy_d;

        const int iu = static_cast<int>(std::round(u));
        const int iv = static_cast<int>(std::round(v));
        if (iu < 0 || iu >= img.cols || iv < 0 || iv >= img.rows) continue;

        if (use_occlusion) {
          const int zu = iu / zbuf_ds, zv = iv / zbuf_ds;
          if (zu >= 0 && zu < zW && zv >= 0 && zv < zH) {
            const float zd = zbuf[static_cast<size_t>(zv) * zW + zu];
            if (static_cast<float>(depth) > zd * (1.0f + occl_tol)) { occluded++; continue; }
          }
        }

        if (!cam_mask.empty()) {
          // When cam_mask dims == img dims (expected when mask_override is a
          // pre-sliced face mask, or when a pinhole mask matches image res),
          // sample 1:1. Otherwise linear rescale -- correct for pinhole with a
          // differently-sized mask, wrong for raw equirect-on-cube (but that
          // wrong path is no longer reachable since expand now slices per face).
          const int mu = cam_mask.cols == img.cols ? iu : static_cast<int>(iu * static_cast<double>(cam_mask.cols) / img.cols);
          const int mv = cam_mask.rows == img.rows ? iv : static_cast<int>(iv * static_cast<double>(cam_mask.rows) / img.rows);
          if (mu >= 0 && mu < cam_mask.cols && mv >= 0 && mv < cam_mask.rows) {
            bool masked = false;
            if (cam_mask.channels() == 1) { masked = cam_mask.at<uint8_t>(mv, mu) == 0; }
            else if (cam_mask.channels() == 3) { const auto& mp = cam_mask.at<cv::Vec3b>(mv, mu); masked = (mp[0] == 0 && mp[1] == 0 && mp[2] == 0); }
            else if (cam_mask.channels() == 4) { const auto& mp = cam_mask.at<cv::Vec4b>(mv, mu); masked = (mp[3] == 0) || (mp[0] == 0 && mp[1] == 0 && mp[2] == 0); }
            if (masked) continue;
          }
        }
        const cv::Vec3b bgr = img.at<cv::Vec3b>(iv, iu);
        Eigen::Vector3f rgb(bgr[2] / 255.0f, bgr[1] / 255.0f, bgr[0] / 255.0f);
        if (params.exposure_normalize) {
          rgb = (rgb * exposure_gain).cwiseMax(0.0f).cwiseMin(1.0f);
        }
        const float dist = std::sqrt(dist_sq);
        if (dist < closest_dist[pi]) {
          closest_dist[pi] = dist;
          closest_color[pi] = rgb;
          has_color[pi] = 1;
        }
        projected++;
      }
      std::cerr << "[Colorize] Camera " << boost::filesystem::path(cam.filepath).filename().string()
                << ": " << projected << " points projected"
                << (use_occlusion ? (" (" + std::to_string(occluded) + " occluded)") : std::string())
                << std::endl;
    }

    // Final colors (always nearest-camera; no blend)
    for (size_t pi = 0; pi < world_points.size(); pi++) {
      if (has_color[pi]) {
        result.colors[pi] = closest_color[pi];
        result.colored++;
      } else {
        // Uncolored: grayscale fallback from intensity
        const float gray = (pi < intensities.size()) ? std::clamp(intensities[pi] / 255.0f, 0.0f, 1.0f) : 0.5f;
        result.colors[pi] = Eigen::Vector3f(gray, gray, gray);
      }
    }

    return result;
  }
};

// Factory entry — called from make_colorizer() in lidar_colorizer.cpp.
std::unique_ptr<ILidarColorizer> make_simple_colorizer() {
  return std::make_unique<LidarColorizerSimple>();
}

}  // namespace glim
