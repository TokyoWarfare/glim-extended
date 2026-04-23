#include <glim/util/lidar_colorizer.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace glim {

/// Weighted view selector.
/// For every (point, camera) pair that projects inside the image, compute
///   w = w_range * w_center * (w_incidence — deferred, requires normals)
/// Top-1 mode: winning camera's color per point.
/// Top-K mode: softmax over the K highest weights, blend their colors.
///
/// Diagnostic outputs (winner_cam, winner_weight) are filled so the alignment
/// check window can visualize coverage and per-point confidence.
class LidarColorizerWeighted : public ILidarColorizer {
public:
  explicit LidarColorizerWeighted(bool topK_mode) : topK_mode_(topK_mode) {}

  ColorizeResult project(
    const std::vector<CameraFrame>& cameras,
    const PinholeIntrinsics& intrinsics,
    const std::vector<Eigen::Vector3f>& world_points,
    const std::vector<float>& intensities,
    const std::vector<Eigen::Vector3f>& world_normals,
    const std::vector<double>& world_point_times,
    const BlendParams& params) override {

    const double max_range = params.max_range;
    const double min_range = params.min_range;
    const cv::Mat& mask = params.mask;
    const float tau = std::max(0.1f, params.range_tau);
    const float center_exp = std::max(0.0f, params.center_exp);
    const float incidence_exp = std::max(0.0f, params.incidence_exp);
    const int K = topK_mode_ ? std::max(2, params.topK) : 1;
    const bool use_incidence = !world_normals.empty() && world_normals.size() == world_points.size() && incidence_exp > 0.0f;
    const bool use_incidence_hard = !world_normals.empty() && world_normals.size() == world_points.size() && params.incidence_hard_cos > 0.0f;
    const bool use_occlusion = params.use_depth_occlusion;
    const int  zbuf_ds = std::max(1, params.occlusion_z_downscale);
    const float occl_tol = std::max(0.001f, params.occlusion_tolerance);
    const bool use_ncc = topK_mode_ && params.ncc_threshold > -1.0f;
    const int  ncc_half = std::max(1, params.ncc_patch_half);
    const bool has_times = !world_point_times.empty() && world_point_times.size() == world_points.size();
    const bool use_time_slice_hard = params.time_slice_hard && has_times;
    const bool use_time_slice_soft = params.time_slice_soft && has_times;
    const float time_sigma = std::max(0.001f, params.time_slice_sigma);
    // Pre-compute closest-in-time camera per point (for hard filter only)
    std::vector<int> closest_cam_per_point;
    if (use_time_slice_hard) {
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
    result.winner_cam.assign(world_points.size(), -1);
    result.winner_weight.assign(world_points.size(), 0.0f);

    // Per-point top-K heap: kept as small sorted arrays for speed over std::priority_queue
    // entries[pi]: up to K (weight, color, cam_idx, patch) sorted descending by weight.
    // Patch is only populated when NCC cross-check is enabled (use_ncc).
    struct Entry {
      float w;
      Eigen::Vector3f rgb;
      int cam;
      std::vector<float> patch;  // grayscale (2*ncc_half+1)^2, only if use_ncc
    };
    std::vector<std::vector<Entry>> topk(world_points.size());

    const double max_range_sq = max_range * max_range;
    const double min_range_sq = min_range * min_range;

    // Centroid for quick camera rejection
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto& p : world_points) centroid += p;
    if (!world_points.empty()) centroid /= static_cast<float>(world_points.size());

    // Normalized-radius denominator (image half-width/half-height)
    const double half_w = std::max(1.0, intrinsics.cx);
    const double half_h = std::max(1.0, intrinsics.cy);

    for (int ci = 0; ci < static_cast<int>(cameras.size()); ci++) {
      const auto& cam = cameras[ci];
      if (!cam.located) continue;

      const Eigen::Vector3f cam_pos = cam.T_world_cam.translation().cast<float>();
      const float dist_to_centroid_sq = (centroid - cam_pos).squaredNorm();
      if (dist_to_centroid_sq > max_range_sq * 9.0) continue;

      // Prefer caller-provided image (e.g. cube-face slice from a spherical source);
      // fall back to reading from disk for ordinary pinhole frames.
      cv::Mat img = cam.image_override ? *cam.image_override : cv::imread(cam.filepath);
      if (img.empty()) continue;

      // Per-cam mask override -- used when the caller pre-sliced an equirect
      // mask into per-face pieces (expand_source_cams_for_projection). Keeps
      // dims matched to img so sampling is 1:1, no linear-rescale tiling.
      const cv::Mat& cam_mask = cam.mask_override ? *cam.mask_override : mask;

      const Eigen::Isometry3d T_cam_world_d = cam.T_world_cam.inverse();
      const Eigen::Matrix3d R_cam = T_cam_world_d.rotation();
      const Eigen::Vector3d t_cam = T_cam_world_d.translation();

      const double fx = intrinsics.fx, fy = intrinsics.fy;
      const double cx_d = intrinsics.cx, cy_d = intrinsics.cy;
      const bool has_distortion = (intrinsics.k1 != 0 || intrinsics.k2 != 0 || intrinsics.p1 != 0 || intrinsics.p2 != 0);

      // Photometric exposure gain. Simple mode = downscaled image mean (bright-balanced,
      // moderate gain on outdoor sun). Surface-pixel mode = mean only over pixels where
      // LiDAR points project (unbiased by sky/ceiling but can over-boost sunny scenes).
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
            surf_sum += 0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0];
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

      // --- Build depth buffer for this camera (optional occlusion culling) ---
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

      // Grayscale view of the current image for NCC patch extraction (lazy)
      cv::Mat img_gray;

      int projected = 0;
      int occluded = 0;
      for (size_t pi = 0; pi < world_points.size(); pi++) {
        // Time-slice hard filter: only the time-closest camera contributes per point
        if (use_time_slice_hard && closest_cam_per_point[pi] != ci) continue;

        const float dist_sq = (world_points[pi] - cam_pos).squaredNorm();
        if (dist_sq > max_range_sq || dist_sq < min_range_sq) continue;

        const Eigen::Vector3d p_cam = R_cam * world_points[pi].cast<double>() + t_cam;
        const double depth = p_cam.x();
        if (depth <= 0.1) continue;

        double xn = -p_cam.y() / depth;
        double yn = -p_cam.z() / depth;

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

        // Depth-buffer occlusion test
        if (use_occlusion) {
          const int zu = iu / zbuf_ds, zv = iv / zbuf_ds;
          if (zu >= 0 && zu < zW && zv >= 0 && zv < zH) {
            const float zd = zbuf[static_cast<size_t>(zv) * zW + zu];
            if (static_cast<float>(depth) > zd * (1.0f + occl_tol)) { occluded++; continue; }
          }
        }

        // Hard incidence gate — drop grazing-angle views outright before they can poison the top-K.
        if (use_incidence_hard) {
          const Eigen::Vector3f ray = (cam_pos - world_points[pi]).normalized();
          const float cos_theta = std::abs(world_normals[pi].dot(ray));
          if (cos_theta < params.incidence_hard_cos) continue;
        }

        if (!cam_mask.empty()) {
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

        // --- Weights ---
        // w_range: closer is better. exp(-dist/tau).
        const float dist = std::sqrt(dist_sq);
        const float w_range = std::exp(-dist / tau);

        // w_center: (1 - r²)^center_exp where r is normalized image radius.
        // Uses the ACTUAL image pixel coordinates so the edge falloff matches what the lens/distortion produces.
        const double rx = (u - cx_d) / half_w;
        const double ry = (v - cy_d) / half_h;
        const double r2_norm = std::min(1.0, rx * rx + ry * ry);
        const float w_center = (center_exp <= 0.0f) ? 1.0f : std::pow(std::max(0.0, 1.0 - r2_norm), center_exp);

        // w_incidence: max(0, |cos θ|)^incidence_exp where θ is angle between
        // surface normal (world frame) and the ray from point to camera.
        // |·| is used because normals from covariance-based estimation have
        // arbitrary sign — what matters is surface orientation, not facing direction.
        float w_incidence = 1.0f;
        if (use_incidence) {
          const Eigen::Vector3f ray = (cam_pos - world_points[pi]).normalized();
          const float cos_theta = std::abs(world_normals[pi].dot(ray));
          w_incidence = std::pow(std::max(0.0f, cos_theta), incidence_exp);
        }

        // w_time: soft time-slicing. Favor cameras whose timestamp is close to the point's sweep time.
        float w_time = 1.0f;
        if (use_time_slice_soft) {
          const double dt = std::abs(cam.timestamp - world_point_times[pi]);
          w_time = std::exp(-static_cast<float>(dt) / time_sigma);
        }

        const float w = w_range * w_center * w_incidence * w_time;
        if (w <= 0.0f) continue;

        const cv::Vec3b bgr = img.at<cv::Vec3b>(iv, iu);
        Eigen::Vector3f rgb(bgr[2] / 255.0f, bgr[1] / 255.0f, bgr[0] / 255.0f);
        if (params.exposure_normalize) {
          rgb = (rgb * exposure_gain).cwiseMax(0.0f).cwiseMin(1.0f);
        }

        // Patch extraction for NCC (top-K only, when enabled)
        std::vector<float> patch;
        if (use_ncc) {
          if (img_gray.empty()) cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
          if (iu - ncc_half >= 0 && iu + ncc_half < img_gray.cols &&
              iv - ncc_half >= 0 && iv + ncc_half < img_gray.rows) {
            const int side = 2 * ncc_half + 1;
            patch.resize(static_cast<size_t>(side) * side);
            for (int dy = -ncc_half; dy <= ncc_half; dy++) {
              for (int dx = -ncc_half; dx <= ncc_half; dx++) {
                patch[static_cast<size_t>((dy + ncc_half)) * side + (dx + ncc_half)] =
                  static_cast<float>(img_gray.at<uint8_t>(iv + dy, iu + dx));
              }
            }
          }
        }

        // Insert into per-point top-K
        auto& heap = topk[pi];
        if (static_cast<int>(heap.size()) < K) {
          heap.push_back({w, rgb, ci, std::move(patch)});
          std::sort(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) { return a.w > b.w; });
        } else if (w > heap.back().w) {
          heap.back() = {w, rgb, ci, std::move(patch)};
          std::sort(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) { return a.w > b.w; });
        }
        projected++;
      }
      std::cerr << "[ColorizeWeighted] Camera " << boost::filesystem::path(cam.filepath).filename().string()
                << ": " << projected << " points projected"
                << (use_occlusion ? (" (" + std::to_string(occluded) + " occluded)") : std::string())
                << std::endl;
    }

    // --- NCC helper (normalized cross-correlation on two equal-size float patches) ---
    auto ncc = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
      if (a.size() != b.size() || a.empty()) return -1.0f;
      double ma = 0, mb = 0;
      for (size_t i = 0; i < a.size(); i++) { ma += a[i]; mb += b[i]; }
      ma /= a.size(); mb /= a.size();
      double num = 0, da = 0, db = 0;
      for (size_t i = 0; i < a.size(); i++) {
        const double va = a[i] - ma, vb = b[i] - mb;
        num += va * vb; da += va * va; db += vb * vb;
      }
      const double denom = std::sqrt(std::max(1e-12, da * db));
      return static_cast<float>(num / denom);
    };
    int ncc_rejections = 0;

    // --- Reduce per-point top-K to a final color ---
    for (size_t pi = 0; pi < world_points.size(); pi++) {
      auto& heap = topk[pi];
      if (heap.empty()) {
        const float gray = (pi < intensities.size()) ? std::clamp(intensities[pi] / 255.0f, 0.0f, 1.0f) : 0.5f;
        result.colors[pi] = Eigen::Vector3f(gray, gray, gray);
        continue;
      }
      // NCC cross-check (top-K only): drop non-winner candidates whose patch doesn't
      // match the winner. For top-1 this is a no-op (nothing to drop).
      if (use_ncc && heap.size() > 1 && !heap.front().patch.empty()) {
        const auto& ref_patch = heap.front().patch;
        heap.erase(std::remove_if(heap.begin() + 1, heap.end(),
          [&](const Entry& e) {
            if (e.patch.empty()) return true;  // no patch -> can't verify -> drop
            const float c = ncc(ref_patch, e.patch);
            if (c < params.ncc_threshold) { ncc_rejections++; return true; }
            return false;
          }), heap.end());
      }
      if (!topK_mode_ || heap.size() == 1) {
        // Top-1 hard winner
        result.colors[pi] = heap.front().rgb;
      } else {
        // Softmax over the K weights — temperature = 1 (direct weights, not exponentiated again)
        float sum_w = 0.0f;
        for (const auto& e : heap) sum_w += e.w;
        if (sum_w <= 0.0f) { result.colors[pi] = heap.front().rgb; }
        else {
          Eigen::Vector3f blended = Eigen::Vector3f::Zero();
          for (const auto& e : heap) blended += (e.w / sum_w) * e.rgb;
          result.colors[pi] = blended;
        }
      }
      result.winner_cam[pi] = heap.front().cam;
      result.winner_weight[pi] = heap.front().w;
      result.colored++;
    }
    if (use_ncc) std::cerr << "[ColorizeWeighted] NCC cross-check rejected " << ncc_rejections << " top-K contributors" << std::endl;

    return result;
  }

private:
  bool topK_mode_;
};

std::unique_ptr<ILidarColorizer> make_weighted_colorizer_top1() {
  return std::make_unique<LidarColorizerWeighted>(false);
}
std::unique_ptr<ILidarColorizer> make_weighted_colorizer_topK() {
  return std::make_unique<LidarColorizerWeighted>(true);
}

}  // namespace glim
