#include <glim/util/colorizer.hpp>
#include <glim/util/geodetic.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <libexif/exif-data.h>

namespace glim {

namespace {

// Helper: extract a rational value from EXIF entry
double exif_rational_to_double(const ExifEntry* entry, int index) {
  ExifByteOrder bo = exif_data_get_byte_order(entry->parent->parent);
  ExifRational r = exif_get_rational(entry->data + index * sizeof(ExifRational), bo);
  return (r.denominator != 0) ? static_cast<double>(r.numerator) / r.denominator : 0.0;
}

// Helper: extract DMS GPS coordinate as decimal degrees
double exif_gps_to_decimal(const ExifEntry* entry) {
  double deg = exif_rational_to_double(entry, 0);
  double min = exif_rational_to_double(entry, 1);
  double sec = exif_rational_to_double(entry, 2);
  return deg + min / 60.0 + sec / 3600.0;
}

// Helper: get string value from EXIF entry
std::string exif_entry_string(ExifEntry* entry) {
  if (!entry) return "";
  char buf[256];
  exif_entry_get_value(entry, buf, sizeof(buf));
  return std::string(buf);
}

// Parse "YYYY:MM:DD HH:MM:SS" + optional subsecond string → UNIX timestamp
double parse_exif_datetime(const std::string& datetime_str, const std::string& subsec_str) {
  if (datetime_str.size() < 19) return 0.0;
  struct tm t = {};
  // Format: "YYYY:MM:DD HH:MM:SS"
  if (sscanf(datetime_str.c_str(), "%d:%d:%d %d:%d:%d",
             &t.tm_year, &t.tm_mon, &t.tm_mday, &t.tm_hour, &t.tm_min, &t.tm_sec) != 6) {
    return 0.0;
  }
  t.tm_year -= 1900;
  t.tm_mon -= 1;
  // Convert to UNIX timestamp (UTC)
  double ts = static_cast<double>(timegm(&t));
  // Add subsecond
  if (!subsec_str.empty()) {
    double frac = std::stod("0." + subsec_str);
    ts += frac;
  }
  return ts;
}

}  // namespace

bool Colorizer::read_exif(const std::string& filepath, double& timestamp, double& lat, double& lon, double& alt) {
  ExifData* ed = exif_data_new_from_file(filepath.c_str());
  if (!ed) return false;

  // GPS Latitude
  ExifEntry* e_lat = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS], static_cast<ExifTag>(0x0002));  // GPSLatitude
  ExifEntry* e_lat_ref = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS], static_cast<ExifTag>(0x0001));  // GPSLatitudeRef
  if (e_lat) {
    lat = exif_gps_to_decimal(e_lat);
    if (e_lat_ref) {
      std::string ref = exif_entry_string(e_lat_ref);
      if (ref == "S" || ref == "s") lat = -lat;
    }
  }

  // GPS Longitude
  ExifEntry* e_lon = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS], static_cast<ExifTag>(0x0004));  // GPSLongitude
  ExifEntry* e_lon_ref = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS], static_cast<ExifTag>(0x0003));  // GPSLongitudeRef
  if (e_lon) {
    lon = exif_gps_to_decimal(e_lon);
    if (e_lon_ref) {
      std::string ref = exif_entry_string(e_lon_ref);
      if (ref == "W" || ref == "w") lon = -lon;
    }
  }

  // GPS Altitude
  ExifEntry* e_alt = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS], static_cast<ExifTag>(0x0006));  // GPSAltitude
  if (e_alt) {
    alt = exif_rational_to_double(e_alt, 0);
    ExifEntry* e_alt_ref = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS], static_cast<ExifTag>(0x0005));
    if (e_alt_ref && exif_entry_string(e_alt_ref) == "1") alt = -alt;  // below sea level
  }

  // DateTimeOriginal + SubSecTimeOriginal → timestamp
  ExifEntry* e_dt = exif_content_get_entry(ed->ifd[EXIF_IFD_EXIF], EXIF_TAG_DATE_TIME_ORIGINAL);
  ExifEntry* e_ss = exif_content_get_entry(ed->ifd[EXIF_IFD_EXIF], EXIF_TAG_SUB_SEC_TIME_ORIGINAL);
  std::string dt_str = e_dt ? exif_entry_string(e_dt) : "";
  std::string ss_str = e_ss ? exif_entry_string(e_ss) : "";
  timestamp = parse_exif_datetime(dt_str, ss_str);

  // Fallback: GPS timestamp if DateTimeOriginal missing
  if (timestamp <= 0.0) {
    ExifEntry* e_gps_date = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS], static_cast<ExifTag>(0x001D));  // GPSDateStamp
    ExifEntry* e_gps_time = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS], static_cast<ExifTag>(0x0007));  // GPSTimeStamp
    if (e_gps_date && e_gps_time) {
      std::string date_str = exif_entry_string(e_gps_date);
      double hours = exif_rational_to_double(e_gps_time, 0);
      double minutes = exif_rational_to_double(e_gps_time, 1);
      double seconds = exif_rational_to_double(e_gps_time, 2);
      // Parse date "YYYY:MM:DD"
      struct tm t = {};
      if (sscanf(date_str.c_str(), "%d:%d:%d", &t.tm_year, &t.tm_mon, &t.tm_mday) == 3) {
        t.tm_year -= 1900; t.tm_mon -= 1;
        timestamp = static_cast<double>(timegm(&t)) + hours * 3600.0 + minutes * 60.0 + seconds;
      }
    }
  }

  exif_data_unref(ed);
  return (lat != 0.0 || lon != 0.0 || timestamp > 0.0);
}

ImageSource Colorizer::load_image_folder(const std::string& folder_path) {
  ImageSource source;
  source.path = folder_path;
  source.name = boost::filesystem::path(folder_path).filename().string();

  // Collect image files
  std::vector<std::string> files;
  for (const auto& entry : boost::filesystem::directory_iterator(folder_path)) {
    if (!boost::filesystem::is_regular_file(entry)) continue;
    const std::string ext = entry.path().extension().string();
    std::string ext_lower = ext;
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    if (ext_lower == ".jpg" || ext_lower == ".jpeg" || ext_lower == ".png") {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());

  // Read EXIF for each
  for (const auto& f : files) {
    CameraFrame frame;
    frame.filepath = f;
    read_exif(f, frame.timestamp, frame.lat, frame.lon, frame.alt);
    source.frames.push_back(std::move(frame));
  }

  return source;
}

Eigen::Isometry3d Colorizer::interpolate_pose(const std::vector<TimedPose>& trajectory, double stamp) {
  if (trajectory.empty()) return Eigen::Isometry3d::Identity();
  if (stamp <= trajectory.front().stamp) return trajectory.front().pose;
  if (stamp >= trajectory.back().stamp) return trajectory.back().pose;

  // Binary search for the interval
  size_t lo = 0, hi = trajectory.size() - 1;
  while (lo + 1 < hi) {
    const size_t mid = (lo + hi) / 2;
    if (trajectory[mid].stamp <= stamp) lo = mid; else hi = mid;
  }

  const double t0 = trajectory[lo].stamp;
  const double t1 = trajectory[hi].stamp;
  const double alpha = (t1 > t0) ? (stamp - t0) / (t1 - t0) : 0.0;

  // Interpolate translation
  const Eigen::Vector3d p = trajectory[lo].pose.translation() * (1.0 - alpha) + trajectory[hi].pose.translation() * alpha;

  // SLERP rotation
  const Eigen::Quaterniond q0(trajectory[lo].pose.rotation());
  const Eigen::Quaterniond q1(trajectory[hi].pose.rotation());
  const Eigen::Quaterniond q = q0.slerp(alpha, q1);

  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translate(p);
  result.rotate(q);
  return result;
}

int Colorizer::locate_by_time(ImageSource& source, const std::vector<TimedPose>& trajectory) {
  if (trajectory.empty()) return 0;

  int located = 0;
  for (auto& frame : source.frames) {
    if (frame.timestamp <= 0.0) continue;

    const double ts = frame.timestamp + source.time_shift;
    // Check if timestamp is within trajectory range (with 1s tolerance)
    if (ts < trajectory.front().stamp - 1.0 || ts > trajectory.back().stamp + 1.0) continue;

    Eigen::Isometry3d T_world_lidar = interpolate_pose(trajectory, ts);

    // Build extrinsic: T_lidar_cam from lever arm + rotation
    const Eigen::Isometry3d T_lidar_cam = build_extrinsic(source.lever_arm, source.rotation_rpy);

    // T_world_cam = T_world_lidar * T_lidar_cam
    frame.T_world_cam = T_world_lidar * T_lidar_cam;
    frame.located = true;
    located++;
  }
  return located;
}

int Colorizer::locate_by_coordinates(ImageSource& source, const std::vector<TimedPose>& trajectory,
                                      int utm_zone, double easting_origin, double northing_origin, double alt_origin) {
  if (trajectory.empty()) return 0;

  int located = 0;
  for (auto& frame : source.frames) {
    if (frame.lat == 0.0 && frame.lon == 0.0) continue;

    // Convert lat/lon to UTM (forced zone to match datum)
    const Eigen::Vector2d utm = glim::wgs84_to_utm_xy(frame.lat, frame.lon, utm_zone);

    // World coordinates = UTM - origin
    const Eigen::Vector3d world_pos(utm.x() - easting_origin, utm.y() - northing_origin, frame.alt - alt_origin);

    // Find nearest trajectory point by XY distance
    double min_dist = std::numeric_limits<double>::max();
    size_t best_idx = 0;
    for (size_t i = 0; i < trajectory.size(); i++) {
      const double dx = trajectory[i].pose.translation().x() - world_pos.x();
      const double dy = trajectory[i].pose.translation().y() - world_pos.y();
      const double d = dx * dx + dy * dy;
      if (d < min_dist) { min_dist = d; best_idx = i; }
    }

    // Use trajectory orientation at nearest point, but GPS position
    Eigen::Isometry3d T_world_lidar = trajectory[best_idx].pose;
    T_world_lidar.translation() = world_pos;

    // Apply extrinsic
    const Eigen::Isometry3d T_lidar_cam = build_extrinsic(source.lever_arm, source.rotation_rpy);
    frame.T_world_cam = T_world_lidar * T_lidar_cam;
    frame.located = true;
    located++;
  }
  return located;
}

ColorizeResult Colorizer::project_colors(
    const std::vector<CameraFrame>& cameras,
    const PinholeIntrinsics& intrinsics,
    const std::vector<Eigen::Vector3f>& world_points,
    const std::vector<float>& intensities,
    double max_range,
    bool blend,
    double min_range,
    const cv::Mat& mask) {

  ColorizeResult result;
  result.total = world_points.size();
  result.points = world_points;
  result.intensities = intensities;
  result.colors.resize(world_points.size(), Eigen::Vector3f::Zero());

  // Per-point accumulation: sum of RGB + count + closest distance
  std::vector<Eigen::Vector3f> color_sum(world_points.size(), Eigen::Vector3f::Zero());
  std::vector<int> color_count(world_points.size(), 0);
  std::vector<float> closest_dist(world_points.size(), std::numeric_limits<float>::max());
  std::vector<Eigen::Vector3f> closest_color(world_points.size(), Eigen::Vector3f::Zero());

  const double max_range_sq = max_range * max_range;
  const double min_range_sq = min_range * min_range;

  // Compute point cloud centroid for quick camera rejection
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto& p : world_points) centroid += p;
  if (!world_points.empty()) centroid /= static_cast<float>(world_points.size());

  for (const auto& cam : cameras) {
    if (!cam.located) continue;

    // Quick rejection: check if camera is roughly near the point cloud
    const Eigen::Vector3f cam_pos = cam.T_world_cam.translation().cast<float>();
    const float dist_to_centroid_sq = (centroid - cam_pos).squaredNorm();
    if (dist_to_centroid_sq > max_range_sq * 9.0) continue;  // 3x margin, skip if way too far

    // Load image
    cv::Mat img = cv::imread(cam.filepath);
    if (img.empty()) continue;

    // Camera transform: world → camera local
    // T_world_cam already includes extrinsic (T_world_lidar * T_lidar_cam)
    const Eigen::Isometry3d T_cam_world_d = cam.T_world_cam.inverse();
    const Eigen::Matrix3d R_cam = T_cam_world_d.rotation();
    const Eigen::Vector3d t_cam = T_cam_world_d.translation();

    const double fx = intrinsics.fx, fy = intrinsics.fy;
    const double cx_d = intrinsics.cx, cy_d = intrinsics.cy;
    const bool has_distortion = (intrinsics.k1 != 0 || intrinsics.k2 != 0 || intrinsics.p1 != 0 || intrinsics.p2 != 0);

    int projected = 0;
    for (size_t pi = 0; pi < world_points.size(); pi++) {
      // Quick range check
      const float dist_sq = (world_points[pi] - cam_pos).squaredNorm();
      if (dist_sq > max_range_sq || dist_sq < min_range_sq) continue;

      // Transform to camera local frame
      // Convention: camera looks along +X (lidar frame), Y=left, Z=up
      // For pinhole: depth = X, project Y and Z
      const Eigen::Vector3d p_cam = R_cam * world_points[pi].cast<double>() + t_cam;

      // Depth = X (forward in our frame)
      const double depth = p_cam.x();
      if (depth <= 0.1) continue;  // behind camera

      // Normalized coordinates (in camera's YZ plane)
      double xn = -p_cam.y() / depth;  // negate Y: LiDAR Y=left → camera right
      double yn = -p_cam.z() / depth;  // negate Z: LiDAR Z=up → camera down

      // Apply Brown-Conrady distortion if present
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

      // Project to pixel
      const double u = fx * xn + cx_d;
      const double v = fy * yn + cy_d;

      // Check bounds
      const int iu = static_cast<int>(std::round(u));
      const int iv = static_cast<int>(std::round(v));
      if (iu < 0 || iu >= img.cols || iv < 0 || iv >= img.rows) continue;

      // Sample color (BGR → RGB)
      // Check mask (if provided) — skip black pixels
      if (!mask.empty()) {
        const int mu = mask.cols == img.cols ? iu : static_cast<int>(iu * static_cast<double>(mask.cols) / img.cols);
        const int mv = mask.rows == img.rows ? iv : static_cast<int>(iv * static_cast<double>(mask.rows) / img.rows);
        if (mu >= 0 && mu < mask.cols && mv >= 0 && mv < mask.rows) {
          bool masked = false;
          if (mask.channels() == 1) { masked = mask.at<uint8_t>(mv, mu) == 0; }
          else if (mask.channels() == 3) { const auto& mp = mask.at<cv::Vec3b>(mv, mu); masked = (mp[0] == 0 && mp[1] == 0 && mp[2] == 0); }
          else if (mask.channels() == 4) { const auto& mp = mask.at<cv::Vec4b>(mv, mu); masked = (mp[3] == 0) || (mp[0] == 0 && mp[1] == 0 && mp[2] == 0); }
          if (masked) continue;
        }
      }
      const cv::Vec3b bgr = img.at<cv::Vec3b>(iv, iu);
      const Eigen::Vector3f rgb(bgr[2] / 255.0f, bgr[1] / 255.0f, bgr[0] / 255.0f);
      color_sum[pi] += rgb;
      color_count[pi]++;
      const float dist = std::sqrt(dist_sq);
      if (dist < closest_dist[pi]) { closest_dist[pi] = dist; closest_color[pi] = rgb; }
      projected++;
    }
    std::cerr << "[Colorize] Camera " << boost::filesystem::path(cam.filepath).filename().string()
              << ": " << projected << " points projected" << std::endl;
  }

  // Final colors
  for (size_t pi = 0; pi < world_points.size(); pi++) {
    if (color_count[pi] > 0) {
      result.colors[pi] = blend ? (color_sum[pi] / static_cast<float>(color_count[pi])) : closest_color[pi];
      result.colored++;
    } else {
      // Uncolored: use intensity as grayscale fallback
      const float gray = (pi < intensities.size()) ? std::clamp(intensities[pi] / 255.0f, 0.0f, 1.0f) : 0.5f;
      result.colors[pi] = Eigen::Vector3f(gray, gray, gray);
    }
  }

  return result;
}

Eigen::Isometry3d Colorizer::build_extrinsic(const Eigen::Vector3d& lever_arm, const Eigen::Vector3d& rotation_rpy) {
  // RPY in degrees → rotation matrix (Z-Y-X order: yaw-pitch-roll)
  const double r = rotation_rpy.x() * M_PI / 180.0;
  const double p = rotation_rpy.y() * M_PI / 180.0;
  const double y = rotation_rpy.z() * M_PI / 180.0;
  Eigen::Matrix3d R = (Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ()) *
                        Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX())).toRotationMatrix();
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = R;
  T.translation() = lever_arm;
  return T;
}

Eigen::Isometry3d Colorizer::solve_extrinsic(
    const std::vector<Eigen::Vector3d>& pts_3d,
    const std::vector<Eigen::Vector2d>& pts_2d,
    const PinholeIntrinsics& intrinsics,
    const Eigen::Isometry3d& T_world_lidar) {

  if (pts_3d.size() < 6 || pts_3d.size() != pts_2d.size()) {
    std::cerr << "[Colorizer] solvePnP needs at least 6 correspondences, got " << pts_3d.size() << std::endl;
    return Eigen::Isometry3d::Identity();
  }

  // Feed raw lidar-local 3D points to solvePnP (no convention rotation).
  // solvePnP will find T that maps lidar-local → OpenCV camera frame (Z fwd, X right, Y down).
  // We then convert the result to our convention (X fwd, Y left, Z up).
  // Log diagnostics
  std::cerr << "[Colorizer] T_world_lidar pos: " << T_world_lidar.translation().transpose() << std::endl;
  std::cerr << "[Colorizer] T_world_lidar fwd: " << T_world_lidar.rotation().col(0).transpose() << std::endl;

  std::vector<cv::Point3d> obj_pts(pts_3d.size());
  std::vector<cv::Point2d> img_pts(pts_2d.size());
  for (size_t i = 0; i < pts_3d.size(); i++) {
    const Eigen::Vector3d p_lidar = T_world_lidar.inverse() * pts_3d[i];
    obj_pts[i] = cv::Point3d(p_lidar.x(), p_lidar.y(), p_lidar.z());
    img_pts[i] = cv::Point2d(pts_2d[i].x(), pts_2d[i].y());
    std::cerr << "[Colorizer] Pair " << i << ": world=(" << pts_3d[i].transpose()
              << ") lidar_local=(" << p_lidar.transpose()
              << ") pixel=(" << pts_2d[i].transpose() << ")" << std::endl;
    // Check: project using our convention to see expected pixel
    if (p_lidar.x() > 0.1) {
      const double eu = intrinsics.fx * (-p_lidar.y() / p_lidar.x()) + intrinsics.cx;
      const double ev = intrinsics.fy * (-p_lidar.z() / p_lidar.x()) + intrinsics.cy;
      std::cerr << "[Colorizer]   → identity projection: (" << eu << ", " << ev << ") vs picked (" << pts_2d[i].transpose() << ")" << std::endl;
    }
  }

  // Camera matrix
  cv::Mat K = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.cx,
                                           0, intrinsics.fy, intrinsics.cy,
                                           0, 0, 1);
  // Distortion coefficients
  cv::Mat dist = (cv::Mat_<double>(5, 1) << intrinsics.k1, intrinsics.k2,
                                             intrinsics.p1, intrinsics.p2, intrinsics.k3);

  // solvePnP: finds R, t such that p_cam_cv = R * p_lidar + t
  // where p_cam_cv is in OpenCV convention (Z fwd, X right, Y down)
  cv::Mat rvec, tvec;
  bool ok = cv::solvePnP(obj_pts, img_pts, K, dist, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
  if (!ok) {
    std::cerr << "[Colorizer] solvePnP failed" << std::endl;
    return Eigen::Isometry3d::Identity();
  }

  // Verify with OpenCV's own reprojection first
  {
    std::vector<cv::Point2d> reproj;
    cv::projectPoints(obj_pts, rvec, tvec, K, dist, reproj);
    double cv_err = 0.0;
    for (size_t i = 0; i < reproj.size(); i++) {
      cv_err += std::sqrt((reproj[i].x - img_pts[i].x) * (reproj[i].x - img_pts[i].x) +
                           (reproj[i].y - img_pts[i].y) * (reproj[i].y - img_pts[i].y));
    }
    std::cerr << "[Colorizer] OpenCV reprojection error: " << cv_err / reproj.size() << " px" << std::endl;
  }

  // Convert solvePnP result to Eigen
  cv::Mat R_cv_mat;
  cv::Rodrigues(rvec, R_cv_mat);
  Eigen::Matrix3d R_pnp;
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) R_pnp(r, c) = R_cv_mat.at<double>(r, c);
  Eigen::Vector3d t_pnp(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

  // R_pnp, t_pnp maps lidar-local points to OpenCV camera frame.
  // Convert to our camera convention (X fwd, Y left, Z up):
  // OpenCV → our: X_our = Z_cv, Y_our = -X_cv, Z_our = -Y_cv
  Eigen::Matrix3d R_cv_to_our;
  R_cv_to_our << 0, 0, 1,
                -1, 0, 0,
                 0,-1, 0;

  // T_our_cam_lidar: transforms lidar-local points to our camera frame
  Eigen::Isometry3d T_our_cam_lidar = Eigen::Isometry3d::Identity();
  T_our_cam_lidar.linear() = R_cv_to_our * R_pnp;
  T_our_cam_lidar.translation() = R_cv_to_our * t_pnp;

  // T_lidar_cam = inverse (from camera to lidar = the mounting offset)
  Eigen::Isometry3d T_lidar_cam = T_our_cam_lidar.inverse();

  // Extract RPY for display
  const Eigen::Matrix3d R_ext = T_lidar_cam.rotation();
  const double yaw = std::atan2(R_ext(1, 0), R_ext(0, 0)) * 180.0 / M_PI;
  const double pitch = std::asin(-std::clamp(R_ext(2, 0), -1.0, 1.0)) * 180.0 / M_PI;
  const double roll = std::atan2(R_ext(2, 1), R_ext(2, 2)) * 180.0 / M_PI;
  std::cerr << "[Colorizer] solvePnP result: lever=[" << T_lidar_cam.translation().transpose()
            << "] RPY=[" << roll << ", " << pitch << ", " << yaw << "] deg" << std::endl;

  // Compute reprojection error using OUR projection convention
  double total_err = 0.0;
  for (size_t i = 0; i < pts_3d.size(); i++) {
    const Eigen::Vector3d p_lidar = T_world_lidar.inverse() * pts_3d[i];
    const Eigen::Vector3d p_cam = T_our_cam_lidar * p_lidar;
    if (p_cam.x() <= 0) continue;
    const double u = intrinsics.fx * (-p_cam.y() / p_cam.x()) + intrinsics.cx;
    const double v = intrinsics.fy * (-p_cam.z() / p_cam.x()) + intrinsics.cy;
    total_err += std::sqrt((u - pts_2d[i].x()) * (u - pts_2d[i].x()) + (v - pts_2d[i].y()) * (v - pts_2d[i].y()));
  }
  std::cerr << "[Colorizer] Our reprojection error: " << total_err / pts_3d.size() << " px" << std::endl;

  return T_lidar_cam;
}

}  // namespace glim
