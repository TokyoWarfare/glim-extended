#include <glim/util/lidar_colorizer.hpp>
#include <glim/util/geodetic.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <libexif/exif-data.h>

namespace glim {

// -----------------------------------------------------------------------------
// Equirectangular <-> cube face slicing
// -----------------------------------------------------------------------------

// 6 fixed rotations that take a face-local frame (X fwd, Y left, Z up) into
// the parent equirect's camera frame. Face forward axis lands on one of the
// six cube directions in that frame.
Eigen::Matrix3d cube_face_rotation(int face_idx) {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  switch (face_idx) {
    case 0: /* +X (forward) */  R = Eigen::Matrix3d::Identity();                                              break;
    case 1: /* -X (back)    */  R = Eigen::AngleAxisd( M_PI,       Eigen::Vector3d::UnitZ()).toRotationMatrix(); break;
    case 2: /* +Y (left)    */  R = Eigen::AngleAxisd( M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix(); break;
    case 3: /* -Y (right)   */  R = Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix(); break;
    case 4: /* +Z (up)      */  R = Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix(); break;
    case 5: /* -Z (down)    */  R = Eigen::AngleAxisd( M_PI / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix(); break;
    default: break;
  }
  return R;
}

PinholeIntrinsics cube_face_intrinsics(int face_size) {
  PinholeIntrinsics k;
  k.fx = k.fy = 0.5 * static_cast<double>(face_size);  // 90 deg FOV cube face
  k.cx = k.cy = 0.5 * static_cast<double>(face_size);
  k.width = k.height = face_size;
  k.k1 = k.k2 = k.k3 = k.p1 = k.p2 = 0.0;
  return k;
}

namespace {

// Build (and cache) the 6 remap tables for a given (equirect size, face size).
// Each face gets two maps (mapx, mapy) of size face_size x face_size in CV_32FC1
// that directly index into the equirect image via cv::remap.
struct FaceRemap {
  std::array<cv::Mat, 6> mapx;
  std::array<cv::Mat, 6> mapy;
};

FaceRemap build_face_remap(int eq_w, int eq_h, int face_size) {
  FaceRemap out;
  for (int f = 0; f < 6; f++) {
    out.mapx[f] = cv::Mat(face_size, face_size, CV_32FC1);
    out.mapy[f] = cv::Mat(face_size, face_size, CV_32FC1);
  }
  // Face forward axes in the equirect (parent) camera frame.
  // Parent convention used here for sampling the equirect: (X fwd, Y left, Z up).
  const Eigen::Matrix3d Rf[6] = {
    cube_face_rotation(0), cube_face_rotation(1), cube_face_rotation(2),
    cube_face_rotation(3), cube_face_rotation(4), cube_face_rotation(5),
  };
  // For each output face pixel (u, v), produce a 3D ray in the face's local
  // frame, rotate into the parent frame, then sample equirect at the matching
  // (longitude, latitude). Longitude = atan2(-Y, X), Latitude = atan2(Z, sqrt(X^2+Y^2))
  // in our X-fwd, Y-left, Z-up convention. Equirect pixel: u = w * (lon + pi) / (2*pi),
  // v = h * (pi/2 - lat) / pi.
  const double inv_f = 2.0 / static_cast<double>(face_size);   // face fx = face_size/2 so 1/fx = 2/face_size
  for (int f = 0; f < 6; f++) {
    for (int yy = 0; yy < face_size; yy++) {
      for (int xx = 0; xx < face_size; xx++) {
        // Pinhole face-local ray. Face-local convention: X fwd, Y left, Z up.
        // Image pixel -> normalized face plane (pinhole):
        //   xn = (cx - u) / fx   (Y_face = left, so positive Y maps to -xn)  [matching existing proj]
        //   yn = (cy - v) / fy   (Z_face = up  , so positive Z maps to -yn)
        // Face-local ray = (1, xn, yn) normalized.
        const double xn = (0.5 * face_size - (xx + 0.5)) * inv_f;
        const double yn = (0.5 * face_size - (yy + 0.5)) * inv_f;
        Eigen::Vector3d ray_face(1.0, xn, yn);
        ray_face.normalize();
        // Rotate into parent (equirect camera) frame.
        Eigen::Vector3d ray = Rf[f] * ray_face;
        // Equirect pixel from ray in parent frame (X fwd, Y left, Z up).
        const double lon = std::atan2(-ray.y(), ray.x());
        const double lat = std::atan2(ray.z(), std::sqrt(ray.x() * ray.x() + ray.y() * ray.y()));
        double eu = (lon + M_PI) / (2.0 * M_PI) * eq_w - 0.5;
        double ev = (0.5 - lat / M_PI) * eq_h - 0.5;
        // Wrap longitude so cv::remap doesn't clamp across the seam.
        if (eu < 0)    eu += eq_w;
        if (eu >= eq_w) eu -= eq_w;
        out.mapx[f].at<float>(yy, xx) = static_cast<float>(eu);
        out.mapy[f].at<float>(yy, xx) = static_cast<float>(ev);
      }
    }
  }
  return out;
}

std::unordered_map<uint64_t, FaceRemap>& face_remap_cache() {
  static std::unordered_map<uint64_t, FaceRemap> cache;
  return cache;
}

uint64_t remap_key(int eq_w, int eq_h, int face_size) {
  return (static_cast<uint64_t>(eq_w) << 42) | (static_cast<uint64_t>(eq_h) << 21) | static_cast<uint64_t>(face_size);
}

}  // anonymous namespace

std::array<std::shared_ptr<cv::Mat>, 6> slice_equirect_cubemap(const cv::Mat& equirect, int face_size) {
  std::array<std::shared_ptr<cv::Mat>, 6> out;
  if (equirect.empty() || face_size <= 0) return out;
  const uint64_t key = remap_key(equirect.cols, equirect.rows, face_size);
  auto& cache = face_remap_cache();
  auto it = cache.find(key);
  if (it == cache.end()) {
    it = cache.emplace(key, build_face_remap(equirect.cols, equirect.rows, face_size)).first;
  }
  const FaceRemap& rm = it->second;
  for (int f = 0; f < 6; f++) {
    out[f] = std::make_shared<cv::Mat>();
    cv::remap(equirect, *out[f], rm.mapx[f], rm.mapy[f], cv::INTER_LINEAR, cv::BORDER_WRAP);
  }
  return out;
}

// -----------------------------------------------------------------------------
// End of cube face slicing
// -----------------------------------------------------------------------------


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

    // Per-frame effective calibration -- anchors (if any) give linear drift
    // correction so a long track can stay aligned at both ends; baseline
    // scalars are used when the anchors vector is empty.
    const auto c = effective_calib(source, frame.timestamp);
    const double ts = frame.timestamp + c.time_shift;
    // Check if timestamp is within trajectory range (with 1s tolerance)
    if (ts < trajectory.front().stamp - 1.0 || ts > trajectory.back().stamp + 1.0) continue;

    Eigen::Isometry3d T_world_lidar = interpolate_pose(trajectory, ts);
    const Eigen::Isometry3d T_lidar_cam = build_extrinsic(c.lever_arm, c.rotation_rpy);

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

    // Apply anchor-aware extrinsic. Coordinate-located frames still have a
    // timestamp; if it's zero, effective_extrinsic will clamp to the first
    // anchor (or use the baseline scalars when no anchors are set).
    const Eigen::Isometry3d T_lidar_cam = effective_extrinsic(source, frame.timestamp);
    frame.T_world_cam = T_world_lidar * T_lidar_cam;
    frame.located = true;
    located++;
  }
  return located;
}

void build_camera_trajectory(ImageSource& src,
                              int utm_zone,
                              double easting_origin,
                              double northing_origin,
                              double alt_origin) {
  src.camera_trajectory.clear();
  struct Entry { double stamp; Eigen::Vector3d world_pos; };
  std::vector<Entry> entries;
  entries.reserve(src.frames.size());
  for (const auto& f : src.frames) {
    if (f.timestamp <= 0.0) continue;
    if (f.lat == 0.0 && f.lon == 0.0) continue;
    const Eigen::Vector2d utm = glim::wgs84_to_utm_xy(f.lat, f.lon, utm_zone);
    const Eigen::Vector3d world_pos(utm.x() - easting_origin,
                                     utm.y() - northing_origin,
                                     f.alt - alt_origin);
    entries.push_back({f.timestamp, world_pos});
  }
  if (entries.size() < 2) return;  // need at least 2 points to fit a path
  std::sort(entries.begin(), entries.end(),
            [](const Entry& a, const Entry& b) { return a.stamp < b.stamp; });

  src.camera_trajectory.reserve(entries.size());
  for (size_t i = 0; i < entries.size(); i++) {
    // Yaw from velocity vector (forward = next - prev, flattened on XY).
    // Edges use one-sided differences.
    Eigen::Vector3d fwd;
    if (i == 0) {
      fwd = entries[1].world_pos - entries[0].world_pos;
    } else if (i + 1 == entries.size()) {
      fwd = entries[i].world_pos - entries[i - 1].world_pos;
    } else {
      fwd = entries[i + 1].world_pos - entries[i - 1].world_pos;
    }
    fwd.z() = 0.0;
    if (fwd.norm() < 1e-6) fwd = Eigen::Vector3d::UnitX();
    fwd.normalize();
    const double yaw = std::atan2(fwd.y(), fwd.x());

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    pose.translation() = entries[i].world_pos;
    src.camera_trajectory.push_back({entries[i].stamp, pose});
  }
}

const std::vector<TimedPose>& trajectory_for(const ImageSource& src,
                                              const std::vector<TimedPose>& slam_trajectory) {
  if (src.params.locate_mode == 2 && !src.camera_trajectory.empty()) {
    return src.camera_trajectory;
  }
  return slam_trajectory;
}

// Strategy factory implementations live in their own .cpp files.
std::unique_ptr<ILidarColorizer> make_simple_colorizer();
std::unique_ptr<ILidarColorizer> make_weighted_colorizer_top1();
std::unique_ptr<ILidarColorizer> make_weighted_colorizer_topK();
// Future: make_matched_colorizer()

std::unique_ptr<ILidarColorizer> make_colorizer(ViewSelectorMode mode) {
  switch (mode) {
    case ViewSelectorMode::SimpleNearest:
      return make_simple_colorizer();
    case ViewSelectorMode::WeightedTop1:
      return make_weighted_colorizer_top1();
    case ViewSelectorMode::WeightedTopK:
      return make_weighted_colorizer_topK();
    case ViewSelectorMode::Matched:
    default:
      // Not implemented yet — fall back to simple so callers don't break.
      return make_simple_colorizer();
  }
}

EffectiveCalib effective_calib(const ImageSource& src, double cam_time) {
  // Lever + rpy always come from source scalars (no per-anchor extrinsic yet).
  // Only time_shift interpolates across anchors.
  EffectiveCalib out{src.time_shift, src.lever_arm, src.rotation_rpy};
  // < 2 anchors: scalar baseline rules. A single anchor is an inert BOOKMARK
  // -- you can pin "this config worked here" without changing runtime behavior
  // for any other frame. Interpolation requires at least 2 anchors to bracket
  // the query time and produce a meaningful drift curve. The alternative
  // (1 anchor = constant everywhere) clobbers previously-correct sections the
  // moment you place an anchor, which is the opposite of what drift-correction
  // should do.
  if (src.anchors.size() < 2) return out;
  // 2+ anchors. Expected sorted by cam_time; lower_bound finds the first
  // anchor whose cam_time >= query. Neighbors (prev, curr) bracket the query.
  const auto& anchors = src.anchors;
  auto it = std::lower_bound(anchors.begin(), anchors.end(), cam_time,
    [](const CalibAnchor& a, double t) { return a.cam_time < t; });
  if (it == anchors.begin()) { out.time_shift = anchors.front().time_shift; return out; }
  if (it == anchors.end())   { out.time_shift = anchors.back().time_shift;  return out; }
  const CalibAnchor& prev = *(it - 1);
  const CalibAnchor& curr = *it;
  const double span = curr.cam_time - prev.cam_time;
  const double u = (span > 1e-9) ? (cam_time - prev.cam_time) / span : 0.0;
  out.time_shift = prev.time_shift + u * (curr.time_shift - prev.time_shift);
  return out;
}

Eigen::Isometry3d Colorizer::effective_extrinsic(const ImageSource& src, double /*cam_time*/) {
  // Scalar pass-through today. When per-anchor lever_arm / rotation_rpy drift
  // lands, re-thread this through effective_calib() to interpolate those too.
  return build_extrinsic(src.lever_arm, src.rotation_rpy);
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
