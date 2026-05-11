#include <glim/util/colmap_export.hpp>
#include <glim/util/geodetic.hpp>  // utm_inverse for EXIF/XMP lat/lon write

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <exiv2/exiv2.hpp>  // EXIF write for Metashape target (lat/lon/alt + heading + datetime)

namespace glim {

// Axis convention: our (X fwd, Y left, Z up)  →  COLMAP cv (X right, Y down, Z fwd)
static const Eigen::Matrix3d R_our_to_cv = (Eigen::Matrix3d() <<
   0, -1,  0,
   0,  0, -1,
   1,  0,  0).finished();

// Rotate world from our (X fwd, Y left, Z up) to 3DGS-style right-handed Y-up
// (X right, Y up, Z back). Applied to points + camera poses so LichtFeld et al.
// display the scene upright. Det = +1 (proper rotation, preserves handedness).
//   new_x (right) = -old_y
//   new_y (up)    =  old_z
//   new_z (back)  = -old_x
static const Eigen::Matrix3d R_yup_from_our = (Eigen::Matrix3d() <<
   0, -1,  0,
   0,  0,  1,
  -1,  0,  0).finished();

// Eigen quaternion (w, x, y, z) from 3x3 rotation; normalized.
static Eigen::Quaterniond rotmat_to_quat(const Eigen::Matrix3d& R) {
  Eigen::Quaterniond q(R);
  q.normalize();
  return q;
}

// Forward declaration -- type defined in colmap_export.hpp (already included).
static void write_geo_priors_for_camera(
    const ExportCameraFrame* c,
    const PinholeIntrinsics& k,
    const std::string& dst_img,
    const std::string& image_subdir,
    const ExportOptions& options,
    const Eigen::Matrix3d& R_world,
    const Eigen::Vector3d& effective_offset);

// ---------------------------------------------------------------------------
// EXIF / XMP helpers (RS + Metashape target writers)
// ---------------------------------------------------------------------------

// Convert a camera position in our world frame (session-local UTM, X=East,
// Y=North, Z=Up) to absolute WGS84 lat/lon/alt using the session datum.
// Returns false if the datum isn't available -- caller skips georef write.
static bool world_to_wgs84(
    const Eigen::Vector3d& world_xyz,
    const ExportOptions& opt,
    double& out_lat, double& out_lon, double& out_alt) {
  if (!opt.datum_available || opt.datum_utm_zone == 0) return false;
  const double abs_easting  = world_xyz.x() + opt.datum_utm_easting_origin;
  const double abs_northing = world_xyz.y() + opt.datum_utm_northing_origin;
  const double abs_alt      = world_xyz.z() + opt.datum_alt;
  const Eigen::Vector2d latlon = utm_inverse(abs_easting, abs_northing,
                                             opt.datum_utm_zone,
                                             opt.datum_southern_hemisphere);
  out_lat = latlon.x();
  out_lon = latlon.y();
  out_alt = abs_alt;
  return true;
}

// Camera "true heading" -- clockwise from geographic North in degrees [0, 360).
// Our world frame: X=East, Y=North, Z=Up. Camera forward (our convention) is
// the +X axis in camera frame, which becomes T_world_cam.rotation().col(0)
// in world frame. Heading is the bearing of that vector projected to the
// horizontal plane.
static double world_heading_deg(const Eigen::Isometry3d& T_world_cam) {
  const Eigen::Vector3d fwd_world = T_world_cam.rotation().col(0);
  // atan2(east, north) gives clockwise-from-north bearing.
  double h = std::atan2(fwd_world.x(), fwd_world.y()) * 180.0 / M_PI;
  while (h < 0)        h += 360.0;
  while (h >= 360.0)   h -= 360.0;
  return h;
}

// Omega / Phi / Kappa in degrees for the trajectory CSV (RS Workflow >
// Trajectory > Custom format). Convention chosen to match RS's "Computer
// vision" axes option + Euler order "zyx" (the same combo the OmniSlam
// example uses in the official RS SLAM tutorial):
//   - World frame: ENU (X=East, Y=North, Z=Up).
//   - Camera frame: Computer Vision (X=right, Y=down, Z=forward), so we
//     re-express our pose (X=fwd, Y=left, Z=up) into CV first.
//   - Decompose R_world_cv = Rz(kappa) * Ry(phi) * Rx(omega) (intrinsic
//     z-y-x). RS reconstructs R from these three values + the Euler order
//     selector + the axes-convention selector, all of which are exposed in
//     the RS Import Trajectory dialog.
// Why this and not YPR: YPR mode in RS assumes a NADIR-facing default
// camera mount (drone style). Our cameras are forward-facing, which sits
// at the gimbal-lock boundary for YPR/NED -- decomposition is ambiguous
// and "nadir + 0 pitch" maps to "looking down". OPK with CV axes has no
// platform-mount assumption and represents any camera orientation
// uniquely (forward-cam is non-degenerate).
static void world_opk_deg(const Eigen::Isometry3d& T_world_cam,
                          double& omega, double& phi, double& kappa) {
  // R_ours_from_cv: maps a vector with Computer-Vision-frame coordinates
  // to the same vector expressed in our-frame coordinates. Columns are
  // CV axes written in our frame:
  //   cv_X (right) = -ours_Y (we have left) -> (0,-1,0) in ours
  //   cv_Y (down)  = -ours_Z (we have up)   -> (0, 0,-1) in ours
  //   cv_Z (fwd)   =  ours_X                 -> (1, 0, 0) in ours
  Eigen::Matrix3d R_ours_from_cv;
  R_ours_from_cv <<
    0,  0, 1,
   -1,  0, 0,
    0, -1, 0;
  const Eigen::Matrix3d R_world_cv = T_world_cam.rotation() * R_ours_from_cv;

  // Decompose R = Rz(kappa) * Ry(phi) * Rx(omega).
  // Closed form (no gimbal lock): sp = -R[2,0], with cp = sqrt(1 - sp^2):
  //   omega = atan2(R[2,1], R[2,2])
  //   phi   = asin(-R[2,0])
  //   kappa = atan2(R[1,0], R[0,0])
  const double sp = -R_world_cv(2, 0);
  if (std::abs(sp) >= 0.9999) {
    // Gimbal lock at phi = +/- 90 deg (camera straight up or down).
    // Standard convention: pin omega = 0, fold the residual into kappa.
    phi   = (sp > 0 ? 90.0 : -90.0);
    omega = 0.0;
    kappa = std::atan2(-R_world_cv(0, 1), R_world_cv(1, 1)) * 180.0 / M_PI;
    return;
  }
  phi   = std::asin(std::clamp(sp, -1.0, 1.0)) * 180.0 / M_PI;
  omega = std::atan2(R_world_cv(2, 1), R_world_cv(2, 2)) * 180.0 / M_PI;
  kappa = std::atan2(R_world_cv(1, 0), R_world_cv(0, 0)) * 180.0 / M_PI;
}

// Write a Reality Scan-format `.xmp` sidecar next to the image. Format
// follows CapturingReality's xcr: namespace -- RS auto-detects same-named
// `.xmp` files when importing the image folder. Pose prior is "initial"
// (= "draft" mode in their UI -- cameras refine during alignment, fixed
// LiDAR provides the anchor). Position is in the same frame as the
// exported point cloud (session-local "absolute"); RS treats both as
// living in one coordinate system regardless of whether it's real-world
// UTM or session-local. Distortion model = brown3 with k1/k2/k3/p1/p2.
static bool write_rs_xmp_sidecar(
    const std::string& xmp_path,
    const Eigen::Isometry3d& T_world_cam,
    const PinholeIntrinsics& K,
    bool y_up_export,                            // matches the PLY's frame (rotated or not)
    const Eigen::Matrix3d& R_world_export,       // identity or R_yup, applied to position+rotation
    const Eigen::Vector3d& datum_offset) {       // added to position when full_utm
  // Apply the same world rotation to the camera that's applied to the points
  // so RS sees them in one frame. Match the PLY writer's order: rotate the
  // SESSION-LOCAL position first, then add the datum offset. Doing it the
  // other way (rotate after offset) would rotate the UTM-scale shift around
  // (0,0,0) and land the cameras hundreds of km away when the region has a
  // non-zero yaw. Zero datum_offset when full_utm is off, so this is a no-op
  // for the local-coords case.
  const Eigen::Vector3d pos_export = R_world_export * T_world_cam.translation() + datum_offset;
  // RS xcr:Rotation expresses the camera's CV-frame axes in world coords
  // (cam X=right, Y=down, Z=forward -- the standard photogrammetry /
  // OpenCV convention RS uses everywhere else). Our T_world_cam.rotation()
  // is cam-to-world in OUR cam frame (X=fwd, Y=left, Z=up), so we post-
  // multiply by R_ours_from_cv to insert the axis change. Without this,
  // RS reads the matrix under CV semantics and the camera ends up looking
  // straight up -- the user's "road goes to the sky" symptom on align.
  Eigen::Matrix3d R_ours_from_cv;
  R_ours_from_cv <<
    0,  0, 1,
   -1,  0, 0,
    0, -1, 0;
  const Eigen::Matrix3d R_export =
    R_world_export * T_world_cam.rotation() * R_ours_from_cv;

  std::ofstream ofs(xmp_path);
  if (!ofs) return false;
  // Force C locale -- RS parses XMP with "." decimals regardless of host
  // locale. A comma-locale system would otherwise write "425000,1234..."
  // which RS parses as 0 -> camera defaults to (0,0,0) = equator.
  ofs.imbue(std::locale::classic());
  ofs << std::setprecision(10) << std::fixed;
  // Build the rotation string up-front so we can put it on the
  // rdf:Description as an ATTRIBUTE (RS-native format) rather than as a
  // child element. RS reads xcr:Rotation off the Description tag; a
  // <xcr:Rotation>...</xcr:Rotation> child is silently ignored by RS,
  // which is why our previous XMPs were "loaded" but pose-priors did
  // nothing useful during alignment.
  std::ostringstream rot_ss;
  rot_ss.imbue(std::locale::classic());
  rot_ss << std::setprecision(10) << std::fixed;
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      rot_ss << R_export(r, c);
      if (!(r == 2 && c == 2)) rot_ss << ' ';
    }
  }
  // Same trick for the distortion coefficients (k1 k2 k3 p1 p2 0): RS-native
  // XMPs put this on the Description tag too. The trailing 0 is RS's
  // reserved slot.
  std::ostringstream dist_ss;
  dist_ss.imbue(std::locale::classic());
  dist_ss << std::setprecision(10) << std::fixed;
  dist_ss << K.k1 << ' ' << K.k2 << ' ' << K.k3 << ' '
          << K.p1 << ' ' << K.p2 << ' ' << 0;
  // RS-native XMP layout: every pose / calibration value lives as an
  // attribute on the rdf:Description tag, and ONLY xcr:Position is a
  // child element with whitespace-separated text content.
  // Reference (a sample copied from RS itself):
  //   <rdf:Description xmlns:xcr="..."
  //                    xcr:Version="3"
  //                    xcr:Rotation="-1 0 0 0 0 -1 0 -1 0"
  //                    xcr:Position-as-attribute? NO -- as child element>
  //     <xcr:Position>0.262... -2.263... 7.038...</xcr:Position>
  //   </rdf:Description>
  // Our previous emit used child elements + x/y/z attributes for Position,
  // both wrong; RS silently fell back to (0,0,0) for the position prior,
  // which surfaced as "cameras land off Ghana / Guinea" (the equator).
  ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  ofs << "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n";
  ofs << "  <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n";
  ofs << "    <rdf:Description xmlns:xcr=\"http://www.capturingreality.com/ns/xcr/1.1#\""
      << " xcr:Version=\"3\""
      << " xcr:PosePrior=\"initial\""
      << " xcr:Coordinates=\"absolute\""
      << " xcr:CalibrationPrior=\"initial\""
      << " xcr:DistortionModel=\"brown3\""
      << " xcr:FocalLength35mm=\"" << (K.fx * 36.0 / std::max(1, K.width)) << "\""
      << " xcr:PrincipalPointU=\"" << K.cx << "\""
      << " xcr:PrincipalPointV=\"" << K.cy << "\""
      << " xcr:Skew=\"0\""
      << " xcr:AspectRatio=\"1\""
      << " xcr:Rotation=\""             << rot_ss.str()  << "\""
      << " xcr:DistortionCoeficients=\"" << dist_ss.str() << "\">\n";
  ofs << "      <xcr:Position>"
      << pos_export.x() << ' ' << pos_export.y() << ' ' << pos_export.z()
      << "</xcr:Position>\n";
  ofs << "    </rdf:Description>\n";
  ofs << "  </rdf:RDF>\n";
  ofs << "</x:xmpmeta>\n";
  (void)y_up_export;  // currently unused -- R_world_export already encodes it
  return static_cast<bool>(ofs);
}

// Format a UNIX seconds-since-epoch timestamp as EXIF DateTimeOriginal
// "YYYY:MM:DD HH:MM:SS" + SubSecTimeOriginal (3 digits ms). UTC.
static void format_exif_datetime(double timestamp_unix,
                                 std::string& out_dt, std::string& out_subsec) {
  if (timestamp_unix <= 0.0) { out_dt.clear(); out_subsec.clear(); return; }
  const time_t whole = static_cast<time_t>(std::floor(timestamp_unix));
  struct tm utc_tm;
  gmtime_r(&whole, &utc_tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y:%m:%d %H:%M:%S", &utc_tm);
  out_dt.assign(buf);
  const double frac = timestamp_unix - static_cast<double>(whole);
  const int ms = std::clamp(static_cast<int>(std::round(frac * 1000.0)), 0, 999);
  char ssbuf[8]; std::snprintf(ssbuf, sizeof(ssbuf), "%03d", ms);
  out_subsec.assign(ssbuf);
}

// Embed EXIF GPS (lat/lon/alt + heading) + DateTimeOriginal in a JPG/TIFF/PNG
// using libexiv2. Used for the Metashape target -- gives Metashape WGS84
// camera positions at import time without any manual georef step.
// Silently skips on failure (corrupt image, unsupported format) so the export
// completes regardless.
static void write_exif_geo_to_image(
    const std::string& image_path,
    double lat, double lon, double alt,
    double heading_deg,                  // -1 to skip heading
    double timestamp_unix,                // <=0 to skip datetime
    const PinholeIntrinsics& K) {         // pass {} to skip focal length
  try {
    auto image = Exiv2::ImageFactory::open(image_path);
    if (image.get() == nullptr) return;
    image->readMetadata();
    Exiv2::ExifData& exif = image->exifData();

    // GPS lat/lon as rational DD/1 MM/1 SSSS/100 (XX'YY''ZZ.ZZ" format).
    auto to_dms_rationals = [](double deg) {
      const double abs_deg = std::abs(deg);
      const int    d = static_cast<int>(std::floor(abs_deg));
      const double rem_min = (abs_deg - d) * 60.0;
      const int    m = static_cast<int>(std::floor(rem_min));
      const double s = (rem_min - m) * 60.0;
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%d/1 %d/1 %d/100", d, m,
                    static_cast<int>(std::round(s * 100.0)));
      return std::string(buf);
    };
    exif["Exif.GPSInfo.GPSLatitudeRef"] = (lat >= 0) ? "N" : "S";
    exif["Exif.GPSInfo.GPSLatitude"]    = to_dms_rationals(lat);
    exif["Exif.GPSInfo.GPSLongitudeRef"] = (lon >= 0) ? "E" : "W";
    exif["Exif.GPSInfo.GPSLongitude"]   = to_dms_rationals(lon);
    {
      char buf[64]; std::snprintf(buf, sizeof(buf), "%d/100",
                                  static_cast<int>(std::round(std::abs(alt) * 100.0)));
      exif["Exif.GPSInfo.GPSAltitudeRef"] = std::string(alt >= 0 ? "0" : "1");
      exif["Exif.GPSInfo.GPSAltitude"]    = std::string(buf);
    }
    if (heading_deg >= 0.0) {
      char buf[64]; std::snprintf(buf, sizeof(buf), "%d/100",
                                  static_cast<int>(std::round(heading_deg * 100.0)));
      exif["Exif.GPSInfo.GPSImgDirectionRef"] = "T";   // True heading
      exif["Exif.GPSInfo.GPSImgDirection"]    = std::string(buf);
    }
    if (timestamp_unix > 0.0) {
      std::string dt, ss; format_exif_datetime(timestamp_unix, dt, ss);
      if (!dt.empty()) {
        exif["Exif.Photo.DateTimeOriginal"]   = dt;
        exif["Exif.Image.DateTime"]           = dt;
        if (!ss.empty()) exif["Exif.Photo.SubSecTimeOriginal"] = ss;
      }
    }
    // Focal length. Cameras are exported undistorted (PINHOLE), so K.fx is the
    // pinhole focal length in pixels. Convert to 35mm-equivalent via
    //   f_35mm = fx * 36.0 / image_width
    // This is what RS / Metashape / most SFM tools read as the initial focal
    // length guess (`Exif.Photo.FocalLengthIn35mmFilm`). Also write
    // `Exif.Photo.FocalLength` (mm, rational) using the same value -- treats
    // the sensor as 36mm (full-frame). Consistent and unambiguous.
    // Image width must be sane (>= 1). Skip if K is unset / degenerate.
    if (K.fx > 0.0 && K.width > 0) {
      const double f35mm = K.fx * 36.0 / static_cast<double>(K.width);
      const int    f35mm_short = std::clamp(static_cast<int>(std::round(f35mm)), 1, 65535);
      exif["Exif.Photo.FocalLengthIn35mmFilm"] = std::to_string(f35mm_short);
      char buf[64]; std::snprintf(buf, sizeof(buf), "%d/100",
                                  static_cast<int>(std::round(f35mm * 100.0)));
      exif["Exif.Photo.FocalLength"] = std::string(buf);
    }
    image->writeMetadata();
  } catch (const Exiv2::Error&) {
    // Silently swallow -- export continues without georef on this image.
  } catch (...) {
    // Same.
  }
}

// Per-camera dispatch: each metadata channel is independently togglable
// so the user can mix-and-match (e.g. RS folder structure + EXIF GPS for
// testing whether RS reads EXIF when XMP is also present, or Metashape
// folder + both EXIF and XMP, etc.). Defaults are set by the UI per
// target_layout but the user has the final say.
//   emit_xmp_sidecar -> writes <basename>.xmp next to the image with the
//                       RS xcr: schema (full pose, brown3 distortion).
//                       Plain XML write, no library; works for any source.
//   emit_exif_gps    -> embeds GPS lat/lon/alt + true heading + DateTime
//                       in the just-copied image via libexiv2. Needs
//                       datum_available to convert session-local world
//                       coords -> WGS84.
// Both silently skip when their preconditions miss (datum / library /
// image format).
static void write_geo_priors_for_camera(
    const ExportCameraFrame* c,
    const PinholeIntrinsics& k,
    const std::string& dst_img,
    const std::string& image_subdir,
    const ExportOptions& options,
    const Eigen::Matrix3d& R_world,
    const Eigen::Vector3d& effective_offset) {
  namespace fs = boost::filesystem;
  if (!c) return;

  if (options.emit_xmp_sidecar) {
    const std::string stem = fs::path(c->export_name).stem().string();
    const std::string xmp_path = options.output_dir + "/" + image_subdir + "/" + stem + ".xmp";
    // effective_offset is R_yup * datum (or just datum when Y-up off, or
    // zero when full_utm off). Caller computes it once at the top of
    // write_colmap_export so points / cameras / XMP all share the same
    // rotated-frame offset.
    write_rs_xmp_sidecar(xmp_path, c->T_world_cam, k,
                          options.rotate_to_y_up, R_world, effective_offset);
  }

  if (options.emit_exif_gps) {
    // EXIF GPS embedded into the just-copied JPG. Skip if datum missing
    // (no way to compute WGS84 from session-local world coords).
    if (!options.datum_available) return;
    if (c->T_world_cam.matrix().hasNaN()) return;
    double lat = 0.0, lon = 0.0, alt = 0.0;
    if (!world_to_wgs84(c->T_world_cam.translation(), options, lat, lon, alt))
      return;
    const double heading = world_heading_deg(c->T_world_cam);
    // Timestamp = c->timestamp (UTC seconds since epoch + source time_shift).
    // 0.0 -> writer skips DateTimeOriginal silently.
    write_exif_geo_to_image(dst_img, lat, lon, alt, heading, c->timestamp, k);
  }
}

ExportStats write_colmap_export(
  const ExportBounds2D& bounds,
  const std::vector<PointSource>& point_sources,
  const std::vector<ExportCameraFrame>& cameras,
  const std::vector<PinholeIntrinsics>& intrinsics_per_source,
  const std::vector<CameraType>& camera_type_per_source,
  const std::vector<bool>& is_virtual_per_source,
  const ExportOptions& options,
  std::string* error_msg) {
  auto type_of = [&](int si) {
    return (si >= 0 && si < static_cast<int>(camera_type_per_source.size()))
      ? camera_type_per_source[si] : CameraType::Pinhole;
  };
  // True when the source is a virtual (LiDAR-rendered) camera. BlocksExchange
  // <Accuracy> uses hardcoded tight sigmas for these so BA treats them as
  // ground-truth anchors independent of the UI slider.
  auto is_virtual = [&](int si) -> bool {
    return (si >= 0 && si < static_cast<int>(is_virtual_per_source.size()))
      ? is_virtual_per_source[si] : false;
  };

  ExportStats stats;
  auto seterr = [&](const std::string& m) { if (error_msg) *error_msg = m; };

  if (options.output_dir.empty()) { seterr("output_dir is empty"); return stats; }
  namespace fs = boost::filesystem;
  fs::create_directories(options.output_dir);
  // Per-target folder names + mask filename suffix:
  //   COLMAP        -> images/ + masks/<stem>.png        (no suffix)
  //   RealityScan   -> .geometry/ + .mask/<stem>.png     (RS auto-detects
  //                    layers by separator-prefixed folder names; "masks"
  //                    plural / no separator does NOT trigger detection)
  //   Metashape     -> images/ + masks/<stem>_mask.png   (Metashape's
  //                    Import Masks default template; user points at the
  //                    folder manually but matches by suffix)
  // Strings are picked once and propagated through the writer; the per-
  // camera image/mask copy spots reference these instead of hardcoding.
  std::string image_subdir = "images";
  std::string mask_subdir  = "masks";
  std::string mask_suffix  = "";  // inserted before .png; e.g. "_mask" for Metashape
  // Point cloud destination subdirectory:
  //   COLMAP    -> sparse/0/   (their multi-model convention; LichtFeld /
  //                gsplat / nerfstudio look here)
  //   RS / MS   -> point_cloud/  (regular folder; the sparse/0/ tree only
  //                makes sense alongside cameras.txt+images.txt, neither
  //                of which non-COLMAP targets need)
  std::string pc_subdir = "sparse/0";
  switch (options.target_layout) {
    case ExportOptions::TargetLayout::RealityScan:
      image_subdir = ".geometry";
      mask_subdir  = ".mask";
      pc_subdir    = "point_cloud";
      break;
    case ExportOptions::TargetLayout::Metashape:
      mask_suffix = "_mask";
      pc_subdir   = "point_cloud";
      break;
    case ExportOptions::TargetLayout::BlocksExchange:
      // BE references images by path so layout is generic. We pick the
      // most universally-compatible folder shape (images/ + masks/) and
      // the simplest mask naming so any downstream tool that opens BE can
      // resolve paths without surprises.
      pc_subdir = "point_cloud";
      break;
    case ExportOptions::TargetLayout::Colmap:
    default:
      break;
  }
  fs::create_directories(options.output_dir + "/" + image_subdir);
  // COLMAP sparse model lives in sparse/0/ (multi-model convention; LichtFeld,
  // gsplat, nerfstudio all look here). For non-COLMAP targets we route the
  // point-cloud output to point_cloud/ instead and skip sparse/0/ entirely.
  const std::string sparse_dir   = options.output_dir + "/sparse/0";
  const std::string pc_full_dir  = options.output_dir + "/" + pc_subdir;
  if (options.emit_colmap) fs::create_directories(sparse_dir);
  fs::create_directories(pc_full_dir);  // PLYs always go SOMEWHERE

  // --- Filter points by bounds, per source ---
  // Each PointSource gets its own kept_points list so we can write one PLY
  // per source -- the merge bug fix. The kept count for stats is summed
  // across all sources (we report total points written, regardless of how
  // many files they're split across).
  struct KeptSource {
    const PointSource* src;
    std::vector<const ColoredPoint*> kept;
  };
  std::vector<KeptSource> kept_per_source;
  kept_per_source.reserve(point_sources.size());
  for (const auto& src : point_sources) {
    KeptSource ks; ks.src = &src;
    ks.kept.reserve(src.points.size());
    for (const auto& p : src.points) if (bounds.contains_xy(p.xyz)) ks.kept.push_back(&p);
    kept_per_source.push_back(std::move(ks));
  }
  // Aggregated kept_points for places that still want a flat view (none
  // currently, but kept the alias name so the diff stays small).
  std::vector<const ColoredPoint*> kept_points;
  for (const auto& ks : kept_per_source) {
    kept_points.insert(kept_points.end(), ks.kept.begin(), ks.kept.end());
  }

  // --- Filter cameras: position inside bounds + overlap_margin ---
  const float marg = std::max(0.0f, options.overlap_margin_m);
  ExportBounds2D expanded{
    bounds.x_min - marg, bounds.x_max + marg,
    bounds.y_min - marg, bounds.y_max + marg,
    bounds.z_min, bounds.z_max,
    bounds.yaw_deg
  };
  std::vector<const ExportCameraFrame*> kept_cams;
  kept_cams.reserve(cameras.size());
  for (const auto& c : cameras) {
    Eigen::Vector3f pos = c.T_world_cam.translation().cast<float>();
    if (expanded.contains_xy(pos)) kept_cams.push_back(&c);
  }

  // --- Compute origin offset (tile center) ---
  Eigen::Vector3d origin_offset = Eigen::Vector3d::Zero();
  if (options.re_origin) {
    origin_offset = Eigen::Vector3d(
      0.5 * (bounds.x_min + bounds.x_max),
      0.5 * (bounds.y_min + bounds.y_max),
      0.5 * (bounds.z_min + bounds.z_max));
  }
  stats.origin_offset = origin_offset;

  // --- Full UTM offset ---
  // When full_utm + datum_available, ADD the session datum's UTM origin
  // to every exported world coord so the output lands in real-world UTM
  // eastings/northings (matching EXIF GPS). Computed once, applied at
  // each per-point/per-camera write site below.
  Eigen::Vector3d datum_offset = Eigen::Vector3d::Zero();
  if (options.full_utm && options.datum_available) {
    datum_offset = Eigen::Vector3d(
      options.datum_utm_easting_origin,
      options.datum_utm_northing_origin,
      options.datum_alt);
  }

  // World rotation applied AFTER re-origin. Only R_yup remains:
  //   R_yup: Z-up -> Y-up (our -> 3DGS convention). Skipped if option off.
  // bounds.yaw_deg is a 2D bounding-box orientation used ONLY by
  // contains_xy() to crop a tilted rectangle around a road or feature.
  // It NEVER rotates output coordinates -- doing so would put PLY/XMP/
  // cameras in a yaw-rotated frame that's inconsistent with EXIF GPS
  // (which writes raw lat/lon via world_to_wgs84, no yaw applied), and
  // for session-local exports rotating the output frame doesn't add
  // anything useful either (downstream tools just see a self-consistent
  // scene either way).
  // R_world = R_yup  (applied to points via P_out = R_world * (P - origin)).
  const Eigen::Matrix3d R_yup_or_id = options.rotate_to_y_up
    ? R_yup_from_our
    : Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R_world = R_yup_or_id;
  // Datum offset has to be rotated by R_yup the same way the cloud is, so
  // points + cameras + offset all land in one consistent frame. Without
  // this, Y-up + full_utm mixes Z-up offset coords with Y-up cloud coords
  // and the cloud ends up shifted to a nonsensical location ("road to
  // sky" in Metashape, Africa in RS).
  // No yaw component participates -- gizmo yaw is a crop orientation
  // only, never a coord transform (see R_world above).
  // No-op when full_utm is off (datum_offset is zero) or rotate_to_y_up
  // is off (R_yup_or_id is identity).
  const Eigen::Vector3d effective_offset = R_yup_or_id * datum_offset;
  // XGrids workflow for COLMAP+RS: keep COLMAP scene-file coords
  // (cameras.txt / images.txt / points3D.{ply,txt}) in session-local
  // frame so the user enters the datum as the TRANSLATION in RS's
  // COLMAP-import dialog. Why: when scene cameras are at UTM scale
  // (~10^6), small per-frame quaternion noise (~10^-3 in matrix
  // entries) gets multiplied by C-magnitude, producing translation
  // noise of meters-to-tens-of-meters in t_cv_world. RS's COLMAP
  // pipeline appears to use float32 for the C = -R^T * t reconstruct,
  // and that precision loss leaves visible jitter on what should be a
  // smooth trajectory. Local coords keep magnitudes small so the
  // double->float conversion is precision-safe.
  // BlocksExchange / RS-XMP / Metashape paths still bake the UTM
  // offset because Metashape uses doubles end-to-end.
  const bool colmap_local =
    (options.target_layout == ExportOptions::TargetLayout::Colmap)
    && options.full_utm
    && options.datum_available;
  const Eigen::Vector3d effective_offset_scene =
    colmap_local ? Eigen::Vector3d::Zero() : effective_offset;

  // --- Write cameras.txt ---
  // One entry per unique source_idx used by kept_cams (all images share a model).
  // IDs are 0-based to match Metashape/COLMAP convention.
  //   PINHOLE (fx fy cx cy)        -- when images are undistorted during export
  //   OPENCV  (fx fy cx cy k1 k2 p1 p2) -- when raw distorted images are emitted
  // Gated on options.emit_colmap so the panel can target ONLY BlocksExchange
  // (or just point clouds) without writing COLMAP files. The image-copy +
  // mask-copy steps below are NOT gated -- both COLMAP and BE need the
  // image files at "<out>/images/<name>", and writing them once covers
  // both targets.
  std::set<int> used_sources;
  for (const auto* c : kept_cams) used_sources.insert(c->source_idx);
  // Count camera definitions (unique intrinsics) regardless of whether
  // cameras.txt is actually written. RS / Metashape targets skip the
  // COLMAP text bundle but the camera definitions still participate
  // (XMP sidecars / EXIF GPS reference them); reporting 0 here would be
  // misleading.
  stats.cameras_written = used_sources.size();

  if (options.emit_colmap) {
    std::ofstream cams_ofs(sparse_dir + "/cameras.txt");
    if (!cams_ofs) { seterr("failed to open cameras.txt"); return stats; }
    // Force C locale -- COLMAP / RS parse with "." as decimal separator. If
    // the host process is in a comma-decimal locale the file would be
    // mis-parsed (cameras land at the equator).
    cams_ofs.imbue(std::locale::classic());
    cams_ofs << "# Camera list with one line of data per camera:\n";
    cams_ofs << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    cams_ofs << "# Number of cameras: " << used_sources.size() << "\n";
    for (int si : used_sources) {
      if (si < 0 || si >= static_cast<int>(intrinsics_per_source.size())) continue;
      const auto& k = intrinsics_per_source[si];
      if (options.export_undistorted) {
        cams_ofs << si << " PINHOLE " << k.width << " " << k.height
                 << " " << std::setprecision(10) << k.fx
                 << " " << k.fy
                 << " " << k.cx
                 << " " << k.cy
                 << "\n";
      } else {
        cams_ofs << si << " OPENCV " << k.width << " " << k.height
                 << " " << std::setprecision(10) << k.fx
                 << " " << k.fy
                 << " " << k.cx
                 << " " << k.cy
                 << " " << k.k1
                 << " " << k.k2
                 << " " << k.p1
                 << " " << k.p2
                 << "\n";
      }
      // (cameras_written set above; not double-counted here)
    }
  }

  // --- Write images.txt + undistort image files + masks ---
  // images.txt is gated on emit_colmap; the per-camera loop ALSO populates
  // the export_cams vector (used by BE writer) and copies image files
  // (used by both targets). So we open the stream conditionally and use a
  // disposable null stream when COLMAP isn't requested -- keeps the per-
  // camera loop simple, no scoping gymnastics.
  std::ofstream imgs_real;
  std::ostringstream imgs_null;  // /dev/null sink for the emit_colmap=false case
  std::ostream* imgs_ptr = &imgs_null;
  if (options.emit_colmap) {
    imgs_real.open(sparse_dir + "/images.txt");
    if (!imgs_real) { seterr("failed to open images.txt"); return stats; }
    imgs_real.imbue(std::locale::classic());  // force "." decimals (RS / COLMAP parse C locale)
    imgs_ptr = &imgs_real;
  }
  imgs_null.imbue(std::locale::classic());
  std::ostream& imgs = *imgs_ptr;
  imgs << "# Image list with two lines of data per image:\n";
  imgs << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
  imgs << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
  imgs << "# Number of images: " << kept_cams.size() << "\n";

  // Masks go to a masks/ sibling folder so Postshot / Lichtfeld pick them up.
  const bool any_masks = [&]{ for (const auto* c : kept_cams) if (!c->source_mask_path.empty()) return true; return false; }();
  if (any_masks) fs::create_directories(options.output_dir + "/" + mask_subdir);

  // Cache undistort maps per source (same intrinsics -> same map; compute once per source).
  struct UndistortMap { cv::Mat mapx, mapy; int w = 0, h = 0; bool valid = false; };
  std::vector<UndistortMap> undist_maps(intrinsics_per_source.size());

  // Collected per-image camera data for the optional BlocksExchange export.
  // R_cv_world + t_cv_world are POST-transform (re-origin + Y-up world rotation
  // already applied); the world-space center is derived on demand as C = -R^T * t.
  struct ExportCam {
    int source_idx;
    std::string export_name;
    Eigen::Matrix3d R_cv_world;
    Eigen::Vector3d t_cv_world;
  };
  std::vector<ExportCam> export_cams;
  if (options.export_blocks_exchange) export_cams.reserve(kept_cams.size());

  int next_image_id = 0;  // 0-based to match Metashape/COLMAP convention
  for (const auto* c : kept_cams) {
    const int cam_id = c->source_idx;  // shared per source, 0-based

    // COLMAP stores T_cam_world (world -> camera, OpenCV convention).
    Eigen::Isometry3d T_our_world = c->T_world_cam.inverse();                 // our cam <- world
    Eigen::Matrix3d R_cv_world = R_our_to_cv * T_our_world.rotation();        // cv cam <- world_our
    Eigen::Vector3d t_cv_world = R_our_to_cv * T_our_world.translation();
    if (options.re_origin) {
      // Points shifted by -origin_offset -> t_new = t_old + R * origin_offset
      // (keeps P_cam = R*(P_world - offset) + t_new identical to R*P_world + t_old).
      t_cv_world += R_cv_world * origin_offset;
    }
    // World-axis rotation (Z-up -> Y-up, region yaw alignment, etc): we now
    // apply R_world to the SESSION-LOCAL frame (before the UTM shift), so:
    //   P_world_new = R_world * P_world_old
    //   R_cv_world_new = R_cv_world_old * R_world^T   (translation unchanged at
    //   this stage). No-op when R_world is identity.
    R_cv_world = R_cv_world * R_world.transpose();
    // Full UTM: AFTER the rotation, the point cloud is shifted by
    // +effective_offset (= R_yup * datum_offset) in the new rotated frame.
    // To keep P_cam unchanged the camera translation must absorb the same
    // shift through the already-rotated camera matrix:
    //   t_new = t_rotated - R_cv_world_new * effective_offset.
    // Using effective_offset (not raw datum_offset) is critical when Y-up is
    // on: it puts the datum shift in the rotated frame so cameras / cloud /
    // offset all live in one consistent coordinate system. With Y-up off,
    // effective_offset == datum_offset and behaviour is unchanged.
    if (options.full_utm && options.datum_available) {
      // effective_offset_scene == effective_offset for non-Colmap targets
      // and zero for Colmap+full_utm (XGrids workflow). See the comment
      // around effective_offset_scene definition for the rationale.
      t_cv_world -= R_cv_world * effective_offset_scene;
    }
    Eigen::Quaterniond q = rotmat_to_quat(R_cv_world);

    imgs << next_image_id
         << " " << std::setprecision(10)
         << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
         << " " << t_cv_world.x() << " " << t_cv_world.y() << " " << t_cv_world.z()
         << " " << cam_id
         << " " << c->export_name
         << "\n";
    imgs << "\n";  // empty POINTS2D line (no feature track info for 3DGS init)
    next_image_id++;

    if (options.export_blocks_exchange) {
      export_cams.push_back({c->source_idx, c->export_name, R_cv_world, t_cv_world});
    }

    // Write image to images/. Two paths:
    //  - export_undistorted: read -> cv::remap undistort -> cv::imwrite (re-encode).
    //    Must COPY (no symlink: the source image is distorted, can't point at it).
    //  - !export_undistorted: copy or symlink the raw file; distortion stays for
    //    the downstream tool to handle via the OPENCV model in cameras.txt.
    const std::string dst_img = options.output_dir + "/" + image_subdir + "/" + c->export_name;
    const int si = c->source_idx;
    if (si >= 0 && si < static_cast<int>(intrinsics_per_source.size())) {
      const auto& k = intrinsics_per_source[si];
      const bool has_distortion =
        (k.k1 != 0.0 || k.k2 != 0.0 || k.k3 != 0.0 || k.p1 != 0.0 || k.p2 != 0.0);

      if (!options.export_undistorted) {
        // Raw copy / symlink path (distortion stays in images, handled by OPENCV intrinsics).
        if (options.copy_images) {
          try { fs::copy_file(c->source_image_path, dst_img, fs::copy_option::overwrite_if_exists); stats.images_copied++; }
          catch (const fs::filesystem_error& e) { seterr(std::string("image copy failed: ") + e.what()); }
        } else {
          try {
            if (fs::exists(dst_img)) fs::remove(dst_img);
            fs::create_symlink(c->source_image_path, dst_img);
            stats.images_copied++;
          } catch (const fs::filesystem_error& e) { seterr(std::string("image symlink failed: ") + e.what()); }
        }
        // Per-image mask copy is SKIPPED when single_mask is on -- the
        // shared per-source mask is written once after the loop instead.
        if (!options.single_mask && !c->source_mask_path.empty() && fs::exists(c->source_mask_path)) {
          const std::string stem = fs::path(c->export_name).stem().string();
          const std::string mask_dst = options.output_dir + "/" + mask_subdir + "/" + stem + mask_suffix + ".png";
          try { fs::copy_file(c->source_mask_path, mask_dst, fs::copy_option::overwrite_if_exists); stats.masks_copied++; }
          catch (const fs::filesystem_error& e) { seterr(std::string("mask copy failed: ") + e.what()); }
        }
        write_geo_priors_for_camera(c, k, dst_img, image_subdir, options, R_world, effective_offset);
        continue;  // next camera frame
      }

      // Undistort path. cv::undistort keeps the original K (no cropping/scaling),
      // so the PINHOLE fx/fy/cx/cy we wrote to cameras.txt is the correct model.
      cv::Mat img = cv::imread(c->source_image_path);
      if (img.empty()) {
        seterr(std::string("failed to read image: ") + c->source_image_path);
        continue;
      }
      cv::Mat out;
      if (has_distortion) {
        // Build / reuse remap tables for this source.
        if (!undist_maps[si].valid ||
            undist_maps[si].w != img.cols || undist_maps[si].h != img.rows) {
          cv::Mat K = (cv::Mat_<double>(3, 3) << k.fx, 0, k.cx, 0, k.fy, k.cy, 0, 0, 1);
          cv::Mat D = (cv::Mat_<double>(1, 5) << k.k1, k.k2, k.p1, k.p2, k.k3);
          cv::initUndistortRectifyMap(K, D, cv::Mat(), K, img.size(), CV_16SC2,
                                      undist_maps[si].mapx, undist_maps[si].mapy);
          undist_maps[si].w = img.cols; undist_maps[si].h = img.rows;
          undist_maps[si].valid = true;
        }
        cv::remap(img, out, undist_maps[si].mapx, undist_maps[si].mapy, cv::INTER_LINEAR);
      } else {
        out = img;
      }
      // Preserve JPEG quality (tools like LichtFeld are sensitive to re-compression).
      std::vector<int> jpg_params = {cv::IMWRITE_JPEG_QUALITY, 95};
      if (cv::imwrite(dst_img, out, jpg_params)) stats.images_copied++;
      else seterr(std::string("failed to write undistorted image: ") + dst_img);

      // Undistort + write mask (nearest-neighbor to preserve binary values).
      // Skipped when single_mask is on -- the shared per-source mask is
      // undistorted once after the loop instead.
      if (!options.single_mask && !c->source_mask_path.empty() && fs::exists(c->source_mask_path)) {
        const std::string stem = fs::path(c->export_name).stem().string();
        const std::string mask_dst = options.output_dir + "/" + mask_subdir + "/" + stem + mask_suffix + ".png";
        cv::Mat mask = cv::imread(c->source_mask_path, cv::IMREAD_UNCHANGED);
        if (!mask.empty()) {
          cv::Mat mask_out;
          if (has_distortion) {
            cv::remap(mask, mask_out, undist_maps[si].mapx, undist_maps[si].mapy, cv::INTER_NEAREST);
          } else {
            mask_out = mask;
          }
          if (cv::imwrite(mask_dst, mask_out)) stats.masks_copied++;
        }
      }
      write_geo_priors_for_camera(c, k, dst_img, image_subdir, options, R_world, effective_offset);
    }
  }

  // --- Single shared mask per source (when options.single_mask is on) ---
  // Replaces the per-image masks that the per-camera loop above skipped.
  // For each unique source_idx that has a non-empty source_mask_path, copy
  // the mask file ONCE to "<mask_subdir>/srcN_mask.png". User points
  // Metashape's Import Masks dialog at the file with "From File" + "Apply
  // to all cameras of the same camera group" -- one click per source
  // covers every image. BE references the same file path-wise.
  if (options.single_mask) {
    namespace fs = boost::filesystem;
    if (!fs::is_directory(options.output_dir + "/" + mask_subdir)) {
      fs::create_directories(options.output_dir + "/" + mask_subdir);
    }
    std::set<int> sources_with_mask;
    for (const auto* c : kept_cams) {
      if (c->source_mask_path.empty()) continue;
      if (!fs::exists(c->source_mask_path)) continue;
      if (sources_with_mask.count(c->source_idx)) continue;
      const std::string mask_name = "src" + std::to_string(c->source_idx) + "_mask.png";
      const std::string mask_dst  = options.output_dir + "/" + mask_subdir + "/" + mask_name;
      try {
        fs::copy_file(c->source_mask_path, mask_dst, fs::copy_option::overwrite_if_exists);
        stats.masks_copied++;
        sources_with_mask.insert(c->source_idx);
      } catch (const fs::filesystem_error& e) {
        seterr(std::string("single mask copy failed: ") + e.what());
      }
    }
  }

  // --- Trajectory CSV (Reality Scan: Workflow > Trajectory) ---
  // RS imports `flight_log.csv` and uses the rows as pose priors. Schema:
  //   Image X/Lon Y/Lat Z/Alt Omega Phi Kappa
  //   X/LonAccuracy Y/LatAccuracy Z/AltAccuracy
  //   OmegaAccuracy PhiAccuracy KappaAccuracy
  // - lat/lon: WGS84 from world_to_wgs84 (datum required, silent skip);
  // - alt: per the session's datum (we don't apply EGM undulation);
  // - OPK: see world_opk_deg() -- world frame ENU, camera frame Computer
  //   Vision (X=right, Y=down, Z=forward), Euler order zyx intrinsic.
  // - accuracies: 1-sigma uncertainties RS uses as BA priors. Position
  //   sigma is given in metres but our X/Y columns are lon/lat in degrees,
  //   so we convert metres -> degrees per row (latitude shrinks lon-deg
  //   per metre by 1/cos(lat)). Rotation accuracies are degrees.
  // The user MUST set the matching options in the RS import dialog:
  //   File format: Custom
  //   Custom format description (paste verbatim):
  //     image x/lon y/lat z/alt omega phi kappa x/lonaccuracy y/lataccuracy z/altaccuracy omegaaccuracy phiaccuracy kappaaccuracy
  //   Euler angles order: zyx
  //   Camera axes convention: Computer vision
  // (Comment block above the data rows repeats the description so the file
  // is self-documenting on revisit.)
  if (options.emit_flight_log && options.datum_available) {
    namespace fs = boost::filesystem;
    const std::string csv_path = options.output_dir + "/flight_log.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
      seterr("failed to open " + csv_path);
    } else {
      csv.imbue(std::locale::classic());  // force "." decimals
      const double pos_sigma_m   = options.pose_pos_sigma_m;
      const double rot_sigma_deg = options.pose_rot_sigma_deg;
      // Approx metres-per-degree for lat (constant) and per-row for lon.
      // 111320 m/deg is the WGS84 mean meridian, accurate enough for BA
      // weighting (the user's pos sigma is itself only a guess).
      const double m_per_deg_lat = 111320.0;
      const double lat_acc_deg = pos_sigma_m / m_per_deg_lat;
      csv << "# Reality Scan trajectory import (Workflow > Trajectory)\n";
      csv << "#   File format: Custom\n";
      csv << "#   Custom format description (paste verbatim):\n";
      csv << "#     image x/lon y/lat z/alt omega phi kappa "
             "x/lonaccuracy y/lataccuracy z/altaccuracy "
             "omegaaccuracy phiaccuracy kappaaccuracy\n";
      csv << "#   Euler angles order: zyx\n";
      csv << "#   Camera axes convention: Computer vision\n";
      csv << "# Image X/Lon Y/Lat Z/Alt Omega Phi Kappa "
             "X/LonAccuracy Y/LatAccuracy Z/AltAccuracy "
             "OmegaAccuracy PhiAccuracy KappaAccuracy\n";
      csv << std::fixed;
      for (const auto* c : kept_cams) {
        if (c->T_world_cam.matrix().hasNaN()) continue;
        double lat = 0.0, lon = 0.0, alt = 0.0;
        if (!world_to_wgs84(c->T_world_cam.translation(), options, lat, lon, alt))
          continue;
        double omega = 0.0, phi = 0.0, kappa = 0.0;
        world_opk_deg(c->T_world_cam, omega, phi, kappa);
        const double cos_lat = std::cos(lat * M_PI / 180.0);
        const double lon_acc_deg =
          (std::abs(cos_lat) > 1e-6)
            ? (pos_sigma_m / (m_per_deg_lat * std::abs(cos_lat)))
            : (pos_sigma_m / m_per_deg_lat);  // poles fallback
        csv << c->export_name
            << ' ' << std::setprecision(9) << lon
            << ' ' << std::setprecision(9) << lat
            << ' ' << std::setprecision(3) << alt
            << ' ' << std::setprecision(4) << omega
            << ' ' << std::setprecision(4) << phi
            << ' ' << std::setprecision(4) << kappa
            << ' ' << std::setprecision(9) << lon_acc_deg
            << ' ' << std::setprecision(9) << lat_acc_deg
            << ' ' << std::setprecision(3) << pos_sigma_m
            << ' ' << std::setprecision(4) << rot_sigma_deg
            << ' ' << std::setprecision(4) << rot_sigma_deg
            << ' ' << std::setprecision(4) << rot_sigma_deg
            << '\n';
      }
    }
  }

  // --- Write per-source PLY files (one per PointSource) ---
  // Each non-empty source gets its own "<filename_stem>.ply" under the
  // target's point-cloud subdir (sparse/0/ for COLMAP, point_cloud/ for
  // RS / Metashape). Field columns (color, intensity, normal) toggled
  // per-source. PLY emission is unconditional -- the cloud is useful to
  // every downstream tool (mesh, 3DGS, RS LiDAR import, etc.). The
  // accompanying .txt sidecar IS gated on emit_colmap because it's a
  // COLMAP-text-format thing only.
  // Single transform used by both the PLY writer and the COLMAP text
  // sidecar -- keeping it in one lambda prevents the two paths from
  // drifting out of sync (the "cloud lands in Italy" bug came from
  // exactly this: rotate-then-shift was correct in one writer, wrong in
  // the other). Order matters: rotate the SESSION-LOCAL frame first,
  // then add the UTM datum offset. Doing it the other way rotates UTM-
  // scale coords around (0,0,0) when there's a non-zero region yaw and
  // catapults the cloud hundreds of km away.
  auto transform_point = [&](const Eigen::Vector3f& xyz_in) -> Eigen::Vector3d {
    Eigen::Vector3d p = xyz_in.cast<double>();
    if (options.re_origin) p -= origin_offset;
    p = R_world * p;
    // Scene-file points use effective_offset_scene -- zero for Colmap
    // target + full_utm (XGrids local coords), normal effective_offset
    // for everything else. Keeps points consistent with cameras in the
    // same scene file.
    if (options.full_utm && options.datum_available) p += effective_offset_scene;
    return p;
  };
  for (const auto& ks : kept_per_source) {
    const auto& src = *ks.src;
    if (ks.kept.empty()) continue;
    const std::string stem = src.filename_stem.empty() ? "points3D" : src.filename_stem;
    const std::string ply_path = pc_full_dir + "/" + stem + ".ply";

    std::ofstream ofs(ply_path, std::ios::binary);
    if (!ofs) { seterr("failed to open " + ply_path); return stats; }
    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << ks.kept.size() << "\n";
    // XYZ precision: double when Full UTM is on (real-world UTM eastings
    // up to ~10^7 m would be quantized to ~0.5 m chunks if stored as
    // float32 -- the cloud literally LOOKS voxelized because the cm-level
    // detail rounds to a coarse grid). Mirror the existing PLY tile export
    // (offline_viewer.cpp line ~18719) which uses double for the same
    // reason. Float32 stays for session-local coords where precision is
    // mm-level and 3DGS / nerfstudio expect single-precision PLY init.
    const bool xyz_double = options.full_utm && options.datum_available;
    if (xyz_double) {
      ofs << "property double x\nproperty double y\nproperty double z\n";
    } else {
      ofs << "property float x\nproperty float y\nproperty float z\n";
    }
    if (src.emit_color) {
      ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    }
    if (src.emit_normal) {
      ofs << "property float nx\nproperty float ny\nproperty float nz\n";
    }
    if (src.emit_intensity) {
      ofs << "property float intensity\n";
    }
    ofs << "end_header\n";
    for (const auto* p : ks.kept) {
      const Eigen::Vector3d p_world = transform_point(p->xyz);
      if (xyz_double) {
        const double x = p_world.x(), y = p_world.y(), z = p_world.z();
        ofs.write(reinterpret_cast<const char*>(&x), 8);
        ofs.write(reinterpret_cast<const char*>(&y), 8);
        ofs.write(reinterpret_cast<const char*>(&z), 8);
      } else {
        const float x = static_cast<float>(p_world.x());
        const float y = static_cast<float>(p_world.y());
        const float z = static_cast<float>(p_world.z());
        ofs.write(reinterpret_cast<const char*>(&x), 4);
        ofs.write(reinterpret_cast<const char*>(&y), 4);
        ofs.write(reinterpret_cast<const char*>(&z), 4);
      }
      if (src.emit_color) {
        const uint8_t r = static_cast<uint8_t>(std::clamp(static_cast<int>(p->rgb.x() * 255.0f), 0, 255));
        const uint8_t g = static_cast<uint8_t>(std::clamp(static_cast<int>(p->rgb.y() * 255.0f), 0, 255));
        const uint8_t b = static_cast<uint8_t>(std::clamp(static_cast<int>(p->rgb.z() * 255.0f), 0, 255));
        ofs.put(static_cast<char>(r));
        ofs.put(static_cast<char>(g));
        ofs.put(static_cast<char>(b));
      }
      if (src.emit_normal) {
        // Rotate normal into output frame (R_world rotates positions; same
        // rotation applies to direction vectors since translation cancels).
        const Eigen::Vector3d n_out = R_world * p->normal.cast<double>();
        const float nx = static_cast<float>(n_out.x());
        const float ny = static_cast<float>(n_out.y());
        const float nz = static_cast<float>(n_out.z());
        ofs.write(reinterpret_cast<const char*>(&nx), 4);
        ofs.write(reinterpret_cast<const char*>(&ny), 4);
        ofs.write(reinterpret_cast<const char*>(&nz), 4);
      }
      if (src.emit_intensity) {
        const float in_val = p->intensity;
        ofs.write(reinterpret_cast<const char*>(&in_val), 4);
      }
    }

    // Optional COLMAP-text sidecar. COLMAP-format only -- skipped when the
    // target isn't COLMAP (emit_colmap=false) since the .txt format is
    // meaningless to RS / Metashape consumers and would just clutter their
    // exports. Sidecar still lands in pc_full_dir alongside the PLY when
    // emitted.
    if (src.emit_text_sidecar && options.emit_colmap) {
      const std::string txt_path = pc_full_dir + "/" + stem + ".txt";
      std::ofstream ofs_txt(txt_path);
      if (ofs_txt) {
        ofs_txt.imbue(std::locale::classic());  // force "." decimals
        // Header-only mode: write the standard COLMAP comment block but
        // claim "Number of points: 0" with no data rows. RS's SLAM
        // tutorial recommends this when feeding the cloud separately via
        // Import LiDAR Scan; avoids the parse error that surfaces as
        // "file not found" in the RS import dialog.
        const bool header_only = options.points3d_text_header_only;
        ofs_txt << "# 3D point list with one line of data per point:\n";
        ofs_txt << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
        ofs_txt << "# Number of points: " << (header_only ? 0 : ks.kept.size()) << "\n";
        if (!header_only) {
          ofs_txt << std::setprecision(6) << std::fixed;
          size_t pid = 1;
          for (const auto* p : ks.kept) {
            const Eigen::Vector3d p_world = transform_point(p->xyz);
            const int r = std::clamp(static_cast<int>(p->rgb.x() * 255.0f), 0, 255);
            const int g = std::clamp(static_cast<int>(p->rgb.y() * 255.0f), 0, 255);
            const int b = std::clamp(static_cast<int>(p->rgb.z() * 255.0f), 0, 255);
            ofs_txt << pid << " " << p_world.x() << " " << p_world.y() << " " << p_world.z()
                    << " " << r << " " << g << " " << b << " 0\n";
            pid++;
          }
        }
      }
    }

    stats.points_written += ks.kept.size();
  }
  // --- Optional BlocksExchange XML export ---
  // ContextCapture / RealityCapture / Metashape-compatible block file. Same
  // camera convention as COLMAP (world -> OpenCV camera frame), so the R/t we
  // already computed drop straight in. Center = -R^T * t (camera position in
  // the exported world). Per-source intrinsics grouped in <Photogroup>.
  if (options.export_blocks_exchange) {
    const std::string xml_path = options.output_dir + "/blocks_exchange.xml";
    std::ofstream ofs(xml_path);
    if (!ofs) { seterr("failed to open blocks_exchange.xml"); }
    else {
      ofs.imbue(std::locale::classic());  // force "." decimals
      ofs << std::setprecision(10);
      ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
      // Version 2.1 is what Metashape's importer targets most reliably; 3.x
      // works in newer CC but Metashape's parser complains about fields it
      // hasn't back-ported.
      ofs << "<BlocksExchange version=\"2.1\">\n";
      // Omit <SpatialReferenceSystems> + <SRSId> entirely: per ContextCapture
      // spec, the default assumption is a local Cartesian coordinate system,
      // which is exactly what our coords are. Declaring a custom LOCAL_CS WKT
      // triggers Metashape's datum-transformation lookup and fails with
      // "unsupported datum transformation" -- the project's default SRS
      // (usually WGS84) can't be transformed to our named Local.
      ofs << "  <Block>\n";
      ofs << "    <Name>GLIM export</Name>\n";
      ofs << "    <Photogroups>\n";

      // One <Photogroup> per unique source_idx (shared intrinsics).
      std::set<int> used_sources_be;
      for (const auto& ec : export_cams) used_sources_be.insert(ec.source_idx);
      for (int si : used_sources_be) {
        if (si < 0 || si >= static_cast<int>(intrinsics_per_source.size())) continue;
        const auto& k = intrinsics_per_source[si];
        const bool is_spherical = (type_of(si) == CameraType::Spherical);
        ofs << "      <Photogroup>\n";
        ofs << "        <Name>src" << si << "</Name>\n";
        ofs << "        <ImageDimensions>\n";
        ofs << "          <Width>"  << k.width  << "</Width>\n";
        ofs << "          <Height>" << k.height << "</Height>\n";
        ofs << "        </ImageDimensions>\n";
        if (is_spherical) {
          // Metashape reads <CameraModelType>Spherical</CameraModelType> for
          // full 360 equirect. No focal / principal point / distortion: the
          // projection is fixed, f derived as w/(2*pi), cx/cy at image centre.
          ofs << "        <CameraModelType>Spherical</CameraModelType>\n";
        } else {
          ofs << "        <CameraModelType>Perspective</CameraModelType>\n";
          ofs << "        <CameraOrientation>XRightYDown</CameraOrientation>\n";
          // Focal length in pixels (BlocksExchange accepts FocalLengthPixels).
          ofs << "        <FocalLengthPixels>" << 0.5 * (k.fx + k.fy) << "</FocalLengthPixels>\n";
          // Aspect ratio fy/fx (1.0 when square pixels).
          ofs << "        <AspectRatio>" << (k.fx > 0 ? k.fy / k.fx : 1.0) << "</AspectRatio>\n";
          ofs << "        <PrincipalPoint>\n";
          ofs << "          <x>" << k.cx << "</x>\n";
          ofs << "          <y>" << k.cy << "</y>\n";
          ofs << "        </PrincipalPoint>\n";
          // Distortion: zero when exporting undistorted images, else source Brown-Conrady.
          const double k1 = options.export_undistorted ? 0.0 : k.k1;
          const double k2 = options.export_undistorted ? 0.0 : k.k2;
          const double k3 = options.export_undistorted ? 0.0 : k.k3;
          const double p1 = options.export_undistorted ? 0.0 : k.p1;
          const double p2 = options.export_undistorted ? 0.0 : k.p2;
          ofs << "        <Distortion>\n";
          ofs << "          <K1>" << k1 << "</K1>\n";
          ofs << "          <K2>" << k2 << "</K2>\n";
          ofs << "          <K3>" << k3 << "</K3>\n";
          ofs << "          <P1>" << p1 << "</P1>\n";
          ofs << "          <P2>" << p2 << "</P2>\n";
          ofs << "        </Distortion>\n";
        }

        // All photos belonging to this source.
        int photo_id = 0;
        for (const auto& ec : export_cams) {
          if (ec.source_idx != si) continue;
          const Eigen::Vector3d C = -ec.R_cv_world.transpose() * ec.t_cv_world;
          ofs << "        <Photo>\n";
          ofs << "          <Id>" << photo_id++ << "</Id>\n";
          ofs << "          <ImagePath>images/" << ec.export_name << "</ImagePath>\n";
          ofs << "          <Pose>\n";
          ofs << "            <Rotation>\n";
          for (int r = 0; r < 3; r++) {
            for (int cc = 0; cc < 3; cc++) {
              ofs << "              <M_" << r << cc << ">" << ec.R_cv_world(r, cc) << "</M_" << r << cc << ">\n";
            }
          }
          ofs << "            </Rotation>\n";
          ofs << "            <Center>\n";
          ofs << "              <x>" << C.x() << "</x>\n";
          ofs << "              <y>" << C.y() << "</y>\n";
          ofs << "              <z>" << C.z() << "</z>\n";
          ofs << "            </Center>\n";
          // Position + rotation accuracy hints -- ContextCapture spec + Metashape
          // read these as BA constraints. Virtual (LiDAR-rendered) cameras
          // always get tight sigmas so BA locks them; real cameras honour the
          // UI slider when pose priors are on.
          const bool virt = is_virtual(ec.source_idx);
          if (virt || options.emit_pose_priors) {
            const double pos_sigma = virt ? 0.001 : options.pose_pos_sigma_m;
            const double rot_sigma = virt ? 0.01  : options.pose_rot_sigma_deg;
            ofs << "            <Accuracy>\n";
            ofs << "              <Horizontal>" << pos_sigma << "</Horizontal>\n";
            ofs << "              <Vertical>"   << pos_sigma << "</Vertical>\n";
            ofs << "              <Rotation>"   << rot_sigma << "</Rotation>\n";
            ofs << "            </Accuracy>\n";
          }
          ofs << "          </Pose>\n";
          ofs << "        </Photo>\n";
        }
        ofs << "      </Photogroup>\n";
      }
      ofs << "    </Photogroups>\n";
      ofs << "  </Block>\n";
      ofs << "</BlocksExchange>\n";
    }
  }

  // --- Manifest (tile metadata; handy for later stitching + debugging) ---
  {
    nlohmann::json m;
    m["bounds_world"] = {
      {"x", {bounds.x_min, bounds.x_max}},
      {"y", {bounds.y_min, bounds.y_max}},
      {"z", {bounds.z_min, bounds.z_max}},
    };
    m["overlap_margin_m"] = options.overlap_margin_m;
    m["re_origined"] = options.re_origin;
    m["origin_offset"] = {origin_offset.x(), origin_offset.y(), origin_offset.z()};
    m["voxel_size_m"] = options.voxel_size_m;
    m["stats"] = {
      {"points", stats.points_written},
      {"cameras", stats.cameras_written},
      {"images", stats.images_copied},
      {"masks", stats.masks_copied},
    };
    m["axis_convention"] = "COLMAP (X right, Y down, Z forward) — converted from our (X fwd, Y left, Z up)";
    // XGrids workflow: when COLMAP scene is in local coords, expose the
    // datum offset so the user can paste it into the RS COLMAP-import
    // dialog as the "translation" field. Order matches the dialog's
    // X / Y / Z (= UTM easting / northing / altitude in the session's
    // CRS). Only emitted when colmap_local is on; for other targets
    // the UTM is already baked into the scene files and there's no
    // offset to apply at import time.
    if (colmap_local) {
      m["offset_xyz"] = {
        options.datum_utm_easting_origin,
        options.datum_utm_northing_origin,
        options.datum_alt
      };
      m["offset_xyz_note"] =
        "RS COLMAP import: paste these three numbers into the dialog's "
        "translation field; leave rotation at 0 deg (or per Y-up setting). "
        "UTM zone " + std::to_string(options.datum_utm_zone)
        + (options.datum_southern_hemisphere ? "S" : "N");
    }
    std::ofstream ofs(options.output_dir + "/manifest.json");
    ofs << std::setw(2) << m << std::endl;
  }

  return stats;
}

}  // namespace glim
