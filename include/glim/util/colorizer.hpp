#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace glim {

/// Pinhole camera intrinsics with optional Brown-Conrady distortion.
struct PinholeIntrinsics {
  double fx = 1920.0, fy = 1920.0;  // focal length in pixels (Elgato Facecam Pro 90° FOV default)
  double cx = 1920.0, cy = 1080.0;  // principal point (image center)
  int width = 3840, height = 2160;   // image resolution
  // Brown-Conrady distortion coefficients
  double k1 = 0.0, k2 = 0.0;       // radial distortion
  double p1 = 0.0, p2 = 0.0;       // tangential distortion
  double k3 = 0.0;                  // higher-order radial
};

/// Single camera frame with EXIF-derived metadata and placed pose.
struct CameraFrame {
  std::string filepath;
  double timestamp = 0.0;           // UNIX epoch (from EXIF DateTimeOriginal + SubSec)
  double lat = 0.0, lon = 0.0, alt = 0.0;  // from EXIF GPS
  Eigen::Isometry3d T_world_cam = Eigen::Isometry3d::Identity();
  bool located = false;
};

/// A folder of images treated as one camera source.
struct ImageSource {
  std::string path;                  // folder path
  std::string name;                  // display name (folder basename)
  std::vector<CameraFrame> frames;
  Eigen::Vector3d lever_arm = Eigen::Vector3d::Zero();  // camera offset from LiDAR in sensor-local frame
  Eigen::Vector3d rotation_rpy = Eigen::Vector3d::Zero();  // camera rotation relative to LiDAR (roll, pitch, yaw) in degrees
  double time_shift = 0.0;          // seconds, slides cameras along trajectory
  PinholeIntrinsics intrinsics;     // camera model
  std::string mask_path;            // path to mask image (black = exclude from projection)
};

/// Trajectory point with timestamp for camera placement.
struct TimedPose {
  double stamp;
  Eigen::Isometry3d pose;  // T_world_lidar
};

/// Result of color projection for a set of points.
struct ColorizeResult {
  std::vector<Eigen::Vector3f> points;   // world-space positions
  std::vector<Eigen::Vector3f> colors;   // RGB [0-1]
  std::vector<float> intensities;        // original intensities (for fallback)
  size_t colored = 0;                    // points that received at least one color sample
  size_t total = 0;
};

class Colorizer {
public:
  /// Scan a folder for images (.jpg/.jpeg/.png), read EXIF GPS + timestamp.
  static ImageSource load_image_folder(const std::string& folder_path);

  /// Place cameras along a trajectory using timestamp matching.
  static int locate_by_time(ImageSource& source, const std::vector<TimedPose>& trajectory);

  /// Place cameras along a trajectory using GPS coordinates.
  static int locate_by_coordinates(ImageSource& source, const std::vector<TimedPose>& trajectory,
                                    int utm_zone, double easting_origin, double northing_origin, double alt_origin);

  /// Project colors from cameras onto world-space points.
  /// Each point gets the average RGB from all cameras that see it.
  /// @param cameras  Located camera frames to project from
  /// @param intrinsics  Pinhole camera model
  /// @param world_points  3D points in world frame
  /// @param intensities  Per-point intensities (preserved in result)
  /// @param max_range  Max distance from camera to point (skip far points)
  /// @param blend  If true, average RGB from all cameras. If false, use closest camera only.
  /// @param mask  Optional mask image (same resolution as camera). Black pixels = don't project.
  static ColorizeResult project_colors(
    const std::vector<CameraFrame>& cameras,
    const PinholeIntrinsics& intrinsics,
    const std::vector<Eigen::Vector3f>& world_points,
    const std::vector<float>& intensities,
    double max_range = 50.0,
    bool blend = true,
    double min_range = 0.0,
    const cv::Mat& mask = cv::Mat());

  /// Interpolate pose on trajectory at given timestamp.
  static Eigen::Isometry3d interpolate_pose(const std::vector<TimedPose>& trajectory, double stamp);

  /// Build T_lidar_camera from lever arm + RPY rotation (degrees).
  static Eigen::Isometry3d build_extrinsic(const Eigen::Vector3d& lever_arm, const Eigen::Vector3d& rotation_rpy);

  /// Solve camera-to-lidar extrinsic from 2D-3D correspondences.
  /// @param pts_3d  World-space 3D points clicked in the viewer
  /// @param pts_2d  Corresponding 2D pixel coords clicked in the image
  /// @param intrinsics  Camera intrinsics
  /// @param T_world_lidar  LiDAR pose at the camera's timestamp
  /// @return T_lidar_camera (transform from camera frame to lidar frame)
  static Eigen::Isometry3d solve_extrinsic(
    const std::vector<Eigen::Vector3d>& pts_3d,
    const std::vector<Eigen::Vector2d>& pts_2d,
    const PinholeIntrinsics& intrinsics,
    const Eigen::Isometry3d& T_world_lidar);

private:
  static bool read_exif(const std::string& filepath, double& timestamp, double& lat, double& lon, double& alt);
};

}  // namespace glim
