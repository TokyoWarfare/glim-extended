#include <glim/util/virtual_cameras.hpp>

#include <algorithm>
#include <cmath>

namespace glim {

std::vector<VirtualCameraAnchor> place_virtual_cameras(
    const std::vector<TrajRecord>& trajectory,
    float interval_m) {
  std::vector<VirtualCameraAnchor> out;
  if (trajectory.size() < 2 || interval_m <= 0.0f) return out;

  const double total_dist = trajectory.back().cumulative_dist;
  if (total_dist <= 0.0) return out;

  // Walk trajectory at constant distance spacing. For each target distance,
  // find the bracketing trajectory records and interpolate pose + stamp.
  // Uses linear interp on translation and SLERP on rotation -- tight enough
  // for virtual-camera placement (sub-metre error at the waypoint is fine,
  // the important thing is that consecutive anchors are roughly interval_m
  // apart along the real path, not straight-line).
  size_t i = 0;
  int anchor_idx = 0;
  for (double d = 0.0; d <= total_dist; d += interval_m) {
    while (i + 1 < trajectory.size() && trajectory[i + 1].cumulative_dist < d) ++i;
    if (i + 1 >= trajectory.size()) break;
    const TrajRecord& a = trajectory[i];
    const TrajRecord& b = trajectory[i + 1];
    const double seg = b.cumulative_dist - a.cumulative_dist;
    const double t = (seg > 1e-9) ? std::clamp((d - a.cumulative_dist) / seg, 0.0, 1.0) : 0.0;

    Eigen::Isometry3d pose;
    pose.translation() = a.pose.translation() + t * (b.pose.translation() - a.pose.translation());
    Eigen::Quaterniond qa(a.pose.rotation());
    Eigen::Quaterniond qb(b.pose.rotation());
    pose.linear() = qa.slerp(t, qb).toRotationMatrix();

    VirtualCameraAnchor anchor;
    anchor.index = anchor_idx++;
    anchor.stamp = a.stamp + t * (b.stamp - a.stamp);
    anchor.T_world_cam = pose;
    anchor.cumulative_dist_m = d;
    out.push_back(anchor);
  }
  return out;
}

}  // namespace glim
