#include <glim/viewer/interactive_viewer.hpp>
#include <glim/util/post_processing.hpp>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <thread>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include <glim/mapping/sub_map.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <glim/util/config.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/trajectory_manager.hpp>

#include <glim/viewer/interactive/manual_loop_close_modal.hpp>
#include <glim/viewer/interactive/bundle_adjustment_modal.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>
#include <gtsam_points/ann/kdtree.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_points/config.hpp>
#include <gtsam_points/factors/integrated_matching_cost_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/optimizers/isam2_ext.hpp>
#include <gtsam_points/optimizers/isam2_result_ext.hpp>

#include <glk/thin_lines.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/spdlog_sink.hpp>
#include <guik/model_control.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <guik/camera/fps_camera_control.hpp>

namespace glim {
using gtsam::symbol_shorthand::X;

// Compute the [lo_pct, hi_pct] percentile range from a vector of samples.
// The vector is sorted in-place; caller should keep a persistent copy or accept the sort.
static Eigen::Vector2f percentile_range(std::vector<float>& samples, float lo_pct = 0.01f, float hi_pct = 0.99f) {
  if (samples.size() < 2) {
    return samples.empty() ? Eigen::Vector2f(0.0f, 1.0f) : Eigen::Vector2f(samples[0], samples[0]);
  }
  std::sort(samples.begin(), samples.end());
  const int lo = static_cast<int>(lo_pct * static_cast<float>(samples.size() - 1));
  const int hi = static_cast<int>(hi_pct * static_cast<float>(samples.size() - 1));
  return Eigen::Vector2f(samples[lo], samples[hi]);
}

// Frustum-AABB intersection test using the 6 clip planes from view-projection matrix.
bool InteractiveViewer::frustum_test_aabb(const Eigen::Matrix4f& vp, const Eigen::AlignedBox3f& box) {
  // Extract 6 frustum planes from VP (rows of VP ± row[3])
  Eigen::Vector4f planes[6];
  for (int i = 0; i < 3; i++) {
    planes[i * 2 + 0] = vp.row(3) + vp.row(i);      // left/bottom/near
    planes[i * 2 + 1] = vp.row(3) - vp.row(i);      // right/top/far
  }
  // Test AABB against each plane: if box is entirely behind any plane, it's outside
  for (int i = 0; i < 6; i++) {
    const Eigen::Vector4f& p = planes[i];
    // Find the "most positive" corner relative to the plane normal
    Eigen::Vector3f positive_corner;
    positive_corner.x() = (p.x() >= 0) ? box.max().x() : box.min().x();
    positive_corner.y() = (p.y() >= 0) ? box.max().y() : box.min().y();
    positive_corner.z() = (p.z() >= 0) ? box.max().z() : box.min().z();
    if (p.head<3>().dot(positive_corner) + p.w() < 0.0f) {
      return false;  // entirely outside this plane
    }
  }
  return true;  // inside or intersecting all planes
}

InteractiveViewer::InteractiveViewer() : logger(create_module_logger("viewer")) {
  glim::Config config(glim::GlobalConfig::get_config_path("config_viewer"));

  kill_switch = false;
  request_to_terminate = false;
  request_to_clear = false;

#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#else
  num_threads = 1;
#endif

  color_mode = 0;
  aux_cmap_range = Eigen::Vector2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
  gps_time_base = 0.0;
  coord_scale = 1.0f;
  sphere_scale = 0.5f;

  draw_current = true;
  draw_traj = true;
  draw_points = true;
  draw_factors = false;
  draw_spheres = true;
  draw_coords = true;

  min_overlap = 0.2f;
  cont_optimize = false;

  needs_session_merge = false;

  enable_partial_rendering = config.param("interactive_viewer", "enable_partial_rendering", false);
  partial_rendering_budget = config.param("interactive_viewer", "partial_rendering_budget", 1024);

  points_alpha = config.param("interactive_viewer", "points_alpha", 1.0);
  factors_alpha = config.param("interactive_viewer", "factors_alpha", 1.0);

  point_size = config.param("interactive_viewer", "point_size", 0.025);
  point_size_metric = config.param("interactive_viewer", "point_size_metric", false);
  point_shape_circle = config.param("interactive_viewer", "point_shape_circle", true);

  z_range = config.param("interactive_viewer", "default_z_range", Eigen::Vector2d(-2.0, 4.0)).cast<float>();

  trajectory.reset(new TrajectoryManager);

  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;

  OdometryEstimationCallbacks::on_new_frame.add(std::bind(&InteractiveViewer::odometry_on_new_frame, this, _1));
  GlobalMappingCallbacks::on_insert_submap.add(std::bind(&InteractiveViewer::globalmap_on_insert_submap, this, _1));
  GlobalMappingCallbacks::on_update_submaps.add(std::bind(&InteractiveViewer::globalmap_on_update_submaps, this, _1));
  GlobalMappingCallbacks::on_smoother_update.add(std::bind(&InteractiveViewer::globalmap_on_smoother_update, this, _1, _2, _3));
  GlobalMappingCallbacks::on_smoother_update_result.add(std::bind(&InteractiveViewer::globalmap_on_smoother_update_result, this, _1, _2));

  thread = std::thread([this] { viewer_loop(); });
}

InteractiveViewer::~InteractiveViewer() {
  // Save memory settings
  {
    const std::string dir = std::string(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") + "/.glim";
    boost::filesystem::create_directories(dir);
    std::ofstream ofs(dir + "/memory_settings.json");
    if (ofs) {
      ofs << "{\n";
      ofs << "  \"lod_enabled\": " << (lod_enabled ? "true" : "false") << ",\n";
      ofs << "  \"lod_hd_enabled\": " << (lod_hd_enabled ? "true" : "false") << ",\n";
      ofs << "  \"lod_show_bboxes\": " << (lod_show_bboxes ? "true" : "false") << ",\n";
      ofs << "  \"lod_sd_range\": " << lod_sd_range << ",\n";
      ofs << "  \"lod_hd_range\": " << lod_hd_range << ",\n";
      ofs << "  \"lod_vram_budget_mb\": " << lod_vram_budget_mb << "\n";
      ofs << "}\n";
    }
  }

  // Stop LOD worker before killing the viewer thread
  lod_worker_kill = true;
  if (lod_worker_thread.joinable()) {
    lod_worker_thread.join();
  }

  kill_switch = true;
  if (thread.joinable()) {
    thread.join();
  }
}

/**
 * @brief Main viewer loop
 */
void InteractiveViewer::viewer_loop() {
  auto viewer = guik::LightViewer::instance(Eigen::Vector2i(2560, 1440));
  viewer->enable_info_buffer();
  viewer->enable_vsync();
  // Install FPV as the startup camera control -- iridescence's canvas defaults
  // to OrbitCameraControlXY. OfflineViewer defaults camera_mode_sel to 1 (FPV)
  // so the UI must match on first paint.
  {
    auto fps = viewer->use_fps_camera_control(60.0);
    fps->set_pose(Eigen::Vector3f(0.0f, 0.0f, 20.0f), 0.0f, -45.0f);
    fps->set_translation_speed(1.0f);
  }
  viewer->shader_setting().add("z_range", z_range);

  viewer->shader_setting().set_point_size(point_size);

  if (point_size_metric) {
    viewer->shader_setting().set_point_scale_metric();
  }

  if (point_shape_circle) {
    viewer->shader_setting().set_point_shape_circle();
  }

  if (enable_partial_rendering) {
    viewer->enable_partial_rendering(0.1);
    viewer->shader_setting().add("dynamic_object", 1);
  }

  viewer->register_ui_callback("display_settings", [this] {
    if (!show_display_settings) return;
    auto viewer = guik::LightViewer::instance();
    ImGui::SetNextWindowSize(ImVec2(250, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Display Settings", &show_display_settings)) {
      bool changed = false;
      float ps = static_cast<float>(point_size);
      if (ImGui::DragFloat("Point size", &ps, 0.001f, 0.001f, 0.5f, "%.3f")) {
        point_size = ps;
        changed = true;
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Size of rendered points.\nSmaller = finer detail, larger = more visible.");
      if (changed) {
        viewer->shader_setting().set_point_size(point_size);
      }
      float pa = static_cast<float>(points_alpha);
      if (ImGui::SliderFloat("Point alpha", &pa, 0.05f, 1.0f, "%.2f")) {
        points_alpha = pa;
        changed = true;
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Point opacity. Lower = more transparent.");
    }
    ImGui::End();
  });
  viewer->register_ui_callback("source_finder", [this] {
    if (!source_finder_active) return;
    auto viewer = guik::LightViewer::instance();

    // Draw gizmo in the main viewport (skip when ICP modal is open to avoid ImGuizmo conflict)
    if (!ImGui::IsPopupOpen("manual loop close")) {
      // Set gizmo operation: translate XYZ for cylinder, translate XYZ + rotate YZ for box
      source_finder_gizmo->set_gizmo_operation(source_finder_shape == 0 ? (1 | 2 | 4) : (1 | 2 | 4 | 16 | 32));

      const auto display_size = ImGui::GetIO().DisplaySize;
      source_finder_gizmo->draw_gizmo(
        0, 0, static_cast<int>(display_size.x), static_cast<int>(display_size.y),
        viewer->view_matrix(), viewer->projection_matrix());

      // Sync stored values from gizmo (after draw, gizmo has latest user input)
      const Eigen::Matrix4f new_m = source_finder_gizmo->model_matrix();
      const Eigen::Vector3f new_pos = new_m.block<3, 1>(0, 3);
      // Extract Euler YZ from rotation matrix
      const float new_yaw = std::atan2(new_m(1, 0), new_m(0, 0)) * 180.0f / static_cast<float>(M_PI);
      const float new_pitch = std::asin(std::clamp(-new_m(2, 0), -1.0f, 1.0f)) * 180.0f / static_cast<float>(M_PI);
      bool gizmo_changed = false;
      if ((new_pos - source_finder_pos).squaredNorm() > 1e-6f) {
        source_finder_pos = new_pos;
        gizmo_changed = true;
      }
      if (std::abs(new_yaw - source_finder_yaw) > 0.1f) {
        source_finder_yaw = new_yaw;
        gizmo_changed = true;
      }
      if (std::abs(new_pitch - source_finder_pitch) > 0.1f) {
        source_finder_pitch = new_pitch;
        gizmo_changed = true;
      }
      if (gizmo_changed) {
        source_finder_update_cylinder();
        if (source_finder_mode == 0) source_finder_scan_fast();
      }
    }

    ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Source Finder", &source_finder_active)) {
      ImGui::Combo("Mode", &source_finder_mode, "Fast (bbox)\0Precise (points)\0");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Fast: uses bounding boxes, updates live.\nPrecise: checks actual points, click Scan to run.");
      bool shape_changed = false;
      shape_changed |= ImGui::Combo("Shape", &source_finder_shape, "Cylinder\0Box\0");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cylinder: radial probe for single features.\nBox: oriented rectangle for linear features (row of trees, curb).");
      ImGui::Separator();

      bool changed = shape_changed;
      changed |= ImGui::DragFloat3("Position", source_finder_pos.data(), 0.2f);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Probe center (world coordinates).\nRight-click a new point to re-center.");
      if (source_finder_shape == 0) {
        changed |= ImGui::DragFloat("Radius", &source_finder_radius, 0.01f, 0.1f, 2.0f, "%.2f m");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Horizontal search radius.");
      } else {
        changed |= ImGui::DragFloat("Length", &source_finder_length, 0.5f, 1.0f, 200.0f, "%.1f m");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Box extent along the yaw direction.");
        changed |= ImGui::DragFloat("Width", &source_finder_width, 0.1f, 0.5f, 20.0f, "%.1f m");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Box extent perpendicular to yaw direction.");
        changed |= ImGui::DragFloat("Yaw", &source_finder_yaw, 1.0f, -180.0f, 180.0f, "%.0f deg");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Box rotation around Z axis (degrees).");
      }
      changed |= ImGui::DragFloat("Height", &source_finder_height, 0.2f, 1.0f, 100.0f, "%.1f m");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Vertical extent (centered on position).");
      if (changed) {
        // Push UI values to gizmo matrix (Rz * Ry)
        const float yr = source_finder_yaw * static_cast<float>(M_PI) / 180.0f;
        const float pr = source_finder_pitch * static_cast<float>(M_PI) / 180.0f;
        Eigen::Matrix3f rot = (Eigen::AngleAxisf(yr, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pr, Eigen::Vector3f::UnitY())).toRotationMatrix();
        Eigen::Matrix4f gizmo_m = Eigen::Matrix4f::Identity();
        gizmo_m.block<3, 3>(0, 0) = rot;
        gizmo_m.block<3, 1>(0, 3) = source_finder_pos;
        source_finder_gizmo->set_model_matrix(gizmo_m);
        source_finder_update_cylinder();
        if (source_finder_mode == 0) source_finder_scan_fast();
      }

      if (source_finder_mode == 1) {
        if (ImGui::Button("Scan")) {
          source_finder_scan_precise();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Check every point in every submap.\nMore accurate but slower.");
      }

      ImGui::Separator();
      ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "%zu submaps found", source_finder_hits.size());

      // Group hits by sequence continuity
      auto teams = glim::group_by_continuity(source_finder_hits);
      if (!teams.empty()) {

        // Apply swap: determine target/source indices
        const int tgt_idx = source_finder_teams_swapped ? 1 : 0;
        const int src_idx = source_finder_teams_swapped ? 0 : 1;

        // Display teams with role labels
        for (size_t ti = 0; ti < teams.size(); ti++) {
          std::string label;
          for (int id : teams[ti]) {
            if (!label.empty()) label += ", ";
            label += std::to_string(id);
          }
          const char* role = (ti == static_cast<size_t>(tgt_idx)) ? " [Target]" : (ti == static_cast<size_t>(src_idx)) ? " [Source]" : " (ignored)";
          const ImVec4 team_color = (ti == static_cast<size_t>(tgt_idx)) ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f) : (ti == static_cast<size_t>(src_idx)) ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 0.7f);
          ImGui::TextColored(team_color, "Team %zu (%zu)%s: %s", ti + 1, teams[ti].size(), role, label.c_str());
        }
      }

      // Auto-align button — needs at least 2 teams
      if (teams.size() >= 2) {
        const int tgt_idx = source_finder_teams_swapped ? 1 : 0;
        const int src_idx = source_finder_teams_swapped ? 0 : 1;

        ImGui::Separator();
        if (ImGui::Button("Swap teams")) {
          source_finder_teams_swapped = !source_finder_teams_swapped;
          source_finder_color_hits();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Swap target and source team roles.");
        ImGui::SameLine();
        if (ImGui::Button("Auto-align teams")) {
          const auto& tgt_team = teams[tgt_idx];
          const auto& src_team = teams[src_idx];

          // Clean up visual noise before opening ICP modal
          viewer->remove_drawable("team_lines");
          viewer->remove_drawable("identify_line");
          // Restore sphere colors to session defaults
          static const float sc2[][3] = {
            {1.0f, 0.0f, 0.0f}, {1.0f, 0.85f, 0.0f}, {0.0f, 0.8f, 0.2f},
            {1.0f, 0.6f, 0.0f}, {0.8f, 0.0f, 0.8f}, {0.0f, 0.8f, 0.8f},
          };
          for (int pi = 0; pi < static_cast<int>(submaps.size()); pi++) {
            if (!submaps[pi]) continue;
            const int ci = submaps[pi]->session_id % 6;
            const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submaps[pi]->id);
            const Eigen::Affine3f sp = submap_poses[pi].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
            viewer->update_drawable("sphere_" + std::to_string(submaps[pi]->id), glk::Primitives::sphere(),
              guik::FlatColor(sc2[ci][0], sc2[ci][1], sc2[ci][2], 0.5f, sp).add("info_values", info).make_transparent());
          }

          // Helper: merge a team's points + intensities into a single cloud
          auto merge_team = [&](const std::vector<int>& team) -> gtsam_points::PointCloudCPU::Ptr {
            const int ref = team[team.size() / 2];
            const Eigen::Isometry3d ref_pose = submap_poses[ref];
            std::vector<Eigen::Vector4d> pts;
            std::vector<double> ints;
            for (int si : team) {
              const auto& sm = submaps[si];
              if (!sm || !sm->frame) continue;
              const Eigen::Isometry3d T = ref_pose.inverse() * submap_poses[si];
              const bool has_int = sm->frame->intensities != nullptr;
              for (size_t pi = 0; pi < sm->frame->size(); pi++) {
                const Eigen::Vector3d p = T * sm->frame->points[pi].head<3>();
                pts.push_back(Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0));
                ints.push_back(has_int ? sm->frame->intensities[pi] : 0.0);
              }
            }
            auto cloud = std::make_shared<gtsam_points::PointCloudCPU>();
            cloud->num_points = pts.size();
            cloud->points_storage = std::move(pts);
            cloud->points = cloud->points_storage.data();
            cloud->intensities_storage = std::move(ints);
            cloud->intensities = cloud->intensities_storage.data();
            return cloud;
          };

          int tgt_ref = tgt_team[tgt_team.size() / 2];
          int src_ref = src_team[src_team.size() / 2];
          const Eigen::Isometry3d tgt_ref_pose = submap_poses[tgt_ref];
          const Eigen::Isometry3d src_ref_pose = submap_poses[src_ref];
          auto tgt_merged = merge_team(tgt_team);
          auto src_merged = merge_team(src_team);

          logger->info("[Auto-align] Team 1 (target): {} submaps, {} pts | Team 2 (source): {} submaps, {} pts",
            tgt_team.size(), tgt_merged->num_points, src_team.size(), src_merged->num_points);

          // Set up modal
          lc_target_frame_id = tgt_ref;
          lc_source_frame_id = src_ref;
          lc_source_group = src_team;
          lc_target_group = tgt_team;

          manual_loop_close_modal->set_target(X(tgt_ref), tgt_merged, tgt_ref_pose);
          manual_loop_close_modal->target_gps_sigma = (tgt_ref < static_cast<int>(submap_gps_sigma.size())) ? submap_gps_sigma[tgt_ref] : -1.0f;

          manual_loop_close_modal->set_source(X(src_ref), src_merged, src_ref_pose);
          manual_loop_close_modal->source_gps_sigma = (src_ref < static_cast<int>(submap_gps_sigma.size())) ? submap_gps_sigma[src_ref] : -1.0f;

          // HD callback for both teams
          manual_loop_close_modal->load_hd_callback = [this, tgt_team, src_team, tgt_ref, src_ref]()
            -> std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr> {
            auto merge_hd = [&](const std::vector<int>& team, int ref_id) -> gtsam_points::PointCloudCPU::Ptr {
              const Eigen::Isometry3d ref_pose = submap_poses[ref_id];
              std::vector<Eigen::Vector4d> all_pts;
              std::vector<double> all_ints;
              for (int si : team) {
                auto hd = load_hd_for_submap(si, false);  // points+intensity, covs deferred
                if (!hd) continue;
                const Eigen::Isometry3d T = ref_pose.inverse() * submap_poses[si];
                const bool has_int = hd->intensities != nullptr;
                for (size_t pi = 0; pi < hd->size(); pi++) {
                  const Eigen::Vector3d p = T * hd->points[pi].head<3>();
                  all_pts.push_back(Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0));
                  all_ints.push_back(has_int ? hd->intensities[pi] : 0.0);
                }
              }
              if (all_pts.empty()) return nullptr;
              auto merged = std::make_shared<gtsam_points::PointCloudCPU>();
              merged->num_points = all_pts.size();
              merged->points_storage = std::move(all_pts);
              merged->points = merged->points_storage.data();
              merged->intensities_storage = std::move(all_ints);
              merged->intensities = merged->intensities_storage.data();
              return merged;
            };
            return {merge_hd(tgt_team, tgt_ref), merge_hd(src_team, src_ref)};
          };
          if (session_hd_paths.empty()) manual_loop_close_modal->load_hd_callback = nullptr;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open ICP alignment between the two teams.\nUse 'Swap teams' to change target/source roles.");
      }
    }
    ImGui::End();
    if (!source_finder_active) {
      // Window was closed via X button
      source_finder_hits.clear();
      viewer->remove_drawable("source_finder_cylinder");
      viewer->remove_drawable("team_lines");
      viewer->remove_drawable("identify_line");
      static const float sc[][3] = {
        {1.0f, 0.0f, 0.0f}, {1.0f, 0.85f, 0.0f}, {0.0f, 0.8f, 0.2f},
        {1.0f, 0.6f, 0.0f}, {0.8f, 0.0f, 0.8f}, {0.0f, 0.8f, 0.8f},
      };
      for (int pi = 0; pi < static_cast<int>(submaps.size()); pi++) {
        if (!submaps[pi]) continue;
        const int ci = submaps[pi]->session_id % 6;
        const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submaps[pi]->id);
        const Eigen::Affine3f sp = submap_poses[pi].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
        viewer->update_drawable("sphere_" + std::to_string(submaps[pi]->id), glk::Primitives::sphere(),
          guik::FlatColor(sc[ci][0], sc[ci][1], sc[ci][2], 0.5f, sp).add("info_values", info).make_transparent());
      }
    }
  });
  viewer->register_ui_callback("selection", [this] { drawable_selection(); });
  viewer->register_ui_callback("on_click", [this] { on_click(); });
  viewer->register_ui_callback("context_menu", [this] { context_menu(); });
  viewer->register_ui_callback("run_modals", [this] { run_modals(); });
  viewer->register_ui_callback("logging", guik::create_logger_ui(glim::get_ringbuffer_sink(), 0.5));

  viewer->register_drawable_filter("filter", [this](const std::string& name) {
    const auto starts_with = [](const std::string& name, const std::string& pattern) {
      return name.size() < pattern.size() ? false : std::equal(pattern.begin(), pattern.end(), name.begin());
    };

    if (!draw_current && name == "current") {
      return false;
    }

    if (!draw_traj && starts_with(name, "traj")) {
      return false;
    }

    if (!draw_points && starts_with(name, "submap_")) {
      return false;
    }
    if (lod_hide_all_submaps && (starts_with(name, "submap_") || starts_with(name, "bbox_"))) {
      return false;
    }
    // HD-only mode: hide SD submaps (those at LOD SD, not HD)
    if (lod_hd_only && starts_with(name, "submap_")) {
      // Extract submap id and check if it's SD (not HD)
      const int sid = std::stoi(name.substr(7));
      for (int ri = 0; ri < static_cast<int>(render_states.size()); ri++) {
        if (ri < static_cast<int>(submaps.size()) && submaps[ri] && submaps[ri]->id == sid) {
          if (render_states[ri].current_lod != SubmapLOD::HD) return false;
          break;
        }
      }
    }
    if (starts_with(name, "bbox_") && (!draw_points || !lod_show_bboxes)) {
      return false;
    }
    if (!draw_factors && name == "factors") {
      return false;
    }
    if (!draw_spheres && starts_with(name, "sphere_")) {
      return false;
    }
    if (!draw_coords && starts_with(name, "coord_")) {
      return false;
    }
    if (!draw_frames && name == "frame_coords") {
      return false;
    }

    return true;
  });

  // Per-frame LOD update — always active, handles distance/frustum streaming
  viewer->register_ui_callback("lod_update", [this] {
    if (submaps.empty()) return;
    if (!lod_enabled && !lod_load_full_sd && !lod_load_full_hd) return;

    auto viewer = guik::LightViewer::instance();
    render_states.resize(submaps.size());

    const Eigen::Matrix4f view_mat = viewer->view_matrix();
    const Eigen::Matrix4f proj_mat = viewer->projection_matrix();
    const Eigen::Matrix4f vp_matrix = proj_mat * view_mat;
    const Eigen::Vector3f camera_pos = -(view_mat.block<3, 3>(0, 0).transpose() * view_mat.block<3, 1>(0, 3));
    const size_t budget_bytes = static_cast<size_t>(lod_vram_budget_mb * 1024 * 1024);

    static const float sc[][3] = {
      {1,0,0},{0,0.5f,1},{0,0.8f,0.2f},{1,0.6f,0},{0.8f,0,0.8f},{0,0.8f,0.8f}
    };

    for (int i = 0; i < static_cast<int>(submaps.size()); i++) {
      const auto& submap = submaps[i];
      auto& rs = render_states[i];
      if (!rs.bbox_computed) continue;
      if (hidden_sessions.count(submap->session_id)) continue;  // skip unloaded sessions

      const Eigen::Affine3f submap_pose = submap_poses[i].cast<float>();
      const float dist = (submap_pose.translation() - camera_pos).norm();
      const bool in_frustum = frustum_test_aabb(vp_matrix, rs.world_bbox);
      const std::string submap_name = "submap_" + std::to_string(submap->id);
      const std::string bbox_name = "bbox_" + std::to_string(submap->id);
      const int ci = submap->session_id % 6;

      // Determine desired LOD
      SubmapLOD desired_lod;
      if (lod_load_full_hd && hd_available) {
        desired_lod = SubmapLOD::HD;
      } else if (lod_load_full_sd) {
        desired_lod = SubmapLOD::SD;
      } else if (lod_hd_enabled && hd_available && dist <= lod_hd_range) {
        desired_lod = SubmapLOD::HD;
      } else if (!in_frustum && dist > lod_sd_range) {
        desired_lod = SubmapLOD::UNLOADED;
      } else if (dist > lod_sd_range * 1.2f) {
        desired_lod = SubmapLOD::BBOX;
      } else if (dist <= lod_sd_range) {
        desired_lod = SubmapLOD::SD;
      } else {
        desired_lod = (rs.current_lod >= SubmapLOD::LOADING) ? rs.current_lod : SubmapLOD::BBOX;
      }

      // Budget enforcement for promotions
      if (desired_lod == SubmapLOD::SD && rs.current_lod < SubmapLOD::LOADING) {
        if (total_gpu_bytes >= budget_bytes) {
          desired_lod = SubmapLOD::BBOX;
        }
      }

      // Ensure UNLOADED submaps with computed bbox get promoted to BBOX
      if (rs.current_lod == SubmapLOD::UNLOADED && desired_lod == SubmapLOD::UNLOADED) {
        // Always show as BBOX if bbox is computed (progressive fill)
        desired_lod = SubmapLOD::BBOX;
      }

      if (desired_lod == rs.current_lod) continue;
      // Don't interfere with in-progress async loads
      if (rs.current_lod == SubmapLOD::LOADING && desired_lod >= SubmapLOD::SD) continue;
      if (rs.current_lod == SubmapLOD::LOADING_HD && desired_lod >= SubmapLOD::SD) continue;

      // --- Demotions ---
      if ((rs.current_lod == SubmapLOD::SD || rs.current_lod == SubmapLOD::HD) && desired_lod < rs.current_lod) {
        // Don't remove if transitioning SD→HD or HD→SD (handled in promotion)
        if (desired_lod < SubmapLOD::SD) {
          viewer->remove_drawable(submap_name);
          if (rs.current_lod == SubmapLOD::HD && rs.hd_points > 0) {
            loaded_hd_points = (loaded_hd_points >= rs.hd_points) ? loaded_hd_points - rs.hd_points : 0;
            rs.hd_points = 0;
          }
          total_gpu_bytes -= rs.gpu_bytes;
          rs.gpu_bytes = 0;
        }
      }
      if ((rs.current_lod == SubmapLOD::LOADING || rs.current_lod == SubmapLOD::LOADING_HD) && desired_lod < SubmapLOD::LOADING) {
        // Worker will check current_lod and skip
      }

      // --- Promotions ---
      // → BBOX: create wire cube (only every 5th submap to reduce density)
      if (desired_lod >= SubmapLOD::BBOX && rs.current_lod < SubmapLOD::BBOX) {
        if (i % 5 == 0) {
          const Eigen::Vector3f center = rs.world_bbox.center();
          const Eigen::Vector3f extents = rs.world_bbox.sizes();
          Eigen::Affine3f bbox_tf = Eigen::Affine3f::Identity();
          bbox_tf.translate(center);
          bbox_tf.scale(extents * 0.5f);
          viewer->update_drawable(bbox_name, glk::Primitives::wire_cube(),
            guik::FlatColor(sc[ci][0], sc[ci][1], sc[ci][2], 0.3f, bbox_tf));
        }

        const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submap->id);
        viewer->update_drawable("coord_" + std::to_string(submap->id),
          glk::Primitives::coordinate_system(),
          guik::VertexColor(submap_pose * Eigen::UniformScaling<float>(coord_scale)).add("info_values", info));
        viewer->update_drawable("sphere_" + std::to_string(submap->id),
          glk::Primitives::sphere(),
          guik::FlatColor(sc[ci][0], sc[ci][1], sc[ci][2], 0.5f,
            submap_pose * Eigen::UniformScaling<float>(sphere_scale)).add("info_values", info).make_transparent());
      }

      // → HD (async)
      if (desired_lod == SubmapLOD::HD && rs.current_lod < SubmapLOD::LOADING_HD) {
        // Demote existing SD drawable if upgrading from SD
        if (rs.current_lod == SubmapLOD::SD) {
          viewer->remove_drawable("submap_" + std::to_string(submap->id));
          total_gpu_bytes -= rs.gpu_bytes;
          rs.gpu_bytes = 0;
        }
        rs.current_lod = SubmapLOD::LOADING_HD;
        const auto hd_it = session_hd_paths.find(submap->session_id);
        std::string hd_path = (hd_it != session_hd_paths.end()) ? hd_it->second : hd_frames_path;
        // Swap to voxelized path if enabled
        if (lod_use_voxelized) {
          const std::string vox_path = hd_path + "_voxelized";
          if (boost::filesystem::exists(vox_path)) hd_path = vox_path;
        }
        lod_work_queue.push_back({i, submap, submap_poses[i], submap->session_id, dist, true, hd_path});
        continue;
      }

      // → SD (async)
      if (desired_lod == SubmapLOD::SD && rs.current_lod < SubmapLOD::LOADING) {
        // Demote existing HD drawable if downgrading from HD
        if (rs.current_lod == SubmapLOD::HD) {
          viewer->remove_drawable("submap_" + std::to_string(submap->id));
          total_gpu_bytes -= rs.gpu_bytes;
          loaded_hd_points = (loaded_hd_points >= rs.hd_points) ? loaded_hd_points - rs.hd_points : 0;
          rs.gpu_bytes = 0;
          rs.hd_points = 0;
        }
        rs.current_lod = SubmapLOD::LOADING;
        lod_work_queue.push_back({i, submap, submap_poses[i], submap->session_id, dist, false});
        continue;
      }

      rs.current_lod = desired_lod;
    }

    // First-paint refresh for auto-promoted scalar coloring. update_viewer() at
    // insert-time ran before the LOD worker had created any drawable; fire once
    // now that at least one submap drawable is live.
    if (scalar_default_refresh_pending) {
      for (const auto& rs : render_states) {
        if (rs.current_lod == SubmapLOD::SD || rs.current_lod == SubmapLOD::HD) {
          update_viewer();
          if (color_mode >= 3 && color_mode - 3 < static_cast<int>(aux_attribute_names.size())) {
            const int sel = scalar_colormap_per_attr[aux_attribute_names[color_mode - 3]];
            guik::LightViewer::instance()->set_colormap(static_cast<glk::COLORMAP>(sel));
          }
          scalar_default_refresh_pending = false;
          break;
        }
      }
    }
  });

  // Memory Manager UI
  viewer->register_ui_callback("memory_manager", [this] {
    if (!show_memory_manager) return;
    ImGui::SetNextWindowSize(ImVec2(340, 250), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Memory Manager", &show_memory_manager)) {
      ImGui::Checkbox("Enable LOD streaming", &lod_enabled);
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Distance-based submap streaming.\nWhen off, use Load/Unload buttons to manage manually.");
      }

      if (!hd_available) ImGui::BeginDisabled();
      ImGui::Checkbox("Enable HD (LOD 0)", &lod_hd_enabled);
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        if (hd_available) {
          ImGui::SetTooltip("Load full-resolution HD frames for nearby submaps.");
        } else {
          ImGui::SetTooltip("No HD frames found.\nRun SLAM with hd_frame_saver module to generate.");
        }
      }
      if (hd_available) {
        if (ImGui::Checkbox("Display HD only", &lod_hd_only)) {
          if (lod_hd_only) lod_hd_enabled = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hide SD submaps, show only HD (LOD 0) data.");
        ImGui::Checkbox("Use voxelized HD", &lod_use_voxelized);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Load from hd_frames_voxelized/ instead of hd_frames/.\nRun Voxelize HD first to generate the data.");
      }
      if (!hd_available) ImGui::EndDisabled();

      ImGui::Checkbox("Show bounding boxes", &lod_show_bboxes);

      ImGui::SliderFloat("SD range (m)", &lod_sd_range, 50.0f, 2000.0f, "%.0f");
      if (lod_hd_enabled && hd_available) {
        ImGui::SliderFloat("HD range (m)", &lod_hd_range, 20.0f, 500.0f, "%.0f");
      }
      ImGui::SliderFloat("VRAM budget (MB)", &lod_vram_budget_mb, 1024.0f, 24000.0f, "%.0f");

      ImGui::Separator();

      // Stats
      const float vram_mb = static_cast<float>(total_gpu_bytes) / (1024.0f * 1024.0f);
      int count_hd = 0, count_sd = 0, count_loading = 0, count_bbox = 0, count_unloaded = 0;
      size_t total_sd_loaded = 0;
      size_t total_sd_all = 0;
      for (int ri = 0; ri < static_cast<int>(render_states.size()); ri++) {
        const auto& rs = render_states[ri];
        const size_t npts = (ri < static_cast<int>(submaps.size()) && submaps[ri] && submaps[ri]->frame)
                            ? submaps[ri]->frame->size() : 0;
        total_sd_all += npts;
        switch (rs.current_lod) {
          case SubmapLOD::HD: count_hd++; break;
          case SubmapLOD::SD: count_sd++; total_sd_loaded += npts; break;
          case SubmapLOD::LOADING_HD:
          case SubmapLOD::LOADING: count_loading++; break;
          case SubmapLOD::BBOX: count_bbox++; break;
          default: count_unloaded++; break;
        }
      }

      ImGui::Text("VRAM: %.0f / %.0f MB", vram_mb, lod_vram_budget_mb);
      const float vram_frac = std::min(vram_mb / lod_vram_budget_mb, 1.0f);
      ImGui::ProgressBar(vram_frac, ImVec2(-1, 0));
      if (count_hd > 0) {
        ImGui::Text("Submaps: %d HD, %d SD, %d loading, %d bbox, %d unloaded",
                     count_hd, count_sd, count_loading, count_bbox, count_unloaded);
      } else {
        ImGui::Text("Submaps: %d SD, %d loading, %d bbox, %d unloaded",
                     count_sd, count_loading, count_bbox, count_unloaded);
      }

      // Point counts
      const double sd_loaded_m = static_cast<double>(total_sd_loaded) / 1e6;
      const double sd_all_m = static_cast<double>(total_sd_all) / 1e6;
      ImGui::Text("SD points: %.1f M / %.1f M loaded", sd_loaded_m, sd_all_m);
      if (hd_available) {
        const double hd_total_m = static_cast<double>(total_hd_points) / 1e6;
        const double hd_loaded_m = static_cast<double>(loaded_hd_points) / 1e6;
        ImGui::Text("HD points: %.1f M / %.1f M available", hd_loaded_m, hd_total_m);
      }

      ImGui::Separator();

      // Action buttons
      if (ImGui::Button("Load full SD map")) {
        lod_load_full_sd = true;
        // Don't set lod_enabled — let lod_load_full_sd suppress demotions
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Load all submaps as SD, closest-first,\nwithin VRAM budget. Disables distance demotion.");
      }

      ImGui::SameLine();
      if (!hd_available) ImGui::BeginDisabled();
      if (ImGui::Button("Load full HD map")) {
        // Unload existing data to make room for HD
        lod_load_full_sd = false;
        loaded_hd_points = 0;
        auto vw2 = guik::LightViewer::instance();
        for (int i = 0; i < static_cast<int>(render_states.size()); i++) {
          auto& rs = render_states[i];
          if (rs.current_lod == SubmapLOD::SD || rs.current_lod == SubmapLOD::HD) {
            vw2->remove_drawable("submap_" + std::to_string(submaps[i]->id));
            total_gpu_bytes -= rs.gpu_bytes;
            rs.gpu_bytes = 0;
            rs.current_lod = SubmapLOD::BBOX;
          } else if (rs.current_lod == SubmapLOD::LOADING || rs.current_lod == SubmapLOD::LOADING_HD) {
            rs.current_lod = SubmapLOD::BBOX;
          }
        }
        lod_load_full_hd = true;
      }
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        if (hd_available) {
          ImGui::SetTooltip("Load HD frames for all submaps, closest-first,\nwithin VRAM budget.");
        } else {
          ImGui::SetTooltip("No HD frames found.\nRun SLAM with hd_frame_saver module to generate.");
        }
      }
      if (!hd_available) ImGui::EndDisabled();

      ImGui::SameLine();
      if (ImGui::Button("Unload all")) {
        lod_load_full_sd = false;
        lod_load_full_hd = false;
        loaded_hd_points = 0;
        total_gpu_bytes = 0;
        auto vw = guik::LightViewer::instance();
        for (int i = 0; i < static_cast<int>(render_states.size()); i++) {
          auto& rs = render_states[i];
          if (i < static_cast<int>(submaps.size()) && submaps[i]) {
            const int sid = submaps[i]->id;
            vw->remove_drawable("submap_" + std::to_string(sid));
            vw->remove_drawable("bbox_" + std::to_string(sid));
            vw->remove_drawable("coord_" + std::to_string(sid));
            vw->remove_drawable("sphere_" + std::to_string(sid));
          }
          rs.gpu_bytes = 0;
          rs.hd_points = 0;
          rs.current_lod = SubmapLOD::UNLOADED;
        }
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Unload all point data, keep bounding boxes only.\nRe-enables distance streaming.");
      }

      if (lod_load_full_sd) {
        ImGui::Text("Loading full SD map...");
        bool all_loaded = true;
        for (const auto& rs : render_states) {
          if (rs.current_lod < SubmapLOD::SD && rs.bbox_computed) { all_loaded = false; break; }
        }
        if (all_loaded || total_gpu_bytes >= static_cast<size_t>(lod_vram_budget_mb * 1024 * 1024)) {
          lod_load_full_sd = false;
        }
      }
      if (lod_load_full_hd) {
        ImGui::Text("Loading full HD map...");
        bool all_hd = true;
        for (const auto& rs : render_states) {
          if (rs.current_lod < SubmapLOD::HD && rs.bbox_computed) { all_hd = false; break; }
        }
        if (all_hd || total_gpu_bytes >= static_cast<size_t>(lod_vram_budget_mb * 1024 * 1024)) {
          lod_load_full_hd = false;
        }
      }
    }
    ImGui::End();
  });

  // Load memory settings from persistent file
  {
    const std::string settings_path = std::string(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") + "/.glim/memory_settings.json";
    if (boost::filesystem::exists(settings_path)) {
      std::ifstream ifs(settings_path);
      const auto j = nlohmann::json::parse(ifs, nullptr, false);
      if (!j.is_discarded()) {
        lod_enabled = j.value("lod_enabled", lod_enabled);
        lod_hd_enabled = j.value("lod_hd_enabled", lod_hd_enabled);
        lod_show_bboxes = j.value("lod_show_bboxes", lod_show_bboxes);
        lod_sd_range = j.value("lod_sd_range", lod_sd_range);
        lod_hd_range = j.value("lod_hd_range", lod_hd_range);
        lod_vram_budget_mb = j.value("lod_vram_budget_mb", lod_vram_budget_mb);
        logger->info("Memory settings loaded from {}", settings_path);
      }
    }
  }

  manual_loop_close_modal.reset(new ManualLoopCloseModal(logger, num_threads));
  bundle_adjustment_modal.reset(new BundleAdjustmentModal);
  source_finder_gizmo.reset(new guik::ModelControl("source_finder_gizmo"));
  source_finder_gizmo->set_gizmo_operation("TRANSLATE");

  // Start async LOD worker thread
  lod_worker_kill = false;
  lod_worker_thread = std::thread([this] { lod_worker_task(); });

  setup_ui();

  logger->info("Starting interactive viewer");

  while (!kill_switch) {
    if (!viewer->spin_once()) {
      request_to_terminate = true;
    }

    std::lock_guard<std::mutex> lock(invoke_queue_mutex);
    for (const auto& task : invoke_queue) {
      task();
    }
    invoke_queue.clear();
  }

  manual_loop_close_modal.reset();
  bundle_adjustment_modal.reset();
  guik::LightViewer::destroy();
}

/**
 * @brief Request to invoke a task on the GUI thread
 */
void InteractiveViewer::invoke(const std::function<void()>& task) {
  if (kill_switch) {
    return;
  }
  std::lock_guard<std::mutex> lock(invoke_queue_mutex);
  invoke_queue.push_back(task);
}

/**
 * @brief Drawable selection UI
 */
void InteractiveViewer::drawable_selection() {
  if (!ImGui::Begin("Selection", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }

  const auto show_note = [](const std::string& text) {
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("%s", text.c_str());
      ImGui::EndTooltip();
    }
    return false;
  };

  // 0=RAINBOW, 1=SESSION, 2=NORMAL, 3+=aux_attributes
  std::vector<const char*> color_modes = {"RAINBOW", "SESSION", "NORMAL"};
  for (const auto& name : aux_attribute_names) {
    color_modes.push_back(name.c_str());
  }
  if (ImGui::Combo("ColorMode", &color_mode, color_modes.data(), color_modes.size())) {
    aux_attr_samples.clear();
    aux_cmap_range = Eigen::Vector2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
    // Force HD tile reload when switching to/from RGB (vertex colors need re-upload)
    const int aidx_new = color_mode - 3;
    const bool switching_rgb = (aidx_new >= 0 && aidx_new < static_cast<int>(aux_attribute_names.size()) && aux_attribute_names[aidx_new] == "RGB");
    static bool was_rgb = false;
    if (switching_rgb != was_rgb) {
      for (auto& rs : render_states) {
        if (rs.current_lod == SubmapLOD::HD) rs.current_lod = SubmapLOD::BBOX;  // force reload
      }
    }
    was_rgb = switching_rgb;
    // Apply the per-scalar remembered colormap for the new attr. For non-aux
    // modes (RAINBOW/SESSION/NORMAL) the canvas colormap is unused so leave it.
    if (aidx_new >= 0 && aidx_new < static_cast<int>(aux_attribute_names.size())) {
      const int sel = scalar_colormap_per_attr[aux_attribute_names[aidx_new]];
      guik::LightViewer::instance()->set_colormap(static_cast<glk::COLORMAP>(sel));
    }
    update_viewer();
  }
  show_note("Color mode for rendering submaps.\n- RAINBOW=Altitude encoding color\n- SESSION=Session ID\n- NORMAL=Surface normal direction\n- others=Per-point attribute colormap");

  // Color scale for the active scalar. Only meaningful for aux-attr modes;
  // RAINBOW/SESSION/NORMAL don't consult the canvas colormap. Choice is stored
  // per-attribute so switching scalars keeps each one's preferred scale.
  if (color_mode >= 3) {
    const int aidx = color_mode - 3;
    if (aidx < static_cast<int>(aux_attribute_names.size())) {
      int& sel = scalar_colormap_per_attr[aux_attribute_names[aidx]];
      static const auto _cmap_names = glk::colormap_names();
      if (ImGui::Combo("Scale", &sel, _cmap_names.data(), static_cast<int>(_cmap_names.size()))) {
        guik::LightViewer::instance()->set_colormap(static_cast<glk::COLORMAP>(sel));
      }
      show_note("Color scale applied to the current scalar field.\n"
                "Turbo = perceptually strong default; Cividis = colour-blind safe;\n"
                "Ocean / Jet / Helix surface different contrast on LiDAR intensity.");
    }
  }

  ImGui::Checkbox("Trajectory", &draw_traj);
  ImGui::SameLine();
  ImGui::Checkbox("Submaps", &draw_points);

  ImGui::Checkbox("Factors", &draw_factors);
  ImGui::SameLine();
  ImGui::Checkbox("Spheres", &draw_spheres);

  ImGui::Checkbox("Coords", &draw_coords);
  ImGui::SameLine();
  ImGui::Checkbox("Cameras", &draw_cameras);
  ImGui::SameLine();
  if (ImGui::Checkbox("Frames", &draw_frames)) update_viewer();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show a small RGB triad at every per-frame sensor pose.");

  if (ImGui::BeginMenu("Display settings")) {
    bool do_update_viewer = false;

    do_update_viewer |= ImGui::DragFloat("coord scale", &coord_scale, 0.01f, 0.01f, 100.0f);
    show_note("Submap coordinate system maker scale.");

    do_update_viewer |= ImGui::DragFloat("sphere scale", &sphere_scale, 0.01f, 0.01f, 100.0f);
    show_note("Submap selection sphere maker scale.");

    do_update_viewer |= ImGui::DragFloat("frame triad scale", &frame_coord_scale, 0.005f, 0.01f, 5.0f);
    show_note("Per-frame RGB triad length (meters).");

    auto viewer = guik::viewer();
    if (ImGui::Checkbox("Cumulative rendering", &enable_partial_rendering)) {
      if (enable_partial_rendering && !viewer->partial_rendering_enabled()) {
        viewer->enable_partial_rendering(1e-1);
        viewer->shader_setting().add("dynamic_object", 1);
      } else {
        viewer->disable_partial_rendering();
      }

      // Update existing submap buffers
      for (int i = 0;; i++) {
        auto found = viewer->find_drawable("submap_" + std::to_string(i));
        if (!found.first) {
          break;
        }

        auto cb = std::dynamic_pointer_cast<const glk::PointCloudBuffer>(found.second);
        auto cloud_buffer = std::const_pointer_cast<glk::PointCloudBuffer>(cb);  // !!

        if (enable_partial_rendering) {
          cloud_buffer->enable_partial_rendering(partial_rendering_budget);
          found.first->add("dynamic_object", 0).make_transparent();
        } else {
          cloud_buffer->disable_partial_rendering();
          found.first->add("dynamic_object", 1);
        }
      }
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::DragInt("Budget", &partial_rendering_budget, 1, 1, 1000000);

    ImGui::EndMenu();

    if (do_update_viewer) {
      update_viewer();
    }
  }

  ImGui::Separator();
  ImGui::DragFloat("Min overlap", &min_overlap, 0.01f, 0.01f, 1.0f);
  show_note("Minimum overlap ratio for finding overlapping submaps.");

  if (needs_session_merge) {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button("Find overlapping submaps") || show_note("Find overlapping submaps and create matching cost factors between them.")) {
    logger->info("finding overlapping submaps...");
    GlobalMappingCallbacks::request_to_find_overlapping_submaps(min_overlap);
  }

  if (ImGui::Button("Recover graph") || show_note("Detect and fix corrupted graph.")) {
    logger->info("recovering graph...");
    GlobalMappingCallbacks::request_to_recover();
  }

  if (needs_session_merge) {
    ImGui::EndDisabled();
  }

  if (submaps.size() && needs_session_merge) {
    if (ImGui::Button("Merge sessions") || show_note("Merge the lastly loaded session with the previous session.")) {
      logger->info("merging sessions...");
      if (submaps.empty()) {
        logger->warn("No submaps");
      } else if (submaps.front()->session_id == submaps.back()->session_id) {
        logger->warn("There is only one session");
      } else {
        const int source_session_id = submaps.back()->session_id;
        const auto source_begin = std::find_if(submaps.begin(), submaps.end(), [=](const SubMap::ConstPtr& submap) { return submap->session_id == source_session_id; });

        const std::vector<SubMap::ConstPtr> target_submaps(submaps.begin(), source_begin);
        const std::vector<SubMap::ConstPtr> source_submaps(source_begin, submaps.end());

        logger->info("|submaps|={} |targets|={} |sources|={} source_session_id={}", submaps.size(), target_submaps.size(), source_submaps.size(), source_session_id);
        manual_loop_close_modal->set_submaps(target_submaps, source_submaps);
      }

      std::vector<SubMap::ConstPtr> target_submaps;
      std::vector<SubMap::ConstPtr> source_submaps;
      for (const auto& submap : submaps) {
        if (target_submaps.empty()) {
          target_submaps.emplace_back(submap);
          continue;
        }
        if (target_submaps.back()->session_id == submap->session_id) {
          target_submaps.emplace_back(submap);
          continue;
        }
      }
    }
  }

  if (needs_session_merge) {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button("Optimize")) {
    logger->info("optimizing...");
    GlobalMappingCallbacks::request_to_optimize();
  }
  show_note("Optimize the graph.");

  ImGui::SameLine();
  ImGui::Checkbox("##Cont optimize", &cont_optimize);
  if (cont_optimize) {
    GlobalMappingCallbacks::request_to_optimize();
  }
  show_note("Continuously optimize the graph.");

  if (needs_session_merge) {
    ImGui::EndDisabled();
  }

  ImGui::End();
}

/**
 * @brief Click callback
 */
void InteractiveViewer::on_click() {
  ImGuiIO& io = ImGui::GetIO();
  if (io.WantCaptureMouse) {
    return;
  }

  auto viewer = guik::LightViewer::instance();
  const auto mouse_pos = ImGui::GetMousePos();
  const Eigen::Vector2i mpos(mouse_pos.x, mouse_pos.y);

  // Double-click left: set orbit rotation center on existing camera
  if (ImGui::IsMouseDoubleClicked(0)) {
    const float depth = viewer->pick_depth(mpos);
    if (depth < 1.0f) {  // valid depth (not background)
      const Eigen::Vector3f point = viewer->unproject(mpos, depth);

      // Record camera position before lookat
      const Eigen::Matrix4f vm_before = viewer->view_matrix();
      const Eigen::Vector3f cam_pos_before = -(vm_before.block<3, 3>(0, 0).transpose() * vm_before.block<3, 1>(0, 3));
      const float dist_before = (cam_pos_before - point).norm();

      viewer->lookat(point);

      // Check if camera ended up too far from the target after lookat
      const Eigen::Matrix4f vm_after = viewer->view_matrix();
      const Eigen::Vector3f cam_pos_after = -(vm_after.block<3, 3>(0, 0).transpose() * vm_after.block<3, 1>(0, 3));
      const float dist_after = (cam_pos_after - point).norm();

      // If camera jumped too far, re-lookat to bring it back (cap at 100m or original distance)
      if (dist_after > 100.0f && dist_after > dist_before * 2.0f) {
        viewer->lookat(point);
      }

      // Visual indicator
      Eigen::Affine3f indicator_tf = Eigen::Affine3f::Identity();
      indicator_tf.translate(point);
      indicator_tf.scale(std::max(0.1f, std::min(dist_before, dist_after) * 0.005f));
      viewer->update_drawable("rotation_center", glk::Primitives::sphere(),
        guik::FlatColor(1.0f, 0.4f, 0.0f, 0.6f, indicator_tf).make_transparent());

      std::thread([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        guik::LightViewer::instance()->invoke([] {
          guik::LightViewer::instance()->remove_drawable("rotation_center");
        });
      }).detach();
    }
    return;
  }

  // Right-click: context menu picking
  if (!ImGui::IsMouseClicked(1)) {
    return;
  }

  right_clicked_info = viewer->pick_info(mpos);
  const float depth = viewer->pick_depth(mpos);
  right_clicked_pos = viewer->unproject(mpos, depth);
}

/**
 * @brief Context menu
 */
gtsam_points::PointCloudCPU::Ptr InteractiveViewer::load_hd_for_submap(int submap_index, bool compute_covs) const {
  if (submap_index < 0 || submap_index >= static_cast<int>(submaps.size()) || !submaps[submap_index]) {
    return nullptr;
  }
  const auto& submap = submaps[submap_index];
  const auto hd_it = session_hd_paths.find(submap->session_id);
  if (hd_it == session_hd_paths.end()) return nullptr;

  // Load and merge all HD frames for this submap
  std::vector<Eigen::Vector4d> all_points;
  std::vector<double> all_intensities;
  std::vector<double> all_times;  // gps_time = frame_stamp + per-point time offset
  std::vector<Eigen::Vector4d> all_normals_disk;  // populated when per-frame normals.bin exists; same filter/order as all_points
  bool any_frame_had_normals = false;
  const Eigen::Isometry3d T_ep = submap->T_world_origin * submap->T_origin_endpoint_L;
  const Eigen::Isometry3d T_odom0 = submap->frames.front()->T_world_imu;

  for (const auto& frame : submap->frames) {
    char dir_name[16];
    std::snprintf(dir_name, sizeof(dir_name), "%08ld", frame->id);
    const std::string frame_dir = hd_it->second + "/" + dir_name;
    const std::string meta_path = frame_dir + "/frame_meta.json";
    if (!boost::filesystem::exists(meta_path)) continue;
    std::ifstream meta_ifs(meta_path);
    const auto meta = nlohmann::json::parse(meta_ifs, nullptr, false);
    if (meta.is_discarded()) continue;
    const int npts = meta.value("num_points", 0);
    if (npts == 0) continue;

    std::vector<Eigen::Vector3f> pts(npts);
    std::vector<float> intensity(npts, 0.0f);
    std::vector<float> frame_times(npts, 0.0f);
    bool has_range = false;
    std::vector<float> range(npts);
    std::vector<Eigen::Vector3f> normals_f(npts, Eigen::Vector3f::Zero());
    bool has_normals = false;
    { std::ifstream f(frame_dir + "/points.bin", std::ios::binary);
      if (!f) continue;
      f.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * npts); }
    { std::ifstream f(frame_dir + "/range.bin", std::ios::binary);
      if (f) { f.read(reinterpret_cast<char*>(range.data()), sizeof(float) * npts); has_range = true; } }
    { std::ifstream f(frame_dir + "/intensities.bin", std::ios::binary);
      if (f) f.read(reinterpret_cast<char*>(intensity.data()), sizeof(float) * npts); }
    { std::ifstream f(frame_dir + "/times.bin", std::ios::binary);
      if (f) f.read(reinterpret_cast<char*>(frame_times.data()), sizeof(float) * npts); }
    { std::ifstream f(frame_dir + "/normals.bin", std::ios::binary);
      if (f) { f.read(reinterpret_cast<char*>(normals_f.data()), sizeof(Eigen::Vector3f) * npts); has_normals = true; any_frame_had_normals = true; } }

    // Transform to submap-local frame (NOT world frame — modal handles pose separately)
    const Eigen::Isometry3d T_w_imu = T_ep * T_odom0.inverse() * frame->T_world_imu;
    const Eigen::Isometry3d T_w_lidar = T_w_imu * frame->T_lidar_imu.inverse();
    const Eigen::Isometry3d T_origin_lidar = submap->T_world_origin.inverse() * T_w_lidar;
    const Eigen::Matrix3d R = T_origin_lidar.rotation();
    const Eigen::Vector3d t = T_origin_lidar.translation();

    const double frame_stamp = frame->stamp;
    for (int pi = 0; pi < npts; pi++) {
      const float r = has_range ? range[pi] : pts[pi].norm();
      if (r < 1.5f) continue;
      const Eigen::Vector3d lp = R * pts[pi].cast<double>() + t;
      all_points.push_back(Eigen::Vector4d(lp.x(), lp.y(), lp.z(), 1.0));
      all_intensities.push_back(static_cast<double>(intensity[pi]));
      all_times.push_back(frame_stamp + static_cast<double>(frame_times[pi]));
      if (any_frame_had_normals) {
        if (has_normals) {
          const Eigen::Vector3d ln = (R * normals_f[pi].cast<double>()).normalized();
          all_normals_disk.push_back(Eigen::Vector4d(ln.x(), ln.y(), ln.z(), 0.0));
        } else {
          all_normals_disk.push_back(Eigen::Vector4d::Zero());  // keep parallel sizing
        }
      }
    }
  }

  if (all_points.empty()) return nullptr;

  // Create PointCloudCPU
  auto cloud = std::make_shared<gtsam_points::PointCloudCPU>();
  cloud->num_points = all_points.size();
  cloud->points_storage = all_points;
  cloud->points = cloud->points_storage.data();

  // Add intensities
  cloud->intensities_storage = std::move(all_intensities);
  cloud->intensities = cloud->intensities_storage.data();

  // Add times (gps_time = frame_stamp + per-point offset)
  cloud->times_storage = std::move(all_times);
  cloud->times = cloud->times_storage.data();

  if (compute_covs) {
    // Compute normals and covariances using k-NN (overrides any disk normals loaded above)
    const int k = 10;
    gtsam_points::KdTree tree(cloud->points, cloud->num_points);
    std::vector<int> neighbors(cloud->num_points * k);
    for (int i = 0; i < static_cast<int>(cloud->num_points); i++) {
      std::vector<size_t> k_indices(k, i);
      std::vector<double> k_sq_dists(k);
      tree.knn_search(cloud->points[i].data(), k, k_indices.data(), k_sq_dists.data());
      std::copy(k_indices.begin(), k_indices.begin() + k, neighbors.begin() + i * k);
    }
    CloudCovarianceEstimation cov_estimator(num_threads);
    std::vector<Eigen::Vector4d> normals;
    std::vector<Eigen::Matrix4d> covs;
    cov_estimator.estimate(cloud->points_storage, neighbors, k, normals, covs);
    cloud->add_normals(normals);
    cloud->add_covs(covs);
    logger->info("[HD] Loaded {} pts for submap {} (with covs)", cloud->num_points, submap_index);
  } else if (any_frame_had_normals && all_normals_disk.size() == cloud->num_points) {
    // Attach the pre-computed normals loaded from disk — cheap, parallel to points.
    cloud->add_normals(all_normals_disk);
    logger->info("[HD] Loaded {} pts for submap {} (points + disk normals)", cloud->num_points, submap_index);
  } else {
    logger->info("[HD] Loaded {} pts for submap {} (points only)", cloud->num_points, submap_index);
  }
  return cloud;
}

void InteractiveViewer::source_finder_update_cylinder() {
  const float half_h = source_finder_height * 0.5f;
  std::vector<Eigen::Vector3f> verts;

  if (source_finder_shape == 0) {
    // Cylinder
    const int N = 32;
    const float r = source_finder_radius;
    for (int ci = 0; ci < 2; ci++) {
      const float z = (ci == 0) ? -half_h : half_h;
      for (int i = 0; i < N; i++) {
        const float a0 = 2.0f * M_PI * i / N;
        const float a1 = 2.0f * M_PI * (i + 1) / N;
        verts.push_back(Eigen::Vector3f(r * std::cos(a0), r * std::sin(a0), z));
        verts.push_back(Eigen::Vector3f(r * std::cos(a1), r * std::sin(a1), z));
      }
    }
    for (int i = 0; i < 4; i++) {
      const float a = 2.0f * M_PI * i / 4;
      verts.push_back(Eigen::Vector3f(r * std::cos(a), r * std::sin(a), -half_h));
      verts.push_back(Eigen::Vector3f(r * std::cos(a), r * std::sin(a), half_h));
    }
  } else {
    // Oriented box with full rotation (Rz * Ry)
    const float hl = source_finder_length * 0.5f;
    const float hw = source_finder_width * 0.5f;
    const float yr = source_finder_yaw * static_cast<float>(M_PI) / 180.0f;
    const float pr = source_finder_pitch * static_cast<float>(M_PI) / 180.0f;
    const Eigen::Matrix3f R = (Eigen::AngleAxisf(yr, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pr, Eigen::Vector3f::UnitY())).toRotationMatrix();
    // 8 corners of the box in local frame, transformed by R
    const Eigen::Vector3f corners[8] = {
      R * Eigen::Vector3f(-hl, -hw, -half_h), R * Eigen::Vector3f(hl, -hw, -half_h),
      R * Eigen::Vector3f(hl, hw, -half_h),   R * Eigen::Vector3f(-hl, hw, -half_h),
      R * Eigen::Vector3f(-hl, -hw, half_h),  R * Eigen::Vector3f(hl, -hw, half_h),
      R * Eigen::Vector3f(hl, hw, half_h),    R * Eigen::Vector3f(-hl, hw, half_h),
    };
    // Bottom face
    verts.push_back(corners[0]); verts.push_back(corners[1]);
    verts.push_back(corners[1]); verts.push_back(corners[2]);
    verts.push_back(corners[2]); verts.push_back(corners[3]);
    verts.push_back(corners[3]); verts.push_back(corners[0]);
    // Top face
    verts.push_back(corners[4]); verts.push_back(corners[5]);
    verts.push_back(corners[5]); verts.push_back(corners[6]);
    verts.push_back(corners[6]); verts.push_back(corners[7]);
    verts.push_back(corners[7]); verts.push_back(corners[4]);
    // Vertical pillars
    for (int i = 0; i < 4; i++) { verts.push_back(corners[i]); verts.push_back(corners[i + 4]); }
  }

  auto lines = std::make_shared<glk::ThinLines>(verts, false);
  Eigen::Affine3f tf = Eigen::Translation3f(source_finder_pos) * Eigen::Scaling(1.0f);
  guik::LightViewer::instance()->update_drawable("source_finder_cylinder", lines,
    guik::FlatColor(1.0f, 1.0f, 0.0f, 1.0f, tf));
}

void InteractiveViewer::source_finder_color_hits() {
  if (source_finder_hits.empty()) {
    guik::LightViewer::instance()->remove_drawable("team_lines");
    return;
  }
  auto teams = glim::group_by_continuity(source_finder_hits);

  const int tgt_idx = source_finder_teams_swapped ? 1 : 0;
  const int src_idx = source_finder_teams_swapped ? 0 : 1;
  std::unordered_set<int> tgt_set, src_set;
  if (teams.size() > static_cast<size_t>(tgt_idx)) for (int id : teams[tgt_idx]) tgt_set.insert(id);
  if (teams.size() > static_cast<size_t>(src_idx)) for (int id : teams[src_idx]) src_set.insert(id);

  auto viewer = guik::LightViewer::instance();
  static const float sc[][3] = {
    {1.0f, 0.0f, 0.0f}, {1.0f, 0.85f, 0.0f}, {0.0f, 0.8f, 0.2f},
    {1.0f, 0.6f, 0.0f}, {0.8f, 0.0f, 0.8f}, {0.0f, 0.8f, 0.8f},
  };
  std::vector<Eigen::Vector3f> tgt_lines, src_lines;
  for (int pi = 0; pi < static_cast<int>(submaps.size()); pi++) {
    if (!submaps[pi]) continue;
    const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submaps[pi]->id);
    const Eigen::Affine3f sp = submap_poses[pi].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
    if (tgt_set.count(pi)) {
      viewer->update_drawable("sphere_" + std::to_string(submaps[pi]->id), glk::Primitives::sphere(),
        guik::FlatColor(1.0f, 1.0f, 0.0f, 1.0f, sp).add("info_values", info));
      tgt_lines.push_back(submap_poses[pi].translation().cast<float>());
      tgt_lines.push_back(source_finder_pos);
    } else if (src_set.count(pi)) {
      viewer->update_drawable("sphere_" + std::to_string(submaps[pi]->id), glk::Primitives::sphere(),
        guik::FlatColor(0.2f, 0.9f, 0.2f, 1.0f, sp).add("info_values", info));
      src_lines.push_back(submap_poses[pi].translation().cast<float>());
      src_lines.push_back(source_finder_pos);
    } else if (source_finder_hits.count(pi)) {
      // Extra teams — grey
      viewer->update_drawable("sphere_" + std::to_string(submaps[pi]->id), glk::Primitives::sphere(),
        guik::FlatColor(0.6f, 0.6f, 0.6f, 0.8f, sp).add("info_values", info));
    } else {
      const int ci = submaps[pi]->session_id % 6;
      viewer->update_drawable("sphere_" + std::to_string(submaps[pi]->id), glk::Primitives::sphere(),
        guik::FlatColor(sc[ci][0], sc[ci][1], sc[ci][2], 0.5f, sp).add("info_values", info).make_transparent());
    }
  }
  // Draw lines: yellow for target, green for source
  std::vector<Eigen::Vector3f> all_lines;
  std::vector<Eigen::Vector4f> all_colors;
  for (size_t i = 0; i < tgt_lines.size(); i++) {
    all_lines.push_back(tgt_lines[i]);
    all_colors.push_back(Eigen::Vector4f(1.0f, 1.0f, 0.0f, 0.6f));
  }
  for (size_t i = 0; i < src_lines.size(); i++) {
    all_lines.push_back(src_lines[i]);
    all_colors.push_back(Eigen::Vector4f(0.2f, 0.9f, 0.2f, 0.6f));
  }
  if (!all_lines.empty()) {
    viewer->update_drawable("team_lines", std::make_shared<glk::ThinLines>(all_lines, all_colors, false),
      guik::VertexColor());
  } else {
    viewer->remove_drawable("team_lines");
  }
}

void InteractiveViewer::source_finder_scan_fast() {
  source_finder_hits.clear();
  const float z_lo = source_finder_pos.z() - source_finder_height * 0.5f;
  const float z_hi = source_finder_pos.z() + source_finder_height * 0.5f;
  const Eigen::Vector2f center_xy(source_finder_pos.x(), source_finder_pos.y());

  // Shape-specific XY test for AABB (fast mode)
  const float r = source_finder_radius;
  const float yaw_rad = source_finder_yaw * static_cast<float>(M_PI) / 180.0f;
  const float cy_r = std::cos(yaw_rad), sy_r = std::sin(yaw_rad);
  const float hl = source_finder_length * 0.5f, hw = source_finder_width * 0.5f;

  // Test if an AABB potentially overlaps our probe shape in XY
  auto aabb_overlaps_xy = [&](const Eigen::AlignedBox3f& box) -> bool {
    if (source_finder_shape == 0) {
      // Cylinder: closest point on AABB to center
      const float cx = std::max(box.min().x(), std::min(center_xy.x(), box.max().x()));
      const float cy = std::max(box.min().y(), std::min(center_xy.y(), box.max().y()));
      const float dx = cx - center_xy.x(), dy = cy - center_xy.y();
      return dx * dx + dy * dy <= r * r;
    } else {
      // Box: test all 4 AABB corners in rotated frame, or AABB vs OBB overlap
      // Simplified: test AABB center against enlarged probe box
      const float bcx = (box.min().x() + box.max().x()) * 0.5f - center_xy.x();
      const float bcy = (box.min().y() + box.max().y()) * 0.5f - center_xy.y();
      const float lx = cy_r * bcx + sy_r * bcy;  // rotate to probe-local
      const float ly = -sy_r * bcx + cy_r * bcy;
      const float bhl = (box.max().x() - box.min().x()) * 0.5f + hl;
      const float bhw = (box.max().y() - box.min().y()) * 0.5f + hw;
      return std::abs(lx) <= bhl && std::abs(ly) <= bhw;
    }
  };

  for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
    if (!submaps[si]) continue;

    if (si < static_cast<int>(render_states.size()) && render_states[si].bbox_computed) {
      const auto& box = render_states[si].world_bbox;
      if (box.max().z() < z_lo || box.min().z() > z_hi) continue;
      if (aabb_overlaps_xy(box)) {
        source_finder_hits.insert(si);
      }
    } else if (submaps[si]->frame) {
      const Eigen::Vector3d origin = submap_poses[si].translation();
      const float dz = static_cast<float>(origin.z());
      if (dz >= z_lo - 50.0f && dz <= z_hi + 50.0f) {
        source_finder_hits.insert(si);
      }
    }
  }

  source_finder_color_hits();
  logger->info("[Source finder/fast] {} submaps (bbox) in probe (r={:.1f} h={:.1f})", source_finder_hits.size(), source_finder_radius, source_finder_height);
}

void InteractiveViewer::source_finder_scan_precise() {
  source_finder_hits.clear();
  const float z_lo = source_finder_pos.z() - source_finder_height * 0.5f;
  const float z_hi = source_finder_pos.z() + source_finder_height * 0.5f;
  const Eigen::Vector2f center_xy(source_finder_pos.x(), source_finder_pos.y());

  // Shape-specific XY point test
  const float r2 = source_finder_radius * source_finder_radius;
  const float yaw_rad = source_finder_yaw * static_cast<float>(M_PI) / 180.0f;
  const float cy_r = std::cos(yaw_rad), sy_r = std::sin(yaw_rad);
  const float hl = source_finder_length * 0.5f, hw = source_finder_width * 0.5f;

  auto point_in_shape_xy = [&](float dx, float dy) -> bool {
    if (source_finder_shape == 0) {
      return dx * dx + dy * dy < r2;
    } else {
      const float lx = cy_r * dx + sy_r * dy;
      const float ly = -sy_r * dx + cy_r * dy;
      return std::abs(lx) <= hl && std::abs(ly) <= hw;
    }
  };

  for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
    if (!submaps[si] || !submaps[si]->frame) continue;
    const auto& frame = submaps[si]->frame;
    const Eigen::Isometry3d& pose = submap_poses[si];

    for (size_t pi = 0; pi < frame->size(); pi++) {
      const Eigen::Vector3d wp = pose * frame->points[pi].head<3>();
      const float wz = static_cast<float>(wp.z());
      if (wz < z_lo || wz > z_hi) continue;
      const float dx = static_cast<float>(wp.x()) - center_xy.x();
      const float dy = static_cast<float>(wp.y()) - center_xy.y();
      if (point_in_shape_xy(dx, dy)) {
        source_finder_hits.insert(si);
        break;
      }
    }
  }

  source_finder_color_hits();
  logger->info("[Source finder/precise] {} submaps (per-point) in probe (r={:.1f} h={:.1f})", source_finder_hits.size(), source_finder_radius, source_finder_height);
}

void InteractiveViewer::context_menu() {
  if (ImGui::BeginPopupContextVoid("context menu")) {
    const PickType type = static_cast<PickType>(right_clicked_info[0]);

    if (type == PickType::FRAME) {
      const int frame_id = right_clicked_info[3];
      if (frame_id >= 0 && frame_id < static_cast<int>(submaps.size()) && submaps[frame_id]) {
        ImGui::TextUnformatted(("Submap ID : " + std::to_string(frame_id)).c_str());
        if (frame_id < static_cast<int>(submap_gps_sigma.size()) && submap_gps_sigma[frame_id] >= 0.0f) {
          char sigma_text[64];
          std::snprintf(sigma_text, sizeof(sigma_text), "GPS sigma: %.3f m", submap_gps_sigma[frame_id]);
          ImGui::TextColored(ImVec4(0.7f, 0.9f, 0.7f, 1.0f), "%s", sigma_text);
        } else {
          ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "GPS sigma: N/A");
        }
        ImGui::Separator();

        // Helper lambda: show submap preview with given color (HD if available, SD fallback)
        auto show_preview = [&](int fid, float r, float g, float b) {
          auto viewer = guik::LightViewer::instance();
          const auto& sm = submaps[fid];
          std::shared_ptr<glk::PointCloudBuffer> cb;
          bool used_hd = false;
          const auto hd_it = session_hd_paths.find(sm->session_id);
          if (hd_it != session_hd_paths.end()) {
            std::vector<Eigen::Vector4d> hd_pts;
            const Eigen::Isometry3d T_ep = sm->T_world_origin * sm->T_origin_endpoint_L;
            const Eigen::Isometry3d T_odom0 = sm->frames.front()->T_world_imu;
            for (const auto& fr : sm->frames) {
              char dn[16]; std::snprintf(dn, sizeof(dn), "%08ld", fr->id);
              const std::string fd = hd_it->second + "/" + dn;
              if (!boost::filesystem::exists(fd + "/frame_meta.json")) continue;
              std::ifstream mf(fd + "/frame_meta.json");
              const auto mj = nlohmann::json::parse(mf, nullptr, false);
              if (mj.is_discarded()) continue;
              const int np = mj.value("num_points", 0);
              if (np == 0) continue;
              std::vector<Eigen::Vector3f> pts(np);
              std::vector<float> rng(np);
              { std::ifstream f(fd + "/points.bin", std::ios::binary); if (!f) continue; f.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * np); }
              { std::ifstream f(fd + "/range.bin", std::ios::binary); if (f) f.read(reinterpret_cast<char*>(rng.data()), sizeof(float) * np); }
              const Eigen::Isometry3d Tw = T_ep * T_odom0.inverse() * fr->T_world_imu;
              const Eigen::Isometry3d Tl = Tw * fr->T_lidar_imu.inverse();
              const Eigen::Matrix3d R = Tl.rotation(); const Eigen::Vector3d t = Tl.translation();
              for (int pi = 0; pi < np; pi++) {
                if (rng[pi] < 1.5f) continue;
                const Eigen::Vector3d wp = R * pts[pi].cast<double>() + t;
                hd_pts.push_back(Eigen::Vector4d(wp.x(), wp.y(), wp.z(), 1.0));
              }
            }
            if (!hd_pts.empty()) {
              cb = std::make_shared<glk::PointCloudBuffer>(hd_pts.data(), hd_pts.size());
              used_hd = true;
            }
          }
          if (!cb) cb = std::make_shared<glk::PointCloudBuffer>(sm->frame->points, sm->frame->size());
          if (used_hd) {
            viewer->update_drawable("lc_preview_" + std::to_string(fid), cb, guik::FlatColor(r, g, b, 0.8f));
          } else {
            viewer->update_drawable("lc_preview_" + std::to_string(fid), cb,
              guik::FlatColor(r, g, b, 0.8f, submap_poses[fid].cast<float>()));
          }
          // Color sphere
          const Eigen::Affine3f sp = submap_poses[fid].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
          const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, sm->id);
          viewer->update_drawable("sphere_" + std::to_string(sm->id), glk::Primitives::sphere(),
            guik::FlatColor(r, g, b, 0.9f, sp).add("info_values", info).make_transparent());
        };

        // Helper: clear all previews and reset sphere colors
        auto clear_all_previews = [&]() {
          auto viewer = guik::LightViewer::instance();
          static const float sc[][3] = {{1,0,0},{1,0.85f,0},{0,0.8f,0.2f},{1,0.6f,0},{0.8f,0,0.8f},{0,0.8f,0.8f}};
          for (int pi = 0; pi < static_cast<int>(submaps.size()); pi++) {
            viewer->remove_drawable("lc_preview_" + std::to_string(pi));
            if (submaps[pi]) {
              const int ci = submaps[pi]->session_id % 6;
              const Eigen::Vector4i inf(static_cast<int>(PickType::FRAME), 0, 0, submaps[pi]->id);
              const Eigen::Affine3f sp = submap_poses[pi].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
              viewer->update_drawable("sphere_" + std::to_string(submaps[pi]->id), glk::Primitives::sphere(),
                guik::FlatColor(sc[ci][0], sc[ci][1], sc[ci][2], 0.5f, sp).add("info_values", inf).make_transparent());
            }
          }
        };

        // Loop end: accumulate sources (green preview), or trigger single-frame ICP if target already set
        {
          const bool already_in_group = std::find(lc_source_group.begin(), lc_source_group.end(), frame_id) != lc_source_group.end();
          // If target already set and no sources accumulated yet, show simple "Loop end" (single-frame mode)
          const bool single_mode = (lc_target_frame_id >= 0 && lc_source_group.empty());
          char end_label[64];
          if (single_mode) {
            std::snprintf(end_label, sizeof(end_label), "Loop end");
          } else {
            std::snprintf(end_label, sizeof(end_label), "Loop end (%zu selected)", lc_source_group.size());
          }
          if (ImGui::MenuItem(end_label, nullptr, already_in_group)) {
            if (already_in_group) {
              // Remove from group
              lc_source_group.erase(std::remove(lc_source_group.begin(), lc_source_group.end(), frame_id), lc_source_group.end());
              // Remove preview for this one
              guik::LightViewer::instance()->remove_drawable("lc_preview_" + std::to_string(frame_id));
            } else if (single_mode) {
              // Target already set — trigger single-frame ICP immediately
              lc_source_frame_id = frame_id;
              show_preview(frame_id, 0.2f, 0.9f, 0.2f);  // green
              manual_loop_close_modal->set_source(X(frame_id), submaps[frame_id]->frame, submap_poses[frame_id]);
              manual_loop_close_modal->source_gps_sigma = (frame_id < static_cast<int>(submap_gps_sigma.size())) ? submap_gps_sigma[frame_id] : -1.0f;
              // HD callback for single source
              manual_loop_close_modal->load_hd_callback = [this]() -> std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr> {
                auto hd_t = (lc_target_frame_id >= 0) ? load_hd_for_submap(lc_target_frame_id, false) : nullptr;
                auto hd_s = (lc_source_frame_id >= 0) ? load_hd_for_submap(lc_source_frame_id, false) : nullptr;
                return {hd_t, hd_s};
              };
              if (session_hd_paths.empty()) manual_loop_close_modal->load_hd_callback = nullptr;
            } else {
              // Accumulate mode: first addition clears existing previews
              if (lc_source_group.empty()) {
                clear_all_previews();
              }
              lc_source_group.push_back(frame_id);
              show_preview(frame_id, 0.2f, 0.9f, 0.2f);  // green
            }
          }
        }

        // Loop begin: set target (red) + trigger modal if sources exist
        if (ImGui::MenuItem("Loop begin", nullptr, lc_target_frame_id == frame_id)) {
          // Clear old target preview if any
          if (lc_target_frame_id >= 0) {
            guik::LightViewer::instance()->remove_drawable("lc_preview_" + std::to_string(lc_target_frame_id));
          }
          lc_target_frame_id = frame_id;
          show_preview(frame_id, 1.0f, 0.3f, 0.3f);  // red

          // Set target on modal
          manual_loop_close_modal->set_target(X(frame_id), submaps[frame_id]->frame, submap_poses[frame_id]);
          manual_loop_close_modal->target_gps_sigma = (frame_id < static_cast<int>(submap_gps_sigma.size())) ? submap_gps_sigma[frame_id] : -1.0f;

          // If source group has entries, merge and trigger modal
          if (!lc_source_group.empty()) {
            // Use central source for the factor key and reference pose
            lc_source_frame_id = lc_source_group[lc_source_group.size() / 2];
            const Eigen::Isometry3d ref_pose = submap_poses[lc_source_frame_id];

            // Collect all points from source group into ref_pose's local frame
            std::vector<Eigen::Vector4d> all_points;
            for (int si : lc_source_group) {
              const auto& sm = submaps[si];
              if (!sm || !sm->frame) continue;
              // Transform: submap-local → world → ref-local
              const Eigen::Isometry3d T_ref_submap = ref_pose.inverse() * submap_poses[si];
              for (size_t pi = 0; pi < sm->frame->size(); pi++) {
                const Eigen::Vector4d wp = sm->frame->points[pi];
                const Eigen::Vector3d p_ref = T_ref_submap * wp.head<3>();
                all_points.push_back(Eigen::Vector4d(p_ref.x(), p_ref.y(), p_ref.z(), 1.0));
              }
            }

            // Build clean PointCloudCPU
            auto merged = std::make_shared<gtsam_points::PointCloudCPU>();
            merged->num_points = all_points.size();
            merged->points_storage = std::move(all_points);
            merged->points = merged->points_storage.data();

            logger->info("[Loop] Merged {} source submaps: {} total points", lc_source_group.size(), merged->num_points);
            manual_loop_close_modal->set_source(X(lc_source_frame_id), merged, ref_pose);
            manual_loop_close_modal->source_gps_sigma = (lc_source_frame_id < static_cast<int>(submap_gps_sigma.size())) ? submap_gps_sigma[lc_source_frame_id] : -1.0f;

            // HD callback
            manual_loop_close_modal->load_hd_callback = [this]() -> std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr> {
              auto hd_t = (lc_target_frame_id >= 0) ? load_hd_for_submap(lc_target_frame_id, false) : nullptr;
              // Merge HD for all source group submaps into ref_pose's local frame
              const Eigen::Isometry3d ref_pose = submap_poses[lc_source_frame_id];
              std::vector<Eigen::Vector4d> all_hd_points;
              std::vector<double> all_hd_ints;
              for (int si : lc_source_group) {
                auto hd = load_hd_for_submap(si, false);
                if (!hd) continue;
                const Eigen::Isometry3d T_ref_submap = ref_pose.inverse() * submap_poses[si];
                const bool has_int = hd->intensities != nullptr;
                for (size_t pi = 0; pi < hd->size(); pi++) {
                  const Eigen::Vector3d p_ref = T_ref_submap * hd->points[pi].head<3>();
                  all_hd_points.push_back(Eigen::Vector4d(p_ref.x(), p_ref.y(), p_ref.z(), 1.0));
                  all_hd_ints.push_back(has_int ? hd->intensities[pi] : 0.0);
                }
              }
              if (all_hd_points.empty()) return {hd_t, nullptr};
              auto hd_merged = std::make_shared<gtsam_points::PointCloudCPU>();
              hd_merged->num_points = all_hd_points.size();
              hd_merged->points_storage = std::move(all_hd_points);
              hd_merged->points = hd_merged->points_storage.data();
              hd_merged->intensities_storage = std::move(all_hd_ints);
              hd_merged->intensities = hd_merged->intensities_storage.data();
              logger->info("[Loop HD] Merged {} source submaps: {} HD points (covs deferred)", lc_source_group.size(), hd_merged->num_points);
              return {hd_t, hd_merged};
            };
            if (session_hd_paths.empty()) manual_loop_close_modal->load_hd_callback = nullptr;
          }
        }

        ImGui::Separator();
        if (ImGui::MenuItem("Preview data")) {
          static int preview_counter = 0;
          const float hue = std::fmod(preview_counter * 0.618f, 1.0f);
          const float r = std::abs(std::sin(hue * 6.28f)) * 0.7f + 0.3f;
          const float g = std::abs(std::sin((hue + 0.33f) * 6.28f)) * 0.7f + 0.3f;
          const float b = std::abs(std::sin((hue + 0.66f) * 6.28f)) * 0.7f + 0.3f;
          show_preview(frame_id, r, g, b);
          preview_counter++;
        }
        if (ImGui::MenuItem("Clear selection")) {
          clear_all_previews();
          lc_source_group.clear();
          lc_target_frame_id = -1;
          lc_source_frame_id = -1;
        }
      }
    }

    if (type == PickType::POINTS) {
      if (ImGui::MenuItem("Bundle adjustment (Plane)")) {
        bundle_adjustment_modal->set_frames(submaps, submap_poses, right_clicked_pos.cast<double>());
      }
      if (ImGui::MenuItem("Identify source")) {
        // Find which submap has the nearest point to the clicked position
        const Eigen::Vector3d click_pos = right_clicked_pos.cast<double>();
        int best_submap = -1;
        double best_dist2 = std::numeric_limits<double>::max();
        for (int si = 0; si < static_cast<int>(submaps.size()); si++) {
          if (!submaps[si] || !submaps[si]->frame) continue;
          const auto& frame = submaps[si]->frame;
          const Eigen::Isometry3d& pose = submap_poses[si];
          // Check a subsample for speed
          const size_t step = std::max<size_t>(1, frame->size() / 500);
          for (size_t pi = 0; pi < frame->size(); pi += step) {
            const Eigen::Vector3d wp = pose * frame->points[pi].head<3>();
            const double d2 = (wp - click_pos).squaredNorm();
            if (d2 < best_dist2) {
              best_dist2 = d2;
              best_submap = si;
            }
          }
        }
        if (best_submap >= 0) {
          // Highlight the source sphere in yellow
          auto viewer = guik::LightViewer::instance();
          const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submaps[best_submap]->id);
          const Eigen::Affine3f sp = submap_poses[best_submap].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
          viewer->update_drawable("sphere_" + std::to_string(submaps[best_submap]->id), glk::Primitives::sphere(),
            guik::FlatColor(1.0f, 1.0f, 0.0f, 1.0f, sp).add("info_values", info));
          // Draw a line from click point to sphere
          std::vector<Eigen::Vector3f> line_verts = {right_clicked_pos, submap_poses[best_submap].translation().cast<float>()};
          viewer->update_drawable("identify_line", std::make_shared<glk::ThinLines>(line_verts, false),
            guik::FlatColor(1.0f, 1.0f, 0.0f, 1.0f));
          logger->info("[Identify] Nearest source: submap {} (dist={:.3f}m)", best_submap, std::sqrt(best_dist2));
        }
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Find which submap owns the nearest point\nto this location. Highlights the sphere in yellow\nwith a connecting line.");
    }

    ImGui::Separator();
    {
      std::lock_guard<std::mutex> lock(last_factor_mutex);
      const bool has_undo = !last_factor_indices.empty();
      if (!has_undo) ImGui::BeginDisabled();
      char undo_label[64];
      std::snprintf(undo_label, sizeof(undo_label), "Undo last factor (%zu)", last_factor_indices.size());
      if (ImGui::MenuItem(undo_label)) {
        request_undo_last = true;
        GlobalMappingCallbacks::request_to_optimize();
        logger->info("[Undo] Requested undo of {} factors", last_factor_indices.size());
      }
      if (!has_undo) {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("No factors to undo.");
        ImGui::EndDisabled();
      } else {
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove the last set of loop closure factors\nand re-optimize the graph.");
      }
    }

    ImGui::Separator();
    if (ImGui::MenuItem("Source finder", nullptr, source_finder_active)) {
      if (!source_finder_active) {
        source_finder_active = true;
        source_finder_pos = right_clicked_pos;
        source_finder_pos.z() += source_finder_height * 0.5f;  // click point = base, offset up
        // Initialize gizmo matrix (Rz * Ry)
        const float yr = source_finder_yaw * static_cast<float>(M_PI) / 180.0f;
        const float pr = source_finder_pitch * static_cast<float>(M_PI) / 180.0f;
        Eigen::Matrix3f rot = (Eigen::AngleAxisf(yr, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pr, Eigen::Vector3f::UnitY())).toRotationMatrix();
        Eigen::Matrix4f gizmo_m = Eigen::Matrix4f::Identity();
        gizmo_m.block<3, 3>(0, 0) = rot;
        gizmo_m.block<3, 1>(0, 3) = source_finder_pos;
        source_finder_gizmo->set_model_matrix(gizmo_m);
        source_finder_update_cylinder();
        source_finder_scan_fast();
      } else {
        source_finder_active = false;
        source_finder_hits.clear();
        auto viewer = guik::LightViewer::instance();
        viewer->remove_drawable("source_finder_cylinder");
      viewer->remove_drawable("team_lines");
      viewer->remove_drawable("identify_line");
        // Restore sphere colors
        static const float sc[][3] = {
          {1.0f, 0.0f, 0.0f}, {1.0f, 0.85f, 0.0f}, {0.0f, 0.8f, 0.2f},
          {1.0f, 0.6f, 0.0f}, {0.8f, 0.0f, 0.8f}, {0.0f, 0.8f, 0.8f},
        };
        for (int pi = 0; pi < static_cast<int>(submaps.size()); pi++) {
          if (!submaps[pi]) continue;
          const int ci = submaps[pi]->session_id % 6;
          const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submaps[pi]->id);
          const Eigen::Affine3f sp = submap_poses[pi].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
          viewer->update_drawable("sphere_" + std::to_string(submaps[pi]->id), glk::Primitives::sphere(),
            guik::FlatColor(sc[ci][0], sc[ci][1], sc[ci][2], 0.5f, sp).add("info_values", info).make_transparent());
        }
      }
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Place a cylinder probe to find which submaps\nhave points in this area. Useful for identifying\nsources of misaligned features.");

    // Extension point for subclass context menu items
    if (extra_context_menu_items) extra_context_menu_items();

    ImGui::EndPopup();
  }
}

/**
 * @brief Run modals
 */
void InteractiveViewer::run_modals() {
  std::vector<gtsam::NonlinearFactor::shared_ptr> factors;

  // Capture relaxation params before run() clears the modal
  const bool do_relax = manual_loop_close_modal->relax_neighbors;
  const int relax_rad = manual_loop_close_modal->relax_radius;
  const float relax_sc = manual_loop_close_modal->relax_scale;
  const bool relax_btw = manual_loop_close_modal->relax_between;
  const bool relax_gps = manual_loop_close_modal->relax_gps;
  const int relax_tgt = lc_target_frame_id;
  const int relax_src = lc_source_frame_id;

  auto manual_loop_close_factor = manual_loop_close_modal->run();
  if (manual_loop_close_factor) {
    needs_session_merge = false;
    // Clear data previews — they no longer reflect the state after factor creation
    auto viewer = guik::LightViewer::instance();
    for (int pi = 0; pi < static_cast<int>(submaps.size()); pi++) {
      viewer->remove_drawable("lc_preview_" + std::to_string(pi));
    }
    // Reset sphere colors
    static const float sc[][3] = {
      {1.0f, 0.0f, 0.0f}, {1.0f, 0.85f, 0.0f}, {0.0f, 0.8f, 0.2f},
      {1.0f, 0.6f, 0.0f}, {0.8f, 0.0f, 0.8f}, {0.0f, 0.8f, 0.8f},
    };
    for (int pi = 0; pi < static_cast<int>(submaps.size()); pi++) {
      if (!submaps[pi]) continue;
      const int ci = submaps[pi]->session_id % 6;
      const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submaps[pi]->id);
      const Eigen::Affine3f sp = submap_poses[pi].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
      viewer->update_drawable("sphere_" + std::to_string(submaps[pi]->id), glk::Primitives::sphere(),
        guik::FlatColor(sc[ci][0], sc[ci][1], sc[ci][2], 0.5f, sp).add("info_values", info).make_transparent());
    }

    // Queue relaxation if enabled
    if (do_relax && relax_tgt >= 0 && relax_src >= 0) {
      RelaxationRequest req;
      req.center_key = relax_tgt;
      req.radius = relax_rad;
      req.scale = relax_sc;
      req.relax_between = relax_btw;
      req.relax_gps = relax_gps;
      pending_relaxations.insert(std::vector<RelaxationRequest>{req});
      // Also queue a second relaxation centered on source
      if (relax_src != relax_tgt) {
        RelaxationRequest req2;
        req2.center_key = relax_src;
        req2.radius = relax_rad;
        req2.scale = relax_sc;
        req2.relax_between = relax_btw;
        req2.relax_gps = relax_gps;
        pending_relaxations.insert(std::vector<RelaxationRequest>{req2});
      }
      logger->info("[Relax] Queued relaxation: radius={} scale={}x between={} gps={}", relax_rad, relax_sc, relax_btw, relax_gps);
    }
  }
  factors.push_back(manual_loop_close_factor);

  // For team alignment: create factors for ALL submaps in both teams
  // The ICP transform is rigid for the whole team — each submap's correction is direct.
  if (manual_loop_close_factor && !lc_target_group.empty() && !lc_source_group.empty()) {
    auto* bf = dynamic_cast<gtsam::BetweenFactor<gtsam::Pose3>*>(manual_loop_close_factor.get());
    if (bf) {
      const Eigen::Isometry3d T_tgt_src(bf->measured().matrix());
      const Eigen::Isometry3d T_tgt_ref = submap_poses[lc_target_frame_id];
      const Eigen::Isometry3d T_src_ref = submap_poses[lc_source_frame_id];
      auto noise = bf->noiseModel();

      int extra_factors = 0;
      // Connect each source submap to the target reference
      for (int si : lc_source_group) {
        if (si == lc_source_frame_id) continue;  // already has the main factor
        const Eigen::Isometry3d T_tgt_ref_si = T_tgt_src * T_src_ref.inverse() * submap_poses[si];
        factors.push_back(gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
          X(lc_target_frame_id), X(si), gtsam::Pose3(T_tgt_ref_si.matrix()), noise));
        extra_factors++;
      }
      // Connect each target submap to the source reference
      for (int ti : lc_target_group) {
        if (ti == lc_target_frame_id) continue;
        const Eigen::Isometry3d T_ti_src_ref = submap_poses[ti].inverse() * T_tgt_ref * T_tgt_src;
        factors.push_back(gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
          X(ti), X(lc_source_frame_id), gtsam::Pose3(T_ti_src_ref.matrix()), noise));
        extra_factors++;
      }
      logger->info("[Team align] Created {} extra factors for {} source + {} target submaps",
                   extra_factors, lc_source_group.size() - 1, lc_target_group.size() - 1);
    }
    lc_target_group.clear();
  }

  factors.push_back(bundle_adjustment_modal->run());

  factors.erase(std::remove(factors.begin(), factors.end(), nullptr), factors.end());

  if (factors.size()) {
    logger->info("optimizing...");
    new_factors.insert(factors);
    GlobalMappingCallbacks::request_to_optimize();

    // Force full LOD refresh (same as "Unload all") so data re-renders at updated poses
    if (lod_enabled) {
      lod_load_full_sd = false;
      lod_load_full_hd = false;
      loaded_hd_points = 0;
      total_gpu_bytes = 0;
      auto vw = guik::LightViewer::instance();
      for (int i = 0; i < static_cast<int>(render_states.size()); i++) {
        auto& rs = render_states[i];
        if (i < static_cast<int>(submaps.size()) && submaps[i]) {
          const int sid = submaps[i]->id;
          vw->remove_drawable("submap_" + std::to_string(sid));
          vw->remove_drawable("bbox_" + std::to_string(sid));
          vw->remove_drawable("coord_" + std::to_string(sid));
          vw->remove_drawable("sphere_" + std::to_string(sid));
        }
        rs.gpu_bytes = 0;
        rs.hd_points = 0;
        rs.current_lod = SubmapLOD::UNLOADED;
      }
      logger->info("[LOD] Full unload for post-optimization refresh");
    }
  }
}

// ---------------------------------------------------------------------------
// Async LOD worker: prepares CPU data on background thread, schedules GL upload
// ---------------------------------------------------------------------------

void InteractiveViewer::detect_hd_frames(const std::string& map_path) {
  const std::string hd_path = map_path + "/hd_frames";
  if (!boost::filesystem::is_directory(hd_path)) {
    hd_available = false;
    logger->warn("[HD] No hd_frames directory at {} (hd_path='{}', map_path='{}')", hd_path, hd_path, map_path);
    return;
  }

  hd_frames_path = hd_path;
  size_t total_pts = 0;
  int frame_count = 0;
  int subdir_count = 0;
  int missing_meta = 0;

  for (boost::filesystem::directory_iterator it(hd_path), end; it != end; ++it) {
    if (!boost::filesystem::is_directory(it->path())) continue;
    subdir_count++;
    const std::string meta_path = it->path().string() + "/frame_meta.json";
    if (!boost::filesystem::exists(meta_path)) { missing_meta++; continue; }

    std::ifstream ifs(meta_path);
    const auto j = nlohmann::json::parse(ifs, nullptr, false);
    if (!j.is_discarded()) {
      total_pts += j.value("num_points", 0);
      frame_count++;
    }
  }

  if (frame_count > 0) {
    hd_available = true;
    total_hd_points = total_pts;
    logger->info("[HD] Found {} HD frames, {:.1f} M total points in {}",
                 frame_count, static_cast<double>(total_pts) / 1e6, hd_path);
  } else {
    logger->warn("[HD] {} has {} subdirs but {} missing frame_meta.json (no HD loaded)",
                 hd_path, subdir_count, missing_meta);
  }
}

void InteractiveViewer::lod_worker_task() {
  while (!lod_worker_kill) {
    auto items = lod_work_queue.get_all_and_clear();
    if (items.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
      continue;
    }

    // Sort by distance — closest submaps get uploaded first
    std::sort(items.begin(), items.end(), [](const LODWorkItem& a, const LODWorkItem& b) {
      return a.distance < b.distance;
    });

    for (auto& item : items) {
      if (lod_worker_kill) break;

      // Check if still wanted (may have been demoted during preparation)
      const SubmapLOD expected_state = item.load_hd ? SubmapLOD::LOADING_HD : SubmapLOD::LOADING;
      if (item.submap_index >= static_cast<int>(render_states.size()) ||
          render_states[item.submap_index].current_lod != expected_state) {
        continue;
      }

      const auto& submap = item.submap;
      if (!submap) continue;

      // ============================================================
      // HD frame loading from disk
      // ============================================================
      if (item.load_hd && !item.hd_path.empty()) {
        // Collect all HD frame data for this submap's frames
        const std::string& session_hd_path = item.hd_path;
        std::vector<Eigen::Vector3f> all_points;
        std::vector<Eigen::Vector4f> all_normal_colors;
        std::vector<float> all_intensities;
        std::vector<float> all_range;
        std::vector<float> all_gps_time;
        std::vector<float> all_ground;
        std::vector<Eigen::Vector4f> all_rgb_colors;
        size_t hd_point_count = 0;
        int frames_found = 0, frames_missing = 0;
        bool any_frame_had_normals_hd = false;  // must stay parallel with all_points

        for (const auto& frame : submap->frames) {
          char dir_name[16];
          std::snprintf(dir_name, sizeof(dir_name), "%08ld", frame->id);
          const std::string frame_dir = session_hd_path + "/" + dir_name;

          // Read frame_meta.json for point count and stamp
          const std::string meta_path = frame_dir + "/frame_meta.json";
          if (!boost::filesystem::exists(meta_path)) { frames_missing++; continue; }
          std::ifstream meta_ifs(meta_path);
          const auto meta = nlohmann::json::parse(meta_ifs, nullptr, false);
          if (meta.is_discarded()) continue;
          const int num_pts = meta.value("num_points", 0);
          const double frame_stamp = meta.value("stamp", 0.0);
          if (num_pts == 0) continue;

          // Read points.bin
          const std::string pts_path = frame_dir + "/points.bin";
          std::ifstream pts_ifs(pts_path, std::ios::binary);
          if (!pts_ifs) continue;
          std::vector<Eigen::Vector3f> frame_points(num_pts);
          pts_ifs.read(reinterpret_cast<char*>(frame_points.data()), sizeof(Eigen::Vector3f) * num_pts);

          // Read normals.bin (optional)
          std::vector<Eigen::Vector3f> frame_normals;
          {
            std::ifstream nrm_ifs(frame_dir + "/normals.bin", std::ios::binary);
            if (nrm_ifs) {
              frame_normals.resize(num_pts);
              nrm_ifs.read(reinterpret_cast<char*>(frame_normals.data()), sizeof(Eigen::Vector3f) * num_pts);
              any_frame_had_normals_hd = true;
            }
          }

          // Read intensities.bin (optional)
          std::vector<float> frame_intensities;
          {
            std::ifstream ifs(frame_dir + "/intensities.bin", std::ios::binary);
            if (ifs) {
              frame_intensities.resize(num_pts);
              ifs.read(reinterpret_cast<char*>(frame_intensities.data()), sizeof(float) * num_pts);
            }
          }

          // Read range.bin (optional)
          std::vector<float> frame_range;
          {
            std::ifstream ifs(frame_dir + "/range.bin", std::ios::binary);
            if (ifs) {
              frame_range.resize(num_pts);
              ifs.read(reinterpret_cast<char*>(frame_range.data()), sizeof(float) * num_pts);
            }
          }

          // Read times.bin + compute gps_time (optional)
          std::vector<float> frame_times;
          {
            std::ifstream ifs(frame_dir + "/times.bin", std::ios::binary);
            if (ifs) {
              frame_times.resize(num_pts);
              ifs.read(reinterpret_cast<char*>(frame_times.data()), sizeof(float) * num_pts);
            }
          }

          // Read aux_ground.bin (optional — from Dynamic filter "Save ground to HD")
          std::vector<float> frame_ground;
          {
            std::ifstream ifs(frame_dir + "/aux_ground.bin", std::ios::binary);
            if (ifs) {
              frame_ground.resize(num_pts);
              ifs.read(reinterpret_cast<char*>(frame_ground.data()), sizeof(float) * num_pts);
            }
          }

          // Read aux_rgb.bin (optional — from Colorize Apply)
          std::vector<float> frame_rgb;
          {
            std::ifstream ifs(frame_dir + "/aux_rgb.bin", std::ios::binary);
            if (ifs) {
              frame_rgb.resize(num_pts * 3);
              ifs.read(reinterpret_cast<char*>(frame_rgb.data()), sizeof(float) * num_pts * 3);
            }
          }

          // Compute optimized world pose for this frame
          const Eigen::Isometry3d T_world_endpoint_L = submap->T_world_origin * submap->T_origin_endpoint_L;
          const Eigen::Isometry3d T_odom_imu0 = submap->frames.front()->T_world_imu;
          const Eigen::Isometry3d T_world_imu = T_world_endpoint_L * T_odom_imu0.inverse() * frame->T_world_imu;
          const Eigen::Isometry3d T_world_lidar = T_world_imu * frame->T_lidar_imu.inverse();
          Eigen::Matrix3f R; Eigen::Vector3f t_vec;
          if (lod_use_voxelized) { R = Eigen::Matrix3f::Identity(); t_vec = Eigen::Vector3f::Zero(); }
          else { R = T_world_lidar.rotation().cast<float>(); t_vec = T_world_lidar.translation().cast<float>(); }

          // Transform points to world frame, filtering by min range
          constexpr float HD_MIN_RANGE = 1.5f;
          for (int pi = 0; pi < num_pts; pi++) {
            const float r = frame_range.empty() ? frame_points[pi].norm() : frame_range[pi];
            if (!lod_use_voxelized && r < HD_MIN_RANGE) continue;  // skip range filter for voxelized

            all_points.push_back(R * frame_points[pi] + t_vec);
            // Keep all_normal_colors in lockstep with all_points:
            //   - if ANY frame so far had normals, every accepted point gets an entry (zero if frame missing)
            //   - until the first normals.bin is seen, no entries are pushed (mirroring unavailability)
            if (!frame_normals.empty()) {
              const Eigen::Vector3f wn = (R * frame_normals[pi]).normalized();
              all_normal_colors.push_back(((wn + Eigen::Vector3f::Ones()) * 0.5f).homogeneous());
            } else if (any_frame_had_normals_hd) {
              all_normal_colors.push_back(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1.0f));
            }
            if (!frame_intensities.empty()) all_intensities.push_back(frame_intensities[pi]);
            if (!frame_range.empty()) all_range.push_back(frame_range[pi]);
            if (!frame_times.empty()) {
              const float base = (gps_time_base > 0.0) ? static_cast<float>(frame_stamp - gps_time_base) : 0.0f;
              all_gps_time.push_back(base + frame_times[pi]);
            }
            if (!frame_ground.empty()) all_ground.push_back(frame_ground[pi]);
            if (!frame_rgb.empty() && pi * 3 + 2 < static_cast<int>(frame_rgb.size())) {
              all_rgb_colors.push_back(Eigen::Vector4f(frame_rgb[pi*3], frame_rgb[pi*3+1], frame_rgb[pi*3+2], 1.0f));
            }
            hd_point_count++;
          }
          frames_found++;
        }

        logger->info("[HD worker] submap {} (id={}): {} frames found, {} missing, {} HD points",
                     item.submap_index, submap->id, frames_found, frames_missing, hd_point_count);
        if (all_points.empty()) continue;

        const int idx = item.submap_index;
        const int sid = item.session_id;
        const int submap_id = submap->id;
        const size_t hd_pts = hd_point_count;
        auto normals_buf = std::make_shared<std::vector<Eigen::Vector4f>>(std::move(all_normal_colors));
        auto intensities_buf = std::make_shared<std::vector<float>>(std::move(all_intensities));
        auto range_buf = std::make_shared<std::vector<float>>(std::move(all_range));
        auto gps_time_buf = std::make_shared<std::vector<float>>(std::move(all_gps_time));
        auto ground_buf = std::make_shared<std::vector<float>>(std::move(all_ground));
        auto rgb_colors_buf = std::make_shared<std::vector<Eigen::Vector4f>>(std::move(all_rgb_colors));

        // Convert Vector3f→Vector4d on worker thread (NOT in invoke — avoids GL thread stall)
        const int n_pts = static_cast<int>(all_points.size());
        auto points_4d = std::make_shared<std::vector<Eigen::Vector4d>>(n_pts);
        for (int pi = 0; pi < n_pts; pi++) {
          const auto& p = all_points[pi];
          (*points_4d)[pi] = Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0);
        }

        guik::LightViewer::instance()->invoke([this, idx, sid, submap_id, points_4d, normals_buf,
                                                intensities_buf, range_buf, gps_time_buf, ground_buf, rgb_colors_buf, hd_pts, n_pts] {
          auto viewer = guik::LightViewer::instance();
          if (idx >= static_cast<int>(render_states.size()) ||
              render_states[idx].current_lod != SubmapLOD::LOADING_HD) {
            return;
          }

          auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points_4d->data(), n_pts);

          // Determine if RGB mode is active
          bool rgb_mode_active = false;
          if (color_mode >= 3) {
            const int aidx = color_mode - 3;
            if (aidx >= 0 && aidx < static_cast<int>(aux_attribute_names.size()) && aux_attribute_names[aidx] == "RGB")
              rgb_mode_active = true;
          }
          // Upload vertex colors: RGB if available and active, otherwise normals.
          // Guard on exact size match — if only some HD frames had normals.bin,
          // normals_buf will be shorter than n_pts and uploading would under-populate
          // a GPU attribute buffer (-> crash at draw time).
          if (rgb_mode_active && rgb_colors_buf->size() == static_cast<size_t>(n_pts)) {
            cloud_buffer->add_color(*rgb_colors_buf);
          } else if (normals_buf->size() == static_cast<size_t>(n_pts)) {
            cloud_buffer->add_color(*normals_buf);
          } else if (!normals_buf->empty()) {
            logger->warn("[HD worker] submap {}: normals buffer size mismatch ({} vs {} points) — skipping normal-color upload",
                         submap_id, normals_buf->size(), n_pts);
          }

          // Upload aux attribute buffers for colormap rendering
          // Also track min/max per attribute from HD data (used as fallback for cmap_range)
          auto track_hd_range = [this](const std::string& name, const std::vector<float>& buf) {
            float bmin = std::numeric_limits<float>::max(), bmax = std::numeric_limits<float>::lowest();
            for (float v : buf) { if (std::isfinite(v)) { bmin = std::min(bmin, v); bmax = std::max(bmax, v); } }
            if (bmin <= bmax) {
              auto it = hd_attr_ranges.find(name);
              if (it == hd_attr_ranges.end()) { hd_attr_ranges[name] = Eigen::Vector2f(bmin, bmax); }
              else { it->second[0] = std::min(it->second[0], bmin); it->second[1] = std::max(it->second[1], bmax); }
            }
          };
          std::string first_aux_name;
          if (intensities_buf->size() == static_cast<size_t>(n_pts)) {
            cloud_buffer->add_buffer("intensity", *intensities_buf);
            track_hd_range("intensity", *intensities_buf);
            if (first_aux_name.empty()) first_aux_name = "intensity";
          }
          if (range_buf->size() == static_cast<size_t>(n_pts)) {
            cloud_buffer->add_buffer("range", *range_buf);
            track_hd_range("range", *range_buf);
            if (first_aux_name.empty()) first_aux_name = "range";
          }
          if (gps_time_buf->size() == static_cast<size_t>(n_pts)) {
            cloud_buffer->add_buffer("gps_time", *gps_time_buf);
            track_hd_range("gps_time", *gps_time_buf);
            if (first_aux_name.empty()) first_aux_name = "gps_time";
          }
          if (ground_buf->size() == static_cast<size_t>(n_pts)) {
            cloud_buffer->add_buffer("ground", *ground_buf);
            track_hd_range("ground", *ground_buf);
            if (first_aux_name.empty()) first_aux_name = "ground";
            // Register "ground" in aux_attribute_names if not already present
            if (std::find(aux_attribute_names.begin(), aux_attribute_names.end(), "ground") == aux_attribute_names.end()) {
              aux_attribute_names.push_back("ground");
            }
          }
          // RGB: register in dropdown but don't override normals vertex color for now
          if (rgb_colors_buf->size() == static_cast<size_t>(n_pts)) {
            if (std::find(aux_attribute_names.begin(), aux_attribute_names.end(), "RGB") == aux_attribute_names.end()) {
              aux_attribute_names.push_back("RGB");
            }
          }

          // Set active colormap buffer
          if (color_mode >= 3) {
            const int aux_idx = color_mode - 3;
            if (aux_idx < static_cast<int>(aux_attribute_names.size())) {
              cloud_buffer->set_colormap_buffer(aux_attribute_names[aux_idx]);
            }
          } else if (!first_aux_name.empty()) {
            cloud_buffer->set_colormap_buffer(first_aux_name);
          }

          const Eigen::Vector4f color = glk::colormap_categoricalf(glk::COLORMAP::TURBO, sid, 6);
          const Eigen::Vector4i info(static_cast<int>(PickType::POINTS), 0, 0, submap_id);
          auto shader_setting = guik::Rainbow().add("info_values", info).set_color(color).set_alpha(points_alpha);
          switch (color_mode) {
            case 0: break;  // RAINBOW
            case 1: shader_setting.set_color_mode(guik::ColorMode::FLAT_COLOR); break;  // SESSION
            case 2: shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLOR); break;  // NORMAL
            default: {
              const int aidx = color_mode - 3;
              if (aidx >= 0 && aidx < static_cast<int>(aux_attribute_names.size()) && aux_attribute_names[aidx] == "RGB")
                shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLOR);
              else shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLORMAP);
              break;
            }
          }

          viewer->update_drawable("submap_" + std::to_string(submap_id), cloud_buffer, shader_setting);
          viewer->remove_drawable("bbox_" + std::to_string(submap_id));

          const size_t new_bytes = cloud_buffer->memory_usage();
          const size_t budget = static_cast<size_t>(lod_vram_budget_mb * 1024 * 1024);
          if (total_gpu_bytes + new_bytes > budget) {
            render_states[idx].current_lod = SubmapLOD::BBOX;
            viewer->remove_drawable("submap_" + std::to_string(submap_id));
            return;
          }

          total_gpu_bytes -= render_states[idx].gpu_bytes;  // remove old SD/HD bytes
          render_states[idx].gpu_bytes = new_bytes;
          total_gpu_bytes += new_bytes;
          // Subtract old HD points before adding new (prevents double-counting on LOD cycling)
          if (render_states[idx].hd_points > 0) {
            loaded_hd_points = (loaded_hd_points >= render_states[idx].hd_points) ? loaded_hd_points - render_states[idx].hd_points : 0;
          }
          render_states[idx].hd_points = hd_pts;
          loaded_hd_points += hd_pts;
          render_states[idx].current_lod = SubmapLOD::HD;
        });
        continue;
      }

      // ============================================================
      // SD submap loading (existing path)
      // ============================================================
      if (!submap->frame || submap->frame->size() == 0) continue;

      const int n = submap->frame->size();

      // --- Prepare CPU-side data on background thread (no GL calls) ---
      // Normal colors
      std::shared_ptr<std::vector<Eigen::Vector4f>> normal_colors_buf;
      if (submap->frame->normals) {
        normal_colors_buf = std::make_shared<std::vector<Eigen::Vector4f>>(n);
        for (int ni = 0; ni < n; ni++) {
          (*normal_colors_buf)[ni] = ((submap->frame->normals[ni].head<3>().cast<float>() + Eigen::Vector3f::Ones()) * 0.5f).homogeneous();
        }
      }

      // Aux attribute buffers
      auto aux_buffers = std::make_shared<std::unordered_map<std::string, std::vector<float>>>();
      for (const auto& attr_name : aux_attribute_names) {
        const auto it = submap->frame->aux_attributes.find(attr_name);
        if (it == submap->frame->aux_attributes.end()) continue;
        const size_t elem_size = it->second.first;
        std::vector<float> vals(n);
        if (elem_size == sizeof(float)) {
          const float* data = static_cast<const float*>(it->second.second);
          std::copy(data, data + n, vals.begin());
        } else if (elem_size == sizeof(double)) {
          const double* data = static_cast<const double*>(it->second.second);
          const double base = (attr_name == "gps_time") ? gps_time_base : 0.0;
          for (int ki = 0; ki < n; ki++) vals[ki] = static_cast<float>(data[ki] - base);
        } else {
          continue;
        }
        (*aux_buffers)[attr_name] = std::move(vals);
      }

      // Capture values for the invoke closure
      const int idx = item.submap_index;
      const int sid = item.session_id;
      const SubMap::ConstPtr submap_capture = submap;  // shared_ptr keeps data alive
      const Eigen::Affine3f submap_pose_f = item.pose.cast<float>();

      // Schedule GL upload on render thread
      guik::LightViewer::instance()->invoke([this, idx, sid, submap_capture, normal_colors_buf, aux_buffers, submap_pose_f] {
        auto viewer = guik::LightViewer::instance();
        // Re-check: still in LOADING state?
        if (idx >= static_cast<int>(render_states.size()) ||
            render_states[idx].current_lod != SubmapLOD::LOADING) {
          return;  // was cancelled
        }

        const int n = submap_capture->frame->size();
        const int submap_id = submap_capture->id;

        // Use the same constructor as the non-LOD path (proven to work)
        auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(submap_capture->frame->points, n);

        if (normal_colors_buf) {
          cloud_buffer->add_color(*normal_colors_buf);
        }

        std::string first_aux_name;
        for (const auto& [name, vals] : *aux_buffers) {
          cloud_buffer->add_buffer(name, vals);
          if (first_aux_name.empty()) first_aux_name = name;
        }

        if (color_mode >= 3) {
          const int aux_idx = color_mode - 3;
          if (aux_idx < static_cast<int>(aux_attribute_names.size())) {
            cloud_buffer->set_colormap_buffer(aux_attribute_names[aux_idx]);
          }
        } else if (!first_aux_name.empty()) {
          cloud_buffer->set_colormap_buffer(first_aux_name);
        }

        const Eigen::Vector4f color = glk::colormap_categoricalf(glk::COLORMAP::TURBO, sid, 6);
        const Eigen::Vector4i info(static_cast<int>(PickType::POINTS), 0, 0, submap_id);

        auto shader_setting = guik::Rainbow(submap_pose_f).add("info_values", info).set_color(color).set_alpha(points_alpha);
        switch (color_mode) {
          case 0: break;
          case 1: shader_setting.set_color_mode(guik::ColorMode::FLAT_COLOR); break;
          case 2: shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLOR); break;
          default: shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLORMAP); break;
        }

        if (enable_partial_rendering) {
          cloud_buffer->enable_partial_rendering(partial_rendering_budget);
          shader_setting.add("dynamic_object", 0).make_transparent();
        }

        viewer->update_drawable("submap_" + std::to_string(submap_id), cloud_buffer, shader_setting);

        // Remove bbox drawable
        viewer->remove_drawable("bbox_" + std::to_string(submap_id));

        // Budget check before committing GPU memory
        const size_t new_bytes = cloud_buffer->memory_usage();
        const size_t budget = static_cast<size_t>(lod_vram_budget_mb * 1024 * 1024);
        if (total_gpu_bytes + new_bytes > budget) {
          // Over budget — discard this upload, revert to BBOX
          render_states[idx].current_lod = SubmapLOD::BBOX;
          return;
        }

        render_states[idx].gpu_bytes = new_bytes;
        total_gpu_bytes += new_bytes;
        render_states[idx].current_lod = SubmapLOD::SD;
      });
    }
  }
}

/**
 * @brief Update viewer
 */
void InteractiveViewer::update_viewer() {
  auto viewer = guik::LightViewer::instance();

  // Ensure render_states vector matches submaps size
  render_states.resize(submaps.size());

  // Initialize gps_time_base from the first submap if not yet set.
  // Must happen before any buffer creation (LOD or non-LOD) to ensure
  // consistent float32 precision when subtracting the base.
  if (gps_time_base == 0.0 && !submaps.empty()) {
    for (const auto& sm : submaps) {
      if (!sm || !sm->frame) continue;
      const auto it = sm->frame->aux_attributes.find("gps_time");
      if (it == sm->frame->aux_attributes.end() || it->second.first != sizeof(double)) continue;
      const double* data = static_cast<const double*>(it->second.second);
      const int n = sm->frame->size();
      for (int k = 0; k < n; k++) {
        if (std::isfinite(data[k]) && data[k] > 0.0) {
          gps_time_base = data[k];
          break;
        }
      }
      if (gps_time_base != 0.0) break;
    }
  }

  Eigen::Vector2f auto_z_range(0.0f, 0.0f);
  for (int i = 0; i < submaps.size(); i++) {
    const auto& submap = submaps[i];
    if (!submap) continue;
    const Eigen::Affine3f submap_pose = submap_poses[i].cast<float>();
    auto& rs = render_states[i];

    auto_z_range[0] = std::min(auto_z_range[0], submap_pose.translation().z());
    auto_z_range[1] = std::max(auto_z_range[1], submap_pose.translation().z());

    // Compute world AABB once per submap (used by lod_update callback)
    if (!rs.bbox_computed && submap->frame && submap->frame->size() > 0) {
      Eigen::Vector3f bmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
      Eigen::Vector3f bmax(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
      for (int pi = 0; pi < submap->frame->size(); pi++) {
        const Eigen::Vector3f wp = (submap_poses[i] * submap->frame->points[pi]).head<3>().cast<float>();
        bmin = bmin.cwiseMin(wp);
        bmax = bmax.cwiseMax(wp);
      }
      rs.world_bbox = Eigen::AlignedBox3f(bmin, bmax);
      rs.bbox_computed = true;
    } else if (!rs.bbox_computed) {
      const Eigen::Vector3f pos = submap_pose.translation();
      rs.world_bbox = Eigen::AlignedBox3f(pos - Eigen::Vector3f::Constant(1.0f), pos + Eigen::Vector3f::Constant(1.0f));
      rs.bbox_computed = true;
    }

    // HD submaps: update color mode but NOT model_matrix (points already in world frame)
    if (rs.current_lod == SubmapLOD::HD) {
      const std::string submap_name = "submap_" + std::to_string(submap->id);
      auto drawable = viewer->find_drawable(submap_name);
      if (drawable.first) {
        if (color_mode >= 3) {
          const int aux_idx = color_mode - 3;
          if (aux_idx < static_cast<int>(aux_attribute_names.size())) {
            auto cb = std::dynamic_pointer_cast<const glk::PointCloudBuffer>(drawable.second);
            auto cloud_buffer = std::const_pointer_cast<glk::PointCloudBuffer>(cb);
            if (cloud_buffer) {
              cloud_buffer->set_colormap_buffer(aux_attribute_names[aux_idx]);
            }
          }
        }
        switch (color_mode) {
          case 0: drawable.first->set_color_mode(guik::ColorMode::RAINBOW); break;
          case 1: drawable.first->set_color_mode(guik::ColorMode::FLAT_COLOR); break;
          case 2: drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLOR); break;
          default: {
            const int aidx2 = color_mode - 3;
            if (aidx2 >= 0 && aidx2 < static_cast<int>(aux_attribute_names.size()) && aux_attribute_names[aidx2] == "RGB")
              drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLOR);
            else drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLORMAP);
            break;
          }
        }
      }
      continue;
    }
    if (rs.current_lod == SubmapLOD::LOADING_HD) {
      continue;
    }

    // When LOD is enabled, lod_update callback handles all transitions.
    // update_viewer() updates poses and color modes on existing SD drawables.
    if (lod_enabled) {
      const std::string submap_name = "submap_" + std::to_string(submap->id);
      auto drawable = viewer->find_drawable(submap_name);
      if (drawable.first && rs.current_lod == SubmapLOD::SD) {
        drawable.first->add("model_matrix", submap_pose.matrix());

        // Color mode switching (same logic as non-LOD path)
        if (color_mode >= 3) {
          const int aux_idx = color_mode - 3;
          if (aux_idx < static_cast<int>(aux_attribute_names.size())) {
            auto cb = std::dynamic_pointer_cast<const glk::PointCloudBuffer>(drawable.second);
            auto cloud_buffer = std::const_pointer_cast<glk::PointCloudBuffer>(cb);
            if (cloud_buffer) {
              cloud_buffer->set_colormap_buffer(aux_attribute_names[aux_idx]);
            }
          }
        }
        switch (color_mode) {
          case 0: drawable.first->set_color_mode(guik::ColorMode::RAINBOW); break;
          case 1: drawable.first->set_color_mode(guik::ColorMode::FLAT_COLOR); break;
          case 2: drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLOR); break;
          default: {
            const int aidx2 = color_mode - 3;
            if (aidx2 >= 0 && aidx2 < static_cast<int>(aux_attribute_names.size()) && aux_attribute_names[aidx2] == "RGB")
              drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLOR);
            else drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLORMAP);
            break;
          }
        }
      }
      continue;
    }

    // --- Non-LOD path: original behavior (all submaps loaded immediately) ---
    const std::string submap_name = "submap_" + std::to_string(submap->id);
    auto drawable = viewer->find_drawable(submap_name);
    if (drawable.first) {
      drawable.first->add("model_matrix", submap_pose.matrix());

      // For aux colormap modes, switch the named buffer without re-uploading
      if (color_mode >= 3) {
        const int aux_idx = color_mode - 3;
        if (aux_idx < static_cast<int>(aux_attribute_names.size())) {
          auto cb = std::dynamic_pointer_cast<const glk::PointCloudBuffer>(drawable.second);
          auto cloud_buffer = std::const_pointer_cast<glk::PointCloudBuffer>(cb);
          if (cloud_buffer) {
            cloud_buffer->set_colormap_buffer(aux_attribute_names[aux_idx]);
          }
        }
      }

      // 0=RAINBOW, 1=SESSION(FLAT), 2=NORMAL, 3+=aux
      switch (color_mode) {
        case 0:
          drawable.first->set_color_mode(guik::ColorMode::RAINBOW);
          break;
        case 1:
          drawable.first->set_color_mode(guik::ColorMode::FLAT_COLOR);
          break;
        case 2:
          drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLOR);
          break;
        default:
          drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLORMAP);
          break;
      }
    } else {
      const Eigen::Vector4f color = glk::colormap_categoricalf(glk::COLORMAP::TURBO, submap->session_id, 6);

      const Eigen::Vector4i info(static_cast<int>(PickType::POINTS), 0, 0, submap->id);
      auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(submap->frame->points, submap->frame->size());

      // Always upload normal colors into the color VBO so NORMAL mode works after mode-switch
      if (submap->frame->normals) {
        const int n = submap->frame->size();
        std::vector<Eigen::Vector4f> normal_colors(n);
        for (int ni = 0; ni < n; ni++) {
          normal_colors[ni] = ((submap->frame->normals[ni].head<3>().cast<float>() + Eigen::Vector3f::Ones()) * 0.5f).homogeneous();
        }
        cloud_buffer->add_color(normal_colors);
      }

      // Always upload ALL float/double aux attributes as named GL buffers so mode-switching works
      std::string first_aux_name;
      for (const auto& attr_name : aux_attribute_names) {
        const auto it = submap->frame->aux_attributes.find(attr_name);
        if (it == submap->frame->aux_attributes.end()) continue;
        const size_t elem_size = it->second.first;
        const int n = submap->frame->size();
        std::vector<float> vals(n);
        if (elem_size == sizeof(float)) {
          const float* data = static_cast<const float*>(it->second.second);
          std::copy(data, data + n, vals.begin());
        } else if (elem_size == sizeof(double)) {
          const double* data = static_cast<const double*>(it->second.second);
          // Set gps_time_base on first encounter. Scan past any zero-sentinel voxels
          // (FIRST-mode boundary voxels store 0) to find the first valid timestamp.
          if (attr_name == "gps_time" && gps_time_base == 0.0) {
            double min_gps = std::numeric_limits<double>::max();
            for (int i = 0; i < n; i++) {
              if (std::isfinite(data[i])) min_gps = std::min(min_gps, data[i]);
            }
            if (min_gps < std::numeric_limits<double>::max()) gps_time_base = min_gps;
          }
          const double base = (attr_name == "gps_time") ? gps_time_base : 0.0;
          for (int i = 0; i < n; i++) vals[i] = static_cast<float>(data[i] - base);
        } else {
          continue;
        }
        cloud_buffer->add_buffer(attr_name, vals);
        if (first_aux_name.empty()) {
          first_aux_name = attr_name;
        }
      }

      // Arm the colormap buffer for the current (or default) aux attribute
      if (color_mode >= 3) {
        const int aux_idx = color_mode - 3;
        if (aux_idx < static_cast<int>(aux_attribute_names.size())) {
          cloud_buffer->set_colormap_buffer(aux_attribute_names[aux_idx]);
        }
      } else if (!first_aux_name.empty()) {
        cloud_buffer->set_colormap_buffer(first_aux_name);
      }

      auto shader_setting = guik::Rainbow(submap_pose).add("info_values", info).set_color(color).set_alpha(points_alpha);

      switch (color_mode) {
        case 0: break;  // RAINBOW
        case 1: shader_setting.set_color_mode(guik::ColorMode::FLAT_COLOR); break;
        case 2: shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLOR); break;
        default: shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLORMAP); break;
      }

      if (enable_partial_rendering) {
        cloud_buffer->enable_partial_rendering(partial_rendering_budget);
        shader_setting.add("dynamic_object", 0).make_transparent();
      }

      viewer->update_drawable("submap_" + std::to_string(submap->id), cloud_buffer, shader_setting);

      // Track GPU memory and LOD state (inline path)
      rs.current_lod = SubmapLOD::SD;
      rs.gpu_bytes = cloud_buffer->memory_usage();
      total_gpu_bytes += rs.gpu_bytes;
    }

    const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submap->id);

    viewer->update_drawable(
      "coord_" + std::to_string(submap->id),
      glk::Primitives::coordinate_system(),
      guik::VertexColor(submap_pose * Eigen::UniformScaling<float>(coord_scale)).add("info_values", info));

    // Session-based sphere colors for multi-map disambiguation
    static const float session_colors[][3] = {
      {1.0f, 0.0f, 0.0f},  // session 0: red
      {1.0f, 0.85f, 0.0f}, // session 1: yellow
      {0.0f, 0.8f, 0.2f},  // session 2: green
      {1.0f, 0.6f, 0.0f},  // session 3: orange
      {0.8f, 0.0f, 0.8f},  // session 4: purple
      {0.0f, 0.8f, 0.8f},  // session 5: cyan
    };
    const int ci = submap->session_id % 6;
    viewer->update_drawable(
      "sphere_" + std::to_string(submap->id),
      glk::Primitives::sphere(),
      guik::FlatColor(session_colors[ci][0], session_colors[ci][1], session_colors[ci][2], 0.5f,
                      submap_pose * Eigen::UniformScaling<float>(sphere_scale)).add("info_values", info).make_transparent());
  }

  viewer->shader_setting().add<Eigen::Vector2f>("z_range", z_range + auto_z_range);

  // Set colormap range for aux attribute rendering
  if (color_mode >= 3) {
    // If range is unset (min > max sentinel), full-scan ALL submaps for global min/max.
    // BUG FIXED: the old path used samples[] capped at AUX_SAMPLE_CAP; if early submaps
    // filled the cap, later submaps never contributed to percentile_range, so the maximum
    // was underestimated and all late-submap points clamped to max color.
    // Fix: track global_min/global_max per-submap (one scalar each), independent of the cap,
    // then set aux_cmap_range from those true global extremes.
    if (aux_cmap_range[0] > aux_cmap_range[1]) {
      const int aux_idx = color_mode - 3;
      if (aux_idx < static_cast<int>(aux_attribute_names.size())) {
        const auto& attr_name = aux_attribute_names[aux_idx];
        auto& samples = aux_attr_samples[attr_name];
        if (samples.empty()) {
          int scanned = 0;
          float global_min = std::numeric_limits<float>::max();
          float global_max = std::numeric_limits<float>::lowest();
          for (const auto& sm : submaps) {
            const auto it = sm->frame->aux_attributes.find(attr_name);
            if (it == sm->frame->aux_attributes.end()) continue;
            scanned++;
            const int n = sm->frame->size();
            float sm_min = std::numeric_limits<float>::max();
            float sm_max = std::numeric_limits<float>::lowest();
            if (it->second.first == sizeof(float)) {
              const float* data = static_cast<const float*>(it->second.second);
              for (int k = 0; k < n; k++) {
                const float v = data[k];
                if (std::isfinite(v)) {
                  sm_min = std::min(sm_min, v);
                  sm_max = std::max(sm_max, v);
                  if (samples.size() < AUX_SAMPLE_CAP) samples.push_back(v);
                }
              }
            } else if (it->second.first == sizeof(double)) {
              const double* data = static_cast<const double*>(it->second.second);
              if (attr_name == "gps_time" && gps_time_base == 0.0) {
                double min_gps = std::numeric_limits<double>::max();
                for (int k = 0; k < n; k++) {
                  if (std::isfinite(data[k])) min_gps = std::min(min_gps, data[k]);
                }
                if (min_gps < std::numeric_limits<double>::max()) gps_time_base = min_gps;
              }
              const double base = (attr_name == "gps_time") ? gps_time_base : 0.0;
              for (int k = 0; k < n; k++) {
                const float v = static_cast<float>(data[k] - base);
                if (std::isfinite(v)) {
                  sm_min = std::min(sm_min, v);
                  sm_max = std::max(sm_max, v);
                  if (samples.size() < AUX_SAMPLE_CAP) samples.push_back(v);
                }
              }
            }
            // Accumulate per-submap extremes into global — this is cap-independent
            if (sm_min <= sm_max) {
              global_min = std::min(global_min, sm_min);
              global_max = std::max(global_max, sm_max);
            }
          }
          std::cerr << "[FULL-SCAN] attr=" << attr_name
                    << " scanned=" << scanned << "/" << submaps.size() << " submaps"
                    << " global_min=" << global_min << " global_max=" << global_max
                    << " samples=" << samples.size() << std::endl;
          // Use global min/max (covers all submaps regardless of cap) as the range.
          // Fall back to percentile of samples if no global range was computed.
          // Final fallback: HD tile ranges (for attributes like "ground" that only exist in HD data).
          if (global_min <= global_max) {
            aux_cmap_range = Eigen::Vector2f(global_min, global_max);
          } else if (!samples.empty()) {
            aux_cmap_range = percentile_range(samples);
          } else {
            auto hd_it = hd_attr_ranges.find(attr_name);
            if (hd_it != hd_attr_ranges.end()) {
              aux_cmap_range = hd_it->second;
              std::cerr << "[FULL-SCAN] Using HD range for " << attr_name << ": [" << hd_it->second[0] << ", " << hd_it->second[1] << "]" << std::endl;
            }
          }
        }
      }
    }
    if (aux_cmap_range[0] <= aux_cmap_range[1]) {
      std::cerr << "[CMAP] uploading cmap_range=[" << aux_cmap_range[0] << ", " << aux_cmap_range[1] << "]" << std::endl;
      viewer->shader_setting().add<Eigen::Vector2f>("cmap_range", aux_cmap_range);
    }
  }

  std::vector<Eigen::Vector3f> factor_lines;
  std::vector<Eigen::Vector4f> factor_colors;
  factor_lines.reserve(global_factors.size() * 2);
  factor_colors.reserve(global_factors.size() * 2);

  const auto get_position = [this](const gtsam::Key key) -> Eigen::Vector3d {
    gtsam::Symbol symbol(key);
    switch (symbol.chr()) {
      case 'x':
        if (symbol.index() < submaps.size() && submaps[symbol.index()]) {
          return submaps[symbol.index()]->T_world_origin.translation();
        }
        return Eigen::Vector3d::Zero();
      case 'e': {
        const int right = symbol.index() % 2;
        const int submap_id = (symbol.index() - right) / 2;

        if (submap_id >= static_cast<int>(submaps.size()) || !submaps[submap_id]) return Eigen::Vector3d::Zero();
        const auto& submap = submaps[submap_id];
        const auto& T_origin_endpoint = right ? submap->T_origin_endpoint_R : submap->T_origin_endpoint_L;
        const Eigen::Isometry3d T_world_endpoint = submap->T_world_origin * T_origin_endpoint;

        return T_world_endpoint.translation();
      }
    }

    std::cout << "warning: unknown symbol " << symbol << std::endl;
    return Eigen::Vector3d(0.0, 0.0, 0.0);
  };

  for (const auto& factor : global_factors) {
    // Skip factors from hidden sessions
    gtsam::Symbol sym1(std::get<1>(factor)), sym2(std::get<2>(factor));
    if (sym1.chr() == 'x' && sym1.index() < submaps.size() && (!submaps[sym1.index()] || hidden_sessions.count(submaps[sym1.index()]->session_id))) continue;
    if (sym2.chr() == 'x' && sym2.index() < submaps.size() && (!submaps[sym2.index()] || hidden_sessions.count(submaps[sym2.index()]->session_id))) continue;

    FactorType type = std::get<0>(factor);
    factor_lines.push_back(get_position(std::get<1>(factor)).cast<float>());
    factor_lines.push_back(get_position(std::get<2>(factor)).cast<float>());

    Eigen::Vector4f color;
    switch (type) {
      case FactorType::MATCHING_COST:
        color = Eigen::Vector4f(0.0f, 1.0f, 0.0f, factors_alpha);
        break;
      case FactorType::BETWEEN:
        color = Eigen::Vector4f(0.0f, 0.0f, 1.0f, factors_alpha);
        break;
      case FactorType::IMU:
        color = Eigen::Vector4f(1.0f, 0.0f, 0.0f, factors_alpha);
        break;
    }

    factor_colors.push_back(color);
    factor_colors.push_back(color);
  }

  viewer->update_drawable("factors", std::make_shared<glk::ThinLines>(factor_lines, factor_colors), guik::VertexColor().set_alpha(factors_alpha));

  // Per-session colored trajectory
  static const Eigen::Vector4f traj_colors[] = {
    {0.2f, 1.0f, 0.2f, 1.0f},   // session 0: green
    {1.0f, 0.85f, 0.0f, 1.0f},  // session 1: yellow
    {0.3f, 0.7f, 1.0f, 1.0f},   // session 2: light blue
    {1.0f, 0.5f, 0.0f, 1.0f},   // session 3: orange
    {0.9f, 0.3f, 0.9f, 1.0f},   // session 4: pink
    {0.0f, 0.9f, 0.9f, 1.0f},   // session 5: cyan
  };
  std::vector<Eigen::Vector3f> traj;
  std::vector<Eigen::Vector4f> traj_cols;
  for (const auto& submap : submaps) {
    if (!submap) continue;
    if (hidden_sessions.count(submap->session_id)) continue;

    const Eigen::Vector4f col = traj_colors[submap->session_id % 6];
    const Eigen::Isometry3d T_world_endpoint_L = submap->T_world_origin * submap->T_origin_endpoint_L;
    const Eigen::Isometry3d T_odom_imu0 = submap->frames.front()->T_world_imu;
    for (const auto& frame : submap->frames) {
      const Eigen::Isometry3d T_world_imu = T_world_endpoint_L * T_odom_imu0.inverse() * frame->T_world_imu;
      traj.emplace_back(T_world_imu.translation().cast<float>());
      traj_cols.push_back(col);
    }
  }

  auto traj_line = std::make_shared<glk::ThinLines>(traj, traj_cols, true);
  traj_line->set_line_width(2.0f);
  viewer->update_drawable("traj", traj_line, guik::VertexColor());

  // Per-frame triads — one ThinLines drawable with 3 axes per frame (RGB)
  if (draw_frames) {
    std::vector<Eigen::Vector3f> fc_pts;
    std::vector<Eigen::Vector4f> fc_cols;
    const float s = frame_coord_scale;
    const Eigen::Vector4f cx(1.0f, 0.2f, 0.2f, 1.0f);
    const Eigen::Vector4f cy(0.2f, 1.0f, 0.2f, 1.0f);
    const Eigen::Vector4f cz(0.3f, 0.5f, 1.0f, 1.0f);
    for (const auto& submap : submaps) {
      if (!submap || submap->frames.empty()) continue;
      if (hidden_sessions.count(submap->session_id)) continue;
      const Eigen::Isometry3d T_world_endpoint_L = submap->T_world_origin * submap->T_origin_endpoint_L;
      const Eigen::Isometry3d T_odom_imu0 = submap->frames.front()->T_world_imu;
      for (const auto& frame : submap->frames) {
        const Eigen::Isometry3d T = T_world_endpoint_L * T_odom_imu0.inverse() * frame->T_world_imu;
        const Eigen::Vector3f o = T.translation().cast<float>();
        const Eigen::Matrix3f R = T.rotation().cast<float>();
        fc_pts.push_back(o); fc_pts.push_back(o + R.col(0) * s); fc_cols.push_back(cx); fc_cols.push_back(cx);
        fc_pts.push_back(o); fc_pts.push_back(o + R.col(1) * s); fc_cols.push_back(cy); fc_cols.push_back(cy);
        fc_pts.push_back(o); fc_pts.push_back(o + R.col(2) * s); fc_cols.push_back(cz); fc_cols.push_back(cz);
      }
    }
    if (!fc_pts.empty()) {
      auto lines = std::make_shared<glk::ThinLines>(fc_pts, fc_cols, false);
      lines->set_line_width(1.5f);
      viewer->update_drawable("frame_coords", lines, guik::VertexColor());
    } else {
      viewer->remove_drawable("frame_coords");
    }
  }
}

void InteractiveViewer::odometry_on_new_frame(const EstimationFrame::ConstPtr& new_frame) {
  invoke([this, new_frame] {
    auto viewer = guik::viewer();
    auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(new_frame->frame->points, new_frame->frame->size());

    trajectory->add_odom(new_frame->stamp, new_frame->T_world_sensor(), 1);
    const Eigen::Isometry3d pose = trajectory->odom2world(new_frame->T_world_sensor());
    viewer->update_drawable("current", cloud_buffer, guik::FlatOrange(pose).set_point_scale(2.0f));
  });
}

/**
 * @brief New submap insertion callback
 */
void InteractiveViewer::globalmap_on_insert_submap(const SubMap::ConstPtr& submap) {
  std::shared_ptr<Eigen::Isometry3d> pose(new Eigen::Isometry3d(submap->T_world_origin));
  invoke([this, submap, pose] {
    if (submaps.size() && submaps.back()->session_id != submap->session_id) {
      needs_session_merge = true;
    }

    trajectory->update_anchor(submap->frames[submap->frames.size() / 2]->stamp, submap->T_world_origin);

    // Populate aux_attribute_names from first submap; guaranteed to run for every new submap.
    if (aux_attribute_names.empty()) {
      for (const auto& attrib : submap->frame->aux_attributes) {
        const size_t elem_size = attrib.second.first;
        if (elem_size == sizeof(float) || elem_size == sizeof(double)) {
          aux_attribute_names.push_back(attrib.first);
        }
      }
      // Auto-promote default ColorMode to "intensity" if present. Only fires on
      // first submap; user edits from the ColorMode combo are preserved.
      if (color_mode == 0) {
        for (size_t i = 0; i < aux_attribute_names.size(); i++) {
          if (aux_attribute_names[i] == "intensity") {
            color_mode = 3 + static_cast<int>(i);
            const int sel = scalar_colormap_per_attr[aux_attribute_names[i]];
            guik::LightViewer::instance()->set_colormap(static_cast<glk::COLORMAP>(sel));
            // Defer a re-apply until at least one SD drawable is live: the
            // update_viewer() at the end of this insert runs before the LOD
            // worker uploads the first drawable, so the color_mode switch
            // wouldn't otherwise propagate to the initial render.
            scalar_default_refresh_pending = true;
            break;
          }
        }
      }
    }

    // Extend per-attr sample pool from this submap, then recompute percentile range
    if (color_mode >= 3) {
      const int aux_idx = color_mode - 3;
      if (aux_idx < static_cast<int>(aux_attribute_names.size())) {
        const auto& attr_name = aux_attribute_names[aux_idx];
        const auto it = submap->frame->aux_attributes.find(attr_name);
        if (it != submap->frame->aux_attributes.end()) {
          auto& samples = aux_attr_samples[attr_name];
          const int n = submap->frame->size();
          if (it->second.first == sizeof(float)) {
            const float* data = static_cast<const float*>(it->second.second);
            for (int k = 0; k < n; k++) {
              const float v = data[k];
              if (std::isfinite(v) && samples.size() < AUX_SAMPLE_CAP) samples.push_back(v);
            }
          } else if (it->second.first == sizeof(double)) {
            const double* data = static_cast<const double*>(it->second.second);
            if (attr_name == "gps_time" && gps_time_base == 0.0) {
              double min_gps = std::numeric_limits<double>::max();
              for (int k = 0; k < n; k++) {
                if (std::isfinite(data[k])) min_gps = std::min(min_gps, data[k]);
              }
              if (min_gps < std::numeric_limits<double>::max()) gps_time_base = min_gps;
            }
            const double base = (attr_name == "gps_time") ? gps_time_base : 0.0;
            for (int k = 0; k < n; k++) {
              const float v = static_cast<float>(data[k] - base);
              if (std::isfinite(v) && samples.size() < AUX_SAMPLE_CAP) samples.push_back(v);
            }
          }
          if (!samples.empty()) {
            aux_cmap_range = percentile_range(samples);
          }
        }
      }
    }

    submap_poses.push_back(*pose);
    submaps.push_back(submap);

    // Apply pending GPS sigma for this submap
    const int si = static_cast<int>(submaps.size()) - 1;
    float sigma = -1.0f;
    const auto sig_it = pending_sigma_map.find(si);
    if (sig_it != pending_sigma_map.end()) {
      sigma = sig_it->second;
      // Add as per-point aux_attribute for colormap rendering
      if (submap->frame && submap->frame->size() > 0) {
        const int n = submap->frame->size();
        auto* sigma_data = new float[n];
        std::fill(sigma_data, sigma_data + n, sigma);
        const_cast<gtsam_points::PointCloud*>(submap->frame.get())->aux_attributes["gps_sigma"] =
          std::make_pair(sizeof(float), static_cast<void*>(sigma_data));
        // Register gps_sigma in aux names if first time
        if (std::find(aux_attribute_names.begin(), aux_attribute_names.end(), "gps_sigma") == aux_attribute_names.end()) {
          aux_attribute_names.push_back("gps_sigma");
        }
      }
    }
    submap_gps_sigma.push_back(sigma);

    update_viewer();
  });
}

/**
 * @brief Submap pose update callback
 */
void InteractiveViewer::globalmap_on_update_submaps(const std::vector<SubMap::Ptr>& updated_submaps) {
  std::vector<Eigen::Isometry3d> poses(updated_submaps.size());
  std::transform(updated_submaps.begin(), updated_submaps.end(), poses.begin(), [](const SubMap::ConstPtr& submap) { return submap->T_world_origin; });

  invoke([this, poses] {
    for (int i = 0; i < std::min(submaps.size(), poses.size()); i++) {
      submap_poses[i] = poses[i];
    }
    update_viewer();
  });
}

/**
 * @brief Smoother update callback
 */
void InteractiveViewer::globalmap_on_smoother_update(gtsam_points::ISAM2Ext& isam2, gtsam::NonlinearFactorGraph& new_factors, gtsam::Values& new_values) {
  // Handle undo request
  if (request_undo_last.exchange(false)) {
    std::lock_guard<std::mutex> lock(last_factor_mutex);
    if (!last_factor_indices.empty()) {
      gtsam::FactorIndices remove_indices(last_factor_indices.begin(), last_factor_indices.end());
      isam2.update(gtsam::NonlinearFactorGraph(), gtsam::Values(), remove_indices);
      logger->info("[Undo] Removed {} factors from ISAM2", last_factor_indices.size());
      last_factor_indices.clear();
    }
  }

  auto factors = this->new_factors.get_all_and_clear();

  // Explicitly move the poses of merged submaps to the coordinate system of the origin of the first session
  for (const auto& factor : factors) {
    auto between = dynamic_cast<const gtsam::BetweenFactor<gtsam::Pose3>*>(factor.get());
    if (!between) {
      continue;
    }

    gtsam::Key target = between->key1();
    gtsam::Key source = between->key2();
    if (!new_values.exists(target) && !new_values.exists(source)) {
      continue;
    }

    if (new_values.exists(target)) {
      logger->error("Target exists in new_values");
      continue;
    }

    logger->info("Moving newly added submaps to the coordinate system of the origin of the first session");
    if (!new_values.exists(source)) {
      std::swap(target, source);
    }

    const gtsam::Pose3 T_target_source = between->measured();
    const gtsam::Pose3 T_worldt_target(submaps[gtsam::Symbol(target).index()]->T_world_origin.matrix());
    const gtsam::Pose3 T_worlds_source(submaps[gtsam::Symbol(source).index()]->T_world_origin.matrix());
    const gtsam::Pose3 T_worldt_worlds = T_worldt_target * T_target_source * T_worlds_source.inverse();

    int moved_count = 0;
    for (const auto& value : new_values) {
      const gtsam::Symbol symbol(value.key);
      if (symbol.chr() != 'x') {
        continue;
      }

      new_values.insert_or_assign(value.key, T_worldt_worlds * value.value.cast<gtsam::Pose3>());
      moved_count++;
    }

    logger->info("Moved {} submaps", moved_count);
  }

  new_factors.add(factors);

  // Record how many viewer factors we added (for undo tracking in smoother_update_result)
  if (!factors.empty()) {
    std::lock_guard<std::mutex> lock(last_factor_mutex);
    last_factor_indices.clear();
    // Store count as a sentinel — actual indices computed after ISAM2 update
    last_factor_indices.resize(factors.size(), SIZE_MAX);
    logger->info("[Undo] {} factors pending index assignment", factors.size());
  }

  // --- Factor relaxation ---
  auto relax_requests = pending_relaxations.get_all_and_clear();
  if (!relax_requests.empty()) {
    // Build set of submap indices to relax
    std::unordered_set<int> relax_keys;
    bool do_between = false, do_gps = false;
    float scale = 1.0f;
    for (const auto& req : relax_requests) {
      const int lo = std::max(0, req.center_key - req.radius);
      const int hi = std::min(static_cast<int>(submaps.size()) - 1, req.center_key + req.radius);
      for (int i = lo; i <= hi; i++) relax_keys.insert(i);
      do_between |= req.relax_between;
      do_gps |= req.relax_gps;
      scale = std::max(scale, req.scale);
    }

    const auto& graph = isam2.getFactorsUnsafe();
    gtsam::FactorIndices remove_indices;
    gtsam::NonlinearFactorGraph relaxed_factors;

    for (size_t fi = 0; fi < graph.size(); fi++) {
      const auto& f = graph[fi];
      if (!f) continue;

      // Check if any key of this factor is in the relax set
      bool involves_relaxed = false;
      for (const auto& key : f->keys()) {
        gtsam::Symbol sym(key);
        if (sym.chr() == 'x' && relax_keys.count(static_cast<int>(sym.index()))) {
          involves_relaxed = true;
          break;
        }
      }
      if (!involves_relaxed) continue;

      try {
        auto* bf = dynamic_cast<gtsam::BetweenFactor<gtsam::Pose3>*>(f.get());
        auto* ptp = dynamic_cast<gtsam::PoseTranslationPrior<gtsam::Pose3>*>(f.get());

        if (bf && do_between) {
          auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(bf->noiseModel());
          if (!gaussian) continue;
          remove_indices.push_back(fi);
          auto scaled_noise = gtsam::noiseModel::Diagonal::Sigmas(gaussian->sigmas() * static_cast<double>(scale));
          relaxed_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(bf->key1(), bf->key2(), bf->measured(), scaled_noise);
        } else if (ptp && do_gps) {
          auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(ptp->noiseModel());
          if (!gaussian) continue;
          remove_indices.push_back(fi);
          auto scaled_noise = gtsam::noiseModel::Diagonal::Sigmas(gaussian->sigmas() * static_cast<double>(scale));
          relaxed_factors.emplace_shared<gtsam::PoseTranslationPrior<gtsam::Pose3>>(ptp->keys()[0], ptp->measured(), scaled_noise);
        }
      } catch (...) {
        logger->warn("[Relax] Skipped factor {} — unsupported noise model", fi);
      }
    }

    if (!remove_indices.empty()) {
      // Atomic: remove originals + add relaxed replacements in one ISAM2 update
      isam2.update(relaxed_factors, gtsam::Values(), remove_indices);
      logger->info("[Relax] Replaced {} factors with relaxed versions (scale {}x, {} submaps affected)", relaxed_factors.size(), scale, relax_keys.size());
    }
  }

  std::vector<std::tuple<FactorType, gtsam::Key, gtsam::Key>> inserted_factors;

  for (const auto& factor : new_factors) {
    if (dynamic_cast<gtsam::BetweenFactor<gtsam::Pose3>*>(factor.get())) {
      inserted_factors.push_back(std::make_tuple(FactorType::BETWEEN, factor->keys()[0], factor->keys()[1]));
    }
    if (dynamic_cast<gtsam_points::IntegratedMatchingCostFactor*>(factor.get())) {
      inserted_factors.push_back(std::make_tuple(FactorType::MATCHING_COST, factor->keys()[0], factor->keys()[1]));
    }
#ifdef GTSAM_POINTS_USE_CUDA
    if (dynamic_cast<gtsam_points::IntegratedVGICPFactorGPU*>(factor.get())) {
      inserted_factors.push_back(std::make_tuple(FactorType::MATCHING_COST, factor->keys()[0], factor->keys()[1]));
    }
#endif
    if (dynamic_cast<gtsam::ImuFactor*>(factor.get())) {
      inserted_factors.push_back(std::make_tuple(FactorType::IMU, factor->keys()[0], factor->keys()[2]));
    }
  }

  // Extract per-submap GPS sigma from PoseTranslationPrior factors
  std::unordered_map<int, float> sigma_map;  // submap_index → avg sigma
  auto extract_sigma = [&](const gtsam::NonlinearFactorGraph& graph) {
    for (const auto& factor : graph) {
      auto* ptp = dynamic_cast<gtsam::PoseTranslationPrior<gtsam::Pose3>*>(factor.get());
      if (!ptp) continue;
      gtsam::Symbol sym(ptp->keys()[0]);
      if (sym.chr() != 'x') continue;
      const int idx = static_cast<int>(sym.index());
      try {
        const auto sigmas = ptp->noiseModel()->sigmas();
        const float avg_sigma = static_cast<float>(sigmas.mean());
        sigma_map[idx] = avg_sigma;
      } catch (...) {}
    }
  };
  extract_sigma(isam2.getFactorsUnsafe());
  extract_sigma(new_factors);
  logger->info("[GPS sigma] Extracted sigma for {} / {} submaps", sigma_map.size(), submaps.size());

  invoke([this, inserted_factors, sigma_map] {
    global_factors.insert(global_factors.end(), inserted_factors.begin(), inserted_factors.end());

    // Store sigma values — will be applied when submaps are inserted
    for (const auto& [idx, sigma] : sigma_map) {
      pending_sigma_map[idx] = sigma;
    }
    logger->info("[GPS sigma] {} sigma values pending for submap assignment", pending_sigma_map.size());
  });
}

/**
 * @brief Smoother update result callback
 */
void InteractiveViewer::globalmap_on_smoother_update_result(gtsam_points::ISAM2Ext& isam2, const gtsam_points::ISAM2ResultExt& result) {
  const std::string text = result.to_string();
  logger->info("--- smoother_updated ---\n{}", text);

  // Assign actual ISAM2 indices to pending undo factors
  // Our loop closure factors are BetweenFactors added at the end of the graph
  std::lock_guard<std::mutex> lock(last_factor_mutex);
  if (!last_factor_indices.empty() && last_factor_indices[0] == SIZE_MAX) {
    const size_t count = last_factor_indices.size();
    last_factor_indices.clear();
    const auto& graph = isam2.getFactorsUnsafe();
    // Scan backwards for the last `count` BetweenFactor<Pose3> entries
    for (int fi = static_cast<int>(graph.size()) - 1; fi >= 0 && last_factor_indices.size() < count; fi--) {
      if (graph[fi] && dynamic_cast<gtsam::BetweenFactor<gtsam::Pose3>*>(graph[fi].get())) {
        last_factor_indices.push_back(fi);
      }
    }
    logger->info("[Undo] Assigned {} BetweenFactor indices for undo (graph size={})", last_factor_indices.size(), graph.size());
  }
}

bool InteractiveViewer::ok() const {
  return !request_to_terminate;
}

void InteractiveViewer::wait() {
  while (!request_to_terminate) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  stop();
}

void InteractiveViewer::stop() {
  std::this_thread::sleep_for(std::chrono::seconds(1));

  kill_switch = true;
  if (thread.joinable()) {
    thread.join();
  }
}

void InteractiveViewer::clear() {
  std::lock_guard<std::mutex> lock(invoke_queue_mutex);
  invoke_queue.clear();

  submaps.clear();
  submap_poses.clear();
  global_factors.clear();
  new_factors.clear();

  guik::LightViewer::instance()->clear_drawables();
}

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::InteractiveViewer();
}
