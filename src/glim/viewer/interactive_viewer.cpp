#include <glim/viewer/interactive_viewer.hpp>

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
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_points/config.hpp>
#include <gtsam_points/factors/integrated_matching_cost_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/optimizers/isam2_result_ext.hpp>

#include <glk/thin_lines.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/spdlog_sink.hpp>
#include <guik/viewer/light_viewer.hpp>

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
  draw_traj = false;
  draw_points = true;
  draw_factors = true;
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
        const std::string hd_path = (hd_it != session_hd_paths.end()) ? hd_it->second : hd_frames_path;
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
        ImGui::Checkbox("Display HD only", &lod_hd_only);
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("Hide SD submaps, show only HD (LOD 0) data.\nUseful for range filter preview.");
        }
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
    aux_attr_samples.clear();  // discard old samples so full-scan re-collects for new attribute
    aux_cmap_range = Eigen::Vector2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
    update_viewer();
  }
  show_note("Color mode for rendering submaps.\n- RAINBOW=Altitude encoding color\n- SESSION=Session ID\n- NORMAL=Surface normal direction\n- others=Per-point attribute colormap");

  ImGui::Checkbox("Trajectory", &draw_traj);
  ImGui::SameLine();
  ImGui::Checkbox("Submaps", &draw_points);

  ImGui::Checkbox("Factors", &draw_factors);
  ImGui::SameLine();
  ImGui::Checkbox("Spheres", &draw_spheres);

  ImGui::Checkbox("Coords", &draw_coords);

  if (ImGui::BeginMenu("Display settings")) {
    bool do_update_viewer = false;

    do_update_viewer |= ImGui::DragFloat("coord scale", &coord_scale, 0.01f, 0.01f, 100.0f);
    show_note("Submap coordinate system maker scale.");

    do_update_viewer |= ImGui::DragFloat("sphere scale", &sphere_scale, 0.01f, 0.01f, 100.0f);
    show_note("Submap selection sphere maker scale.");

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

      // Just change the orbit center — don't create a new camera.
      // The camera will shift to orbit around the new point.
      viewer->lookat(point);

      // Visual indicator
      const Eigen::Matrix4f vm = viewer->view_matrix();
      const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));
      const float dist = (cam_pos - point).norm();

      Eigen::Affine3f indicator_tf = Eigen::Affine3f::Identity();
      indicator_tf.translate(point);
      indicator_tf.scale(std::max(0.1f, dist * 0.005f));
      viewer->update_drawable("rotation_center", glk::Primitives::sphere(),
        guik::FlatColor(1.0f, 0.4f, 0.0f, 0.6f, indicator_tf).make_transparent());

      std::thread([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
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
gtsam_points::PointCloudCPU::Ptr InteractiveViewer::load_hd_for_submap(int submap_index) const {
  if (submap_index < 0 || submap_index >= static_cast<int>(submaps.size()) || !submaps[submap_index]) {
    return nullptr;
  }
  const auto& submap = submaps[submap_index];
  const auto hd_it = session_hd_paths.find(submap->session_id);
  if (hd_it == session_hd_paths.end()) return nullptr;

  // Load and merge all HD frames for this submap
  std::vector<Eigen::Vector4d> all_points;
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
    std::vector<float> range(npts);
    { std::ifstream f(frame_dir + "/points.bin", std::ios::binary);
      if (!f) continue;
      f.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * npts); }
    { std::ifstream f(frame_dir + "/range.bin", std::ios::binary);
      if (f) f.read(reinterpret_cast<char*>(range.data()), sizeof(float) * npts); }

    // Transform to submap-local frame (NOT world frame — modal handles pose separately)
    const Eigen::Isometry3d T_w_imu = T_ep * T_odom0.inverse() * frame->T_world_imu;
    const Eigen::Isometry3d T_w_lidar = T_w_imu * frame->T_lidar_imu.inverse();
    const Eigen::Isometry3d T_origin_lidar = submap->T_world_origin.inverse() * T_w_lidar;
    const Eigen::Matrix3d R = T_origin_lidar.rotation();
    const Eigen::Vector3d t = T_origin_lidar.translation();

    for (int pi = 0; pi < npts; pi++) {
      if (range[pi] < 1.5f) continue;
      const Eigen::Vector3d lp = R * pts[pi].cast<double>() + t;
      all_points.push_back(Eigen::Vector4d(lp.x(), lp.y(), lp.z(), 1.0));
    }
  }

  if (all_points.empty()) return nullptr;

  // Create PointCloudCPU
  auto cloud = std::make_shared<gtsam_points::PointCloudCPU>();
  cloud->num_points = all_points.size();
  cloud->points_storage = all_points;
  cloud->points = cloud->points_storage.data();

  // Compute normals and covariances using k-NN
  const int k = 10;
  // Build KdTree and find neighbors
  gtsam_points::KdTree tree(cloud->points, cloud->num_points);
  std::vector<int> neighbors(cloud->num_points * k);
  for (int i = 0; i < static_cast<int>(cloud->num_points); i++) {
    std::vector<size_t> k_indices(k, i);
    std::vector<double> k_sq_dists(k);
    tree.knn_search(cloud->points[i].data(), k, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.begin() + k, neighbors.begin() + i * k);
  }

  // Estimate normals and covariances
  CloudCovarianceEstimation cov_estimator(num_threads);
  std::vector<Eigen::Vector4d> normals;
  std::vector<Eigen::Matrix4d> covs;
  cov_estimator.estimate(cloud->points_storage, neighbors, k, normals, covs);
  cloud->add_normals(normals);
  cloud->add_covs(covs);

  logger->info("[HD ICP] Loaded {} HD points for submap {}, computed normals+covs", cloud->num_points, submap_index);
  return cloud;
}

void InteractiveViewer::context_menu() {
  if (ImGui::BeginPopupContextVoid("context menu")) {
    const PickType type = static_cast<PickType>(right_clicked_info[0]);

    if (type == PickType::FRAME) {
      const int frame_id = right_clicked_info[3];
      if (frame_id >= 0 && frame_id < static_cast<int>(submaps.size()) && submaps[frame_id]) {
        ImGui::TextUnformatted(("Submap ID : " + std::to_string(frame_id)).c_str());
        if (ImGui::MenuItem("Loop begin", nullptr, manual_loop_close_modal->is_target_set())) {
          manual_loop_close_modal->set_target(X(frame_id), submaps[frame_id]->frame, submap_poses[frame_id]);
          lc_target_frame_id = frame_id;
          // Update HD callback
          manual_loop_close_modal->load_hd_callback = [this]() -> std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr> {
            auto hd_t = (lc_target_frame_id >= 0) ? load_hd_for_submap(lc_target_frame_id) : nullptr;
            auto hd_s = (lc_source_frame_id >= 0) ? load_hd_for_submap(lc_source_frame_id) : nullptr;
            return {hd_t, hd_s};
          };
          if (session_hd_paths.empty()) manual_loop_close_modal->load_hd_callback = nullptr;
        }
        if (ImGui::MenuItem("Loop end", nullptr, manual_loop_close_modal->is_source_set())) {
          manual_loop_close_modal->set_source(X(frame_id), submaps[frame_id]->frame, submap_poses[frame_id]);
          lc_source_frame_id = frame_id;
          // Update HD callback
          manual_loop_close_modal->load_hd_callback = [this]() -> std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr> {
            auto hd_t = (lc_target_frame_id >= 0) ? load_hd_for_submap(lc_target_frame_id) : nullptr;
            auto hd_s = (lc_source_frame_id >= 0) ? load_hd_for_submap(lc_source_frame_id) : nullptr;
            return {hd_t, hd_s};
          };
          if (session_hd_paths.empty()) manual_loop_close_modal->load_hd_callback = nullptr;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Preview data")) {
          // Random color for this preview
          static int preview_counter = 0;
          const float hue = std::fmod(preview_counter * 0.618f, 1.0f);  // golden ratio spacing
          const float r = std::abs(std::sin(hue * 6.28f)) * 0.7f + 0.3f;
          const float g = std::abs(std::sin((hue + 0.33f) * 6.28f)) * 0.7f + 0.3f;
          const float b = std::abs(std::sin((hue + 0.66f) * 6.28f)) * 0.7f + 0.3f;

          auto viewer = guik::LightViewer::instance();
          const auto& submap = submaps[frame_id];

          // Try to load HD frames, fall back to SD
          std::shared_ptr<glk::PointCloudBuffer> cb;
          bool used_hd = false;
          const auto hd_it = session_hd_paths.find(submap->session_id);
          if (hd_it != session_hd_paths.end()) {
            // Load and merge HD frames for this submap
            std::vector<Eigen::Vector4d> hd_points;
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
              std::vector<float> range(npts);
              { std::ifstream f(frame_dir + "/points.bin", std::ios::binary);
                if (!f) continue;
                f.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * npts); }
              { std::ifstream f(frame_dir + "/range.bin", std::ios::binary);
                if (f) f.read(reinterpret_cast<char*>(range.data()), sizeof(float) * npts); }
              // Transform to world frame
              const Eigen::Isometry3d T_w_imu = T_ep * T_odom0.inverse() * frame->T_world_imu;
              const Eigen::Isometry3d T_w_lidar = T_w_imu * frame->T_lidar_imu.inverse();
              const Eigen::Matrix3d R = T_w_lidar.rotation();
              const Eigen::Vector3d t = T_w_lidar.translation();
              for (int pi = 0; pi < npts; pi++) {
                if (range[pi] < 1.5f) continue;
                const Eigen::Vector3d wp = R * pts[pi].cast<double>() + t;
                hd_points.push_back(Eigen::Vector4d(wp.x(), wp.y(), wp.z(), 1.0));
              }
            }
            if (!hd_points.empty()) {
              cb = std::make_shared<glk::PointCloudBuffer>(hd_points.data(), hd_points.size());
              used_hd = true;
              logger->info("[Preview] Loaded {} HD points for submap {}", hd_points.size(), frame_id);
            }
          }
          if (!cb) {
            cb = std::make_shared<glk::PointCloudBuffer>(submap->frame->points, submap->frame->size());
          }
          const std::string preview_name = "lc_preview_" + std::to_string(frame_id);
          if (used_hd) {
            // HD points already in world frame — no model_matrix
            viewer->update_drawable(preview_name, cb, guik::FlatColor(r, g, b, 0.8f));
          } else {
            // SD points in submap-local frame — need pose transform
            viewer->update_drawable(preview_name, cb,
              guik::FlatColor(r, g, b, 0.8f, submap_poses[frame_id].cast<float>()));
          }

          // Color the sphere to match the preview
          const Eigen::Affine3f sphere_pose = submap_poses[frame_id].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
          const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submap->id);
          viewer->update_drawable(
            "sphere_" + std::to_string(submap->id),
            glk::Primitives::sphere(),
            guik::FlatColor(r, g, b, 0.9f, sphere_pose).add("info_values", info).make_transparent());

          preview_counter++;
        }
        if (ImGui::MenuItem("Clear previews")) {
          auto viewer = guik::LightViewer::instance();
          // Remove all preview drawables and reset sphere colors
          static const float session_colors[][3] = {
            {1.0f, 0.0f, 0.0f}, {1.0f, 0.85f, 0.0f}, {0.0f, 0.8f, 0.2f},
            {1.0f, 0.6f, 0.0f}, {0.8f, 0.0f, 0.8f}, {0.0f, 0.8f, 0.8f},
          };
          for (int pi = 0; pi < static_cast<int>(submaps.size()); pi++) {
            viewer->remove_drawable("lc_preview_" + std::to_string(pi));
            if (submaps[pi]) {
              const int ci = submaps[pi]->session_id % 6;
              const Eigen::Vector4i info(static_cast<int>(PickType::FRAME), 0, 0, submaps[pi]->id);
              const Eigen::Affine3f sp = submap_poses[pi].cast<float>() * Eigen::UniformScaling<float>(sphere_scale);
              viewer->update_drawable(
                "sphere_" + std::to_string(submaps[pi]->id),
                glk::Primitives::sphere(),
                guik::FlatColor(session_colors[ci][0], session_colors[ci][1], session_colors[ci][2], 0.5f, sp)
                  .add("info_values", info).make_transparent());
            }
          }
        }
      }
    }

    if (type == PickType::POINTS) {
      if (ImGui::MenuItem("Bundle adjustment (Plane)")) {
        bundle_adjustment_modal->set_frames(submaps, submap_poses, right_clicked_pos.cast<double>());
      }
    }

    ImGui::EndPopup();
  }
}

/**
 * @brief Run modals
 */
void InteractiveViewer::run_modals() {
  std::vector<gtsam::NonlinearFactor::shared_ptr> factors;

  auto manual_loop_close_factor = manual_loop_close_modal->run();
  if (manual_loop_close_factor) {
    needs_session_merge = false;
  }
  factors.push_back(manual_loop_close_factor);
  factors.push_back(bundle_adjustment_modal->run());

  factors.erase(std::remove(factors.begin(), factors.end(), nullptr), factors.end());

  if (factors.size()) {
    logger->info("optimizing...");
    new_factors.insert(factors);
    GlobalMappingCallbacks::request_to_optimize();
  }
}

// ---------------------------------------------------------------------------
// Async LOD worker: prepares CPU data on background thread, schedules GL upload
// ---------------------------------------------------------------------------

void InteractiveViewer::detect_hd_frames(const std::string& map_path) {
  const std::string hd_path = map_path + "/hd_frames";
  if (!boost::filesystem::is_directory(hd_path)) {
    hd_available = false;
    return;
  }

  hd_frames_path = hd_path;
  size_t total_pts = 0;
  int frame_count = 0;

  for (boost::filesystem::directory_iterator it(hd_path), end; it != end; ++it) {
    if (!boost::filesystem::is_directory(it->path())) continue;
    const std::string meta_path = it->path().string() + "/frame_meta.json";
    if (!boost::filesystem::exists(meta_path)) continue;

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
        size_t hd_point_count = 0;
        int frames_found = 0, frames_missing = 0;

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

          // Compute optimized world pose for this frame
          const Eigen::Isometry3d T_world_endpoint_L = submap->T_world_origin * submap->T_origin_endpoint_L;
          const Eigen::Isometry3d T_odom_imu0 = submap->frames.front()->T_world_imu;
          const Eigen::Isometry3d T_world_imu = T_world_endpoint_L * T_odom_imu0.inverse() * frame->T_world_imu;
          const Eigen::Isometry3d T_world_lidar = T_world_imu * frame->T_lidar_imu.inverse();
          const Eigen::Matrix3f R = T_world_lidar.rotation().cast<float>();
          const Eigen::Vector3f t_vec = T_world_lidar.translation().cast<float>();

          // Transform points to world frame, filtering by min range
          constexpr float HD_MIN_RANGE = 1.5f;  // skip near-sensor noise
          for (int pi = 0; pi < num_pts; pi++) {
            // Range filter: skip points too close to sensor
            const float r = frame_range.empty() ? frame_points[pi].norm() : frame_range[pi];
            if (r < HD_MIN_RANGE) continue;

            all_points.push_back(R * frame_points[pi] + t_vec);
            if (!frame_normals.empty()) {
              const Eigen::Vector3f wn = (R * frame_normals[pi]).normalized();
              all_normal_colors.push_back(((wn + Eigen::Vector3f::Ones()) * 0.5f).homogeneous());
            }
            if (!frame_intensities.empty()) all_intensities.push_back(frame_intensities[pi]);
            if (!frame_range.empty()) all_range.push_back(frame_range[pi]);
            if (!frame_times.empty()) {
              const float base = (gps_time_base > 0.0) ? static_cast<float>(frame_stamp - gps_time_base) : 0.0f;
              all_gps_time.push_back(base + frame_times[pi]);
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

        // Convert Vector3f→Vector4d on worker thread (NOT in invoke — avoids GL thread stall)
        const int n_pts = static_cast<int>(all_points.size());
        auto points_4d = std::make_shared<std::vector<Eigen::Vector4d>>(n_pts);
        for (int pi = 0; pi < n_pts; pi++) {
          const auto& p = all_points[pi];
          (*points_4d)[pi] = Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0);
        }

        guik::LightViewer::instance()->invoke([this, idx, sid, submap_id, points_4d, normals_buf,
                                                intensities_buf, range_buf, gps_time_buf, hd_pts, n_pts] {
          auto viewer = guik::LightViewer::instance();
          if (idx >= static_cast<int>(render_states.size()) ||
              render_states[idx].current_lod != SubmapLOD::LOADING_HD) {
            return;
          }

          auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points_4d->data(), n_pts);

          if (!normals_buf->empty()) {
            cloud_buffer->add_color(*normals_buf);
          }

          // Upload aux attribute buffers for colormap rendering
          std::string first_aux_name;
          if (intensities_buf->size() == static_cast<size_t>(n_pts)) {
            cloud_buffer->add_buffer("intensity", *intensities_buf);
            if (first_aux_name.empty()) first_aux_name = "intensity";
          }
          if (range_buf->size() == static_cast<size_t>(n_pts)) {
            cloud_buffer->add_buffer("range", *range_buf);
            if (first_aux_name.empty()) first_aux_name = "range";
          }
          if (gps_time_buf->size() == static_cast<size_t>(n_pts)) {
            cloud_buffer->add_buffer("gps_time", *gps_time_buf);
            if (first_aux_name.empty()) first_aux_name = "gps_time";
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
            default: shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLORMAP); break;  // AUX
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
          default: drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLORMAP); break;
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
          default: drawable.first->set_color_mode(guik::ColorMode::VERTEX_COLORMAP); break;
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
                if (std::isfinite(v) && v > 0.0f) {
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
          if (global_min <= global_max) {
            aux_cmap_range = Eigen::Vector2f(global_min, global_max);
          } else if (!samples.empty()) {
            aux_cmap_range = percentile_range(samples);
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
              if (std::isfinite(v) && v > 0.0f && samples.size() < AUX_SAMPLE_CAP) samples.push_back(v);
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

  invoke([this, inserted_factors] { global_factors.insert(global_factors.end(), inserted_factors.begin(), inserted_factors.end()); });
}

/**
 * @brief Smoother update result callback
 */
void InteractiveViewer::globalmap_on_smoother_update_result(gtsam_points::ISAM2Ext& isam2, const gtsam_points::ISAM2ResultExt& result) {
  const std::string text = result.to_string();
  logger->info("--- smoother_updated ---\n{}", text);
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
