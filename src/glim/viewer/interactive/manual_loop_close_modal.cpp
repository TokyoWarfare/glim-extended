#include <glim/viewer/interactive/manual_loop_close_modal.hpp>

#include <gtsam/base/Matrix.h>
#include <spdlog/spdlog.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/features/fpfh_estimation.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/registration/ransac.hpp>
#include <gtsam_points/registration/graduated_non_convexity.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <glim/util/convert_to_string.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>

#include <glk/colormap.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/gl_canvas.hpp>
#include <guik/camera/basic_projection_control.hpp>
#include <guik/model_control.hpp>
#include <guik/progress_modal.hpp>
#include <guik/viewer/light_viewer.hpp>

#ifdef GTSAM_USE_TBB
#include <tbb/task_arena.h>
#endif

namespace glim {

ManualLoopCloseModal::ManualLoopCloseModal(const std::shared_ptr<spdlog::logger>& logger, int num_threads) : num_threads(num_threads), request_to_open(false), logger(logger) {
  target_pose.setIdentity();
  source_pose.setIdentity();

  seed = 53123;
  min_distance = 0.5f;
  fpfh_radius = 5.0f;
  global_registration_type = 0;

  ransac_max_iterations = 5000;
  ransac_early_stop_rate = 0.9;
  ransac_inlier_voxel_resolution = 1.0;
  global_registration_4dof = true;

  gnc_max_samples = 10000;

  information_scale = 1.0f;
  max_correspondence_distance = 1.0f;

  canvas.reset(new guik::GLCanvas(Eigen::Vector2i(512, 512)));
  progress_modal.reset(new guik::ProgressModal("manual_loop_close_progress"));
  model_control.reset(new guik::ModelControl("model_control"));
  helper_model_control.reset(new guik::ModelControl("helper_gizmo"));

#ifdef GTSAM_USE_TBB
  tbb_task_arena = std::make_shared<tbb::task_arena>(1);
#endif
}

ManualLoopCloseModal::~ManualLoopCloseModal() {}

bool ManualLoopCloseModal::is_target_set() const {
  return target_key != -1 && target && target_drawable;
}

bool ManualLoopCloseModal::is_source_set() const {
  return source_key != -1 && source && source_drawable;
}

void ManualLoopCloseModal::set_target(const gtsam::Key target_key, const gtsam_points::PointCloud::ConstPtr& target, const Eigen::Isometry3d& target_pose) {
  this->target_key = target_key;
  this->target_pose = target_pose;
  this->target = gtsam_points::PointCloudCPU::clone(*target);

  // Gravity (Z-axis) alignment
  Eigen::Isometry3d T_world_local = Eigen::Isometry3d::Identity();
  T_world_local.linear() = target_pose.linear();
  gtsam_points::transform_inplace(this->target, T_world_local);

  this->target_drawable = std::make_shared<glk::PointCloudBuffer>(this->target->points, this->target->size());
  this->sd_target = gtsam_points::PointCloudCPU::clone(*this->target);  // cache SD
}

void ManualLoopCloseModal::set_source(const gtsam::Key source_key, const gtsam_points::PointCloud::ConstPtr& source, const Eigen::Isometry3d& source_pose) {
  this->source_key = source_key;
  this->source_pose = source_pose;
  this->source = gtsam_points::PointCloudCPU::clone(*source);

  // Gravity (Z-axis) alignment
  Eigen::Isometry3d T_world_local = Eigen::Isometry3d::Identity();
  T_world_local.linear() = source_pose.linear();
  gtsam_points::transform_inplace(this->source, T_world_local);

  this->source_drawable = std::make_shared<glk::PointCloudBuffer>(this->source->points, this->source->size());
  this->sd_source = gtsam_points::PointCloudCPU::clone(*this->source);  // cache SD
  request_to_open = true;
}

void ManualLoopCloseModal::set_submaps(const std::vector<SubMap::ConstPtr>& target_submaps, const std::vector<SubMap::ConstPtr>& source_submaps) {
  logger->info("|targets|={} |sources|={}", target_submaps.size(), source_submaps.size());

  this->target_key = -1;
  this->target = nullptr;
  this->target_pose.setIdentity();
  this->target_drawable = nullptr;
  this->target_submaps = target_submaps;

  this->source_key = -1;
  this->source = nullptr;
  this->source_pose.setIdentity();
  this->source_drawable = nullptr;
  this->source_submaps = source_submaps;

  request_to_open = true;
}

void ManualLoopCloseModal::replace_with_hd(const gtsam_points::PointCloudCPU::Ptr& hd_target, const gtsam_points::PointCloudCPU::Ptr& hd_source) {
  if (hd_target) {
    this->target = hd_target;
    // Apply gravity alignment (rotation only)
    Eigen::Isometry3d T_world_local = Eigen::Isometry3d::Identity();
    T_world_local.linear() = target_pose.linear();
    gtsam_points::transform_inplace(this->target, T_world_local);
    this->target_drawable = std::make_shared<glk::PointCloudBuffer>(this->target->points, this->target->size());
    // Clear cached FPFH
    this->target->aux_attributes.erase("fpfh");
    this->target_fpfh_tree = nullptr;
  }
  if (hd_source) {
    this->source = hd_source;
    Eigen::Isometry3d T_world_local = Eigen::Isometry3d::Identity();
    T_world_local.linear() = source_pose.linear();
    gtsam_points::transform_inplace(this->source, T_world_local);
    this->source_drawable = std::make_shared<glk::PointCloudBuffer>(this->source->points, this->source->size());
    this->source->aux_attributes.erase("fpfh");
    this->source_fpfh_tree = nullptr;
  }
  logger->info("[HD ICP] Replaced with HD data: target={} pts, source={} pts",
               hd_target ? hd_target->size() : 0, hd_source ? hd_source->size() : 0);
}

void ManualLoopCloseModal::clear() {
  helper_gizmo_active = false;
  target_key = -1;
  source_key = -1;
  target = nullptr;
  source = nullptr;
  target_fpfh_tree = nullptr;
  source_fpfh_tree = nullptr;
  target_drawable = nullptr;
  source_drawable = nullptr;
  modal_target_intensity_drawable = nullptr;
  modal_source_intensity_drawable = nullptr;
  modal_intensity_mode = false;
  sd_target = nullptr;
  sd_source = nullptr;
  target_submaps.clear();
  source_submaps.clear();
}

gtsam::NonlinearFactor::shared_ptr ManualLoopCloseModal::run() {
  gtsam::NonlinearFactor::shared_ptr factor;

  if (request_to_open && target && source) {
    // Setup for submap vs submap loop closure
    Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();

    Eigen::Isometry3d R_world_target = Eigen::Isometry3d::Identity();
    R_world_target.linear() = target_pose.linear();

    Eigen::Isometry3d R_world_source = Eigen::Isometry3d::Identity();
    R_world_source.linear() = source_pose.linear();

    init_T_target_source.translation() = (R_world_target * target_pose.inverse() * source_pose * R_world_source.inverse()).translation();
    model_control->set_model_matrix(init_T_target_source);
    initial_model_matrix = init_T_target_source.cast<float>().matrix();

    // Open the manual loop close modal
    ImGui::OpenPopup("manual loop close");
  } else if (request_to_open && target_submaps.size() && source_submaps.size()) {
    // Setup for session vs session loop closure
    model_control->set_model_matrix(Eigen::Matrix4f::Identity().eval());
    ImGui::OpenPopup("preprocess maps");
  }
  request_to_open = false;

  bool open_preprocess_modal = false;  // Request to open session preprocessing progress modal
  // Preprocess parameter setting modal
  if (ImGui::BeginPopupModal("preprocess maps", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Set default parameters:");
    show_note("Select default parameters for minimum distance for downsampling and neighbor search radius for FPFH extraction.");

    ImGui::SameLine();
    if (ImGui::Button("Indoor")) {
      min_distance = 0.25f;
      fpfh_radius = 2.5f;
      max_correspondence_distance = 1.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Outdoor")) {
      min_distance = 0.5f;
      fpfh_radius = 5.0f;
      max_correspondence_distance = 3.0f;
    }

    ImGui::DragFloat("Min distance", &min_distance, 0.01f, 0.01f, 100.0f) || show_note("Minimum distance between points for downsampling.");

    if (ImGui::Button("OK")) {
      open_preprocess_modal = true;
      ImGui::CloseCurrentPopup();
    }

    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      ImGui::CloseCurrentPopup();
      clear();
    }

    ImGui::EndPopup();
  }

  // Open the preprocessing progress modal
  if (open_preprocess_modal) {
    progress_modal->open<std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr>>("preprocess", [this](guik::ProgressInterface& progress) {
      return preprocess_maps(progress);
    });
  }

  // Run the preprocessing progress modal and get the preprocessed point clouds
  auto preprocessed = progress_modal->run<std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr>>("preprocess");
  if (preprocessed) {
    target = preprocessed->first;
    source = preprocessed->second;

    target_drawable = std::make_shared<glk::PointCloudBuffer>(target->points, target->size());
    source_drawable = std::make_shared<glk::PointCloudBuffer>(source->points, source->size());

    model_control->set_model_matrix(Eigen::Matrix4f::Identity().eval());

    // Open the manual loop close modal
    ImGui::OpenPopup("manual loop close");
  }

  // Manual loop close modal
  ImGui::SetNextWindowSize(ImVec2(800, 700), ImGuiCond_FirstUseEver);
  if (ImGui::BeginPopupModal("manual loop close", nullptr, 0)) {
    // SD/HD data source buttons
    ImGui::Text("Data source:");
    ImGui::SameLine();
    if (ImGui::Button("SD")) {
      if (sd_target) {
        this->target = gtsam_points::PointCloudCPU::clone(*sd_target);
        this->target_drawable = std::make_shared<glk::PointCloudBuffer>(this->target->points, this->target->size());
        this->target->aux_attributes.erase("fpfh");
        this->target_fpfh_tree = nullptr;
      }
      if (sd_source) {
        this->source = gtsam_points::PointCloudCPU::clone(*sd_source);
        this->source_drawable = std::make_shared<glk::PointCloudBuffer>(this->source->points, this->source->size());
        this->source->aux_attributes.erase("fpfh");
        this->source_fpfh_tree = nullptr;
      }
    }
    ImGui::SameLine();
    if (!load_hd_callback) ImGui::BeginDisabled();
    if (ImGui::Button("HD")) {
      if (load_hd_callback) {
        auto [hd_target, hd_source] = load_hd_callback();
        replace_with_hd(hd_target, hd_source);
      }
    }
    if (!load_hd_callback) {
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("No HD frames available.");
      }
      ImGui::EndDisabled();
    } else {
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Load HD frames with computed covariances\nfor higher-quality registration.");
      }
    }
    ImGui::SameLine();
    if (ImGui::Button(modal_intensity_mode ? "Color" : "Intensity")) {
      modal_intensity_mode = !modal_intensity_mode;
      if (modal_intensity_mode) {
        // Build intensity-colored drawables using TURBO colormap (same as range filter)
        auto build_intensity_drawable = [this](const gtsam_points::PointCloudCPU::Ptr& cloud, const std::string& label) -> glk::Drawable::ConstPtr {
          if (!cloud || !cloud->intensities || cloud->size() == 0) {
            logger->warn("[ICP] No intensity data for {}", label);
            return nullptr;
          }
          // Use 1st-99th percentile for range (avoids outlier stretching)
          std::vector<float> sorted_vals;
          sorted_vals.reserve(std::min<size_t>(cloud->num_points, 50000));
          const size_t step = std::max<size_t>(1, cloud->num_points / 50000);
          for (size_t i = 0; i < cloud->num_points; i += step) {
            const float v = static_cast<float>(cloud->intensities[i]);
            if (std::isfinite(v)) sorted_vals.push_back(v);
          }
          std::sort(sorted_vals.begin(), sorted_vals.end());
          const float imin = sorted_vals.empty() ? 0.0f : sorted_vals[sorted_vals.size() / 100];
          const float imax = sorted_vals.empty() ? 1.0f : sorted_vals[sorted_vals.size() * 99 / 100];
          const float inv_range = (imax > imin) ? 1.0f / (imax - imin) : 1.0f;
          // Build vertex colors
          std::vector<float> cmap_vals(cloud->num_points);
          for (size_t i = 0; i < cloud->num_points; i++) {
            cmap_vals[i] = std::clamp((static_cast<float>(cloud->intensities[i]) - imin) * inv_range, 0.0f, 1.0f);
          }
          auto buf = std::make_shared<glk::PointCloudBuffer>(cloud->points, cloud->size());
          buf->add_intensity(glk::COLORMAP::TURBO, cmap_vals);
          logger->info("[ICP] Intensity drawable for {}: {} pts, range [{:.1f}, {:.1f}]", label, cloud->num_points, imin, imax);
          return buf;
        };
        modal_target_intensity_drawable = build_intensity_drawable(target, "target");
        modal_source_intensity_drawable = build_intensity_drawable(source, "source");
      }
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle between red/green flat colors\nand intensity grayscale rendering.");

    // GPS sigma display (red=target, green=source, matching canvas colors)
    ImGui::SameLine();
    ImGui::Text("  | GPS sigma:");
    ImGui::SameLine();
    if (target_gps_sigma >= 0.0f) {
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Target: %.3f m", target_gps_sigma);
    } else {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Target: N/A");
    }
    ImGui::SameLine();
    if (source_gps_sigma >= 0.0f) {
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Source: %.3f m", source_gps_sigma);
    } else {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Source: N/A");
    }
    ImGui::Separator();

    // Draw canvas — adapt to available space (width, and height minus ~250px for controls)
    const float canvas_w = std::max(256.0f, ImGui::GetContentRegionAvail().x - 10.0f);
    const float canvas_h = std::max(200.0f, ImGui::GetContentRegionAvail().y - 450.0f);
    const int cw = static_cast<int>(canvas_w);
    const int ch = static_cast<int>(canvas_h);
    if (canvas->size[0] != cw || canvas->size[1] != ch) {
      canvas.reset(new guik::GLCanvas(Eigen::Vector2i(cw, ch)));
    }
    ImGui::BeginChild(
      "canvas",
      ImVec2(canvas_w, canvas_h),
      false,
      ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoNavFocus);
    if (ImGui::IsWindowFocused() && !model_control->is_guizmo_using()) {
      canvas->mouse_control();
    }
    draw_canvas();
    ImGui::Image(reinterpret_cast<void*>(canvas->frame_buffer->color().id()), ImVec2(canvas_w, canvas_h), ImVec2(0, 1), ImVec2(1, 0));

    ImVec2 canvas_rect_min = ImGui::GetItemRectMin();
    ImVec2 canvas_rect_max = ImGui::GetItemRectMax();

    if (helper_gizmo_active) {
      // Helper mode: render ONLY the helper gizmo (avoids ImGuizmo conflict)
      // Compute delta from helper gizmo movement and apply to real model_control
      const Eigen::Matrix4f helper_current = helper_model_control->model_matrix();
      const Eigen::Vector3f delta_t = helper_current.block<3, 1>(0, 3) - helper_prev_matrix.block<3, 1>(0, 3);

      if (delta_t.squaredNorm() > 1e-8f) {
        Eigen::Matrix4f m = model_control->model_matrix();
        m.block<3, 1>(0, 3) += delta_t;
        model_control->set_model_matrix(m);
      }
      helper_prev_matrix = helper_current;

      helper_model_control->draw_gizmo(
        canvas_rect_min.x,
        canvas_rect_min.y,
        canvas_rect_max.x - canvas_rect_min.x,
        canvas_rect_max.y - canvas_rect_min.y,
        canvas->camera_control->view_matrix(),
        canvas->projection_control->projection_matrix(),
        true);
    } else {
      // Normal mode: render original gizmo
      model_control->draw_gizmo(
        canvas_rect_min.x,
        canvas_rect_min.y,
        canvas_rect_max.x - canvas_rect_min.x,
        canvas_rect_max.y - canvas_rect_min.y,
        canvas->camera_control->view_matrix(),
        canvas->projection_control->projection_matrix(),
        true);
    }

    ImGui::EndChild();

    /*** Helper gizmo + manual controls ***/
    ImGui::Separator();

    if (ImGui::Button(helper_gizmo_active ? "Hide helper gizmo" : "Helper gizmo")) {
      helper_gizmo_active = !helper_gizmo_active;
      if (helper_gizmo_active) {
        // Place helper at the 3D point at the center of the canvas viewport
        const Eigen::Matrix4f vm = canvas->camera_control->view_matrix();
        const Eigen::Vector3f cam_pos = -(vm.block<3, 3>(0, 0).transpose() * vm.block<3, 1>(0, 3));
        const Eigen::Vector3f cam_fwd = -vm.block<1, 3>(2, 0).transpose().normalized();
        // Place helper 20 units in front of camera, at source Z
        const float source_z = model_control->model_matrix()(2, 3);
        Eigen::Vector3f helper_pos = cam_pos + cam_fwd * 20.0f;
        helper_pos.z() = source_z;
        Eigen::Matrix4f helper_m = Eigen::Matrix4f::Identity();
        helper_m.block<3, 1>(0, 3) = helper_pos;
        helper_model_control->set_model_matrix(helper_m);
        helper_prev_matrix = helper_m;
      }
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show a helper gizmo at the view center.\nDrag it to translate the source cloud\nwithout needing to see the original gizmo.");

    ImGui::SameLine();
    {
      auto proj = std::dynamic_pointer_cast<guik::BasicProjectionControl>(canvas->projection_control);
      if (proj) {
        static bool ortho_mode = false;
        if (ImGui::Button(ortho_mode ? "Perspective" : "Ortho")) {
          ortho_mode = !ortho_mode;
          proj->set_projection_mode(ortho_mode ? 1 : 0);
          if (ortho_mode) proj->set_ortho_width(100.0);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle orthographic/perspective view.\nOrtho is useful for top-down alignment.");
      }
    }

    // XYZ translation + rotation on two rows
    {
      Eigen::Matrix4f m = model_control->model_matrix();
      Eigen::Vector3f t = m.block<3, 1>(0, 3);
      ImGui::SetNextItemWidth(200.0f);
      if (ImGui::DragFloat3("Translate", t.data(), 0.05f, -500.0f, 500.0f, "%.2f")) {
        m.block<3, 1>(0, 3) = t;
        model_control->set_model_matrix(m);
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Manual XYZ translation offset (metres).\nDrag or Ctrl+click to type exact values.");
    }
    {
      Eigen::Matrix4f m = model_control->model_matrix();
      Eigen::Matrix3f rot = m.block<3, 3>(0, 0);
      Eigen::Vector3f euler = rot.eulerAngles(2, 1, 0) * 180.0f / static_cast<float>(M_PI);
      ImGui::SetNextItemWidth(200.0f);
      if (ImGui::DragFloat3("Rotate (deg)", euler.data(), 0.1f, -180.0f, 180.0f, "%.1f")) {
        Eigen::Matrix3f new_rot;
        new_rot = Eigen::AngleAxisf(euler.x() * static_cast<float>(M_PI) / 180.0f, Eigen::Vector3f::UnitZ())
                * Eigen::AngleAxisf(euler.y() * static_cast<float>(M_PI) / 180.0f, Eigen::Vector3f::UnitY())
                * Eigen::AngleAxisf(euler.z() * static_cast<float>(M_PI) / 180.0f, Eigen::Vector3f::UnitX());
        m.block<3, 3>(0, 0) = new_rot;
        model_control->set_model_matrix(m);
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Manual rotation in degrees (Yaw, Pitch, Roll).\nDrag or Ctrl+click to type exact values.");
    }

    /*** Global registration ***/
    ImGui::Separator();

    ImGui::Combo("Global registration type", &global_registration_type, "RANSAC\0GNC\0");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("RANSAC: fast, random sampling based.\nGNC: graduated non-convexity, more robust but slower.");

    if (ImGui::DragFloat("fpfh_radius", &fpfh_radius, 0.01f, 0.01f, 100.0f)) {
      target->aux_attributes.erase("fpfh");
      source->aux_attributes.erase("fpfh");
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Neighbour search radius for FPFH feature extraction.\n~2.5m indoors, ~5.0m outdoors.\nChanging this clears cached features.");
    if (target->aux_attributes.count("fpfh")) {
      ImGui::SameLine();
      ImGui::Text("[Cached]");
    }

    switch (global_registration_type) {
      case 0:  // RANSAC
        ImGui::DragInt("max_iterations", &ransac_max_iterations, 100, 1, 100000);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum RANSAC iterations.\nHigher = better chance of finding correct alignment, slower.");
        ImGui::DragFloat("inlier_voxel_resolution", &ransac_inlier_voxel_resolution, 0.01f, 0.01f, 100.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Voxel resolution for RANSAC inlier counting.\nSmaller = stricter matching.");
        break;
      case 1:
        ImGui::DragInt("max_samples", &gnc_max_samples, 100, 1, 100000);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum feature samples for GNC registration.");
        break;
    }
    ImGui::Checkbox("4dof", &global_registration_4dof);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Restrict to 4-DOF (XYZ + yaw) instead of full 6-DOF.\nUseful for ground vehicles / outdoor MMS.");

    bool open_align_global_modal = false;
    if (ImGui::Button("Run global registration")) {
      open_align_global_modal = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Compute initial alignment using FPFH features.\nUse when clouds are far apart or poorly initialised.");

    /*** Fine registration ***/

    ImGui::Separator();
    ImGui::DragFloat("max_corr_dist", &max_correspondence_distance, 0.01f, 0.01f, 100.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum correspondence distance for ICP.\nPoints farther apart are ignored.\nIncrease if clouds are still misaligned.");
    ImGui::DragFloat("inf_scale", &information_scale, 0.0f, 1.0f, 10000.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Information (confidence) scale for the loop factor.\nHigher = optimizer trusts this constraint more.\nDefault 1.0 is usually fine.");

    bool open_align_modal = false;
    bool open_align_icp_modal = false;
    if (ImGui::Button("GICP")) {
      open_align_modal = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Run GICP (point-to-plane) registration.\nUses surface covariances for better accuracy.");
    ImGui::SameLine();
    if (ImGui::Button("ICP")) {
      open_align_icp_modal = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Run ICP (point-to-point) registration.\nSimpler, no covariances needed. For comparison.");

    if (open_align_global_modal) {
      progress_modal->open<std::shared_ptr<Eigen::Isometry3d>>("align", [this](guik::ProgressInterface& progress) { return align_global(progress); });
    }
    if (open_align_modal) {
      progress_modal->open<std::shared_ptr<Eigen::Isometry3d>>("align", [this](guik::ProgressInterface& progress) { return align(progress); });
    }
    if (open_align_icp_modal) {
      progress_modal->open<std::shared_ptr<Eigen::Isometry3d>>("align", [this](guik::ProgressInterface& progress) {
        progress.set_title("Aligning frames (ICP)");
        progress.set_maximum(200);
        progress.set_text("Creating graph");
        gtsam::NonlinearFactorGraph graph;
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
        auto f = gtsam::make_shared<gtsam_points::IntegratedICPFactor>(0, 1, target, source);
        f->set_num_threads(num_threads);
        f->set_max_correspondence_distance(max_correspondence_distance);
        graph.add(f);
        gtsam::Values values;
        values.insert(0, gtsam::Pose3::Identity());
        values.insert(1, gtsam::Pose3(model_control->model_matrix().cast<double>()));
        progress.set_text("Optimizing");
        gtsam_points::LevenbergMarquardtExtParams lm_params;
        lm_params.setMaxIterations(200);
        lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values&) {
          progress.increment();
          progress.set_text(fmt::format("ICP iter:{} error:{:.3f}", status.iterations, status.error));
        };
        gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
#ifdef GTSAM_USE_TBB
        auto arena = static_cast<tbb::task_arena*>(tbb_task_arena.get());
        arena->execute([&] { values = optimizer.optimize(); });
#else
        values = optimizer.optimize();
#endif
        const gtsam::Pose3 estimated = values.at<gtsam::Pose3>(0).inverse() * values.at<gtsam::Pose3>(1);
        return std::shared_ptr<Eigen::Isometry3d>(new Eigen::Isometry3d(estimated.matrix()));
      });
    }
    auto align_result = progress_modal->run<std::shared_ptr<Eigen::Isometry3d>>("align");
    if (align_result) {
      model_control->set_model_matrix((*align_result)->cast<float>().matrix());
    }

    /*** Neighbour relaxation ***/

    ImGui::Separator();
    ImGui::Checkbox("Relax neighbours", &relax_neighbors);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Temporarily weaken nearby factors so the optimizer\nblends the loop correction smoothly over several submaps\ninstead of creating a hard jump at the closure point.");
    if (relax_neighbors) {
      ImGui::Indent();
      ImGui::DragInt("Radius (submaps)", &relax_radius, 1, 1, 30);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of submaps on each side of the loop\nendpoints whose factors will be relaxed.");
      ImGui::DragFloat("Scale", &relax_scale, 0.5f, 1.5f, 50.0f, "%.1fx");
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Factor by which to multiply the noise sigma.\nHigher = looser constraints = more movement allowed.\n5x is a good starting point.");
      ImGui::Checkbox("Between factors", &relax_between);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Relax inter-submap odometry/matching constraints.");
      ImGui::SameLine();
      ImGui::Checkbox("GPS factors", &relax_gps);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Relax GNSS position constraints.\nUseful when GPS anchoring prevents submaps from moving.");
      ImGui::Unindent();
    }

    /*** Factor creation ***/

    ImGui::Separator();
    if (ImGui::Button("Create Factor")) {
      factor = create_factor();
      ImGui::CloseCurrentPopup();
      clear();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Create a loop factor with the current transformation\nand close the modal. Triggers optimisation.");

    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      ImGui::CloseCurrentPopup();
      clear();
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
      model_control->set_model_matrix(initial_model_matrix);
      helper_gizmo_active = false;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset source cloud to its initial position.");

    ImGui::EndPopup();
  }
  return factor;
}

std::pair<gtsam_points::PointCloudCPU::Ptr, gtsam_points::PointCloudCPU::Ptr> ManualLoopCloseModal::preprocess_maps(guik::ProgressInterface& progress) {
  logger->info("preprocessing maps");
  progress.set_title("Preprocessing maps");
  progress.set_maximum(target_submaps.size() + source_submaps.size());

  progress.set_text("Downsampling");
  const auto preprocess = [&](const std::vector<SubMap::ConstPtr>& submaps) {
    logger->info("Downsampling");
    gtsam_points::iVox ivox(min_distance * 5.0);
    ivox.set_lru_horizon(1000000);
    ivox.voxel_insertion_setting().max_num_points_in_cell = 50;
    ivox.voxel_insertion_setting().min_sq_dist_in_cell = std::pow(min_distance, 2);
    for (const auto& submap : submaps) {
      progress.increment();
      auto transformed = gtsam_points::transform(submap->frame, submap->T_world_origin);
      ivox.insert(*transformed);
    }

    auto points = std::make_shared<gtsam_points::PointCloudCPU>(ivox.voxel_points());

    logger->info("Finding neighbors");
    const int k = 10;
    gtsam_points::KdTree2<gtsam_points::PointCloud> tree(points);
    std::vector<int> neighbors(points->size() * k);

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (int i = 0; i < points->size(); i++) {
      std::vector<size_t> k_indices(k);
      std::vector<double> k_sq_dists(k);
      tree.knn_search(points->points[i].data(), k, k_indices.data(), k_sq_dists.data());
      std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + i * k);
    }

    logger->info("Estimate covariances");
    glim::CloudCovarianceEstimation covest(num_threads);
    covest.estimate(points->points_storage, neighbors, points->normals_storage, points->covs_storage);

    points->normals = points->normals_storage.data();
    points->covs = points->covs_storage.data();

    return points;
  };

  logger->info("preprocessing");
  gtsam_points::PointCloudCPU::Ptr target = preprocess(target_submaps);
  gtsam_points::PointCloudCPU::Ptr source = preprocess(source_submaps);

  progress.set_text("Done");

  return {target, source};
}

std::shared_ptr<Eigen::Isometry3d> ManualLoopCloseModal::align_global(guik::ProgressInterface& progress) {
  logger->info("Aligning global maps");
  progress.set_title("Global registration");
  progress.set_maximum(10);

  progress.increment();
  logger->info("Creating KdTree");
  progress.set_text("Creating KdTree");
  auto target_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(target);
  progress.increment();
  auto source_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(source);

  gtsam_points::FPFHEstimationParams fpfh_params;
  fpfh_params.num_threads = num_threads;
  fpfh_params.search_radius = fpfh_radius;

  // Helper: ensure normals+covs exist (needed for FPFH). Merged clouds may lack covs.
  auto ensure_normals = [&](gtsam_points::PointCloudCPU::Ptr& cloud, const std::string& label) {
    if (cloud->has_normals()) return;
    if (!cloud->has_covs()) {
      logger->info("Estimating {} covariances", label);
      progress.set_text("Estimating " + label + " covariances");
      const int k = 10;
      gtsam_points::KdTree tree(cloud->points, cloud->num_points);
      std::vector<int> neighbors(cloud->num_points * k);
      for (size_t i = 0; i < cloud->num_points; i++) {
        std::vector<size_t> k_indices(k, i);
        std::vector<double> k_sq_dists(k);
        tree.knn_search(cloud->points[i].data(), k, k_indices.data(), k_sq_dists.data());
        std::copy(k_indices.begin(), k_indices.begin() + k, neighbors.begin() + i * k);
      }
      glim::CloudCovarianceEstimation cov_est(num_threads);
      std::vector<Eigen::Vector4d> normals;
      std::vector<Eigen::Matrix4d> covs;
      cov_est.estimate(cloud->points_storage, neighbors, k, normals, covs);
      cloud->add_normals(normals);
      cloud->add_covs(covs);
    } else {
      logger->info("Estimating {} normals", label);
      progress.set_text("Estimating " + label + " normals");
      cloud->add_normals(gtsam_points::estimate_normals(cloud->points, cloud->covs, cloud->size(), num_threads));
    }
  };

  progress.increment();
  if (!target->aux_attributes.count("fpfh")) {
    ensure_normals(target, "target");

    logger->info("Estimating target FPFH features");
    progress.set_text("Estimating target FPFH features");
    const auto fpfh = gtsam_points::estimate_fpfh(target->points, target->normals, target->size(), *target_tree, fpfh_params);
    target->add_aux_attribute("fpfh", fpfh);

    logger->info("Constructing target FPFH KdTree");
    progress.set_text("Constructing target FPFH KdTree");
    const auto target_fpfh = target->aux_attribute<gtsam_points::FPFHSignature>("fpfh");
    target_fpfh_tree = std::make_shared<gtsam_points::KdTreeX<gtsam_points::FPFH_DIM>>(target_fpfh, target->size());
  }

  progress.increment();
  if (!source->aux_attributes.count("fpfh")) {
    ensure_normals(source, "source");

    logger->info("Estimating source FPFH features");
    progress.set_text("Estimating source FPFH features");
    const auto fpfh = gtsam_points::estimate_fpfh(source->points, source->normals, source->size(), *source_tree, fpfh_params);
    source->add_aux_attribute("fpfh", fpfh);

    logger->info("Constructing source FPFH KdTree");
    progress.set_text("Constructing source FPFH KdTree");
    const auto source_fpfh = source->aux_attribute<gtsam_points::FPFHSignature>("fpfh");
    source_fpfh_tree = std::make_shared<gtsam_points::KdTreeX<gtsam_points::FPFH_DIM>>(source_fpfh, source->size());
  }

  const auto target_fpfh = target->aux_attribute<gtsam_points::FPFHSignature>("fpfh");
  const auto source_fpfh = source->aux_attribute<gtsam_points::FPFHSignature>("fpfh");

  progress.increment();

  gtsam_points::RegistrationResult result;

  if (global_registration_type == 0) {
    logger->info("Estimating transformation RANSAC (seed={})", seed);
    progress.set_text("Estimating transformation RANSAC");

    gtsam_points::RANSACParams ransac_params;
    ransac_params.max_iterations = ransac_max_iterations;
    ransac_params.early_stop_inlier_rate = ransac_early_stop_rate;
    ransac_params.inlier_voxel_resolution = ransac_inlier_voxel_resolution;
    ransac_params.dof = global_registration_4dof ? 4 : 6;
    ransac_params.seed = (seed += 4322);
    ransac_params.num_threads = num_threads;

    result = gtsam_points::estimate_pose_ransac(*target, *source, target_fpfh, source_fpfh, *target_tree, *target_fpfh_tree, ransac_params);

  } else {
    logger->info("Estimating transformation GNC (seed={})", seed);
    progress.set_text("Estimating transformation (GNC)");

    gtsam_points::GNCParams gnc_params;
    gnc_params.max_init_samples = gnc_max_samples;
    gnc_params.reciprocal_check = true;
    gnc_params.tuple_check = false;
    gnc_params.max_num_tuples = 5000;
    gnc_params.dof = global_registration_4dof ? 4 : 6;
    gnc_params.seed = (seed += 4322);
    gnc_params.num_threads = num_threads;

    result = gtsam_points::estimate_pose_gnc(*target, *source, target_fpfh, source_fpfh, *target_tree, *target_fpfh_tree, *source_fpfh_tree, gnc_params);
  }

  logger->info("Registration result");
  logger->info("T_target_source={}", convert_to_string(result.T_target_source));
  logger->info("inlier_rate={}", result.inlier_rate);

  std::shared_ptr<Eigen::Isometry3d> trans(new Eigen::Isometry3d(result.T_target_source));

  return trans;
}

void ManualLoopCloseModal::ensure_covs(gtsam_points::PointCloudCPU::Ptr& cloud, const std::string& label) {
  if (!cloud || cloud->has_covs()) return;
  logger->info("Computing covariances for {} ({} pts)", label, cloud->num_points);
  const int k = 10;
  gtsam_points::KdTree tree(cloud->points, cloud->num_points);
  std::vector<int> neighbors(cloud->num_points * k);
  for (size_t i = 0; i < cloud->num_points; i++) {
    std::vector<size_t> k_indices(k, i);
    std::vector<double> k_sq_dists(k);
    tree.knn_search(cloud->points[i].data(), k, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.begin() + k, neighbors.begin() + i * k);
  }
  glim::CloudCovarianceEstimation cov_est(num_threads);
  std::vector<Eigen::Vector4d> normals;
  std::vector<Eigen::Matrix4d> covs;
  cov_est.estimate(cloud->points_storage, neighbors, k, normals, covs);
  cloud->add_normals(normals);
  cloud->add_covs(covs);
}

std::shared_ptr<Eigen::Isometry3d> ManualLoopCloseModal::align(guik::ProgressInterface& progress) {
  progress.set_title("Aligning frames");
  int num_iterations = 20;
  progress.set_maximum(num_iterations);

  // Ensure covariances exist for GICP
  progress.set_text("Checking covariances");
  ensure_covs(target, "target");
  ensure_covs(source, "source");

  progress.set_text("Creating graph");
  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));

  auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source);
  factor->set_num_threads(num_threads);
  factor->set_max_correspondence_distance(max_correspondence_distance);
  graph.add(factor);

  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Identity());
  values.insert(1, gtsam::Pose3(model_control->model_matrix().cast<double>()));

  progress.set_text("Optimizing");
  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.setMaxIterations(num_iterations);
  lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    progress.increment();
    progress.set_text(fmt::format("Optimizing iter:{} error:{:.3f}", status.iterations, status.error));
  };

  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);

#ifdef GTSAM_USE_TBB
  auto arena = static_cast<tbb::task_arena*>(tbb_task_arena.get());
  arena->execute([&] {
#endif
    values = optimizer.optimize();
#ifdef GTSAM_USE_TBB
  });
#endif

  const gtsam::Pose3 estimated = values.at<gtsam::Pose3>(0).inverse() * values.at<gtsam::Pose3>(1);

  return std::shared_ptr<Eigen::Isometry3d>(new Eigen::Isometry3d(estimated.matrix()));
}

gtsam::NonlinearFactor::shared_ptr ManualLoopCloseModal::create_factor() {
  using gtsam::symbol_shorthand::X;

  if (target_submaps.size()) {
    const Eigen::Isometry3d T_target_source(model_control->model_matrix().cast<double>());

    double min_distance = std::numeric_limits<double>::max();
    std::pair<SubMap::ConstPtr, SubMap::ConstPtr> best_pair;

    for (const auto& target : target_submaps) {
      for (const auto& source : source_submaps) {
        const double distance = (target->T_world_origin.translation() - T_target_source * source->T_world_origin.translation()).norm();
        if (distance < min_distance) {
          min_distance = distance;
          best_pair = std::make_pair(target, source);
        }
      }
    }

    const auto target_anchor = best_pair.first;
    const auto source_anchor = best_pair.second;
    logger->info("target_id:{} source_id:{} dist={}", target_anchor->id, source_anchor->id, min_distance);

    const Eigen::Isometry3d T_target_anchort = target_anchor->T_world_origin;
    const Eigen::Isometry3d T_source_anchors = source_anchor->T_world_origin;
    const Eigen::Isometry3d T_anchort_anchors = T_target_anchort.inverse() * T_target_source * T_source_anchors;

    return gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      X(target_anchor->id),
      X(source_anchor->id),
      gtsam::Pose3(T_anchort_anchors.matrix()),
      gtsam::noiseModel::Isotropic::Sigma(6, 1e-6));
  }

  const gtsam::Pose3 T_target_source(model_control->model_matrix().cast<double>());

  gtsam::Values values;
  values.insert(0, gtsam::Pose3::Identity());
  values.insert(1, T_target_source);

  // Ensure covariances exist for GICP
  ensure_covs(target, "target");
  ensure_covs(source, "source");

  auto icp_factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(0, 1, target, source);
  icp_factor->set_num_threads(num_threads);
  icp_factor->set_max_correspondence_distance(max_correspondence_distance);

  const auto linearized = icp_factor->linearize(values);
  const gtsam::Matrix H = linearized->hessianBlockDiagonal()[1];

  // Cancel out the gravity alignment
  const gtsam::Pose3 T_gb_b = gtsam::Pose3(gtsam::Rot3(source_pose.linear()), gtsam::Vector3::Zero());
  const gtsam::Pose3 T_ga_a = gtsam::Pose3(gtsam::Rot3(target_pose.linear()), gtsam::Vector3::Zero());
  const gtsam::Pose3 T_a_b = T_ga_a.inverse() * T_target_source * T_gb_b;

  return gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(target_key, source_key, T_a_b, gtsam::noiseModel::Gaussian::Information(information_scale * H));
}

void ManualLoopCloseModal::draw_canvas() {
  canvas->bind();
  canvas->shader->set_uniform("color_mode", guik::ColorMode::VERTEX_COLOR);
  canvas->shader->set_uniform("model_matrix", Eigen::Matrix4f::Identity().eval());

  glk::Primitives::coordinate_system()->draw(*canvas->shader);

  if (modal_intensity_mode && modal_target_intensity_drawable && modal_source_intensity_drawable) {
    // Intensity vertex-color mode
    canvas->shader->set_uniform("color_mode", guik::ColorMode::VERTEX_COLOR);
    canvas->shader->set_uniform("model_matrix", Eigen::Matrix4f::Identity().eval());
    modal_target_intensity_drawable->draw(*canvas->shader);
    canvas->shader->set_uniform("model_matrix", model_control->model_matrix());
    modal_source_intensity_drawable->draw(*canvas->shader);
  } else {
    // Flat red/green mode
    canvas->shader->set_uniform("color_mode", guik::ColorMode::FLAT_COLOR);
    canvas->shader->set_uniform("material_color", Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));
    canvas->shader->set_uniform("model_matrix", Eigen::Matrix4f::Identity().eval());
    target_drawable->draw(*canvas->shader);

    canvas->shader->set_uniform("material_color", Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));
    canvas->shader->set_uniform("model_matrix", model_control->model_matrix());
    source_drawable->draw(*canvas->shader);
  }

  canvas->unbind();
}

bool ManualLoopCloseModal::show_note(const std::string& note) {
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("%s", note.c_str());
    ImGui::EndTooltip();
  }
  return false;
}

}  // namespace glim
