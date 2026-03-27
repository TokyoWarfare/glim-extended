#include <glim/viewer/offline_viewer.hpp>

#include <cstring>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <gtsam_points/config.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>
#include <glim/util/config.hpp>

#include <spdlog/spdlog.h>
#include <imgui.h>
#include <glk/io/ply_io.hpp>
#include <guik/recent_files.hpp>
#include <guik/progress_modal.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace glim {

OfflineViewer::OfflineViewer(const std::string& init_map_path) : init_map_path(init_map_path) {
  dialog_path_buf[0] = '\0';
  dialog_optimize_on_load = true;
  show_load_error = false;
}

OfflineViewer::~OfflineViewer() {}

void OfflineViewer::setup_ui() {
  auto viewer = guik::LightViewer::instance();
  viewer->register_ui_callback("main_menu", [this] { main_menu(); });

  progress_modal.reset(new guik::ProgressModal("offline_viewer_progress"));

#ifdef GTSAM_POINTS_USE_CUDA
  gtsam_points::LinearizationHook::register_hook([] { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif
}

void OfflineViewer::main_menu() {
  // Track which dialog to open this frame
  bool open_dialog = !init_map_path.empty();  // auto-trigger if init path provided
  bool close_confirm = false;
  bool save_dialog = false;
  bool export_dialog = false;
  bool quit_confirm = false;

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (!async_global_mapping) {
        if (ImGui::MenuItem("Open New Map")) {
          open_dialog = true;
        }
      } else {
        if (ImGui::MenuItem("Open Additional Map")) {
          open_dialog = true;
        }
      }

      if (ImGui::MenuItem("Close Map")) {
        close_confirm = true;
      }

      if (ImGui::BeginMenu("Save")) {
        if (ImGui::MenuItem("Save Map")) {
          save_dialog = true;
        }
        if (ImGui::MenuItem("Export Points")) {
          export_dialog = true;
        }
        ImGui::EndMenu();
      }

      if (ImGui::MenuItem("Quit")) {
        quit_confirm = true;
      }

      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  // --- Open map ---
  if (open_dialog) {
    if (!init_map_path.empty()) {
      // Auto-load from command-line path: skip dialog, start immediately
      const std::string map_path = init_map_path;
      init_map_path.clear();
      do_open_map(map_path, true);
    } else {
      guik::RecentFiles recent_files("offline_viewer_open");
      const std::string recent = recent_files.most_recent();
      strncpy(dialog_path_buf, recent.c_str(), sizeof(dialog_path_buf) - 1);
      dialog_path_buf[sizeof(dialog_path_buf) - 1] = '\0';
      dialog_optimize_on_load = true;
      ImGui::OpenPopup("Open Map##dlg");
    }
  }
  if (ImGui::BeginPopupModal("Open Map##dlg", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Map directory:");
    ImGui::SetNextItemWidth(400.0f);
    ImGui::InputText("##open_path", dialog_path_buf, sizeof(dialog_path_buf));
    ImGui::Checkbox("Enable optimization", &dialog_optimize_on_load);
    ImGui::Separator();
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      const std::string map_path(dialog_path_buf);
      if (!map_path.empty()) {
        guik::RecentFiles("offline_viewer_open").push(map_path);
        do_open_map(map_path, dialog_optimize_on_load);
      }
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  // Process open result
  auto open_result = progress_modal->run<std::shared_ptr<GlobalMapping>>("open");
  if (open_result) {
    if (!(*open_result)) {
      show_load_error = true;
    } else {
      async_global_mapping.reset(new glim::AsyncGlobalMapping(*open_result, 1e6));
    }
  }
  if (show_load_error) {
    ImGui::OpenPopup("Load Error##dlg");
    show_load_error = false;
  }
  if (ImGui::BeginPopupModal("Load Error##dlg", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Failed to load map.");
    ImGui::Separator();
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  // --- Close map ---
  if (close_confirm) {
    ImGui::OpenPopup("Close Map?##dlg");
  }
  if (ImGui::BeginPopupModal("Close Map?##dlg", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Close the current map?");
    ImGui::Separator();
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      if (async_global_mapping) {
        logger->info("Closing map");
        async_global_mapping->join();
        async_global_mapping.reset();
        clear();
      }
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  // --- Save map ---
  if (save_dialog) {
    if (!async_global_mapping) {
      logger->warn("No map data to save");
    } else {
      guik::RecentFiles recent_files("offline_viewer_save");
      const std::string recent = recent_files.most_recent();
      strncpy(dialog_path_buf, recent.empty() ? "/tmp/glim_map" : recent.c_str(), sizeof(dialog_path_buf) - 1);
      dialog_path_buf[sizeof(dialog_path_buf) - 1] = '\0';
      ImGui::OpenPopup("Save Map##dlg");
    }
  }
  if (ImGui::BeginPopupModal("Save Map##dlg", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Save directory:");
    ImGui::SetNextItemWidth(400.0f);
    ImGui::InputText("##save_path", dialog_path_buf, sizeof(dialog_path_buf));
    ImGui::Separator();
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      const std::string path(dialog_path_buf);
      if (!path.empty()) {
        guik::RecentFiles("offline_viewer_save").push(path);
        progress_modal->open<bool>("save", [this, path](guik::ProgressInterface& progress) { return save_map(progress, path); });
      }
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  progress_modal->run<bool>("save");

  // --- Export points ---
  if (export_dialog) {
    guik::RecentFiles recent_files("offline_viewer_export");
    const std::string recent = recent_files.most_recent();
    strncpy(dialog_path_buf, recent.empty() ? "/tmp/glim_points.ply" : recent.c_str(), sizeof(dialog_path_buf) - 1);
    dialog_path_buf[sizeof(dialog_path_buf) - 1] = '\0';
    ImGui::OpenPopup("Export Points##dlg");
  }
  if (ImGui::BeginPopupModal("Export Points##dlg", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Output PLY file:");
    ImGui::SetNextItemWidth(400.0f);
    ImGui::InputText("##export_path", dialog_path_buf, sizeof(dialog_path_buf));
    ImGui::Separator();
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      const std::string path(dialog_path_buf);
      if (!path.empty()) {
        guik::RecentFiles("offline_viewer_export").push(path);
        progress_modal->open<bool>("export", [this, path](guik::ProgressInterface& progress) { return export_map(progress, path); });
      }
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  progress_modal->run<bool>("export");

  // --- Quit ---
  if (quit_confirm) {
    ImGui::OpenPopup("Quit?##dlg");
  }
  if (ImGui::BeginPopupModal("Quit?##dlg", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Quit the application?");
    ImGui::Separator();
    if (ImGui::Button("OK", ImVec2(120, 0))) {
      request_to_terminate = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void OfflineViewer::do_open_map(const std::string& map_path, bool optimize) {
  logger->debug("open map from {}", map_path);

  if (boost::filesystem::exists(map_path + "/config")) {
    logger->info("Use config from {}", map_path + "/config");
    GlobalConfig::instance(map_path + "/config", true);
  } else {
    logger->warn("No config found in {}", map_path);
  }

  const Config config_ros(GlobalConfig::get_config_path("config_ros"));
  const std::vector<std::string> ext_module_names = config_ros.param<std::vector<std::string>>("glim_ros", "extension_modules", {});
  for (const auto& name : ext_module_names) {
    if (name.find("viewer") != std::string::npos || name.find("monitor") != std::string::npos) {
      continue;
    }
    if (imported_shared_libs.count(name)) {
      logger->debug("Extension module {} already loaded", name);
      continue;
    }
    logger->info("Export classes from {}", name);
    ExtensionModule::export_classes(name);
    imported_shared_libs.insert(name);
  }

  std::shared_ptr<GlobalMapping> existing_mapping;
  if (async_global_mapping) {
    logger->info("global map already exists, loading new map into existing global map");
    existing_mapping = std::dynamic_pointer_cast<GlobalMapping>(async_global_mapping->get_global_mapping());
  }

  progress_modal->open<std::shared_ptr<GlobalMapping>>(
    "open",
    [this, map_path, existing_mapping, optimize](guik::ProgressInterface& progress) { return load_map_with_optimize(progress, map_path, existing_mapping, optimize); });
}

std::shared_ptr<glim::GlobalMapping> OfflineViewer::load_map_with_optimize(
  guik::ProgressInterface& progress,
  const std::string& path,
  std::shared_ptr<GlobalMapping> global_mapping,
  bool optimize) {
  progress.set_title("Load map");
  progress.set_text("Now loading");
  progress.set_maximum(1);

  if (global_mapping == nullptr) {
    glim::GlobalMappingParams params;
    params.isam2_relinearize_skip = 1;
    params.isam2_relinearize_thresh = 0.0;
    params.enable_optimization = optimize;
    logger->info("enable_optimization={}", params.enable_optimization);
    global_mapping.reset(new glim::GlobalMapping(params));
  }

  if (!global_mapping->load(path)) {
    logger->error("failed to load {}", path);
    return nullptr;
  }

  return global_mapping;
}

bool OfflineViewer::save_map(guik::ProgressInterface& progress, const std::string& path) {
  progress.set_title("Save map");
  progress.set_text("Now saving");
  async_global_mapping->save(path);
  return true;
}

bool OfflineViewer::export_map(guik::ProgressInterface& progress, const std::string& path) {
  progress.set_title("Export points");
  progress.set_text("Concatenating submaps");
  progress.set_maximum(3);
  progress.increment();

  if (submaps.empty()) {
    logger->warn("No submaps available for export");
    return false;
  }

  // Determine which fields are present across all submaps
  bool has_normals = true;
  bool has_intensities = true;
  size_t total_points = 0;
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }
    total_points += submap->frame->size();
    if (!submap->frame->normals) {
      has_normals = false;
    }
    if (!submap->frame->has_intensities()) {
      has_intensities = false;
    }
  }

  if (total_points == 0) {
    logger->warn("No points available for export");
    return false;
  }

  // Collect float aux_attribute names present in all submaps, excluding primary PLY properties
  static const std::unordered_set<std::string> primary_ply_props = {"x", "y", "z", "nx", "ny", "nz", "intensity", "r", "g", "b", "a"};
  std::vector<std::string> aux_names;
  if (submaps[0] && submaps[0]->frame) {
    for (const auto& attrib : submaps[0]->frame->aux_attributes) {
      if (attrib.second.first != sizeof(float)) {
        continue;
      }
      if (primary_ply_props.count(attrib.first)) {
        continue;
      }
      bool all_have = true;
      for (const auto& sm : submaps) {
        if (!sm || !sm->frame) {
          all_have = false;
          break;
        }
        const auto it = sm->frame->aux_attributes.find(attrib.first);
        if (it == sm->frame->aux_attributes.end() || it->second.first != sizeof(float)) {
          all_have = false;
          break;
        }
      }
      if (all_have) {
        aux_names.push_back(attrib.first);
      }
    }
  }

  progress.set_text("Writing to file");
  progress.increment();

  glk::PLYData ply;
  ply.vertices.reserve(total_points);
  if (has_normals) {
    ply.normals.reserve(total_points);
  }
  if (has_intensities) {
    ply.intensities.reserve(total_points);
  }

  std::unordered_map<std::string, std::vector<float>> aux_data;
  for (const auto& name : aux_names) {
    aux_data[name].reserve(total_points);
  }

  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }
    const auto& frame = submap->frame;
    const Eigen::Matrix3d R = submap->T_world_origin.rotation();

    for (int i = 0; i < frame->size(); i++) {
      ply.vertices.push_back((submap->T_world_origin * frame->points[i]).head<3>().cast<float>());

      if (has_normals) {
        ply.normals.push_back((R * frame->normals[i].head<3>()).cast<float>().normalized());
      }
      if (has_intensities) {
        ply.intensities.push_back(static_cast<float>(frame->intensities[i]));
      }
    }

    for (const auto& name : aux_names) {
      const float* src = static_cast<const float*>(frame->aux_attributes.at(name).second);
      aux_data[name].insert(aux_data[name].end(), src, src + frame->size());
    }
  }

  for (const auto& name : aux_names) {
    ply.add_prop<float>(name, aux_data[name].data(), aux_data[name].size());
  }

  glk::save_ply_binary(path, ply);
  return true;
}

}  // namespace glim