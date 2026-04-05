#include <glim/viewer/offline_viewer.hpp>

#include <algorithm>
#include <fstream>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
#include <gtsam_points/config.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>
#include <glim/util/config.hpp>

#include <spdlog/spdlog.h>
#include <portable-file-dialogs.h>
#include <glk/io/ply_io.hpp>
#include <guik/recent_files.hpp>
#include <guik/progress_modal.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace glim {

OfflineViewer::OfflineViewer(const std::string& init_map_path) : init_map_path(init_map_path) {}

OfflineViewer::~OfflineViewer() {}

void OfflineViewer::load_gnss_datum() {
  gnss_datum_available = false;

  const std::string datum_path = GlobalConfig::get_config_path("gnss_datum");
  if (!boost::filesystem::exists(datum_path)) {
    logger->debug("gnss_datum.json not found at {}", datum_path);
    return;
  }

  std::ifstream ifs(datum_path);
  const auto j = nlohmann::json::parse(ifs, nullptr, /*exceptions=*/false);
  if (j.is_discarded()) {
    logger->warn("Failed to parse {}", datum_path);
    return;
  }

  gnss_utm_zone              = j.value("utm_zone", 0);
  gnss_utm_easting_origin    = j.value("utm_easting_origin", 0.0);
  gnss_utm_northing_origin   = j.value("utm_northing_origin", 0.0);
  gnss_datum_alt             = j.value("altitude", 0.0);

  // T_enu_world: flat row-major 12-element array [R00 R01 R02 tx | R10 ... | R20 ...]
  gnss_T_enu_world = Eigen::Isometry3d::Identity();
  if (j.contains("T_enu_world") && j["T_enu_world"].size() == 12) {
    const auto& T = j["T_enu_world"];
    gnss_T_enu_world.linear()(0, 0) = T[0];  gnss_T_enu_world.linear()(0, 1) = T[1];
    gnss_T_enu_world.linear()(0, 2) = T[2];  gnss_T_enu_world.translation()(0) = T[3];
    gnss_T_enu_world.linear()(1, 0) = T[4];  gnss_T_enu_world.linear()(1, 1) = T[5];
    gnss_T_enu_world.linear()(1, 2) = T[6];  gnss_T_enu_world.translation()(1) = T[7];
    gnss_T_enu_world.linear()(2, 0) = T[8];  gnss_T_enu_world.linear()(2, 1) = T[9];
    gnss_T_enu_world.linear()(2, 2) = T[10]; gnss_T_enu_world.translation()(2) = T[11];
  } else {
    logger->warn("gnss_datum.json missing or malformed T_enu_world — UTM export will use identity");
  }

  gnss_datum_available = true;
  logger->info(
    "GNSS datum loaded: UTM zone {} E={:.3f} N={:.3f} alt={:.3f}",
    gnss_utm_zone, gnss_utm_easting_origin, gnss_utm_northing_origin, gnss_datum_alt);

  // DEBUG: print the T_enu_world that will be applied to all points during UTM export.
  // If this is near-identity the SVD rotation was not captured in gnss_datum.json.
  {
    const Eigen::Matrix3d R = gnss_T_enu_world.linear();
    const Eigen::Vector3d t = gnss_T_enu_world.translation();
    logger->info("[DEBUG] gnss_T_enu_world loaded from JSON (will be applied in UTM export):");
    logger->info("[DEBUG]   R row0: [{:.6f}, {:.6f}, {:.6f}]", R(0,0), R(0,1), R(0,2));
    logger->info("[DEBUG]   R row1: [{:.6f}, {:.6f}, {:.6f}]", R(1,0), R(1,1), R(1,2));
    logger->info("[DEBUG]   R row2: [{:.6f}, {:.6f}, {:.6f}]", R(2,0), R(2,1), R(2,2));
    logger->info("[DEBUG]   translation: [{:.4f}, {:.4f}, {:.4f}]", t(0), t(1), t(2));
    const double yaw_deg = std::atan2(R(1,0), R(0,0)) * 180.0 / M_PI;
    logger->info("[DEBUG] yaw of world +X relative to geographic East: {:.2f} deg "
                 "(0=East, 90=North, ±180=West; near-zero means identity — no heading correction)",
                 yaw_deg);
  }
}

void OfflineViewer::setup_ui() {
  auto viewer = guik::LightViewer::instance();
  viewer->register_ui_callback("main_menu", [this] { main_menu(); });

  progress_modal.reset(new guik::ProgressModal("offline_viewer_progress"));

#ifdef GTSAM_POINTS_USE_CUDA
  gtsam_points::LinearizationHook::register_hook([] { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif
}

void OfflineViewer::main_menu() {
  bool start_open_map = !init_map_path.empty();
  bool start_close_map = false;
  bool start_save_map = false;
  bool start_export_map = false;

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (!async_global_mapping) {
        if (ImGui::MenuItem("Open New Map")) {
          start_open_map = true;
        }
      } else {
        if (ImGui::MenuItem("Open Additional Map")) {
          start_open_map = true;
        }
      }

      if (ImGui::MenuItem("Close Map")) {
        if (pfd::message("Warning", "Close the map?").result() == pfd::button::ok) {
          start_close_map = true;
        }
      }

      if (ImGui::BeginMenu("Save")) {
        if (ImGui::MenuItem("Save Map")) {
          start_save_map = true;
        }
        if (ImGui::MenuItem("Export Points")) {
          start_export_map = true;
        }
        ImGui::Separator();
        if (!gnss_datum_available) {
          ImGui::BeginDisabled();
        }
        ImGui::MenuItem("Export in UTM", nullptr, &export_in_utm);
        if (!gnss_datum_available) {
          if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("No GNSS datum available (gnss_datum.json not found)");
          }
          ImGui::EndDisabled();
        }
        ImGui::EndMenu();
      }

      if (ImGui::MenuItem("Quit")) {
        if (pfd::message("Warning", "Quit?").result() == pfd::button::ok) {
          request_to_terminate = true;
        }
      }

      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  // --- Open map ---
  if (start_open_map) {
    logger->debug("open map");
    std::string map_path;

    guik::RecentFiles recent_files("offline_viewer_open");
    if (init_map_path.empty()) {
      map_path = pfd::select_folder("Select a dump directory", recent_files.most_recent()).result();
    } else {
      map_path = init_map_path;
      init_map_path.clear();
    }

    if (!map_path.empty()) {
      logger->debug("open map from {}", map_path);
      recent_files.push(map_path);

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
        [this, map_path, existing_mapping](guik::ProgressInterface& progress) { return load_map(progress, map_path, existing_mapping); });
    }
  }

  auto open_result = progress_modal->run<std::shared_ptr<GlobalMapping>>("open");
  if (open_result) {
    if (!(*open_result)) {
      pfd::message("Error", "Failed to load map").result();
    } else {
      async_global_mapping.reset(new glim::AsyncGlobalMapping(*open_result, 1e6));
      load_gnss_datum();
    }
  }

  // --- Save map ---
  if (start_save_map) {
    if (!async_global_mapping) {
      logger->warn("No map data to save");
    } else {
      guik::RecentFiles recent_files("offline_viewer_save");
      const std::string path = pfd::select_folder("Select a directory to save the map", recent_files.most_recent()).result();
      if (!path.empty()) {
        recent_files.push(path);
        progress_modal->open<bool>("save", [this, path](guik::ProgressInterface& progress) { return save_map(progress, path); });
      }
    }
  }
  auto save_result = progress_modal->run<bool>("save");

  // --- Export points ---
  if (start_export_map) {
    guik::RecentFiles recent_files("offline_viewer_export");
    const std::string path = pfd::save_file("Select the file destination", recent_files.most_recent(), {"PLY", "*.ply"}).result();
    if (!path.empty()) {
      recent_files.push(path);
      progress_modal->open<bool>("export", [this, path](guik::ProgressInterface& progress) { return export_map(progress, path); });
    }
  }
  auto export_result = progress_modal->run<bool>("export");

  // --- Close map ---
  if (start_close_map) {
    if (async_global_mapping) {
      logger->info("Closing map");
      async_global_mapping->join();
      async_global_mapping.reset();
      clear();
    } else {
      logger->warn("No map to close");
    }
  }
}

std::shared_ptr<glim::GlobalMapping> OfflineViewer::load_map(
  guik::ProgressInterface& progress,
  const std::string& path,
  std::shared_ptr<GlobalMapping> global_mapping) {
  progress.set_title("Load map");
  progress.set_text("Now loading");
  progress.set_maximum(1);

  if (global_mapping == nullptr) {
    glim::GlobalMappingParams params;
    params.isam2_relinearize_skip = 1;
    params.isam2_relinearize_thresh = 0.0;

    const auto result = pfd::message("Confirm", "Do optimization?", pfd::choice::yes_no).result();
    params.enable_optimization = (result == pfd::button::ok) || (result == pfd::button::yes);

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

  // Collect float/double aux_attribute names present in all submaps, excluding primary PLY properties.
  // "intensity" is excluded because it collides with ply.intensities (primary double field);
  // it will be exported separately as "intensity_aux" for pipeline comparison.
  static const std::unordered_set<std::string> primary_ply_props = {"x", "y", "z", "nx", "ny", "nz", "intensity", "r", "g", "b", "a"};
  std::vector<std::string> aux_names;
  std::unordered_map<std::string, size_t> aux_elem_sizes;
  if (submaps[0] && submaps[0]->frame) {
    for (const auto& attrib : submaps[0]->frame->aux_attributes) {
      const size_t elem_size = attrib.second.first;
      if (elem_size != sizeof(float) && elem_size != sizeof(double)) {
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
        if (it == sm->frame->aux_attributes.end() || it->second.first != elem_size) {
          all_have = false;
          break;
        }
      }
      if (all_have) {
        aux_names.push_back(attrib.first);
        aux_elem_sizes[attrib.first] = elem_size;
      }
    }
  }

  progress.set_text("Writing to file");
  progress.increment();

  // Check whether aux_attributes["intensity"] (float) is present in all submaps.
  // This is distinct from frame->intensities (primary double field exported via ply.intensities).
  bool has_aux_intensity = true;
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }
    const auto it = submap->frame->aux_attributes.find("intensity");
    if (it == submap->frame->aux_attributes.end() || it->second.first != sizeof(float)) {
      has_aux_intensity = false;
      break;
    }
  }

  // Split aux names by element size so double attrs (gps_time) are written as
  // "property double" in the PLY header — float32 loses ~128 s precision on GPS epoch values.
  std::vector<std::string> aux_names_float, aux_names_double;
  for (const auto& name : aux_names) {
    if (aux_elem_sizes.at(name) == sizeof(double)) {
      aux_names_double.push_back(name);
    } else {
      aux_names_float.push_back(name);
    }
  }

  glk::PLYData ply;
  ply.vertices.reserve(total_points);
  if (has_normals) {
    ply.normals.reserve(total_points);
  }
  if (has_intensities) {
    ply.intensities.reserve(total_points);
  }

  std::unordered_map<std::string, std::vector<float>> aux_data_float;
  std::unordered_map<std::string, std::vector<double>> aux_data_double;
  for (const auto& name : aux_names_float) {
    aux_data_float[name].reserve(total_points);
  }
  for (const auto& name : aux_names_double) {
    aux_data_double[name].reserve(total_points);
  }
  std::vector<float> aux_intensity_data;
  if (has_aux_intensity) {
    aux_intensity_data.reserve(total_points);
  }

  size_t total_nan_filtered = 0;

  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }
    const auto& frame = submap->frame;
    const int n = frame->size();
    const Eigen::Matrix3d R = submap->T_world_origin.rotation();

    // Build per-point valid mask: exclude points where any aux attribute is non-finite or
    // where gps_time == 0.0 exactly (sentinel left by voxels merged before gps_time was
    // populated — these would corrupt MIN-blend colorisation by pulling the range to zero).
    std::vector<bool> valid(n, true);
    for (const auto& name : aux_names_float) {
      const float* src = static_cast<const float*>(frame->aux_attributes.at(name).second);
      for (int i = 0; i < n; i++) {
        if (valid[i] && !std::isfinite(src[i])) valid[i] = false;
      }
    }
    for (const auto& name : aux_names_double) {
      const double* src = static_cast<const double*>(frame->aux_attributes.at(name).second);
      for (int i = 0; i < n; i++) {
        if (valid[i] && !std::isfinite(src[i])) valid[i] = false;
      }
    }
    // Filter gps_time == 0.0: these are voxels that inherited a zero stamp from a keyframe
    // that was processed before the per-point GPS timestamps were available.
    {
      const auto gps_it = frame->aux_attributes.find("gps_time");
      if (gps_it != frame->aux_attributes.end() && gps_it->second.first == sizeof(double)) {
        const double* gps_src = static_cast<const double*>(gps_it->second.second);
        for (int i = 0; i < n; i++) {
          if (valid[i] && gps_src[i] == 0.0) valid[i] = false;
        }
      }
    }
    if (has_aux_intensity) {
      const float* src = static_cast<const float*>(frame->aux_attributes.at("intensity").second);
      for (int i = 0; i < n; i++) {
        if (valid[i] && !std::isfinite(src[i])) valid[i] = false;
      }
    }
    const size_t sm_nan = static_cast<size_t>(std::count(valid.begin(), valid.end(), false));
    total_nan_filtered += sm_nan;

    // Write geometry arrays, skipping NaN points
    for (int i = 0; i < n; i++) {
      if (!valid[i]) continue;
      ply.vertices.push_back((submap->T_world_origin * frame->points[i]).head<3>().cast<float>());
      if (has_normals) {
        ply.normals.push_back((R * frame->normals[i].head<3>()).cast<float>().normalized());
      }
      if (has_intensities) {
        ply.intensities.push_back(static_cast<float>(frame->intensities[i]));
      }
    }

    // Write float aux attributes, skipping NaN points
    for (const auto& name : aux_names_float) {
      const float* src = static_cast<const float*>(frame->aux_attributes.at(name).second);
      for (int i = 0; i < n; i++) {
        if (valid[i]) aux_data_float[name].push_back(src[i]);
      }
    }

    // Write double aux attributes as double (preserves full GPS time precision), skipping NaN points
    for (const auto& name : aux_names_double) {
      const double* src = static_cast<const double*>(frame->aux_attributes.at(name).second);
      for (int i = 0; i < n; i++) {
        if (valid[i]) aux_data_double[name].push_back(src[i]);
      }
    }

    if (has_aux_intensity) {
      const float* src = static_cast<const float*>(frame->aux_attributes.at("intensity").second);
      for (int i = 0; i < n; i++) {
        if (valid[i]) aux_intensity_data.push_back(src[i]);
      }
    }
  }

  if (total_nan_filtered > 0) {
    logger->info("PLY export: filtered {} / {} points with NaN aux attributes", total_nan_filtered, total_points);
  } else {
    logger->info("PLY export: no NaN points filtered ({} points total)", total_points);
  }

  for (const auto& name : aux_names_float) {
    ply.add_prop<float>(name, aux_data_float[name].data(), aux_data_float[name].size());
  }
  for (const auto& name : aux_names_double) {
    // Written as "property double <name>" — full 64-bit precision in the PLY file.
    ply.add_prop<double>(name, aux_data_double[name].data(), aux_data_double[name].size());
  }
  if (has_aux_intensity) {
    ply.add_prop<float>("intensity_aux", aux_intensity_data.data(), aux_intensity_data.size());
  }

  // Print gps_time range summary so we can verify precision in the exported file.
  const auto gps_it = aux_data_double.find("gps_time");
  if (gps_it != aux_data_double.end() && !gps_it->second.empty()) {
    const auto& gps_vec = gps_it->second;
    const double gps_min = *std::min_element(gps_vec.begin(), gps_vec.end());
    const double gps_max = *std::max_element(gps_vec.begin(), gps_vec.end());
    logger->info("PLY export: gps_time range [{:.9f}, {:.9f}] ({} points)", gps_min, gps_max, gps_vec.size());
  }

  // Apply UTM coordinate transform if requested.
  // Each vertex is currently in GLIM's world frame.  We apply T_enu_world to
  // get the ENU offset from the datum, then add the absolute UTM origin.
  //
  // Absolute UTM northing (~4.6 M m) cannot be represented with sub-metre
  // precision in float32 (ULP = 0.5 m), which causes visible banding.
  // We therefore write x/y/z as property double instead of property float.
  // To achieve this we clear ply.vertices (which would emit float x/y/z) and
  // add the coordinates as double generic properties, which the iridescence
  // PLY writer emits verbatim as "property double x/y/z".
  if (export_in_utm && gnss_datum_available) {
    // DEBUG: confirm which transform is in use at export time.
    {
      const Eigen::Matrix3d R = gnss_T_enu_world.linear();
      const Eigen::Vector3d t = gnss_T_enu_world.translation();
      logger->info("[DEBUG] PLY UTM export — gnss_T_enu_world at export time:");
      logger->info("[DEBUG]   R row0: [{:.6f}, {:.6f}, {:.6f}]", R(0,0), R(0,1), R(0,2));
      logger->info("[DEBUG]   R row1: [{:.6f}, {:.6f}, {:.6f}]", R(1,0), R(1,1), R(1,2));
      logger->info("[DEBUG]   R row2: [{:.6f}, {:.6f}, {:.6f}]", R(2,0), R(2,1), R(2,2));
      logger->info("[DEBUG]   translation: [{:.4f}, {:.4f}, {:.4f}]", t(0), t(1), t(2));
    }
    const size_t n = ply.vertices.size();
    std::vector<double> utm_x(n), utm_y(n), utm_z(n);
    for (size_t i = 0; i < n; i++) {
      const Eigen::Vector3d world_pt = ply.vertices[i].cast<double>();
      const Eigen::Vector3d enu_pt   = gnss_T_enu_world * world_pt;
      utm_x[i] = gnss_utm_easting_origin  + enu_pt.x();
      utm_y[i] = gnss_utm_northing_origin + enu_pt.y();
      utm_z[i] = gnss_datum_alt           + enu_pt.z();
    }
    ply.vertices.clear();  // prevent float x/y/z from shadowing the double properties
    ply.add_prop<double>("x", utm_x.data(), n);
    ply.add_prop<double>("y", utm_y.data(), n);
    ply.add_prop<double>("z", utm_z.data(), n);
    logger->info(
      "PLY export: {} vertices converted to UTM zone {} (E_origin={:.3f} N_origin={:.3f})",
      n, gnss_utm_zone, gnss_utm_easting_origin, gnss_utm_northing_origin);
  }

  glk::save_ply_binary(path, ply);
  return true;
}

}  // namespace glim
