#pragma once

#include <unordered_set>
#include <Eigen/Geometry>
#include <glim/mapping/global_mapping.hpp>
#include <glim/mapping/async_global_mapping.hpp>
#include <glim/viewer/interactive_viewer.hpp>

namespace guik {
class ProgressModal;
class ProgressInterface;
}  // namespace guik

namespace glim {

class OfflineViewer : public InteractiveViewer {
public:
  OfflineViewer(const std::string& init_map_path = "");
  virtual ~OfflineViewer() override;

private:
  virtual void setup_ui() override;

  void main_menu();

  std::shared_ptr<GlobalMapping> load_map(guik::ProgressInterface& progress, const std::string& path, std::shared_ptr<GlobalMapping> global_mapping);
  bool save_map(guik::ProgressInterface& progress, const std::string& path);
  bool export_map(guik::ProgressInterface& progress, const std::string& path);

  /// Tries to load gnss_datum.json from the current GlobalConfig config path.
  /// Sets gnss_datum_available and populates UTM origin + T_enu_world on success.
  void load_gnss_datum();

private:
  std::string init_map_path;
  std::unique_ptr<guik::ProgressModal> progress_modal;

  std::unordered_set<std::string> imported_shared_libs;
  std::unique_ptr<AsyncGlobalMapping> async_global_mapping;

  // GNSS datum loaded from gnss_datum.json (written by GNSSGlobal::on_save).
  bool             gnss_datum_available  = false;
  int              gnss_utm_zone         = 0;
  double           gnss_utm_easting_origin  = 0.0;
  double           gnss_utm_northing_origin = 0.0;
  double           gnss_datum_alt           = 0.0;
  Eigen::Isometry3d gnss_T_enu_world{Eigen::Isometry3d::Identity()};

  // PLY export options (persistent across export invocations).
  bool export_in_utm = false;
};

}  // namespace glim
