#pragma once

#include <unordered_set>
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
  void do_open_map(const std::string& map_path, bool optimize);

  std::shared_ptr<GlobalMapping> load_map_with_optimize(guik::ProgressInterface& progress, const std::string& path, std::shared_ptr<GlobalMapping> global_mapping, bool optimize);
  bool save_map(guik::ProgressInterface& progress, const std::string& path);
  bool export_map(guik::ProgressInterface& progress, const std::string& path);

private:
  std::string init_map_path;
  std::unique_ptr<guik::ProgressModal> progress_modal;

  std::unordered_set<std::string> imported_shared_libs;
  std::unique_ptr<AsyncGlobalMapping> async_global_mapping;

  // ImGui dialog state (replaces pfd/zenity)
  char dialog_path_buf[512];
  bool dialog_optimize_on_load;
  bool show_load_error;
};

}  // namespace glim
