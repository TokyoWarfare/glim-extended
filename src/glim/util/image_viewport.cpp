#include <glim/util/image_viewport.hpp>

#include <algorithm>

namespace glim {

bool apply_wheel_zoom_around_cursor(float& scale, const ImVec2& canvas_origin,
                                     float min_scale, float max_scale) {
  if (!ImGui::IsWindowHovered() || ImGui::GetIO().KeyCtrl) return false;
  const float wheel = ImGui::GetIO().MouseWheel;
  if (wheel == 0.0f) return false;

  const ImVec2 mpos = ImGui::GetMousePos();
  const float old_scale = std::max(1e-4f, scale);
  // Image pixel under cursor before zoom.
  const float ix = (mpos.x - canvas_origin.x) / old_scale;
  const float iy = (mpos.y - canvas_origin.y) / old_scale;
  const float factor = (wheel > 0.0f) ? 1.15f : (1.0f / 1.15f);
  scale = std::clamp(old_scale * factor, min_scale, max_scale);
  if (scale == old_scale) return false;
  // Shift scroll so that the same image pixel stays under the cursor.
  const float desired_origin_x = mpos.x - ix * scale;
  const float desired_origin_y = mpos.y - iy * scale;
  ImGui::SetScrollX(ImGui::GetScrollX() + (canvas_origin.x - desired_origin_x));
  ImGui::SetScrollY(ImGui::GetScrollY() + (canvas_origin.y - desired_origin_y));
  return true;
}

}  // namespace glim
