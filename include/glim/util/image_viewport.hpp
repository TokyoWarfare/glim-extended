#pragma once

#include <imgui.h>

namespace glim {

/// Mouse-wheel zoom-around-cursor helper for an image drawn inside an ImGui
/// child window with scrollbars. Call AFTER the image was drawn with
/// ImGui::Image(), while still inside the child's scope, so ImGui::GetScrollX/Y
/// and ImGui::IsWindowHovered refer to the right window.
///
/// @param scale          in/out: display scale (pixels per image pixel).
/// @param canvas_origin  screen-space top-left corner of the image (what
///                       GetCursorScreenPos() returned just before Image()).
/// @param min_scale      lower clamp.
/// @param max_scale      upper clamp.
/// @return true if the scale actually changed this frame.
///
/// Implementation: computes the image-pixel under the cursor, applies a fixed
/// zoom factor (1.15x per wheel tick), then shifts the scroll so that same
/// pixel stays under the cursor after the zoom.
bool apply_wheel_zoom_around_cursor(float& scale, const ImVec2& canvas_origin,
                                     float min_scale = 0.05f, float max_scale = 10.0f);

}  // namespace glim
