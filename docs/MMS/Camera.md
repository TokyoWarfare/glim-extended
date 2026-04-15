# GLIM Camera System

## Camera Modes

Three camera modes available in the offline viewer under **Tools → Camera**:

### Orbit (default)
Standard orbit camera rotating around a center point.

- **LMB drag**: Rotate around center
- **RMB drag**: Pan
- **Scroll**: Zoom (distance to center)
- **W/S**: Dolly toward/away from center (true distance change, not FOV)
- **Shift+W/S**: 5× faster dolly
- **Double-click**: Set orbit center to clicked 3D point (orange sphere indicator)

### FPV (First Person View)
Free-flight camera for close inspection of point cloud data.

- **WASD**: Move forward/back/left/right
- **Mouse**: Look around
- **Shift**: Speed boost (configurable multiplier, default 3×)
- **Position smoothing**: Configurable (default 0.7) — removes jerkiness from keyboard movement while keeping rotation crisp

### Follow Trajectory
Camera follows the SLAM trajectory path with cinematic smoothing.

- **Catmull-Rom spline**: Smooth cubic interpolation through trajectory points — no sharp corners
- **Exponential position smoothing**: Drone-like inertia (configurable, default 0.05)
- **Yaw wrap-around**: Smooth 360° transitions without spinning
- **RMB drag**: Turret look-around (360° video feel) — smooth return to forward on release
- **Space**: Play/pause (pause = speed 0, unpause = restore previous speed)
- **W/S**: Accelerate/decelerate (works from pause — builds speed naturally)
- **Shift+W/S**: 5× acceleration rate
- **Negative speed**: S below 0 = reverse along trajectory
- **Progress slider**: Click to jump, drag to scrub (0-100%)
- **Max speed**: ±500 km/h
- **HUD overlay**: Target speed, actual speed, distance, play state, key mappings

#### HUD Display
```
[=========|========================] 42.5%
188 km/h (actual 165)  |  5100 / 12000 m  |  Playing
Space=play/pause  W/S=speed  RMB=look around
```

## Camera Settings (Tools → Camera → Settings)

### FPV Settings
- **FPV speed**: Translation speed (0.01-1.0)
- **Shift multiplier**: Speed boost factor (1.5-10×)
- **FPV smoothness**: Position smoothing (0.05=very smooth, 1.0=raw)

### Follow Trajectory Settings
- **Speed (km/h)**: Playback speed (-500 to +500)
- **Smoothness**: Position smoothing (0.01=drone-like, 0.50=tight)

## Standard Viewer (Live SLAM)

Camera mode dropdown: Orbit, FPV, TopDown.
- FPV mode uses sensor-tracking `lookat(pose)` for smooth chase-cam during live SLAM
- TopDown provides bird's-eye view for coverage review

## Technical Notes

### Why RMB for Follow Trajectory turret (not LMB)
Iridescence's built-in camera controls process LMB events internally for orbit rotation. In Follow Trajectory mode, the FPS camera's LMB handler conflicts with our turret rotation — the orbit camera fights back, causing jerky movement. RMB avoids this entirely since Iridescence uses it for panning (which our `set_pose` overrides every frame anyway).

### Why FPV rotation is not smoothed
The FPS camera handles rotation natively via mouse input. Our smoothing loop reads the view matrix, smooths it, and calls `set_pose` — but `set_pose` overrides the camera's internal state. Next frame, the FPS camera's input starts from our smoothed position, creating a feedback loop. Position smoothing works because it's additive (camera moves, we lag behind). Rotation smoothing fights because it dampens the camera's own input.

### Why orbit double-click sometimes feels odd
The orbit camera's `lookat(point)` changes the orbit center but preserves the current distance and viewing angles. If the new center is far from the old one, the camera appears to "jump" because it suddenly orbits a distant point. A proper fix requires modifying Iridescence's `OrbitCameraControlXY` to support `set_center_preserving_camera_position()` — planned for future.

### Iridescence modifications planned (future)
The Iridescence library ([github.com/koide3/iridescence](https://github.com/koide3/iridescence)) is currently used as a pre-compiled dependency. Future modifications planned:

1. **Orbit camera dolly**: Replace scroll-zoom-via-FOV with true distance dolly. Current workaround: W/S keys call `scroll()` which adjusts distance.
2. **Input mode isolation**: Disable orbit camera input when in FPV/Follow mode to prevent LMB conflicts. Would allow LMB turret in Follow mode.
3. **Set center preserving position**: New method on OrbitCameraControlXY that changes the orbit center while computing new distance/theta/phi to keep the camera position fixed.
4. **Configurable scroll behavior**: Scroll = distance (default), Ctrl+Scroll = FOV.
