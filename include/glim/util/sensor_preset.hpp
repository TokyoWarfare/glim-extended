#pragma once

#include <string>
#include <vector>

#include <glim/util/erasor_filter.hpp>

namespace glim {

/**
 * @brief Per-sensor tuning preset bundle.
 *
 * Holds the algorithm params that vary by sensor model for both Dynamic-Erasor
 * and Dynamic (MapCleaner) cleaners. The Data Cleaner UI exposes a "Sensor
 * preset" dropdown that copies the preset's values into the active UI fields.
 *
 * Why a struct: ERASOR's defaults (and to a lesser extent MapCleaner's) depend
 * heavily on scan geometry -- Livox Horizon's ~82degx25deg non-repetitive vs
 * Pandar 128's 360degx40deg uniform spinner imply very different ring counts,
 * max-range envelopes, ratio thresholds, and V-FOV gates. Coding these as
 * a single set of "generic" defaults makes the UI brittle (a Pandar user
 * overwriting Horizon defaults loses the safe state). A preset bundle lets us
 * ship verified defaults per sensor and ramp up new sensors as we acquire data.
 *
 * The MapCleaner side of the bundle is currently a placeholder structure --
 * MapCleanerFilter::Params lives in map_cleaner.hpp and we don't yet have
 * per-sensor MapCleaner tunings to ship. The hooks are here so adding them
 * later is a one-struct change.
 */
struct SensorPreset {
  std::string             name;
  glim::ErasorFilter::Params erasor;
  // MapCleaner (Dynamic mode) per-sensor tunings to be filled in once we have
  // measurements. Kept separate from erasor so each cleaner pulls only what
  // it needs.
  // glim::MapCleanerFilter::Params dynamic;  // TODO: enable when needed
};

/**
 * @brief Built-in sensor preset list.
 *
 * Order matches the Sensor preset combo in the Data Cleaner UI (index = combo
 * selection). To add a new sensor: append a new entry, populate its fields,
 * rebuild. No other UI edits required.
 */
inline std::vector<SensorPreset> built_in_sensor_presets() {
  std::vector<SensorPreset> presets;

  // Livox Horizon -- forward-facing ~82degx25deg non-repetitive scan. Tuned
  // to over-detect (low ratio_threshold) so the scalar-save + manual-trim
  // workflow has plenty of candidates to work with.
  {
    SensorPreset p;
    p.name = "Livox Horizon";
    p.erasor.num_rings = 30;
    p.erasor.num_sectors = 108;
    p.erasor.max_range = 30.0f;
    p.erasor.min_range = 1.0f;
    p.erasor.ratio_threshold = 0.01f;
    p.erasor.exclude_ground_pw = true;
    p.erasor.frame_skip = 0;
    p.erasor.sensor_v_fov_half_deg = 12.5f;
    presets.push_back(std::move(p));
  }

  // Pandar 128 / Velodyne / Ouster slots will land here once we have data.
  // Recommended starting points (NOT tuned, do not enable without testing):
  //   Pandar 128  -- num_rings=20, max_range=80, ratio_threshold=0.20,
  //                  v_fov_half=22.0  (uniform 360deg, density holds at range)
  //   VLP-16      -- num_rings=16, max_range=60, ratio_threshold=0.20,
  //                  v_fov_half=15.0
  //   Ouster OS1  -- num_rings=20, max_range=80, ratio_threshold=0.20,
  //                  v_fov_half=22.0

  return presets;
}

}  // namespace glim
