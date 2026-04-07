#include <deque>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <thread>
#include <numeric>
#include <filesystem>
#include <Eigen/Core>

#include <boost/format.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>

#ifdef GLIM_ROS2
#include <glim/util/extension_module_ros2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS2;
using NavSatFix = sensor_msgs::msg::NavSatFix;
using NavSatFixConstPtr = sensor_msgs::msg::NavSatFix::ConstSharedPtr;

template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.sec + stamp.nanosec / 1e9;
}
#else
#include <glim/util/extension_module_ros.hpp>
using ExtensionModuleBase = glim::ExtensionModuleROS;
#endif

#include <spdlog/spdlog.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <glim/util/convert_to_string.hpp>
#include <glim/util/config.hpp>
#include <glim/util/geodetic.hpp>

namespace glim {

using gtsam::symbol_shorthand::X;

/**
 * @brief GNSS global constraints from NavSatFix (lat/lon/alt -> UTM).
 *
 * The GLIM world frame IS the UTM frame (with origin subtracted).
 * No rotation estimation is needed: UTM East = +X, UTM North = +Y, Up = +Z.
 *
 * The first accepted fix sets the UTM origin. All subsequent fixes are
 * projected to the same UTM zone and expressed as offsets from that origin.
 * These offsets are used directly as PoseTranslationPrior targets on the
 * submap pose nodes.
 */
class GNSSGlobal : public ExtensionModuleBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int STATUS_NO_FIX   = -1;
  static constexpr int STATUS_FIX      =  0;
  static constexpr int STATUS_SBAS_FIX =  1;
  static constexpr int STATUS_GBAS_FIX =  2;

  GNSSGlobal() : logger(create_module_logger("gnss_global")) {
    logger->info("initializing GNSS global constraints (NavSatFix -> UTM)");
    const std::string config_path = glim::GlobalConfig::get_config_path("config_gnss_global");

    glim::Config config(config_path);
    gnss_topic         = config.param<std::string>("gnss", "gnss_topic", "/navsatfix");
    prior_inf_scale    = config.param<Eigen::Vector3d>("gnss", "prior_inf_scale", Eigen::Vector3d(1e3, 1e3, 1e3));
    min_fix_status     = config.param<int>("gnss", "min_fix_status", STATUS_SBAS_FIX);
    sbas_noise_inflation = config.param<double>("gnss", "sbas_noise_inflation", 10.0);

    logger->info("config: topic={} min_fix_status={} sbas_noise_inflation={:.1f}",
                 gnss_topic, min_fix_status, sbas_noise_inflation);

    datum_initialized  = false;
    datum_json_written = false;
    datum_utm_zone     = 0;
    datum_utm_easting  = 0.0;
    datum_utm_northing = 0.0;

    kill_switch = false;
    thread = std::thread([this] { backend_task(); });

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    GlobalMappingCallbacks::on_insert_submap.add(std::bind(&GNSSGlobal::on_insert_submap, this, _1));
    GlobalMappingCallbacks::on_smoother_update.add(std::bind(&GNSSGlobal::on_smoother_update, this, _1, _2, _3));
  }

  ~GNSSGlobal() {
    kill_switch = true;
    thread.join();
  }

  virtual std::vector<GenericTopicSubscription::Ptr> create_subscriptions() override {
    const auto sub = std::make_shared<TopicSubscription<NavSatFix>>(
      gnss_topic, [this](const NavSatFixConstPtr msg) { navsatfix_callback(msg); });
    return {sub};
  }

  void navsatfix_callback(const NavSatFixConstPtr& msg) {
    const int status = msg->status.status;
    if (status == STATUS_NO_FIX) return;
    if (status < min_fix_status) return;

    // Set UTM datum from the first accepted fix.
    if (!datum_initialized) {
      datum_lat = msg->latitude;
      datum_lon = msg->longitude;
      datum_alt = msg->altitude;
      datum_utm_zone = ecef_to_utm_zone(datum_lat, datum_lon);
      const Eigen::Vector2d utm_xy = wgs84_to_utm_xy(datum_lat, datum_lon);
      datum_utm_easting  = utm_xy.x();
      datum_utm_northing = utm_xy.y();
      datum_initialized = true;
      logger->info("UTM datum set: lat={:.9f} lon={:.9f} alt={:.3f} zone={} E={:.3f} N={:.3f}",
                   datum_lat, datum_lon, datum_alt, datum_utm_zone, datum_utm_easting, datum_utm_northing);
    }

    // Project to UTM (forced to datum zone) and express as offset from origin.
    const Eigen::Vector2d utm_xy = wgs84_to_utm_xy(msg->latitude, msg->longitude, datum_utm_zone);
    const double dx = utm_xy.x() - datum_utm_easting;
    const double dy = utm_xy.y() - datum_utm_northing;
    const double dz = msg->altitude - datum_alt;

    // Extract per-axis sigmas from position_covariance.
    constexpr int COVARIANCE_TYPE_UNKNOWN = 0;
    constexpr double SIGMA_FLOOR = 0.005;
    constexpr double SIGMA_CAP   = 20.0;

    Eigen::Vector3d sigma(-1.0, -1.0, -1.0);
    if (msg->position_covariance_type != COVARIANCE_TYPE_UNKNOWN) {
      const double raw_E = std::sqrt(msg->position_covariance[0]);
      const double raw_N = std::sqrt(msg->position_covariance[4]);
      const double raw_U = std::sqrt(msg->position_covariance[8]);
      if (std::isnan(raw_E) || std::isnan(raw_N) || std::isnan(raw_U) ||
          raw_E > SIGMA_CAP || raw_N > SIGMA_CAP || raw_U > SIGMA_CAP) {
        return;
      }
      auto floored = [&](double raw) { return raw < SIGMA_FLOOR ? SIGMA_FLOOR : raw; };
      sigma = {floored(raw_E), floored(raw_N), floored(raw_U)};
    }

    // Queue: [stamp, utm_dx, utm_dy, utm_dz, status, sigma_E, sigma_N, sigma_U]
    Eigen::Matrix<double, 8, 1> entry;
    entry << to_sec(msg->header.stamp), dx, dy, dz,
             static_cast<double>(status), sigma.x(), sigma.y(), sigma.z();
    input_gnss_queue.push_back(entry);
  }

  void on_insert_submap(const SubMap::ConstPtr& submap) {
    input_submap_queue.push_back(submap);
  }

  void on_smoother_update(
    gtsam_points::ISAM2Ext& isam2,
    gtsam::NonlinearFactorGraph& new_factors,
    gtsam::Values& new_values)
  {
    const auto factors = output_factors.get_all_and_clear();
    if (!factors.empty()) {
      new_factors.add(factors);
    }
  }

  void backend_task() {
    std::deque<Eigen::Matrix<double, 8, 1>> gnss_queue;
    std::deque<SubMap::ConstPtr> submap_queue;

    while (!kill_switch) {
      const auto gnss_data = input_gnss_queue.get_all_and_clear();
      gnss_queue.insert(gnss_queue.end(), gnss_data.begin(), gnss_data.end());

      const auto new_submaps = input_submap_queue.get_all_and_clear();
      if (new_submaps.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }
      submap_queue.insert(submap_queue.end(), new_submaps.begin(), new_submaps.end());

      // Discard submaps that predate the oldest GNSS measurement.
      while (!gnss_queue.empty() && !submap_queue.empty() &&
             submap_queue.front()->frames.front()->stamp < gnss_queue.front()[0]) {
        submap_queue.pop_front();
      }

      // Temporally interpolate GNSS to each submap's mid-frame stamp.
      while (
        !gnss_queue.empty() && !submap_queue.empty() &&
        submap_queue.front()->frames.front()->stamp > gnss_queue.front()[0] &&
        submap_queue.front()->frames.back()->stamp  < gnss_queue.back()[0])
      {
        const auto& submap = submap_queue.front();
        const double stamp = submap->frames[submap->frames.size() / 2]->stamp;

        const auto right = std::lower_bound(
          gnss_queue.begin(), gnss_queue.end(), stamp,
          [](const Eigen::Matrix<double, 8, 1>& e, double t) { return e[0] < t; });
        if (right == gnss_queue.end() || right == gnss_queue.begin()) break;
        const auto left = right - 1;

        const double p = (stamp - (*left)[0]) / ((*right)[0] - (*left)[0]);
        Eigen::Matrix<double, 8, 1> interp = (1.0 - p) * (*left) + p * (*right);
        interp[4] = std::min((*left)[4], (*right)[4]);
        if ((*left)[5] < 0.0 || (*right)[5] < 0.0) {
          interp[5] = -1.0; interp[6] = -1.0; interp[7] = -1.0;
        }

        matched_submaps.push_back(submap);
        matched_coords.push_back(interp);

        submap_queue.pop_front();
        gnss_queue.erase(gnss_queue.begin(), left);
      }

      // Emit GPS factor for the latest matched submap.
      // Target is already in world frame (UTM-origin coords). No rotation needed.
      if (datum_initialized && !matched_coords.empty()) {
        const auto& coord = matched_coords.back();
        const Eigen::Vector3d xyz = coord.segment<3>(1);
        const int fix_status = static_cast<int>(coord[4]);

        Eigen::Vector3d inf_scale;
        if (coord[5] >= 0.0) {
          inf_scale = Eigen::Vector3d(
            1.0 / (coord[5] * coord[5]),
            1.0 / (coord[6] * coord[6]),
            1.0 / (coord[7] * coord[7]));
        } else {
          inf_scale = prior_inf_scale;
        }

        if (fix_status < STATUS_GBAS_FIX) {
          const double k2 = sbas_noise_inflation * sbas_noise_inflation;
          inf_scale /= k2;
        }

        const auto model = gtsam::noiseModel::Diagonal::Precisions(inf_scale);
        output_factors.push_back(gtsam::NonlinearFactor::shared_ptr(
          new gtsam::PoseTranslationPrior<gtsam::Pose3>(X(matched_submaps.back()->id), xyz, model)));
      }
    }

    // Write gnss_datum.json at end of run.
    if (datum_initialized) {
      write_datum_json("/tmp/dump/config/gnss_datum.json");
    }
  }

  void write_datum_json(const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream ofs(path);
    if (!ofs) {
      logger->error("failed to write gnss_datum.json to {}", path);
      return;
    }
    ofs << std::setprecision(15) << std::fixed;
    ofs << "{\n";
    ofs << "  \"latitude\": "            << datum_lat          << ",\n";
    ofs << "  \"longitude\": "           << datum_lon          << ",\n";
    ofs << "  \"altitude\": "            << datum_alt          << ",\n";
    ofs << "  \"utm_zone\": "            << datum_utm_zone     << ",\n";
    ofs << "  \"utm_easting_origin\": "  << datum_utm_easting  << ",\n";
    ofs << "  \"utm_northing_origin\": " << datum_utm_northing << "\n";
    ofs << "}\n";
    logger->info("gnss_datum.json written to {}", path);
  }

private:
  std::atomic_bool kill_switch;
  std::thread thread;

  ConcurrentVector<Eigen::Matrix<double, 8, 1>> input_gnss_queue;
  ConcurrentVector<SubMap::ConstPtr> input_submap_queue;
  ConcurrentVector<gtsam::NonlinearFactor::shared_ptr> output_factors;

  std::vector<SubMap::ConstPtr> matched_submaps;
  std::vector<Eigen::Matrix<double, 8, 1>> matched_coords;  // [stamp, utm_dx, utm_dy, utm_dz, status, sigma_E, sigma_N, sigma_U]

  std::string gnss_topic;
  Eigen::Vector3d prior_inf_scale;
  int min_fix_status;
  double sbas_noise_inflation;

  bool datum_initialized;
  bool datum_json_written;
  double datum_lat, datum_lon, datum_alt;
  int    datum_utm_zone;
  double datum_utm_easting;
  double datum_utm_northing;

  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::GNSSGlobal();
}
