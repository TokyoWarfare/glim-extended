#include <glim/util/colmap_export.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <set>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace glim {

// Axis convention: our (X fwd, Y left, Z up)  →  COLMAP cv (X right, Y down, Z fwd)
static const Eigen::Matrix3d R_our_to_cv = (Eigen::Matrix3d() <<
   0, -1,  0,
   0,  0, -1,
   1,  0,  0).finished();

// Rotate world from our (X fwd, Y left, Z up) to 3DGS-style right-handed Y-up
// (X right, Y up, Z back). Applied to points + camera poses so LichtFeld et al.
// display the scene upright. Det = +1 (proper rotation, preserves handedness).
//   new_x (right) = -old_y
//   new_y (up)    =  old_z
//   new_z (back)  = -old_x
static const Eigen::Matrix3d R_yup_from_our = (Eigen::Matrix3d() <<
   0, -1,  0,
   0,  0,  1,
  -1,  0,  0).finished();

// Eigen quaternion (w, x, y, z) from 3x3 rotation; normalized.
static Eigen::Quaterniond rotmat_to_quat(const Eigen::Matrix3d& R) {
  Eigen::Quaterniond q(R);
  q.normalize();
  return q;
}

ExportStats write_colmap_export(
  const ExportBounds2D& bounds,
  const std::vector<ColoredPoint>& points,
  const std::vector<ExportCameraFrame>& cameras,
  const std::vector<PinholeIntrinsics>& intrinsics_per_source,
  const std::vector<CameraType>& camera_type_per_source,
  const ExportOptions& options,
  std::string* error_msg) {
  auto type_of = [&](int si) {
    return (si >= 0 && si < static_cast<int>(camera_type_per_source.size()))
      ? camera_type_per_source[si] : CameraType::Pinhole;
  };

  ExportStats stats;
  auto seterr = [&](const std::string& m) { if (error_msg) *error_msg = m; };

  if (options.output_dir.empty()) { seterr("output_dir is empty"); return stats; }
  namespace fs = boost::filesystem;
  fs::create_directories(options.output_dir);
  fs::create_directories(options.output_dir + "/images");
  // COLMAP sparse model lives in sparse/0/ (multi-model convention; LichtFeld,
  // gsplat, nerfstudio all look here). images/ and masks/ stay at dataset root.
  const std::string sparse_dir = options.output_dir + "/sparse/0";
  fs::create_directories(sparse_dir);

  // --- Filter points by bounds ---
  std::vector<const ColoredPoint*> kept_points;
  kept_points.reserve(points.size());
  for (const auto& p : points) if (bounds.contains_xy(p.xyz)) kept_points.push_back(&p);

  // --- Filter cameras: position inside bounds + overlap_margin ---
  const float marg = std::max(0.0f, options.overlap_margin_m);
  ExportBounds2D expanded{
    bounds.x_min - marg, bounds.x_max + marg,
    bounds.y_min - marg, bounds.y_max + marg,
    bounds.z_min, bounds.z_max,
    bounds.yaw_deg
  };
  std::vector<const ExportCameraFrame*> kept_cams;
  kept_cams.reserve(cameras.size());
  for (const auto& c : cameras) {
    Eigen::Vector3f pos = c.T_world_cam.translation().cast<float>();
    if (expanded.contains_xy(pos)) kept_cams.push_back(&c);
  }

  // --- Compute origin offset (tile center) ---
  Eigen::Vector3d origin_offset = Eigen::Vector3d::Zero();
  if (options.re_origin) {
    origin_offset = Eigen::Vector3d(
      0.5 * (bounds.x_min + bounds.x_max),
      0.5 * (bounds.y_min + bounds.y_max),
      0.5 * (bounds.z_min + bounds.z_max));
  }
  stats.origin_offset = origin_offset;

  // World rotation applied AFTER re-origin. Two stages, composed in order:
  //   1. R_yaw:  rotate world around Z by -bounds.yaw_deg so the region's local
  //              X/Y aligns with output X/Y (tile is axis-aligned in the export).
  //   2. R_yup:  Z-up -> Y-up (our -> 3DGS convention). Skipped if option off.
  // R_world = R_yup * R_yaw  (applied to points via P_out = R_world * (P - origin)).
  const double yaw_rad = -static_cast<double>(bounds.yaw_deg) * M_PI / 180.0;
  Eigen::Matrix3d R_yaw = Eigen::Matrix3d::Identity();
  {
    const double c = std::cos(yaw_rad), s = std::sin(yaw_rad);
    R_yaw << c, -s, 0,
             s,  c, 0,
             0,  0, 1;
  }
  const Eigen::Matrix3d R_yup_or_id = options.rotate_to_y_up
    ? R_yup_from_our
    : Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R_world = R_yup_or_id * R_yaw;

  // --- Write cameras.txt ---
  // One entry per unique source_idx used by kept_cams (all images share a model).
  // IDs are 0-based to match Metashape/COLMAP convention.
  //   PINHOLE (fx fy cx cy)        -- when images are undistorted during export
  //   OPENCV  (fx fy cx cy k1 k2 p1 p2) -- when raw distorted images are emitted
  std::set<int> used_sources;
  for (const auto* c : kept_cams) used_sources.insert(c->source_idx);
  std::ofstream cams_ofs(sparse_dir + "/cameras.txt");
  if (!cams_ofs) { seterr("failed to open cameras.txt"); return stats; }
  cams_ofs << "# Camera list with one line of data per camera:\n";
  cams_ofs << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
  cams_ofs << "# Number of cameras: " << used_sources.size() << "\n";
  for (int si : used_sources) {
    if (si < 0 || si >= static_cast<int>(intrinsics_per_source.size())) continue;
    const auto& k = intrinsics_per_source[si];
    if (options.export_undistorted) {
      cams_ofs << si << " PINHOLE " << k.width << " " << k.height
               << " " << std::setprecision(10) << k.fx
               << " " << k.fy
               << " " << k.cx
               << " " << k.cy
               << "\n";
    } else {
      cams_ofs << si << " OPENCV " << k.width << " " << k.height
               << " " << std::setprecision(10) << k.fx
               << " " << k.fy
               << " " << k.cx
               << " " << k.cy
               << " " << k.k1
               << " " << k.k2
               << " " << k.p1
               << " " << k.p2
               << "\n";
    }
    stats.cameras_written++;
  }

  // --- Write images.txt + undistort image files + masks ---
  std::ofstream imgs(sparse_dir + "/images.txt");
  if (!imgs) { seterr("failed to open images.txt"); return stats; }
  imgs << "# Image list with two lines of data per image:\n";
  imgs << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
  imgs << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
  imgs << "# Number of images: " << kept_cams.size() << "\n";

  // Masks go to a masks/ sibling folder so Postshot / Lichtfeld pick them up.
  const bool any_masks = [&]{ for (const auto* c : kept_cams) if (!c->source_mask_path.empty()) return true; return false; }();
  if (any_masks) fs::create_directories(options.output_dir + "/masks");

  // Cache undistort maps per source (same intrinsics -> same map; compute once per source).
  struct UndistortMap { cv::Mat mapx, mapy; int w = 0, h = 0; bool valid = false; };
  std::vector<UndistortMap> undist_maps(intrinsics_per_source.size());

  // Collected per-image camera data for optional Bundler / BlocksExchange export.
  // R_cv_world + t_cv_world are POST-transform (re-origin + Y-up world rotation
  // already applied), so downstream format writers just need a frame conversion
  // if their camera convention differs (Bundler: OpenGL). The world-space center
  // is derived on demand as C = -R^T * t.
  struct ExportCam {
    int source_idx;
    std::string export_name;
    Eigen::Matrix3d R_cv_world;
    Eigen::Vector3d t_cv_world;
  };
  std::vector<ExportCam> export_cams;
  if (options.export_bundler || options.export_blocks_exchange) export_cams.reserve(kept_cams.size());

  int next_image_id = 0;  // 0-based to match Metashape/COLMAP convention
  for (const auto* c : kept_cams) {
    const int cam_id = c->source_idx;  // shared per source, 0-based

    // COLMAP stores T_cam_world (world -> camera, OpenCV convention).
    Eigen::Isometry3d T_our_world = c->T_world_cam.inverse();                 // our cam <- world
    Eigen::Matrix3d R_cv_world = R_our_to_cv * T_our_world.rotation();        // cv cam <- world_our
    Eigen::Vector3d t_cv_world = R_our_to_cv * T_our_world.translation();
    if (options.re_origin) {
      // Points shifted by -origin_offset -> t_new = t_old + R * origin_offset
      // (keeps P_cam = R*(P_world - offset) + t_new identical to R*P_world + t_old).
      t_cv_world += R_cv_world * origin_offset;
    }
    // World-axis rotation (Z-up -> Y-up): P_world_new = R_world * P_world_old,
    // so camera rotation needs R_cv_world_new = R_cv_world_old * R_world^T
    // (translation unchanged). No-op when R_world is identity.
    R_cv_world = R_cv_world * R_world.transpose();
    Eigen::Quaterniond q = rotmat_to_quat(R_cv_world);

    imgs << next_image_id
         << " " << std::setprecision(10)
         << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
         << " " << t_cv_world.x() << " " << t_cv_world.y() << " " << t_cv_world.z()
         << " " << cam_id
         << " " << c->export_name
         << "\n";
    imgs << "\n";  // empty POINTS2D line (no feature track info for 3DGS init)
    next_image_id++;

    if (options.export_bundler || options.export_blocks_exchange) {
      export_cams.push_back({c->source_idx, c->export_name, R_cv_world, t_cv_world});
    }

    // Write image to images/. Two paths:
    //  - export_undistorted: read -> cv::remap undistort -> cv::imwrite (re-encode).
    //    Must COPY (no symlink: the source image is distorted, can't point at it).
    //  - !export_undistorted: copy or symlink the raw file; distortion stays for
    //    the downstream tool to handle via the OPENCV model in cameras.txt.
    const std::string dst_img = options.output_dir + "/images/" + c->export_name;
    const int si = c->source_idx;
    if (si >= 0 && si < static_cast<int>(intrinsics_per_source.size())) {
      const auto& k = intrinsics_per_source[si];
      const bool has_distortion =
        (k.k1 != 0.0 || k.k2 != 0.0 || k.k3 != 0.0 || k.p1 != 0.0 || k.p2 != 0.0);

      if (!options.export_undistorted) {
        // Raw copy / symlink path (distortion stays in images, handled by OPENCV intrinsics).
        if (options.copy_images) {
          try { fs::copy_file(c->source_image_path, dst_img, fs::copy_option::overwrite_if_exists); stats.images_copied++; }
          catch (const fs::filesystem_error& e) { seterr(std::string("image copy failed: ") + e.what()); }
        } else {
          try {
            if (fs::exists(dst_img)) fs::remove(dst_img);
            fs::create_symlink(c->source_image_path, dst_img);
            stats.images_copied++;
          } catch (const fs::filesystem_error& e) { seterr(std::string("image symlink failed: ") + e.what()); }
        }
        if (!c->source_mask_path.empty() && fs::exists(c->source_mask_path)) {
          const std::string stem = fs::path(c->export_name).stem().string();
          const std::string mask_dst = options.output_dir + "/masks/" + stem + ".png";
          try { fs::copy_file(c->source_mask_path, mask_dst, fs::copy_option::overwrite_if_exists); stats.masks_copied++; }
          catch (const fs::filesystem_error& e) { seterr(std::string("mask copy failed: ") + e.what()); }
        }
        continue;  // next camera frame
      }

      // Undistort path. cv::undistort keeps the original K (no cropping/scaling),
      // so the PINHOLE fx/fy/cx/cy we wrote to cameras.txt is the correct model.
      cv::Mat img = cv::imread(c->source_image_path);
      if (img.empty()) {
        seterr(std::string("failed to read image: ") + c->source_image_path);
        continue;
      }
      cv::Mat out;
      if (has_distortion) {
        // Build / reuse remap tables for this source.
        if (!undist_maps[si].valid ||
            undist_maps[si].w != img.cols || undist_maps[si].h != img.rows) {
          cv::Mat K = (cv::Mat_<double>(3, 3) << k.fx, 0, k.cx, 0, k.fy, k.cy, 0, 0, 1);
          cv::Mat D = (cv::Mat_<double>(1, 5) << k.k1, k.k2, k.p1, k.p2, k.k3);
          cv::initUndistortRectifyMap(K, D, cv::Mat(), K, img.size(), CV_16SC2,
                                      undist_maps[si].mapx, undist_maps[si].mapy);
          undist_maps[si].w = img.cols; undist_maps[si].h = img.rows;
          undist_maps[si].valid = true;
        }
        cv::remap(img, out, undist_maps[si].mapx, undist_maps[si].mapy, cv::INTER_LINEAR);
      } else {
        out = img;
      }
      // Preserve JPEG quality (tools like LichtFeld are sensitive to re-compression).
      std::vector<int> jpg_params = {cv::IMWRITE_JPEG_QUALITY, 95};
      if (cv::imwrite(dst_img, out, jpg_params)) stats.images_copied++;
      else seterr(std::string("failed to write undistorted image: ") + dst_img);

      // Undistort + write mask (nearest-neighbor to preserve binary values).
      if (!c->source_mask_path.empty() && fs::exists(c->source_mask_path)) {
        const std::string stem = fs::path(c->export_name).stem().string();
        const std::string mask_dst = options.output_dir + "/masks/" + stem + ".png";
        cv::Mat mask = cv::imread(c->source_mask_path, cv::IMREAD_UNCHANGED);
        if (!mask.empty()) {
          cv::Mat mask_out;
          if (has_distortion) {
            cv::remap(mask, mask_out, undist_maps[si].mapx, undist_maps[si].mapy, cv::INTER_NEAREST);
          } else {
            mask_out = mask;
          }
          if (cv::imwrite(mask_dst, mask_out)) stats.masks_copied++;
        }
      }
    }
  }

  // --- Write points3D.ply (binary little-endian; standard 3DGS init format) ---
  // Lichtfeld / gaussian-splatting / nerfstudio all accept float xyz + uchar rgb.
  // The parallel points3D.txt writer was removed: Metashape + every downstream
  // tool we target reads the .ply directly, and the .txt was just a bloat copy.
  {
    const std::string ply_path = sparse_dir + "/points3D.ply";
    std::ofstream ofs(ply_path, std::ios::binary);
    if (!ofs) { seterr("failed to open points3D.ply"); return stats; }
    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << kept_points.size() << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ofs << "end_header\n";
    for (const auto* p : kept_points) {
      Eigen::Vector3d p_world = p->xyz.cast<double>();
      if (options.re_origin) p_world -= origin_offset;
      p_world = R_world * p_world;
      const float x = static_cast<float>(p_world.x());
      const float y = static_cast<float>(p_world.y());
      const float z = static_cast<float>(p_world.z());
      const uint8_t r = static_cast<uint8_t>(std::clamp(static_cast<int>(p->rgb.x() * 255.0f), 0, 255));
      const uint8_t g = static_cast<uint8_t>(std::clamp(static_cast<int>(p->rgb.y() * 255.0f), 0, 255));
      const uint8_t b = static_cast<uint8_t>(std::clamp(static_cast<int>(p->rgb.z() * 255.0f), 0, 255));
      ofs.write(reinterpret_cast<const char*>(&x), 4);
      ofs.write(reinterpret_cast<const char*>(&y), 4);
      ofs.write(reinterpret_cast<const char*>(&z), 4);
      ofs.put(static_cast<char>(r));
      ofs.put(static_cast<char>(g));
      ofs.put(static_cast<char>(b));
    }
    stats.points_written = kept_points.size();
  }
  // --- Optional Bundler export (bundle.out at dataset root) ---
  // Metashape accepts Bundler via File -> Import Cameras... -> Bundler.
  // Bundler uses OpenGL camera convention (X right, Y up, Z back), distinct
  // from COLMAP's OpenCV convention (X right, Y down, Z fwd). Conversion:
  //   R_bundler = diag(1, -1, -1) * R_cv
  //   t_bundler = diag(1, -1, -1) * t_cv
  if (options.export_bundler) {
    const std::string bpath = options.output_dir + "/bundle.out";
    std::ofstream ofs(bpath);
    if (!ofs) { seterr("failed to open bundle.out"); }
    else {
      const Eigen::Matrix3d R_bundler_from_cv = (Eigen::Matrix3d() <<
        1,  0,  0,
        0, -1,  0,
        0,  0, -1).finished();
      // Cameras-only export. Bundler requires a point count in the header
      // but is fine with zero points (Metashape just loads the cameras and
      // prompts for a separate point cloud import when needed).
      ofs << "# Bundle file v0.3\n";
      ofs << export_cams.size() << " " << 0 << "\n";
      for (const auto& bc : export_cams) {
        const auto& k = intrinsics_per_source[bc.source_idx];
        // Bundler intrinsics: focal (single, in pixels) + k1 k2 (radial).
        // With undistorted images we pass zero distortion, with raw images
        // we pass the original k1/k2 (Bundler can't represent k3/p1/p2).
        const double f = 0.5 * (k.fx + k.fy);  // assume nearly equal
        const double k1 = options.export_undistorted ? 0.0 : k.k1;
        const double k2 = options.export_undistorted ? 0.0 : k.k2;
        ofs << std::setprecision(10) << f << " " << k1 << " " << k2 << "\n";
        Eigen::Matrix3d R_b = R_bundler_from_cv * bc.R_cv_world;
        Eigen::Vector3d t_b = R_bundler_from_cv * bc.t_cv_world;
        for (int r = 0; r < 3; r++)
          ofs << R_b(r, 0) << " " << R_b(r, 1) << " " << R_b(r, 2) << "\n";
        ofs << t_b.x() << " " << t_b.y() << " " << t_b.z() << "\n";
      }
    }
    // Sidecar image list that Bundler+Metashape workflow expects. export_cams
    // is 1:1 parallel with kept_cams (pushed inside the same loop, no skips).
    std::ofstream ilist(options.output_dir + "/bundle.out.list.txt");
    if (ilist) {
      for (const auto& ec : export_cams) ilist << "images/" << ec.export_name << "\n";
    }
  }

  // --- Optional BlocksExchange XML export ---
  // ContextCapture / RealityCapture / Metashape-compatible block file. Same
  // camera convention as COLMAP (world -> OpenCV camera frame), so the R/t we
  // already computed drop straight in. Center = -R^T * t (camera position in
  // the exported world). Per-source intrinsics grouped in <Photogroup>.
  if (options.export_blocks_exchange) {
    const std::string xml_path = options.output_dir + "/blocks_exchange.xml";
    std::ofstream ofs(xml_path);
    if (!ofs) { seterr("failed to open blocks_exchange.xml"); }
    else {
      ofs << std::setprecision(10);
      ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
      // Version 2.1 is what Metashape's importer targets most reliably; 3.x
      // works in newer CC but Metashape's parser complains about fields it
      // hasn't back-ported.
      ofs << "<BlocksExchange version=\"2.1\">\n";
      // Omit <SpatialReferenceSystems> + <SRSId> entirely: per ContextCapture
      // spec, the default assumption is a local Cartesian coordinate system,
      // which is exactly what our coords are. Declaring a custom LOCAL_CS WKT
      // triggers Metashape's datum-transformation lookup and fails with
      // "unsupported datum transformation" -- the project's default SRS
      // (usually WGS84) can't be transformed to our named Local.
      ofs << "  <Block>\n";
      ofs << "    <Name>GLIM export</Name>\n";
      ofs << "    <Photogroups>\n";

      // One <Photogroup> per unique source_idx (shared intrinsics).
      std::set<int> used_sources_be;
      for (const auto& ec : export_cams) used_sources_be.insert(ec.source_idx);
      for (int si : used_sources_be) {
        if (si < 0 || si >= static_cast<int>(intrinsics_per_source.size())) continue;
        const auto& k = intrinsics_per_source[si];
        const bool is_spherical = (type_of(si) == CameraType::Spherical);
        ofs << "      <Photogroup>\n";
        ofs << "        <Name>src" << si << "</Name>\n";
        ofs << "        <ImageDimensions>\n";
        ofs << "          <Width>"  << k.width  << "</Width>\n";
        ofs << "          <Height>" << k.height << "</Height>\n";
        ofs << "        </ImageDimensions>\n";
        if (is_spherical) {
          // Metashape reads <CameraModelType>Spherical</CameraModelType> for
          // full 360 equirect. No focal / principal point / distortion: the
          // projection is fixed, f derived as w/(2*pi), cx/cy at image centre.
          ofs << "        <CameraModelType>Spherical</CameraModelType>\n";
        } else {
          ofs << "        <CameraModelType>Perspective</CameraModelType>\n";
          ofs << "        <CameraOrientation>XRightYDown</CameraOrientation>\n";
          // Focal length in pixels (BlocksExchange accepts FocalLengthPixels).
          ofs << "        <FocalLengthPixels>" << 0.5 * (k.fx + k.fy) << "</FocalLengthPixels>\n";
          // Aspect ratio fy/fx (1.0 when square pixels).
          ofs << "        <AspectRatio>" << (k.fx > 0 ? k.fy / k.fx : 1.0) << "</AspectRatio>\n";
          ofs << "        <PrincipalPoint>\n";
          ofs << "          <x>" << k.cx << "</x>\n";
          ofs << "          <y>" << k.cy << "</y>\n";
          ofs << "        </PrincipalPoint>\n";
          // Distortion: zero when exporting undistorted images, else source Brown-Conrady.
          const double k1 = options.export_undistorted ? 0.0 : k.k1;
          const double k2 = options.export_undistorted ? 0.0 : k.k2;
          const double k3 = options.export_undistorted ? 0.0 : k.k3;
          const double p1 = options.export_undistorted ? 0.0 : k.p1;
          const double p2 = options.export_undistorted ? 0.0 : k.p2;
          ofs << "        <Distortion>\n";
          ofs << "          <K1>" << k1 << "</K1>\n";
          ofs << "          <K2>" << k2 << "</K2>\n";
          ofs << "          <K3>" << k3 << "</K3>\n";
          ofs << "          <P1>" << p1 << "</P1>\n";
          ofs << "          <P2>" << p2 << "</P2>\n";
          ofs << "        </Distortion>\n";
        }

        // All photos belonging to this source.
        int photo_id = 0;
        for (const auto& ec : export_cams) {
          if (ec.source_idx != si) continue;
          const Eigen::Vector3d C = -ec.R_cv_world.transpose() * ec.t_cv_world;
          ofs << "        <Photo>\n";
          ofs << "          <Id>" << photo_id++ << "</Id>\n";
          ofs << "          <ImagePath>images/" << ec.export_name << "</ImagePath>\n";
          ofs << "          <Pose>\n";
          ofs << "            <Rotation>\n";
          for (int r = 0; r < 3; r++) {
            for (int cc = 0; cc < 3; cc++) {
              ofs << "              <M_" << r << cc << ">" << ec.R_cv_world(r, cc) << "</M_" << r << cc << ">\n";
            }
          }
          ofs << "            </Rotation>\n";
          ofs << "            <Center>\n";
          ofs << "              <x>" << C.x() << "</x>\n";
          ofs << "              <y>" << C.y() << "</y>\n";
          ofs << "              <z>" << C.z() << "</z>\n";
          ofs << "            </Center>\n";
          if (options.emit_pose_priors) {
            // Position + rotation accuracy hints -- ContextCapture spec + Metashape
            // read these as BA constraints. Same sigma for all three axes is the
            // usual assumption when we don't have per-axis uncertainty.
            ofs << "            <Accuracy>\n";
            ofs << "              <Horizontal>" << options.pose_pos_sigma_m << "</Horizontal>\n";
            ofs << "              <Vertical>"   << options.pose_pos_sigma_m << "</Vertical>\n";
            ofs << "              <Rotation>"   << options.pose_rot_sigma_deg << "</Rotation>\n";
            ofs << "            </Accuracy>\n";
          }
          ofs << "          </Pose>\n";
          ofs << "        </Photo>\n";
        }
        ofs << "      </Photogroup>\n";
      }
      ofs << "    </Photogroups>\n";
      ofs << "  </Block>\n";
      ofs << "</BlocksExchange>\n";
    }
  }

  // --- Manifest (tile metadata; handy for later stitching + debugging) ---
  {
    nlohmann::json m;
    m["bounds_world"] = {
      {"x", {bounds.x_min, bounds.x_max}},
      {"y", {bounds.y_min, bounds.y_max}},
      {"z", {bounds.z_min, bounds.z_max}},
    };
    m["overlap_margin_m"] = options.overlap_margin_m;
    m["re_origined"] = options.re_origin;
    m["origin_offset"] = {origin_offset.x(), origin_offset.y(), origin_offset.z()};
    m["voxel_size_m"] = options.voxel_size_m;
    m["stats"] = {
      {"points", stats.points_written},
      {"cameras", stats.cameras_written},
      {"images", stats.images_copied},
      {"masks", stats.masks_copied},
    };
    m["axis_convention"] = "COLMAP (X right, Y down, Z forward) — converted from our (X fwd, Y left, Z up)";
    std::ofstream ofs(options.output_dir + "/manifest.json");
    ofs << std::setw(2) << m << std::endl;
  }

  return stats;
}

}  // namespace glim
