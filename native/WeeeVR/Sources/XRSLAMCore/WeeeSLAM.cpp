// C wrapper around XRSLAM's core (XRSLAM::Detail), replacing its singleton C API (xrslam-interface):
// we need reset, velocity and landmarks, and a header Swift can import.
#include "XRSLAMCore.h"
#include "Relocalizer.h"

#include <xrslam/common.h>
#include <xrslam/core/detail.h>
#include <xrslam/core/feature_tracker.h>
#include <xrslam/core/frontend_worker.h>
#include <xrslam/extra/opencv_image.h>
#include <xrslam/geometry/stereo.h>
#include <xrslam/inspection.h>
#include <xrslam/map/frame.h>
#include <xrslam/map/map.h>
#include <xrslam/map/track.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

namespace {

using namespace xrslam;

/// Configuration from code instead of XRSLAM's YAML files. Values not set from WeeeSLAMConfig
/// follow XRSLAM's iOS app (xrslam-ios/visualizer/configs: slam_params.yaml + device yaml).
class WeeeConfig : public Config {
  public:
    explicit WeeeConfig(const WeeeSLAMConfig &c) : c(c) {}

    vector<2> camera_resolution() const override { return {double(c.width), double(c.height)}; }
    matrix<3> camera_intrinsic() const override {
        matrix<3> k;
        k << c.fx, 0, c.cx, 0, c.fy, c.cy, 0, 0, 1;
        return k;
    }
    vector<4> camera_distortion() const override { return vector<4>::Zero(); }
    quaternion camera_to_body_rotation() const override {
        return quaternion(c.q_bc[3], c.q_bc[0], c.q_bc[1], c.q_bc[2]).normalized();
    }
    vector<3> camera_to_body_translation() const override { return {c.p_bc[0], c.p_bc[1], c.p_bc[2]}; }
    size_t camera_distortion_flag() const override { return 0; }
    double camera_time_offset() const override { return c.time_offset; }
    quaternion imu_to_body_rotation() const override { return quaternion::Identity(); }
    vector<3> imu_to_body_translation() const override { return vector<3>::Zero(); }

    matrix<2> keypoint_noise_cov() const override { return matrix<2>::Identity() * 0.5; }
    matrix<3> gyroscope_noise_cov() const override { return matrix<3>::Identity() * 2.8791302399999997e-08; }
    matrix<3> accelerometer_noise_cov() const override { return matrix<3>::Identity() * 4.0e-6; }
    matrix<3> gyroscope_bias_noise_cov() const override { return matrix<3>::Identity() * 3.7608844899999997e-10; }
    matrix<3> accelerometer_bias_noise_cov() const override { return matrix<3>::Identity() * 9.0e-6; }

    size_t sliding_window_size() const override { return size_t(c.window_size); }
    size_t sliding_window_tracker_frequent() const override { return size_t(c.tracker_frequent); }
    double feature_tracker_min_keypoint_distance() const override { return c.min_keypoint_distance; }
    size_t feature_tracker_max_keypoint_detection() const override { return size_t(c.max_keypoints); }
    size_t feature_tracker_max_frames() const override { return 100; }
    size_t solver_iteration_limit() const override { return size_t(c.solver_iterations); }
    double solver_time_limit() const override { return c.solver_time_limit; }

    bool visual_localization_enable() const override { return false; }
    bool parsac_flag() const override { return false; }
    double parsac_dynamic_probability() const override { return 0.15; }
    double parsac_threshold() const override { return 1.0; }
    double parsac_norm_scale() const override { return 1.0; }
    size_t parsac_keyframe_check_size() const override { return 1; }

  private:
    WeeeSLAMConfig c;
};

void clear_landmarks() {
    inspect_debug(sliding_window_landmarks, landmarks) { landmarks = std::any(); }
}

} // namespace

struct WeeeSLAM {
    WeeeSLAMConfig config;
    std::shared_ptr<WeeeConfig> xrslamConfig;
    // Guards `detail` against reset: shared by get_state and the relocalization snapshot (so neither
    // waits for the other), exclusive only for reset's swap.
    std::shared_mutex mutex;
    std::unique_ptr<XRSLAM::Detail> detail;
    uint64_t images = 0;
    cv::Mat undistortX, undistortY;            // remap tables when the lens has residual distortion

    // Relocalization (XRSLAMCore.h). Images: the last few tracking images by XRSLAM time, so the
    // relocalization step can take the one its pose belongs to.
    std::mutex image_mutex;
    std::deque<std::pair<double, cv::Mat>> recent;

    std::mutex align_mutex;                    // the session/map bookkeeping below
    uint64_t map_id = 0, session_id = 1;
    bool session_has_map = false;
    bool tracking = false;                     // XRSLAM status seen by the last get_state
    bool aligned = true;
    double map_start = 0;                      // first tracking time of the current map
    matrix<3> R_sm = matrix<3>::Identity();    // session <- map: yaw about gravity...
    vector<3> t_sm = vector<3>::Zero();        // ...and translation
    int keyframes = 0, relocalizations = 0;
    bool clear_database = false;               // reset/new session: the step clears its database

    // Relocalization step's thread only.
    std::unique_ptr<weee::Relocalizer> relocalizer;
    struct Candidate {
        uint64_t map_id;
        double t, yaw;
        vector<3> translation;
    };
    std::optional<Candidate> candidate;        // first of two agreeing relocalizations
};

namespace {

constexpr size_t kRecentImages = 8;          // 2.4 MB: covers the optimized frame's lag
constexpr double kRelocTimeout = 20;          // s on a new map without relocalizing: new session
constexpr double kKeyframeSettleTime = 1;     // s of tracking on a map before its keyframes are stored
constexpr size_t kKeyframeMinLandmarks = 40;

/// get_state's view of XRSLAM's status, under align_mutex: a new map begins whenever tracking
/// (re)starts, because XRSLAM always reinitializes into a new world.
void observe_status(WeeeSLAM *slam, bool tracking, double t) {
    if (tracking && !slam->tracking) {
        slam->map_id += 1;
        slam->map_start = t;
        if (!slam->session_has_map) {
            slam->session_has_map = true;           // the session's first map defines its world
            slam->aligned = true;
            slam->R_sm.setIdentity();
            slam->t_sm.setZero();
        } else if (slam->keyframes == 0) {
            // Nothing to relocalize against: this map starts a new session in its own world.
            slam->session_id += 1;
            slam->clear_database = true;
            slam->aligned = true;
            slam->R_sm.setIdentity();
            slam->t_sm.setZero();
        } else {
            slam->aligned = false;
        }
    }
    slam->tracking = tracking;
}

/// The latest frame the sliding window optimized, under slam->mutex: its time, camera pose
/// (world <- OpenCV camera, current map) and the tracker's observations of triangulated landmarks
/// in it (pixel, world point), all from XRSLAM's locked mirror of its window (keymap). Not the
/// feature tracker's latest state (what get_state reports): that one is IMU-predicted for the
/// newest frame and a few pixels off, which would bias every keyframe and relocalization.
bool optimized_frame(WeeeSLAM *slam, double &t, matrix<3> &R_wc, vector<3> &p_wc,
                     std::vector<cv::Point2f> &pixels, std::vector<vector<3>> &points) {
    auto [time, frame_id, pose, motion] = slam->detail->frontend->get_latest_state();
    if (frame_id == nil()) return false;
    bool found = false;
    synchronized(slam->detail->feature_tracker->keymap) {
        Map *keymap = slam->detail->feature_tracker->keymap.get();
        size_t index = keymap->frame_index_by_id(frame_id);
        if (index != nil()) {
            Frame *frame = keymap->get_frame(index);
            PoseState camera = frame->get_pose(frame->camera);
            t = frame->image->t;
            R_wc = camera.q.toRotationMatrix();
            p_wc = camera.p;
            for (size_t k = 0; k < frame->keypoint_num(); ++k) {
                Track *track = frame->get_track(k);
                if (!track || !track->all_tagged(TT_VALID, TT_TRIANGULATED)) continue;
                vector<2> u = apply_k(frame->get_keypoint(k), frame->K);
                pixels.emplace_back(float(u.x()), float(u.y()));
                points.push_back(track->get_landmark_point());
            }
            found = true;
        }
    }
    return found;
}

} // namespace

extern "C" {

WeeeSLAM *weee_slam_create(const WeeeSLAMConfig *config) {
    try {
        auto slam = std::make_unique<WeeeSLAM>();
        slam->config = *config;
        slam->xrslamConfig = std::make_shared<WeeeConfig>(*config);
        clear_landmarks();
        slam->detail = std::make_unique<XRSLAM::Detail>(slam->xrslamConfig);
        const double *d = config->distortion;
        if (d[0] != 0 || d[1] != 0 || d[2] != 0 || d[3] != 0) {
            // Undistort into the same pinhole camera; the tables are built once, remap is ~1 ms.
            cv::Mat k = (cv::Mat_<double>(3, 3) << config->fx, 0, config->cx, 0, config->fy, config->cy, 0, 0, 1);
            cv::Mat coeffs = (cv::Mat_<double>(1, 4) << d[0], d[1], d[2], d[3]);
            cv::initUndistortRectifyMap(k, coeffs, cv::Mat(), k, cv::Size(config->width, config->height), CV_16SC2,
                                        slam->undistortX, slam->undistortY);
        }
        log_message(XRSLAM_LOG_INFO,
                    "created %s: %dx%d fx=%.1f fy=%.1f cx=%.1f cy=%.1f dist=%.4f,%.4f,%.5f,%.5f window=%d keypoints=%d",
                    XRSLAM_VERSION_STRING, config->width, config->height, config->fx, config->fy, config->cx,
                    config->cy, d[0], d[1], d[2], d[3], config->window_size, config->max_keypoints);
        return slam.release();
    } catch (const std::exception &e) {
        log_message(XRSLAM_LOG_ERR, "create failed: %s", e.what());
        return nullptr;
    }
}

void weee_slam_destroy(WeeeSLAM *slam) { delete slam; }

void weee_slam_reset(WeeeSLAM *slam) {
    // The new core is swapped in under the lock and the old one torn down afterwards: its destructor
    // joins XRSLAM's threads (waiting out a bundle adjustment, up to ~0.1 s), which would otherwise
    // stall get_state on the camera thread.
    auto fresh = std::make_unique<XRSLAM::Detail>(slam->xrslamConfig);
    std::unique_ptr<XRSLAM::Detail> old;
    {
        std::unique_lock<std::shared_mutex> lock(slam->mutex);
        old = std::move(slam->detail);
        slam->detail = std::move(fresh);
        slam->images = 0;
        std::lock_guard<std::mutex> align(slam->align_mutex);
        slam->session_id += 1;
        slam->session_has_map = false;
        slam->tracking = false;
        slam->aligned = true;
        slam->keyframes = 0;
        slam->clear_database = true;
    }
    old.reset();
    clear_landmarks();      // after the old threads are gone; the new core has no landmarks yet
}

void weee_slam_push_gyro(WeeeSLAM *slam, double t, double x, double y, double z) {
    slam->detail->track_gyroscope(t, x, y, z);
}

void weee_slam_push_accel(WeeeSLAM *slam, double t, double x, double y, double z) {
    slam->detail->track_accelerometer(t, x, y, z);
}

void weee_slam_push_image(WeeeSLAM *slam, double t, const uint8_t *gray, int width, int height, int stride) {
    if (width != slam->config.width || height != slam->config.height) return;
    auto image = std::make_shared<extra::OpenCvImage>();
    image->t = t + slam->config.time_offset;
    cv::Mat source(height, width, CV_8UC1, const_cast<uint8_t *>(gray), size_t(stride));
    if (slam->undistortX.empty()) {
        image->image = source.clone();
    } else {
        cv::remap(source, image->image, slam->undistortX, slam->undistortY, cv::INTER_LINEAR);
    }
    image->raw = image->image;
    {
        std::lock_guard<std::mutex> lock(slam->image_mutex);
        slam->recent.emplace_back(image->t, image->image.clone());
        if (slam->recent.size() > kRecentImages) slam->recent.pop_front();
    }
    slam->detail->track_camera(image);
    slam->images += 1;
}

void weee_slam_get_state(WeeeSLAM *slam, WeeeSLAMState *out) {
    *out = WeeeSLAMState{};
    std::shared_lock<std::shared_mutex> lock(slam->mutex);
    out->images = slam->images;

    SysState system = slam->detail->get_system_state();
    auto latest = slam->detail->feature_tracker->get_latest_state();
    if (system == SYS_TRACKING && latest &&
        std::get<0>(*latest) < std::get<0>(slam->detail->frontend->get_latest_state())) {
        // Just reinitialized: until it tracks the next frame, the feature tracker still holds the
        // previous map's last state (older than the new map's first optimized frame). Reporting it
        // would start the new map at the wrong time.
        system = SYS_INITIALIZING;
        latest.reset();
    }
    if (system == SYS_INITIALIZING) {
        out->status = WEEE_SLAM_INITIALIZING;
    } else if (system == SYS_TRACKING && latest) {
        out->status = WEEE_SLAM_TRACKING;
    } else {
        out->status = WEEE_SLAM_LOST;
    }

    matrix<3> R_sm = matrix<3>::Identity();
    vector<3> t_sm = vector<3>::Zero();
    {
        std::lock_guard<std::mutex> align(slam->align_mutex);
        observe_status(slam, out->status == WEEE_SLAM_TRACKING && latest, latest ? std::get<0>(*latest) : 0);
        out->map_id = slam->map_id;
        out->session_id = slam->session_id;
        out->aligned = slam->aligned;
        out->keyframes = slam->keyframes;
        out->relocalizations = slam->relocalizations;
        if (slam->aligned) {
            R_sm = slam->R_sm;
            t_sm = slam->t_sm;
        }
    }
    if (!latest) return;

    auto [t, pose, motion] = latest.value();
    const WeeeConfig &config = *slam->xrslamConfig;
    quaternion q_wc = pose.q * config.camera_to_body_rotation();
    vector<3> p_wc = pose.p + pose.q * config.camera_to_body_translation();
    // Reported in the session world when aligned (identity for a session's first map).
    quaternion q_out = quaternion(R_sm) * q_wc;
    vector<3> p_out = R_sm * p_wc + t_sm;
    vector<3> v_out = R_sm * motion.v;
    out->has_pose = 1;
    out->t = t;
    out->q_wc[0] = q_out.x(); out->q_wc[1] = q_out.y(); out->q_wc[2] = q_out.z(); out->q_wc[3] = q_out.w();
    for (int i = 0; i < 3; ++i) {
        out->p_wc[i] = p_out(i);
        out->v_wb[i] = v_out(i);
        out->bg[i] = motion.bg(i);
    }

    // Depth proxy: median camera-frame depth of landmarks near the optical axis (±20°), in the map.
    inspect_debug(sliding_window_landmarks, landmarks) {
        if (!landmarks.has_value()) return;
        const auto *points = std::any_cast<std::vector<Landmark>>(&landmarks);
        if (!points) return;
        out->landmarks = int(points->size());
        matrix<3> r_cw = q_wc.conjugate().toRotationMatrix();
        const double tan20 = 0.364;
        std::vector<double> depths;
        depths.reserve(points->size());
        for (const auto &point : *points) {
            if (!point.triangulated) continue;
            vector<3> c = r_cw * (point.p - p_wc);
            if (c.z() > 0.2 && std::abs(c.x()) < tan20 * c.z() && std::abs(c.y()) < tan20 * c.z()) {
                depths.push_back(c.z());
            }
        }
        if (depths.size() >= 5) {
            auto mid = depths.begin() + depths.size() / 2;
            std::nth_element(depths.begin(), mid, depths.end());
            out->depth_median = *mid;
        }
    }
}

const char *weee_slam_version(void) { return XRSLAM_VERSION_STRING; }

void weee_slam_reloc_step(WeeeSLAM *slam) {
    // 1. Snapshot under the locks: the latest optimized frame (pose, observed landmarks) of the
    //    current map, its image, the alignment. Everything expensive happens afterwards, unlocked.
    double t = 0;
    matrix<3> R_mc;
    vector<3> p_mc;
    std::vector<cv::Point2f> pixels;
    std::vector<vector<3>> points;
    {
        std::shared_lock<std::shared_mutex> lock(slam->mutex);
        if (slam->detail->get_system_state() != SYS_TRACKING) return;
        if (!optimized_frame(slam, t, R_mc, p_mc, pixels, points)) return;
    }
    cv::Mat image;
    {
        std::lock_guard<std::mutex> lock(slam->image_mutex);
        for (const auto &[time, mat] : slam->recent)
            if (std::abs(time - t) < 0.002) image = mat;
    }
    if (image.empty()) return;

    uint64_t map_id;
    bool aligned, clear;
    double map_start;
    matrix<3> R_sm;
    vector<3> t_sm;
    {
        std::lock_guard<std::mutex> align(slam->align_mutex);
        if (!slam->tracking) return;
        map_id = slam->map_id;
        aligned = slam->aligned;
        map_start = slam->map_start;
        R_sm = slam->R_sm;
        t_sm = slam->t_sm;
        clear = slam->clear_database;
        slam->clear_database = false;
    }
    if (!slam->relocalizer) {
        const auto &c = slam->config;
        slam->relocalizer = std::make_unique<weee::Relocalizer>(cv::Matx33d(c.fx, 0, c.cx, 0, c.fy, c.cy, 0, 0, 1));
    }
    if (clear) {
        slam->relocalizer->clear();
        slam->candidate.reset();
    }

    // 2. Aligned: store keyframes of the session (in the session world).
    if (aligned) {
        if (t - map_start < kKeyframeSettleTime || points.size() < kKeyframeMinLandmarks) return;
        weee::CameraPose pose{R_sm * R_mc, R_sm * p_mc + t_sm};
#ifdef XRSLAM_REPLAY_TRACE
        // Offline replay only: where relocalization would put the camera while tracking normally,
        // i.e. the tracker's drift relative to the stored keyframes.
        static double last_check = 0;
        if (t - last_check > 1 && slam->relocalizer->keyframe_count() > 5) {
            last_check = t;
            int n = 0;
            if (auto found = slam->relocalizer->locate(image, &n)) {
                vector<3> d = pose.p - found->p;
                fprintf(stderr, "RELOCCHECK t=%.3f inliers=%d tracker-minus-reloc %.3f %.3f %.3f\n", t, n, d.x(),
                        d.y(), d.z());
            }
        }
#endif
        for (auto &p : points) p = R_sm * p + t_sm;
        if (slam->relocalizer->add_if_new(image, pose, pixels, points)) {
            std::lock_guard<std::mutex> align(slam->align_mutex);
            if (slam->map_id == map_id) slam->keyframes = int(slam->relocalizer->keyframe_count());
        }
        return;
    }

    // 3. Not aligned: find the camera in the session's keyframes. Too long without: new session.
    if (t - map_start > kRelocTimeout) {
        std::lock_guard<std::mutex> align(slam->align_mutex);
        if (slam->map_id == map_id && !slam->aligned) {
            slam->session_id += 1;
            slam->aligned = true;
            slam->R_sm.setIdentity();
            slam->t_sm.setZero();
            slam->keyframes = 0;
            slam->clear_database = true;
            log_message(XRSLAM_LOG_INFO, "relocalization: no match in %.0f s, new session", kRelocTimeout);
        }
        return;
    }
    int inliers = 0;
    auto found = slam->relocalizer->locate(image, &inliers);
    if (!found) return;
    // session <- map. Both worlds are gravity-aligned (z up), so it must be a rotation about z.
    matrix<3> R = found->R * R_mc.transpose();
    double tilt = std::acos(std::clamp((R * vector<3>::UnitZ()).z(), -1.0, 1.0));
    if (tilt > 5 * M_PI / 180) return;
    double yaw = std::atan2(R(1, 0) - R(0, 1), R(0, 0) + R(1, 1));
    matrix<3> R_yaw = Eigen::AngleAxisd(yaw, vector<3>::UnitZ()).toRotationMatrix();
    vector<3> translation = found->p - R_yaw * p_mc;

    // Accept two agreeing results from different frames within 3 s (a single PnP can be wrong).
    auto &c = slam->candidate;
    bool agrees = c && c->map_id == map_id && t - c->t < 3 && t - c->t > 0.01 &&
                  (c->translation - translation).norm() < 0.10 &&
                  std::abs(std::remainder(c->yaw - yaw, 2 * M_PI)) < 3 * M_PI / 180;
    if (!agrees) {
        c = WeeeSLAM::Candidate{map_id, t, yaw, translation};
        return;
    }
    double mean_yaw = c->yaw + std::remainder(yaw - c->yaw, 2 * M_PI) / 2;
    vector<3> mean_translation = (c->translation + translation) / 2;
    c.reset();
    std::lock_guard<std::mutex> align(slam->align_mutex);
    if (slam->map_id != map_id || slam->aligned) return;
    slam->R_sm = Eigen::AngleAxisd(mean_yaw, vector<3>::UnitZ()).toRotationMatrix();
    slam->t_sm = mean_translation;
    slam->aligned = true;
    slam->relocalizations += 1;
    log_message(XRSLAM_LOG_INFO, "relocalized map %llu (%d inliers, yaw %.1f deg)", (unsigned long long)map_id,
                inliers, mean_yaw * 180 / M_PI);
}

} // extern "C"
