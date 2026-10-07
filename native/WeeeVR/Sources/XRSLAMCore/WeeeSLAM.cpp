// C wrapper around XRSLAM's core (XRSLAM::Detail), replacing its singleton C API (xrslam-interface):
// we need reset, velocity and landmarks, and a header Swift can import.
#include "XRSLAMCore.h"

#include <xrslam/common.h>
#include <xrslam/core/detail.h>
#include <xrslam/core/feature_tracker.h>
#include <xrslam/core/frontend_worker.h>
#include <xrslam/extra/opencv_image.h>
#include <xrslam/inspection.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>

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
    std::mutex mutex;                          // guards `detail` against get_state during reset
    std::unique_ptr<XRSLAM::Detail> detail;
    uint64_t images = 0;
    cv::Mat undistortX, undistortY;            // remap tables when the lens has residual distortion
};

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
    std::lock_guard<std::mutex> lock(slam->mutex);
    slam->detail.reset();   // joins the feature-tracker and solver threads
    clear_landmarks();
    slam->detail = std::make_unique<XRSLAM::Detail>(slam->xrslamConfig);
    slam->images = 0;
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
    slam->detail->track_camera(image);
    slam->images += 1;
}

void weee_slam_get_state(WeeeSLAM *slam, WeeeSLAMState *out) {
    *out = WeeeSLAMState{};
    std::lock_guard<std::mutex> lock(slam->mutex);
    out->images = slam->images;

    SysState system = slam->detail->get_system_state();
    auto latest = slam->detail->feature_tracker->get_latest_state();
    if (system == SYS_INITIALIZING) {
        out->status = WEEE_SLAM_INITIALIZING;
    } else if (system == SYS_TRACKING && latest) {
        out->status = WEEE_SLAM_TRACKING;
    } else {
        out->status = WEEE_SLAM_LOST;
    }
    if (!latest) return;

    auto [t, pose, motion] = latest.value();
    const WeeeConfig &config = *slam->xrslamConfig;
    quaternion q_wc = pose.q * config.camera_to_body_rotation();
    vector<3> p_wc = pose.p + pose.q * config.camera_to_body_translation();
    out->has_pose = 1;
    out->t = t;
    out->q_wc[0] = q_wc.x(); out->q_wc[1] = q_wc.y(); out->q_wc[2] = q_wc.z(); out->q_wc[3] = q_wc.w();
    for (int i = 0; i < 3; ++i) {
        out->p_wc[i] = p_wc(i);
        out->v_wb[i] = motion.v(i);
        out->bg[i] = motion.bg(i);
    }

    // Depth proxy: median camera-frame depth of landmarks near the optical axis (±20°).
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

} // extern "C"
