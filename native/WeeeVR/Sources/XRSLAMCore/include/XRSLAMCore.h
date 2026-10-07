// Pure-C interface to XRSLAM (monocular visual-inertial odometry), imported by Swift as `XRSLAMCore`.
//
// Threading: create/reset/push_* must all be called from one serial queue (XRSLAM's IMU/camera
// synchronisation is not locked). get_state may be called from any thread.
//
// Frames: world is gravity-aligned, z up, arbitrary yaw. Camera is OpenCV-style (x right, y down,
// z forward) in the image buffer's orientation. Body = IMU = CoreMotion device frame.
// Units: seconds (host clock, same as CACurrentMediaTime / CMLogItem.timestamp), metres, rad/s,
// m/s^2 with +g reading upward at rest (CoreMotion acceleration * -9.80665).
#ifndef WEEE_XRSLAM_CORE_H
#define WEEE_XRSLAM_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int width, height;              // tracking image size in pixels
    double fx, fy, cx, cy;          // pinhole intrinsics at that size
    double distortion[4];           // radial-tangential k1 k2 p1 p2 (Kalibr pinhole-radtan); zeros = none
    double q_bc[4];                 // camera -> body rotation, x y z w
    double p_bc[3];                 // camera origin in the body frame, metres
    double time_offset;             // seconds added to camera timestamps to align them with the IMU
    int window_size;                // sliding-window keyframes (XRSLAM iOS: 5)
    int tracker_frequent;           // run bundle adjustment every n-th frame (iOS: 3)
    int max_keypoints;              // new keypoints detected per frame (iOS: 300)
    double min_keypoint_distance;   // pixels (iOS: 25)
    int solver_iterations;          // (iOS: 10)
    double solver_time_limit;       // seconds (iOS: 0.1)
} WeeeSLAMConfig;

typedef enum {
    WEEE_SLAM_INITIALIZING = 0,     // collecting parallax; move the camera sideways
    WEEE_SLAM_TRACKING = 1,
    WEEE_SLAM_LOST = 2,             // no pose yet / tracking failed; XRSLAM reinitialises itself
} WeeeSLAMStatus;

typedef struct {
    int status;                     // WeeeSLAMStatus
    int has_pose;                   // 1 when t / q_wc / p_wc / v_wb are valid
    double t;                       // time of the pose (host clock, after time_offset)
    double q_wc[4];                 // camera -> world rotation, x y z w
    double p_wc[3];                 // camera position in world, metres
    double v_wb[3];                 // body velocity in world, m/s
    double bg[3];                   // gyroscope bias estimate, rad/s
    int landmarks;                  // valid landmarks in the sliding window
    double depth_median;            // median camera depth of landmarks within ±20° of the axis, 0 if unknown
    uint64_t images;                // images pushed since create/reset
} WeeeSLAMState;

typedef struct WeeeSLAM WeeeSLAM;

/// Returns NULL on failure (the reason is logged).
WeeeSLAM *weee_slam_create(const WeeeSLAMConfig *config);
void weee_slam_destroy(WeeeSLAM *slam);
/// Drops the map and all state; tracking reinitialises from scratch.
void weee_slam_reset(WeeeSLAM *slam);

void weee_slam_push_gyro(WeeeSLAM *slam, double t, double x, double y, double z);
void weee_slam_push_accel(WeeeSLAM *slam, double t, double x, double y, double z);
/// 8-bit grayscale, `width` x `height` must match the config. The pixels are copied.
void weee_slam_push_image(WeeeSLAM *slam, double t, const uint8_t *gray, int width, int height, int stride);

void weee_slam_get_state(WeeeSLAM *slam, WeeeSLAMState *state);
const char *weee_slam_version(void);

/// Hand models: sample a `size` x `size` RGB crop (planar float, 0..1, R then G then B planes) from a
/// bi-planar 4:2:0 full-range YCbCr image through the affine map from crop pixel (u, v) to image
/// pixel (x, y): x = m[0] u + m[1] v + m[2], y = m[3] u + m[4] v + m[5]. Outside the image is black.
void weee_crop_rgb(const uint8_t *luma, int luma_stride, const uint8_t *chroma, int chroma_stride,
                   int width, int height, const float m[6], int size, float *out);

#ifdef __cplusplus
}
#endif

#endif // WEEE_XRSLAM_CORE_H
