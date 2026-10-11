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

    // Relocalization. Each tracking loss makes XRSLAM start a new map with its own world. While a
    // map is `aligned`, the pose above is in the session world (the world of the session's first
    // map); otherwise it is in the map's own world. A new map is aligned when the camera recognises
    // the session's keyframes (`relocalizations` counts these), or after 20 s without that it starts
    // a new session in its own world (`session_id` changes).
    uint64_t map_id;                // changes with every new XRSLAM map
    uint64_t session_id;            // changes on reset and when a map gives up relocalizing
    int aligned;                    // 1: pose in the session world
    int keyframes;                  // relocalization keyframes stored for the session
    int relocalizations;            // maps aligned to the session by relocalization
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

/// Relocalization work (keyframe capture while aligned, matching while not): call a few times a
/// second from a low-priority thread, never concurrently with destroy. It only briefly takes the
/// locks shared with the other calls; descriptor extraction and matching run outside them.
void weee_slam_reloc_step(WeeeSLAM *slam);

/// Hand models: sample a `size` x `size` RGB crop (planar float, 0..1, R then G then B planes) from a
/// bi-planar 4:2:0 full-range YCbCr image through the affine map from crop pixel (u, v) to image
/// pixel (x, y): x = m[0] u + m[1] v + m[2], y = m[3] u + m[4] v + m[5]. Outside the image is black.
void weee_crop_rgb(const uint8_t *luma, int luma_stride, const uint8_t *chroma, int chroma_stride,
                   int width, int height, const float m[6], int size, float *out);

/// The smart paddle's AprilTag (tagStandard41h12) in a camera image, with its pose.
typedef struct {
    int id;
    int hamming;                    // bits corrected
    float margin;                   // decode margin
    double centre[2];               // undistorted image pixels
    double corners[8];              // undistorted image pixels, AprilTag order (x0 y0 x1 y1 ...)
    int solutions;                  // 1 or 2 (the planar pose ambiguity)
    double rotation[2][9];          // camera <- tag, OpenCV camera frame (x right, y down, z forward)
    double translation[2][3];       // tag centre in the camera frame, metres
    double error[2];                // object-space error of each solution
} WeeeTagDetection;

typedef struct WeeeTagDetector WeeeTagDetector;

WeeeTagDetector *weee_tag_create(int threads);
void weee_tag_destroy(WeeeTagDetector *detector);
/// Looks for the tags `ids` (black-square sizes `sizes`, m) in the `w` x `h` rectangle at (`x0`, `y0`)
/// of an 8-bit image (quads found at 1/`decimate` resolution, decoded at full). Corners are undistorted
/// with the radtan model (`camera` = fx, fy, cx, cy in these pixels; `distortion` = k1, k2, p1, p2)
/// before each pose is estimated. Fills up to `max_out` of `out` (one per ID) and returns how many.
int weee_tag_detect(WeeeTagDetector *detector, const uint8_t *image, int stride, int x0, int y0, int w, int h,
                    float decimate, const int *ids, const double *sizes, int id_count, const double camera[4],
                    const double distortion[4], WeeeTagDetection *out, int max_out);
/// Diagnostics: the IDs of every tag (any ID, this family) the last detect call decoded; returns how
/// many (at most `max` are written).
int weee_tag_last_ids(const WeeeTagDetector *detector, int *ids, int max);

#ifdef __cplusplus
}
#endif

#endif // WEEE_XRSLAM_CORE_H
